/**
 * C2 core: switch back to a session whose reply completes while the switch-back
 * hydrate is in flight.
 *
 * Session B streams (provider held) -> user opens A -> user returns to B. The
 * return issues a REST page read (/api/sessions/B/messages) that races the
 * live message.complete. Both orders are forced deterministically: the REST
 * read is parked in main (ipcMain 'hermes:api' wrapper) until the test opens
 * it, and the provider stream is parked on a gate. No sleeps.
 *
 * complete-before-hydrate was red on base (reply rendered twice: committed row
 * plus a live `assistant-stream-*` row, sometimes frozen on its first chunk);
 * fixed in "render a reply once when it completes during the switch-back
 * hydrate". Oracle: every message exactly once, DOM == persisted, at every
 * sampled frame.
 *
 * Each order also runs with a tool turn (narration, a tool, then the held
 * answer). History folds that turn into one bubble keyed by its first row
 * while the live reply is keyed by its last (#128809), and the folded text
 * carries the narration too. complete-before-hydrate (tool turn) was red on
 * base (the settled live reply painted beside the folded bubble); the overlay
 * now matches the reply's stored row instead of its words.
 */

import { expect, type Page, test } from '@playwright/test'

import {
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  send,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { assertTranscriptOracle, installDuplicateSampler, type OracleTarget } from './oracle'
import { gate, startScriptedProvider } from './provider'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`
const AI = (n: number) => `A${n}i-${nonce}`

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

const cases = (['complete-before-hydrate', 'hydrate-before-complete'] as const).flatMap(order =>
  (['reply', 'tool turn'] as const).map(shape => ({ order, shape }))
)

for (const { order, shape } of cases) {
  test(`switch back while away session completes: ${order}${shape === 'reply' ? '' : ` (${shape})`}`, async () => {
    const provider = await startScriptedProvider()
    const sandbox = createCoreSandbox('race')
    writeProviderHome(sandbox.hermesHome, provider.url)
    const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
    const ws = recordWebSockets(page)

    try {
      await waitForInteractive(app, page)
      await installDuplicateSampler(page)
      provider.script(U(1), [{ text: [`${A(1)} `, 'one'] }])
      await send(page, `${U(1)} a`, 'Enter', ws)
      await expect.poll(() => provider.completions.some(c => c.marker === U(1) && c.finished)).toBe(true)
      await expect.poll(() => currentSessionId(page)).not.toBe('')
      const a = await currentSessionId(page)
      await expect(viewport(page)).toContainText(A(1))

      await page.evaluate(() => {
        window.location.hash = '#/'
      })
      await expect.poll(() => currentSessionId(page)).toBe('')
      const hold = gate()
      const answer = { text: [`${A(2)} `, 'finished ', 'while ', 'away'], holdAfterFirstChunk: hold }
      provider.script(
        U(2),
        shape === 'reply'
          ? [answer]
          : [
              {
                text: [`${AI(2)} `, 'checking'],
                toolCalls: [{ name: 'terminal', args: { command: 'echo core-race-tool' } }]
              },
              answer
            ]
      )
      await send(page, `${U(2)} b`, 'Enter', ws)
      await provider.streamStarted(U(2), shape === 'reply' ? 0 : 1)
      await expect.poll(() => currentSessionId(page)).not.toBe('')
      const b = await currentSessionId(page)
      await expect(viewport(page)).toContainText(A(2))

      await page.evaluate(id => {
        window.location.hash = `#/${id}`
      }, a)
      await expect(viewport(page)).toContainText(A(1))

      {
        const wrapped = await app.evaluate(({ ipcMain }, b) => {
          const g = globalThis as any
          g.__gated = []
          g.__gateOpen = false
          g.__gateWaiters = [] as (() => void)[]
          // Electron keeps invoke handlers in a private map; if that ever moves,
          // fail loudly here rather than silently not gating.
          const handlers = (ipcMain as any)._invokeHandlers as Map<string, (...args: any[]) => Promise<any>> | undefined
          const original = handlers?.get('hermes:api')

          if (!original) {
            return false
          }

          ipcMain.removeHandler('hermes:api')
          ipcMain.handle('hermes:api', async (event: any, request: any) => {
            const p = String(request?.path ?? '')

            if (p.includes(`/api/sessions/${b}`) && !g.__gateOpen) {
              g.__gated.push(p)
              await new Promise<void>(resolve => g.__gateWaiters.push(resolve))
            }

            return original(event, request)
          })

          return true
        }, b)

        expect(wrapped, 'hermes:api handler wrapped for gating').toBe(true)
      }

      await page.evaluate(id => {
        window.location.hash = `#/${id}`
      }, b)
      await expect.poll(() => currentSessionId(page)).toBe(b)

      const openGate = () =>
        app.evaluate(() => {
          const g = globalThis as any
          g.__gateOpen = true

          for (const w of g.__gateWaiters) {
            w()
          }

          return g.__gated
        })

      const completed = () =>
        ws.events.some(e => e.type === 'message.complete' && String(e.payload?.text ?? '').startsWith(A(2)))

      const gatedCount = () => app.evaluate(() => ((globalThis as any).__gated ?? []).length as number)

      if (order === 'complete-before-hydrate') {
        await expect.poll(gatedCount).toBeGreaterThan(0)
        hold.open()
        await expect.poll(completed).toBe(true)
        await openGate()
      } else if (order === 'hydrate-before-complete') {
        await expect.poll(gatedCount).toBeGreaterThan(0)
        await openGate()
        hold.open()
        await expect.poll(completed).toBe(true)
      }

      const target: OracleTarget = { sessionId: b, expectUserMarkers: [U(2)] }
      await assertTranscriptOracle(page, ws, provider, target, `switch-back race ${order}`)
    } finally {
      await app.close().catch(() => undefined)
      await provider.close()
      sandbox.cleanup()
    }
  })
}
