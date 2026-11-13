/**
 * A socket drop while the answer after a tool streams must not leave the reply
 * on its first chunk.
 *
 * The reconnect replay (`session.events.since`) is answered before
 * `session.resume` rebinds the new socket, so a turn that commits in between
 * sends its tail and message.complete to the dropped socket. This spec forces
 * that window: it holds the reconnect's `session.resume`, releases the tail
 * right after the replay answer, and lets the turn finish while the resume is
 * still held. Only the idle resume answer can then tell the view the reply is
 * durable, so the full reply must render promptly after it, not after a
 * periodic session sweep tens of seconds later.
 *
 * Real Electron + real gateway; only the LLM is scripted.
 */

import { expect, type Page, test } from '@playwright/test'

import {
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  routePrimaryWebSocket,
  send,
  startTcpProxy,
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

const words = (marker: string, ...rest: string[]) => [
  `${marker} `,
  ...rest.map((w, i) => (i === rest.length - 1 ? w : `${w} `))
]

const viewport = (page: Page) => page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
const stopButton = (page: Page) => page.locator('[data-slot="composer-root"] button[aria-label="Stop"]')

const RESUME_HOLD_MS = 1500
const FINISH_DELAY_MS = 800

const parse = (raw: string | Buffer) => {
  try {
    return JSON.parse(String(raw))
  } catch {
    return null
  }
}

test('a reply that commits between the reconnect replay and the resume renders in full', async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('reply-tail')
  writeProviderHome(sandbox.hermesHome, provider.url)
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws = recordWebSockets(page)
  const session: OracleTarget = { sessionId: '', expectUserMarkers: [] }

  // After the drop: hold the first resume, and run `afterReplay` once the
  // reconnect's replay request is answered.
  const gap = { armed: false, held: false, replayIds: new Set<unknown>(), afterReplay: null as null | (() => void) }

  try {
    await waitForInteractive(app, page)
    provider.script(U(1), [{ text: words(A(1), 'first', 'answer') }])
    await send(page, `${U(1)} warm up`, 'Enter', ws)
    await expect(viewport(page)).toContainText(`${A(1)} first answer`, { timeout: 60_000 })
    await expect(stopButton(page)).toHaveCount(0, { timeout: 60_000 })
    session.sessionId = await currentSessionId(page)
    session.expectUserMarkers.push(U(1))

    const backendPort = Number(new URL(ws.sockets[0]!.url).port)
    const proxy = await startTcpProxy(backendPort)

    try {
      await page.routeWebSocket(/\/api\/ws/, client => {
        const server = client.connectToServer()
        const socket = ws.sockets.push({ id: ws.sockets.length, url: client.url(), closed: false }) - 1

        client.onMessage(raw => {
          const msg = parse(raw)

          if (typeof msg?.method === 'string') {
            ws.sent.push({ socket, method: msg.method, params: msg.params })
          }

          if (gap.armed && msg?.method === 'session.events.since') {
            gap.replayIds.add(msg.id)
          }

          if (gap.armed && !gap.held && (msg?.method === 'session.resume' || msg?.method === 'session.activate')) {
            gap.held = true
            setTimeout(() => server.send(raw), RESUME_HOLD_MS)

            return
          }

          server.send(raw)
        })
        server.onMessage(raw => {
          const msg = parse(raw)

          if (msg?.method === 'event' && msg.params) {
            ws.events.push({
              socket,
              type: String(msg.params.type ?? ''),
              sessionId: String(msg.params.session_id ?? ''),
              seq: typeof msg.params.seq === 'number' ? msg.params.seq : null,
              payload: msg.params.payload
            } as (typeof ws.events)[number])
          }

          client.send(raw)

          if (msg && gap.replayIds.has(msg.id) && gap.afterReplay) {
            const release = gap.afterReplay
            gap.afterReplay = null
            release()
          }
        })
        client.onClose(() => {
          ws.sockets[socket]!.closed = true
          server.close()
        })
        server.onClose(() => {
          ws.sockets[socket]!.closed = true
          client.close()
        })
      })
      await routePrimaryWebSocket(app, backendPort, proxy.port)
      await page.reload()
      await waitForInteractive(app, page)
      await installDuplicateSampler(page)
      await expect.poll(() => proxy.connections()).toBeGreaterThan(0)
      await page.evaluate(id => {
        window.location.hash = `#/${id}`
      }, session.sessionId)
      await expect(viewport(page)).toContainText(A(1), { timeout: 30_000 })

      const tail = gate()
      const finish = gate()
      provider.script(U(2), [
        {
          text: words(AI(2), 'narrating', 'before', 'the', 'drop'),
          toolCalls: [{ name: 'terminal', args: { command: 'echo core-reply-tail' } }]
        },
        { text: words(A(2), 'answer', 'across', 'the', 'drop'), holdAfterFirstChunk: tail, holdBeforeFinish: finish }
      ])
      void tail.opened.then(() => setTimeout(finish.open, FINISH_DELAY_MS))
      await send(page, `${U(2)} across a drop`, 'Enter', ws)
      await provider.streamStarted(U(2), 1)
      await expect(viewport(page)).toContainText(A(2))
      session.expectUserMarkers.push(U(2))

      gap.afterReplay = tail.open
      gap.armed = true
      const socketsBefore = ws.sockets.length
      expect(proxy.dropAll()).toBeGreaterThan(0)
      await expect.poll(() => ws.sockets.length, { timeout: 60_000 }).toBeGreaterThan(socketsBefore)
      await expect.poll(() => gap.held, { timeout: 60_000, message: 'reconnect resume held' }).toBe(true)
      await tail.opened

      // Resume hold + finish delay + margin: well under the periodic sweep that
      // otherwise repairs the view 20-50 s later.
      await expect(viewport(page)).toContainText(`${A(2)} answer across the drop`, { timeout: 10_000 })
      await expect(stopButton(page)).toHaveCount(0, { timeout: 30_000 })
      await page.screenshot({ path: test.info().outputPath('reply-after-reconnect.png') })
      session.lossyWire = [U(2)]
      await assertTranscriptOracle(page, ws, provider, session, 'commit between replay and resume')
    } finally {
      await proxy.close()
    }
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
})
