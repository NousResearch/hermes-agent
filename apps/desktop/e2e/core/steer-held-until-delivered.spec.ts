/**
 * C2 core: a steer waits under the reply the model is still writing.
 *
 * With `busy_input_mode: steer`, a message sent while the model thinks is
 * accepted at once but joins the conversation only after the turn's next tool
 * batch, where the agent appends it. Until then the model keeps thinking and
 * calls the tool it was already heading for. That reasoning and tool belong to
 * one assistant bubble above the steer; only the reply to the steer goes below
 * it. Sealing at arrival split the thinking block around the steer bubble.
 *
 * The transcript oracle leaves reasoning out of its order check, so this spec
 * asserts the bubble structure directly: while the tool runs (steer held),
 * after `steer.delivered`, and after a reload (durable order).
 */

import fs from 'node:fs'
import path from 'node:path'

import { expect, type Page, test } from '@playwright/test'

import {
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  send,
  waitForInteractive,
  writeProviderHome,
  type WsRecorder
} from './harness'
import { assertTranscriptOracle, installDuplicateSampler, type OracleTarget } from './oracle'
import { shellQuote } from './process-census'
import { gate, startScriptedProvider } from './provider'

const STEER_MODE = `display:
  busy_input_mode: steer
`

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`
const R = (n: number) => `R${n}-${nonce}`

const words = (marker: string, ...rest: string[]) => [
  `${marker} `,
  ...rest.map((w, i) => (i === rest.length - 1 ? w : `${w} `))
]

const shots = process.env.HERMES_E2E_SHOTS_DIR

async function shot(page: Page, name: string) {
  if (shots) {
    fs.mkdirSync(shots, { recursive: true })
    await page.screenshot({ path: path.join(shots, `${name}.png`) })
  }
}

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

/** Bubbles top to bottom: role, and whether the bubble carries a tool row. */
function bubbles(page: Page) {
  return page.evaluate(() => {
    const view = ([...document.querySelectorAll('[data-slot="aui_thread-viewport"]')] as HTMLElement[]).find(
      el => el.offsetParent !== null
    )

    return [...(view?.querySelectorAll('[data-slot="aui_user-message-root"], [data-slot="aui_assistant-message-root"]') ?? [])].map(
      el => ({
        role: el.getAttribute('data-slot') === 'aui_user-message-root' ? 'user' : 'assistant',
        text: (el.textContent ?? '').replace(/\s+/g, ' ').trim()
      })
    )
  })
}

/** The running turn's working indicator: an assistant root with no content but
 *  the elapsed time, drawn under a trailing user row. Not a reply bubble. */
const isWorkingIndicator = (row: { role: string; text: string }, last: boolean) =>
  last && row.role === 'assistant' && /^((\d+m\s*)?\d+s)?$/.test(row.text)

/** Compact layout: `role` plus the markers it shows (`tool` = carries the held tool's output). */
async function layout(page: Page, markers: string[], toolTag: string) {
  const rows = await bubbles(page)

  return rows
    .filter((row, i) => !isWorkingIndicator(row, i === rows.length - 1))
    .map(({ role, text }) => {
      const tags = markers.filter(marker => text.includes(marker))

      if (role === 'assistant' && text.includes(toolTag)) {
        tags.push('tool')
      }

      return tags.length ? `${role}:${tags.join('+')}` : role
    })
}

async function settled(page: Page, ws: WsRecorder, marker: string) {
  await expect
    .poll(() => ws.events.some(e => e.type === 'message.complete' && String(e.payload?.text ?? '').includes(marker)), {
      timeout: 120_000,
      message: `message.complete for ${marker}`
    })
    .toBe(true)
  await expect(page.locator('[data-slot="composer-root"] button[aria-label="Stop"]')).toHaveCount(0, {
    timeout: 30_000
  })
}

test('a steer sent while the model thinks stays below that thinking until the model reads it', async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('steer-held')
  writeProviderHome(sandbox.hermesHome, provider.url, STEER_MODE)
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws = recordWebSockets(page)

  try {
    await waitForInteractive(app, page)
    await installDuplicateSampler(page)
    const session: OracleTarget = { sessionId: '', expectUserMarkers: [] }
    const thinking = gate()
    const ready = path.join(sandbox.root, `held-tool-running-${nonce}`)
    const release = path.join(sandbox.root, `held-tool-release-${nonce}`)
    const toolTag = `held-tool-running-${nonce}`
    const markers = [U(1), U(2), R(1), A(2)]

    // Turn 1: think, then call a tool that blocks until the test releases it.
    provider.script(U(1), [
      {
        reasoning: words(R(1), 'weighing', 'option', 'A', 'then', 'option', 'B'),
        toolCalls: [
          {
            name: 'terminal',
            args: {
              command: `touch ${shellQuote(ready)}; while [ ! -e ${shellQuote(release)} ]; do sleep 0.05; done; echo core-held`
            }
          }
        ],
        holdAfterFirstChunk: thinking
      }
    ])
    // After the batch the steer is the newest user row: its reply is the next call.
    provider.script(U(2), [{ text: words(A(2), 'going', 'with', 'B') }])

    await test.step('steer while the model is still thinking', async () => {
      await send(page, `${U(1)} pick an option`, 'Enter', ws)
      await provider.streamStarted(U(1))
      await send(page, `${U(2)} prefer B`, 'Enter', ws)
      await expect
        .poll(() => ws.sent.some(m => m.method === 'session.steer'), { message: 'Enter while busy steered' })
        .toBe(true)
      session.sessionId = await currentSessionId(page)
      session.expectUserMarkers.push(U(1), U(2))
    })

    await test.step('the thinking and the tool it leads to stay one bubble above the held steer', async () => {
      thinking.open()
      await expect.poll(() => fs.existsSync(ready), { timeout: 60_000, message: 'tool started' }).toBe(true)
      await expect(viewport(page)).toContainText(U(2))
      // The steer is still unread: no reply bubble below it yet.
      await expect
        .poll(() => layout(page, markers, toolTag), { timeout: 30_000 })
        .toEqual([`user:${U(1)}`, expect.stringMatching(/^assistant:(.+\+)?tool$/), `user:${U(2)}`])
      expect(ws.events.some(e => e.type === 'steer.delivered')).toBe(false)
      await shot(page, '1-steer-held-while-tool-runs')
    })

    await test.step('the reply to the steer streams below it once delivered', async () => {
      fs.writeFileSync(release, '')
      await settled(page, ws, A(2))
      expect(ws.events.some(e => e.type === 'steer.delivered')).toBe(true)
      expect(await layout(page, markers, toolTag)).toEqual([
        `user:${U(1)}`,
        expect.stringMatching(/^assistant:(.+\+)?tool$/),
        `user:${U(2)}`,
        `assistant:${A(2)}`
      ])
      await shot(page, '2-after-delivery')
      await assertTranscriptOracle(page, ws, provider, session, 'steer held until delivered')
    })

    await test.step('reload paints the same order from history', async () => {
      await page.reload()
      await expect(viewport(page)).toContainText(A(2), { timeout: 60_000 })
      await installDuplicateSampler(page)
      await assertTranscriptOracle(page, ws, provider, session, 'held steer after reload')
      await shot(page, '3-after-reload')
    })
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
})
