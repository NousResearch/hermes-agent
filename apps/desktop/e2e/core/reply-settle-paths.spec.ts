/**
 * C2 core: the timings that turn one reply into two bubbles.
 *
 * One stored reply reaches the renderer twice: as live events and again in
 * the history the app fetches on reload, switch-back or reconnect. Each path
 * that merges the two decides on its own that they are the same message, so a
 * reply duplicates only when a turn's shape and timing reach a path that gets
 * that decision wrong. transcript-integrity walks each transition once. These
 * steps force the shapes the duplicate-reply reports describe, with gates,
 * and hold every one to the same oracle:
 *
 * - an answer the model sent before a housekeeping tool, which the backend
 *   repeats as the final reply (interim + identical final), with and without
 *   a review.summary row delivered before that turn's message.complete (the
 *   #131626 report: a slow end-of-turn flush lets the review fork's row win);
 * - a steer at each point of a tool turn: reasoning, the narration before a
 *   tool, the running tool, the answer after it, and twice in a row;
 * - then a reload and a switch away and back, where the tool turns fold into
 *   fewer bubbles than the live view drew.
 */

import fs from 'node:fs'
import path from 'node:path'

import { type ElectronApplication, expect, type Page, test } from '@playwright/test'

import {
  coreAppEnv,
  type CoreSandbox,
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
import { gate, type ScriptedProvider, startScriptedProvider } from './provider'

// Review after every turn, so a review fork is in flight during the next one.
const REVIEW_EVERY_TURN = `memory:
  nudge_interval: 1
`

const REVIEW_PROMPT = 'Review the conversation above'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`
const AI = (n: number) => `A${n}i-${nonce}`
const R = (n: number) => `R${n}-${nonce}`

const words = (marker: string, ...rest: string[]) => [
  `${marker} `,
  ...rest.map((w, i) => (i === rest.length - 1 ? w : `${w} `))
]

const echo = (tag: string) => ({ name: 'terminal', args: { command: `echo core-${tag}` } })

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

function stopButton(page: Page) {
  return page.locator('[data-slot="composer-root"] button[aria-label="Stop"]')
}

function finished(provider: ScriptedProvider, marker: string, step = 0) {
  return expect
    .poll(() => provider.completions.some(c => c.marker === marker && c.step === step && c.finished), {
      timeout: 120_000,
      message: `provider finished ${marker}#${step}`
    })
    .toBe(true)
}

async function settled(page: Page, ws: WsRecorder, marker: string) {
  await expect
    .poll(() => ws.events.some(e => e.type === 'message.complete' && String(e.payload?.text ?? '').includes(marker)), {
      timeout: 120_000,
      message: `message.complete for ${marker}`
    })
    .toBe(true)
  await expect(stopButton(page)).toHaveCount(0, { timeout: 30_000 })
}

/** Session switch through the route, the way the sidebar does it. */
async function openSession(page: Page, sessionId: string, waitFor: string) {
  await page.evaluate(id => {
    window.location.hash = id ? `#/${id}` : '#/'
  }, sessionId)
  await expect.poll(() => currentSessionId(page)).toBe(sessionId)

  if (waitFor) {
    await expect(viewport(page)).toContainText(waitFor, { timeout: 30_000 })
  }
}

async function withApp(
  label: string,
  extraYaml: string,
  body: (ctx: {
    app: ElectronApplication
    page: Page
    provider: ScriptedProvider
    sandbox: CoreSandbox
    ws: WsRecorder
  }) => Promise<void>
) {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox(label)
  writeProviderHome(sandbox.hermesHome, provider.url, extraYaml)
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws = recordWebSockets(page)

  try {
    await waitForInteractive(app, page)
    await installDuplicateSampler(page)
    await body({ app, page, provider, sandbox, ws })
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
}

/**
 * Pass the primary socket through the test, holding back the first server
 * frame `hold` matches until a frame `until` matches has been delivered: the
 * order a slow end-of-turn flush produces on the wire. Routed sockets are fed
 * into `ws` here (the recorder dedupes by session + seq).
 */
async function routeHeldFrame(page: Page, ws: WsRecorder) {
  const ctl = {
    hold: null as null | ((msg: any) => boolean),
    until: null as null | ((msg: any) => boolean),
    held: [] as string[],
    released: false
  }

  const parse = (raw: string | Buffer) => {
    try {
      return JSON.parse(String(raw))
    } catch {
      return null
    }
  }

  await page.routeWebSocket(/\/api\/ws/, client => {
    const server = client.connectToServer()
    const socket = ws.sockets.push({ id: ws.sockets.length, url: client.url(), closed: false }) - 1

    client.onMessage(raw => {
      const msg = parse(raw)

      if (typeof msg?.method === 'string') {
        ws.sent.push({ socket, method: msg.method, params: msg.params })
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
        })
      }

      if (ctl.hold?.(msg) && ctl.held.length === 0) {
        ctl.held.push(String(raw))

        return
      }

      client.send(raw)

      if (ctl.held.length && ctl.until?.(msg)) {
        ctl.hold = ctl.until = null
        ctl.held.splice(0).forEach(frame => client.send(frame))
        ctl.released = true
      }
    })
  })

  return ctl
}

const eventOf = (type: string, text?: string) => (msg: any) =>
  msg?.method === 'event' &&
  msg.params?.type === type &&
  (text === undefined || String(msg.params.payload?.text ?? '').includes(text))

test('an answer repeated as the final reply renders once, with or without a row between', async () => {
  await withApp('repeat-final', REVIEW_EVERY_TURN, async ({ app, page, provider, ws }) => {
    const session: OracleTarget = { sessionId: '', expectUserMarkers: [] }
    const frames = await routeHeldFrame(page, ws)
    await page.reload()
    await waitForInteractive(app, page)
    await installDuplicateSampler(page)
    const rows = page.locator('[data-slot="aui_system-message-root"]')

    // Every review fork saves one memory, so it publishes a review.summary row.
    provider.scriptPrompt('review', REVIEW_PROMPT, [
      { toolCalls: [{ name: 'memory', args: { action: 'add', target: 'memory', content: `core e2e ${nonce}` } }] },
      { text: ['Saved.'] }
    ])

    // The answer arrives beside a housekeeping tool and the follow-up is empty,
    // so the backend reuses the answer as the final reply: one stored answer,
    // sent as message.interim and again as message.complete.
    const repeatedFinal = (n: number) =>
      provider.script(U(n), [
        {
          text: words(A(n), 'answered', 'before', 'tidying'),
          toolCalls: [
            { name: 'memory', args: { action: 'add', target: 'memory', content: `core e2e turn ${n} ${nonce}` } }
          ]
        },
        { text: [] }
      ])

    await test.step('interim answer + identical final settle into one bubble', async () => {
      repeatedFinal(1)
      await send(page, `${U(1)} answer then tidy`, 'Enter', ws)
      await settled(page, ws, A(1))
      session.sessionId = await currentSessionId(page)
      session.expectUserMarkers.push(U(1))
      await assertTranscriptOracle(page, ws, provider, session, 'interim + identical final')
      // Let this turn's review publish its row before the next prompt supersedes it.
      await expect(rows).toHaveCount(1, { timeout: 60_000 })
    })

    await test.step('a review row delivered before the final', async () => {
      // The review fork starts before the turn's message.complete is sent; when
      // the end-of-turn flush is slow its row reaches the client first (#131626).
      frames.hold = eventOf('message.complete', A(2))
      frames.until = eventOf('review.summary')
      repeatedFinal(2)
      await send(page, `${U(2)} answer then tidy again`, 'Enter', ws)
      await expect
        .poll(() => frames.released, { timeout: 120_000, message: 'review row before message.complete' })
        .toBe(true)
      await expect(rows).toHaveCount(2, { timeout: 30_000 })
      await settled(page, ws, A(2))
      session.expectUserMarkers.push(U(2))
      await assertTranscriptOracle(page, ws, provider, session, 'review row before the final')
    })
  })
})

test('a steer at every point of a tool turn, then reload and switch back', async () => {
  await withApp('steer-matrix', '', async ({ page, provider, sandbox, ws }) => {
    const session: OracleTarget = { sessionId: '', expectUserMarkers: [] }

    const steer = async (label: string, turn: number, by: number, onAir: () => Promise<void>) => {
      provider.script(U(by), [{ text: words(A(by), 'steered', 'after', label) }])
      await send(page, `${U(turn)} ${label}`, 'Enter', ws)
      await onAir()
      await send(page, `${U(by)} change course`, 'Enter', ws)
      await finished(provider, U(by))
      await settled(page, ws, A(by))
      session.sessionId ||= await currentSessionId(page)
      session.expectUserMarkers.push(U(turn), U(by))
      await assertTranscriptOracle(page, ws, provider, session, `steer during ${label}`)
    }

    await test.step('steer while reasoning streams', async () => {
      const hold = gate()
      provider.script(U(1), [
        { reasoning: words(R(1), 'weighing'), text: words(A(1), 'never'), holdAfterFirstChunk: hold }
      ])
      await steer('reasoning', 1, 2, () => provider.streamStarted(U(1)))
      hold.open()
    })

    await test.step('steer while the narration before a tool streams', async () => {
      const hold = gate()
      provider.script(U(3), [
        { text: words(AI(3), 'checking', 'first'), toolCalls: [echo('narration')], holdAfterFirstChunk: hold },
        { text: words(A(3), 'unreached') }
      ])
      await steer('narration', 3, 4, async () => {
        await provider.streamStarted(U(3))
        await expect(viewport(page)).toContainText(AI(3))
      })
      hold.open()
    })

    await test.step('steer while the tool runs', async () => {
      const ready = path.join(sandbox.root, `tool-running-${nonce}`)
      const release = path.join(sandbox.root, `tool-release-${nonce}`)
      provider.script(U(5), [
        {
          text: words(AI(5), 'running', 'a', 'tool'),
          toolCalls: [
            {
              name: 'terminal',
              args: { command: `touch ${ready}; while [ ! -e ${release} ]; do sleep 0.05; done; echo core-ran` }
            }
          ]
        },
        { text: words(A(5), 'tool', 'finished') }
      ])
      await steer('tool', 5, 6, () =>
        expect.poll(() => fs.existsSync(ready), { timeout: 60_000, message: 'tool started' }).toBe(true)
      )
      // The steer hands the running command to the background; finishing it
      // starts a notification turn (a process_complete row, not a user bubble).
      provider.scriptPrompt('process', 'Background process', [{ text: words(A(20), 'noted', 'the', 'process') }])
      fs.writeFileSync(release, '')
      await settled(page, ws, A(20))
      await assertTranscriptOracle(page, ws, provider, session, 'background process after a steered tool')
    })

    await test.step('steer while the answer after a tool streams', async () => {
      const hold = gate()
      provider.script(U(7), [
        { text: words(AI(7), 'checking', 'first'), toolCalls: [echo('answer')] },
        { text: words(A(7), 'answer', 'after', 'tool'), holdAfterFirstChunk: hold }
      ])
      await steer('answer', 7, 8, async () => {
        await provider.streamStarted(U(7), 1)
        await expect(viewport(page)).toContainText(A(7))
      })
      hold.open()
    })

    await test.step('two steers back to back', async () => {
      const first = gate()
      const second = gate()
      provider.script(U(9), [{ text: words(A(9), 'slow', 'reply'), holdAfterFirstChunk: first }])
      provider.script(U(10), [{ text: words(A(10), 'first', 'steer'), holdAfterFirstChunk: second }])
      await steer('a steer', 9, 11, async () => {
        await provider.streamStarted(U(9))
        await expect(viewport(page)).toContainText(A(9))
        await send(page, `${U(10)} first steer`, 'Enter', ws)
        session.expectUserMarkers.push(U(10))
        await provider.streamStarted(U(10))
        await expect(viewport(page)).toContainText(A(10))
      })
      first.open()
      second.open()
    })

    await test.step('a finished two-tool turn, then reload', async () => {
      provider.script(U(12), [
        { text: words(AI(12), 'looking'), toolCalls: [echo('fold-a')] },
        { toolCalls: [echo('fold-b')] },
        { text: words(A(12), 'folded', 'answer') }
      ])
      await send(page, `${U(12)} two tools`, 'Enter', ws)
      await settled(page, ws, A(12))
      session.expectUserMarkers.push(U(12))
      await assertTranscriptOracle(page, ws, provider, session, 'tool turn live')
      await page.reload()
      await expect(viewport(page)).toContainText(A(12), { timeout: 60_000 })
      await installDuplicateSampler(page)
      await assertTranscriptOracle(page, ws, provider, session, 'tool turn after reload')
    })

    await test.step('switch away and back', async () => {
      await openSession(page, '', '')
      await openSession(page, session.sessionId, A(12))
      await assertTranscriptOracle(page, ws, provider, session, 'switch back')
    })
  })
})
