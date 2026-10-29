/**
 * C2 core: a window that re-attaches to a turn that is still running.
 *
 * The turn persists its narrated tool round before the tool runs, so the
 * history a re-attaching window fetches already holds that round while the
 * live stream keeps drawing the same segment (#127288, #127665). Each step
 * leaves the chat mid-turn by a different road (switch away and back during
 * the answer or the running tool, a reload, a steer, a socket drop) and holds
 * the settled turn to the transcript oracle, whose in-page sampler also fails a
 * transient double. The last two specs re-send frames the backend already
 * delivered under new seqs; the turn's wire identity (turn_id, segment, offset)
 * is what lets the renderer upsert them.
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
  routePrimaryWebSocket,
  send,
  startTcpProxy,
  waitForInteractive,
  writeProviderHome,
  type WsRecorder
} from './harness'
import { assertTranscriptOracle, installDuplicateSampler, type OracleTarget } from './oracle'
import { shellQuote } from './process-census'
import { gate, type ScriptedProvider, startScriptedProvider } from './provider'

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

const echo = (tag: string) => ({ name: 'terminal', args: { command: `echo core-${tag}` } })

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

function stopButton(page: Page) {
  return page.locator('[data-slot="composer-root"] button[aria-label="Stop"]')
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
  writeProviderHome(sandbox.hermesHome, provider.url)
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

test('re-attaching to a running tool turn renders each segment once', async () => {
  await withApp('reattach', async ({ app, page, provider, sandbox, ws }) => {
    const session: OracleTarget = { sessionId: '', expectUserMarkers: [] }

    await test.step('switch away and back while the answer after a tool streams', async () => {
      const hold = gate()
      provider.script(U(1), [
        { text: words(AI(1), 'narrating', 'before', 'the', 'tool'), toolCalls: [echo('reattach-a')] },
        { text: words(A(1), 'answer', 'after', 'the', 'tool'), holdAfterFirstChunk: hold }
      ])
      await send(page, `${U(1)} narrate then answer`, 'Enter', ws)
      await provider.streamStarted(U(1), 1)
      await expect(viewport(page)).toContainText(A(1))
      session.sessionId = await currentSessionId(page)
      session.expectUserMarkers.push(U(1))
      await openSession(page, '', '')
      await openSession(page, session.sessionId, AI(1))
      hold.open()
      await settled(page, ws, A(1))
      await assertTranscriptOracle(page, ws, provider, session, 're-attach during the answer')
    })

    await test.step('switch away and back while the tool runs', async () => {
      const ready = path.join(sandbox.root, `reattach-running-${nonce}`)
      const release = path.join(sandbox.root, `reattach-release-${nonce}`)
      provider.script(U(2), [
        {
          text: words(AI(2), 'running', 'a', 'slow', 'tool'),
          toolCalls: [
            {
              name: 'terminal',
              args: {
                command: `touch ${shellQuote(ready)}; while [ ! -e ${shellQuote(release)} ]; do sleep 0.05; done; echo core-ran`
              }
            }
          ]
        },
        { text: words(AI(3), 'second', 'round'), toolCalls: [echo('reattach-b')] },
        { text: words(A(2), 'answer', 'after', 'two', 'rounds') }
      ])
      await send(page, `${U(2)} slow tool`, 'Enter', ws)
      await expect.poll(() => fs.existsSync(ready), { timeout: 60_000, message: 'tool started' }).toBe(true)
      session.expectUserMarkers.push(U(2))
      await openSession(page, '', '')
      await openSession(page, session.sessionId, AI(2))
      fs.writeFileSync(release, '')
      await settled(page, ws, A(2))
      await assertTranscriptOracle(page, ws, provider, session, 're-attach during the tool')
    })

    await test.step('reload while the answer after a tool streams', async () => {
      const hold = gate()
      const finish = gate()
      provider.script(U(4), [
        { text: words(AI(4), 'narrating', 'again'), toolCalls: [echo('reattach-c')] },
        { text: words(A(4), 'answer', 'across', 'a', 'reload'), holdAfterFirstChunk: hold, holdBeforeFinish: finish }
      ])
      await send(page, `${U(4)} reload mid answer`, 'Enter', ws)
      await provider.streamStarted(U(4), 1)
      await expect(viewport(page)).toContainText(A(4))
      session.expectUserMarkers.push(U(4))
      await page.reload()
      await expect(viewport(page)).toContainText(AI(4), { timeout: 60_000 })
      await installDuplicateSampler(page)
      hold.open()

      // Before the turn completes, the rest of the answer extends the reply the
      // reloaded window restored: one bubble, not a second one beside it.
      const answer = viewport(page)
        .locator('[data-slot="aui_assistant-message-root"]')
        .filter({ hasText: 'across a reload' })

      await expect(answer).toHaveCount(1, { timeout: 60_000 })
      await page.screenshot({ path: test.info().outputPath('reload-mid-turn.png') })
      await expect(answer).toContainText(A(4))
      finish.open()
      await settled(page, ws, A(4))
      await assertTranscriptOracle(page, ws, provider, session, 're-attach after a reload')
    })
    await test.step('a steer during the answer, then switch away and back mid-turn', async () => {
      const hold = gate()
      const steerHold = gate()
      provider.script(U(5), [
        { text: words(AI(5), 'narrating', 'before', 'a', 'steer'), toolCalls: [echo('reattach-s')] },
        { text: words(A(5), 'answer', 'interrupted', 'by', 'steer'), holdAfterFirstChunk: hold }
      ])
      provider.script(U(6), [
        { text: words(AI(6), 'steered', 'narration'), toolCalls: [echo('reattach-s2')] },
        { text: words(A(6), 'steered', 'answer'), holdAfterFirstChunk: steerHold }
      ])
      await send(page, `${U(5)} steer me`, 'Enter', ws)
      await provider.streamStarted(U(5), 1)
      await expect(viewport(page)).toContainText(A(5))
      await send(page, `${U(6)} change course`, 'Enter', ws)
      session.expectUserMarkers.push(U(5), U(6))
      hold.open()
      await provider.streamStarted(U(6), 1)
      await openSession(page, '', '')
      await openSession(page, session.sessionId, AI(6))
      steerHold.open()
      await settled(page, ws, A(6))
      await assertTranscriptOracle(page, ws, provider, session, 'steer then re-attach')
    })

    await test.step('socket drop and reconnect while the answer after a tool streams', async () => {
      const backendPort = Number(new URL(ws.sockets[0]!.url).port)
      const proxy = await startTcpProxy(backendPort)

      try {
        await routePrimaryWebSocket(app, backendPort, proxy.port)
        await page.reload()
        await waitForInteractive(app, page)
        await installDuplicateSampler(page)
        await expect.poll(() => proxy.connections()).toBeGreaterThan(0)
        await openSession(page, session.sessionId, A(6))
        const hold = gate()
        provider.script(U(7), [
          { text: words(AI(7), 'narrating', 'before', 'the', 'drop'), toolCalls: [echo('reattach-d')] },
          { text: words(A(7), 'answer', 'across', 'the', 'drop'), holdAfterFirstChunk: hold }
        ])
        await send(page, `${U(7)} across a drop`, 'Enter', ws)
        await provider.streamStarted(U(7), 1)
        await expect(viewport(page)).toContainText(A(7))
        session.expectUserMarkers.push(U(7))
        const socketsBefore = ws.sockets.length
        expect(proxy.dropAll()).toBeGreaterThan(0)
        await expect.poll(() => ws.sockets.length, { timeout: 60_000 }).toBeGreaterThan(socketsBefore)
        await expect.poll(() => proxy.connections(), { timeout: 60_000 }).toBeGreaterThan(0)
        hold.open()
        // The tail, and even message.complete, can fall in the reconnect gap
        // (a lossy wire), so the view, not a frame, says the turn settled.
        await expect(stopButton(page)).toHaveCount(0, { timeout: 120_000 })
        session.lossyWire = [U(7)]
        await assertTranscriptOracle(page, ws, provider, session, 'reconnect during a tool turn')
      } finally {
        await proxy.close()
      }
    })

  })
})

/**
 * Route the primary socket through the test and, once, right after the first
 * server event `pick` matches, deliver copies of the frames `replay` selects
 * from what has been delivered so far, under seqs the backend never used: the
 * shape a replay after a reconnect produces (#140397, #144609). Every other
 * frame passes through unchanged and is recorded in `ws`.
 */
async function routeRedelivery(
  page: Page,
  ws: WsRecorder,
  pick: (msg: any) => boolean,
  replay: (delivered: any[], picked: any) => any[] = (_delivered, picked) => [picked]
) {
  const delivered: any[] = []

  const state = { redelivered: false }

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

      client.send(raw)

      if (msg?.method === 'event') {
        delivered.push(msg)
      }

      if (!state.redelivered && pick(msg)) {
        state.redelivered = true
        replay(delivered, msg).forEach((frame, index) =>
          client.send(JSON.stringify({ ...frame, params: { ...frame.params, seq: frame.params.seq + 100_000 + index } }))
        )
      }
    })
  })

  return state
}

test('a tool completion redelivered under a newer seq renders one tool row', async () => {
  await withApp('redeliver', async ({ app, page, provider, ws }) => {
    const session: OracleTarget = { sessionId: '', expectUserMarkers: [] }
    const tag = `redeliver-${nonce}`

    const redelivery = await routeRedelivery(
      page,
      ws,
      msg => msg?.method === 'event' && msg.params?.type === 'tool.complete' && JSON.stringify(msg.params.payload ?? {}).includes(tag)
    )

    await page.reload()
    await waitForInteractive(app, page)
    await installDuplicateSampler(page)
    provider.script(U(1), [
      { text: words(AI(1), 'narrating', 'before', 'the', 'tool'), toolCalls: [echo(tag)] },
      { text: words(A(1), 'answer', 'after', 'the', 'tool') }
    ])
    await send(page, `${U(1)} one tool`, 'Enter', ws)
    await settled(page, ws, A(1))
    session.sessionId = await currentSessionId(page)
    session.expectUserMarkers.push(U(1))
    expect(redelivery.redelivered).toBe(true)
    await page.screenshot({ path: test.info().outputPath('redelivered-turn.png') })
    await assertTranscriptOracle(page, ws, provider, session, 'redelivered tool completion')
    await page.screenshot({ path: test.info().outputPath('redelivered-tool-row.png') })
    const rows = viewport(page).locator('[data-tool-row]').filter({ hasText: `core-${tag}` })
    await expect(rows).toHaveCount(1)
  })
})

test('a narrated segment replayed after a message.start renders once', async () => {
  await withApp('replay-segment', async ({ app, page, provider, ws }) => {
    const session: OracleTarget = { sessionId: '', expectUserMarkers: [] }

    const isEvent = (msg: any, type: string) => msg?.method === 'event' && msg.params?.type === type

    // When the tool starts, replay the turn's message.start and the narration
    // deltas after it: the segment a superseded attempt re-streams.
    const redelivery = await routeRedelivery(
      page,
      ws,
      msg => isEvent(msg, 'tool.start') && JSON.stringify(msg.params.payload ?? {}).includes(`replay-${nonce}`),
      delivered => {
        const start = delivered.findLastIndex(frame => isEvent(frame, 'message.start'))

        return [delivered[start], ...delivered.slice(start + 1).filter(frame => isEvent(frame, 'message.delta'))]
      }
    )

    await page.reload()
    await waitForInteractive(app, page)
    await installDuplicateSampler(page)
    const hold = gate()
    provider.script(U(1), [
      { text: words(AI(1), 'commentary', 'before', 'the', 'tool'), toolCalls: [echo(`replay-${nonce}`)] },
      { text: words(A(1), 'a', 'different', 'final', 'answer'), holdAfterFirstChunk: hold }
    ])
    await send(page, `${U(1)} commentary then answer`, 'Enter', ws)
    // Mid-turn, with the re-sent narration delivered and the answer held:
    // the frame a reader sees while the turn is still running.
    await provider.streamStarted(U(1), 1)
    await expect(viewport(page)).toContainText(A(1))
    expect(redelivery.redelivered).toBe(true)
    await page.screenshot({ path: test.info().outputPath('replayed-segment-mid-turn.png') })
    hold.open()
    await settled(page, ws, A(1))
    session.sessionId = await currentSessionId(page)
    session.expectUserMarkers.push(U(1))
    expect(redelivery.redelivered).toBe(true)
    await page.screenshot({ path: test.info().outputPath('replayed-segment.png') })
    session.lossyWire = [U(1)]
    await assertTranscriptOracle(page, ws, provider, session, 'replayed narrated segment')
  })
})
