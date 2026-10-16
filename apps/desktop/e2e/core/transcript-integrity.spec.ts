/**
 * C2 core: transcript integrity across every transition that has shipped a
 * duplicate / vanishing / reordered message bug.
 *
 * One real Electron app + one real `hermes serve` backend (only the LLM is
 * faked, by a scripted recording provider). After every transition the
 * transcript oracle (./oracle.ts) asserts: each persisted user/assistant
 * message is rendered exactly once, in order, nothing unpersisted is
 * rendered, no marker was ever rendered twice even transiently, and the
 * backend's concatenated deltas equal what the provider streamed.
 *
 * Transitions (in order, sharing the app so each builds on the last):
 *   stream → tool-call turn → reasoning turn → steer mid-stream →
 *   queued follow-up →
 *   session switch mid-stream → warm resume → reload →
 *   WebSocket drop + reconnect mid-stream → second socket to the same
 *   backend (#120005 topology) → final reload re-checks every session.
 */

import * as path from 'node:path'

import { expect, type Page, test } from '@playwright/test'

import {
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  routePrimaryWebSocket,
  send,
  splitProfileRoute,
  startTcpProxy,
  storedSessionForMarker,
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
const R = (n: number) => `R${n}-${nonce}`

const words = (marker: string, ...rest: string[]) => [
  `${marker} `,
  ...rest.map((w, i) => (i === rest.length - 1 ? w : `${w} `))
]

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

async function openSession(page: Page, sessionId: string, mustShow: string) {
  await page.evaluate(id => {
    window.location.hash = `#/${encodeURIComponent(id)}`
  }, sessionId)
  await expect.poll(() => currentSessionId(page)).toBe(sessionId)
  await expect(viewport(page)).toContainText(mustShow, { timeout: 60_000 })
}

test('transcript oracle holds across every transition', async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('transcript')
  writeProviderHome(sandbox.hermesHome, provider.url)
  writeProviderHome(path.join(sandbox.hermesHome, 'profiles', 'p2'), provider.url)
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws = recordWebSockets(page)
  const proxies: { close: () => Promise<void> }[] = []

  const finished = (marker: string, step = 0) =>
    expect
      .poll(() => provider.completions.some(c => c.marker === marker && c.step === step && c.finished), {
        timeout: 120_000,
        message: `provider finished ${marker} step ${step}`
      })
      .toBe(true)

  const sessionA: OracleTarget = { sessionId: '', expectUserMarkers: [] }
  const sessionB: OracleTarget = { sessionId: '', expectUserMarkers: [] }

  try {
    await waitForInteractive(app, page)
    await installDuplicateSampler(page)

    await test.step('stream: multi-chunk reply', async () => {
      provider.script(U(1), [{ text: words(A(1), 'alpha', 'bravo', 'charlie', 'delta', 'echo') }])
      await send(page, `${U(1)} hello`, 'Enter', ws)
      await finished(U(1))
      await expect.poll(() => currentSessionId(page)).not.toBe('')
      sessionA.sessionId = await currentSessionId(page)
      sessionA.expectUserMarkers.push(U(1))
      await assertTranscriptOracle(page, ws, provider, sessionA, 'stream')
    })

    await test.step('tool-call turn: interim text + real terminal tool + final', async () => {
      provider.script(U(2), [
        {
          text: words(AI(2), 'checking', 'first'),
          toolCalls: [{ name: 'terminal', args: { command: 'echo core-tool-ok' } }]
        },
        { text: words(A(2), 'tool', 'said', 'ok') }
      ])
      await send(page, `${U(2)} run a tool`, 'Enter', ws)
      await finished(U(2), 1)
      sessionA.expectUserMarkers.push(U(2))
      await assertTranscriptOracle(page, ws, provider, sessionA, 'tool-call turn')
    })

    await test.step('reasoning turn: reasoning deltas never leak into or double the reply', async () => {
      provider.script(U(3), [
        { reasoning: words(R(3), 'weighing', 'options'), text: words(A(3), 'reasoned', 'answer') }
      ])
      await send(page, `${U(3)} think first`, 'Enter', ws)
      await finished(U(3))
      sessionA.expectUserMarkers.push(U(3))
      await assertTranscriptOracle(page, ws, provider, sessionA, 'reasoning turn')
    })

    await test.step('steer mid-stream: Enter while busy redirects the live turn', async () => {
      const hold = gate()
      provider.script(U(4), [{ text: words(A(4), 'slow', 'first', 'reply'), holdAfterFirstChunk: hold }])
      provider.script(U(5), [{ text: words(A(5), 'steered', 'reply') }])
      await send(page, `${U(4)} slow one`, 'Enter', ws)
      await provider.streamStarted(U(4))
      await expect(viewport(page)).toContainText(A(4))
      await send(page, `${U(5)} change course`, 'Enter', ws)
      await finished(U(5))
      hold.open()
      sessionA.expectUserMarkers.push(U(4), U(5))
      await assertTranscriptOracle(page, ws, provider, sessionA, 'steer mid-stream')
    })

    await test.step('queued follow-up: Ctrl+Enter while busy runs after the live turn, once', async () => {
      const hold = gate()
      provider.script(U(11), [{ text: words(A(11), 'long', 'running', 'reply'), holdAfterFirstChunk: hold }])
      provider.script(U(12), [{ text: words(A(12), 'queued', 'reply') }])
      await send(page, `${U(11)} long one`, 'Enter', ws)
      await provider.streamStarted(U(11))
      await expect(viewport(page)).toContainText(A(11))
      await send(page, `${U(12)} after that`, 'Control+Enter', ws)
      hold.open()
      await finished(U(11))
      await finished(U(12))
      // The queued message is a turn of its own, strictly after the first.
      const u11 = provider.completions.findIndex(c => c.marker === U(11))
      const u12 = provider.completions.findIndex(c => c.marker === U(12))
      expect(u12).toBeGreaterThan(u11)
      sessionA.expectUserMarkers.push(U(11), U(12))
      await assertTranscriptOracle(page, ws, provider, sessionA, 'queued follow-up')
    })

    await test.step('session switch mid-stream: the away session completes exactly once', async () => {
      // NEW_CHAT_ROUTE ('/') — the route the sidebar's New session button opens.
      await page.evaluate(() => {
        window.location.hash = '#/'
      })
      await expect.poll(() => currentSessionId(page)).toBe('')
      await expect(viewport(page)).not.toContainText(U(1))
      const hold = gate()
      provider.script(U(6), [{ text: words(A(6), 'finished', 'while', 'away'), holdAfterFirstChunk: hold }])
      await send(page, `${U(6)} in session b`, 'Enter', ws)
      await provider.streamStarted(U(6))
      await expect.poll(() => currentSessionId(page)).not.toBe('')
      sessionB.sessionId = await currentSessionId(page)
      expect(sessionB.sessionId).not.toBe(sessionA.sessionId)
      sessionB.expectUserMarkers.push(U(6))
      await openSession(page, sessionA.sessionId, A(12))
      hold.open()
      await finished(U(6))
      await assertTranscriptOracle(page, ws, provider, sessionA, 'switch: session A while B streams')
      await openSession(page, sessionB.sessionId, U(6))
      await assertTranscriptOracle(page, ws, provider, sessionB, 'switch: back to B after it completed away')
    })

    await test.step('warm resume: revisit cached sessions and continue one', async () => {
      await openSession(page, sessionA.sessionId, A(12))
      await assertTranscriptOracle(page, ws, provider, sessionA, 'warm resume A')
      await openSession(page, sessionB.sessionId, A(6))
      provider.script(U(7), [{ text: words(A(7), 'warm', 'continue') }])
      await send(page, `${U(7)} continue b`, 'Enter', ws)
      await finished(U(7))
      sessionB.expectUserMarkers.push(U(7))
      await assertTranscriptOracle(page, ws, provider, sessionB, 'warm resume B + new turn')
    })

    // From here on the primary socket dials through a loopback proxy the test
    // controls, so a network drop can be injected mid-stream.
    const backendPort = Number(new URL(ws.sockets[0]!.url).port)
    const proxy = await startTcpProxy(backendPort)
    proxies.push(proxy)
    await routePrimaryWebSocket(app, backendPort, proxy.port)

    await test.step('reload: hydrated transcript equals persisted', async () => {
      await page.reload()
      await waitForInteractive(app, page)
      await installDuplicateSampler(page)
      await expect.poll(() => proxy.connections()).toBeGreaterThan(0)
      await openSession(page, sessionB.sessionId, A(7))
      await assertTranscriptOracle(page, ws, provider, sessionB, 'reload B')
      await openSession(page, sessionA.sessionId, A(12))
      await assertTranscriptOracle(page, ws, provider, sessionA, 'reload A')
    })

    await test.step('WebSocket drop + reconnect mid-stream: no lost, doubled or replayed text', async () => {
      const hold = gate()
      provider.script(U(8), [{ text: words(A(8), 'survives', 'the', 'drop'), holdAfterFirstChunk: hold }])
      await send(page, `${U(8)} across a drop`, 'Enter', ws)
      await provider.streamStarted(U(8))
      await expect(viewport(page)).toContainText(A(8))
      const socketsBefore = ws.sockets.length
      expect(proxy.dropAll()).toBeGreaterThan(0)
      // The renderer must dial a fresh socket (through the proxy) on its own.
      await expect.poll(() => ws.sockets.length, { timeout: 60_000 }).toBeGreaterThan(socketsBefore)
      await expect.poll(() => proxy.connections(), { timeout: 60_000 }).toBeGreaterThan(0)
      hold.open()
      await finished(U(8))
      sessionA.expectUserMarkers.push(U(8))
      sessionA.lossyWire = [U(8)]
      await assertTranscriptOracle(page, ws, provider, sessionA, 'ws drop + reconnect')
    })

    // Sockets that delivered the message.complete of the turn whose reply opens with `marker`.
    const completeSockets = (marker: string) =>
      new Set(
        ws.events
          .filter(e => e.type === 'message.complete' && String(e.payload?.text ?? '').startsWith(marker))
          .map(e => e.socket)
      )

    let sessionP2: OracleTarget = { sessionId: '', profile: 'p2', expectUserMarkers: [] }

    await test.step('non-default profile on the shared host backend: one socket per session', async () => {
      await page
        .getByRole('button', { name: 'Bots', exact: true })
        .or(page.getByRole('tab', { name: 'Bots', exact: true }))
        .first()
        .click()
      const row = page.locator('[data-slot="bots-roster"] [data-roster-key="local::p2"]')
      await expect(row).toBeVisible({ timeout: 60_000 })
      await row.click()
      await expect(page.getByRole('tab', { name: /Draft/ })).toBeVisible({ timeout: 60_000 })
      provider.script(U(9), [{ text: words(A(9), 'profile', 'two', 'first') }])
      await send(page, `${U(9)} on p2`, 'Enter', ws)
      await finished(U(9))
      await expect(viewport(page)).toContainText(A(9))
      let p2Session: null | string = null
      await expect
        .poll(() => (p2Session = storedSessionForMarker(sandbox, 'p2', U(9))), {
          message: 'p2 turn persisted in the p2 state.db'
        })
        .not.toBeNull()
      sessionP2 = { sessionId: p2Session!, profile: 'p2', expectUserMarkers: [U(9)] }
      await assertTranscriptOracle(page, ws, provider, sessionP2, 'p2 first turn')

      // #120005: the SECOND message of a non-default-profile chat is where a
      // session-owner call used to dial a second socket to the same process.
      provider.script(U(13), [{ text: words(A(13), 'profile', 'two', 'second') }])
      await send(page, `${U(13)} again on p2`, 'Enter', ws)
      await finished(U(13))
      sessionP2.expectUserMarkers.push(U(13))
      await assertTranscriptOracle(page, ws, provider, sessionP2, 'p2 second turn')
      expect([...completeSockets(A(9))].length, 'p2 turn 1 delivered on exactly one socket').toBe(1)
      expect([...completeSockets(A(13))].length, 'p2 turn 2 delivered on exactly one socket').toBe(1)
      // One backend process, one socket: the host backend serves p2 too, so
      // the renderer must not hold a second live socket to it (#120006).
      const sameBackend = new Set([String(backendPort), String(proxy.port)])
      await expect
        .poll(
          () =>
            ws.sockets
              .filter(s => !s.closed && sameBackend.has(new URL(s.url).port))
              .map(s => s.url.replace(/token=[^&]+/, 'token=…')),
          {
            timeout: 30_000,
            message: 'live sockets to the one host backend'
          }
        )
        .toHaveLength(1)
    })

    await test.step('forced second socket to the same backend: every event still renders once', async () => {
      // Force the #120005 topology regardless of routing: from the next
      // session call on, the p2 route is served by a second socket to the same
      // backend while the primary stays joined. The renderer must still apply
      // every event exactly once (tool turn → interim bubble + final).
      const socketsBefore = ws.sockets.length
      await splitProfileRoute(app, 'p2')
      provider.script(U(10), [
        {
          text: words(AI(10), 'looking', 'it', 'up'),
          toolCalls: [{ name: 'terminal', args: { command: 'echo core-two-sockets' } }]
        },
        { text: words(A(10), 'two', 'sockets', 'one', 'render') }
      ])
      await send(page, `${U(10)} tool on p2`, 'Enter', ws)
      await finished(U(10), 1)
      await expect.poll(() => ws.sockets.length).toBeGreaterThan(socketsBefore)
      // Precondition of the scenario: the turn really was fanned out to 2+ sockets.
      await expect
        .poll(() => completeSockets(A(10)).size, {
          timeout: 60_000,
          message: 'the forced turn reached more than one socket'
        })
        .toBeGreaterThan(1)
      sessionP2.expectUserMarkers.push(U(10))
      await assertTranscriptOracle(page, ws, provider, sessionP2, 'p2 tool turn over two sockets')
    })

    await test.step('final reload: every session re-hydrates exactly once', async () => {
      await page.reload()
      await waitForInteractive(app, page)
      await installDuplicateSampler(page)
      await openSession(page, sessionA.sessionId, A(8))
      await assertTranscriptOracle(page, ws, provider, sessionA, 'final reload A')
      await openSession(page, sessionB.sessionId, A(7))
      await assertTranscriptOracle(page, ws, provider, sessionB, 'final reload B')
    })
  } finally {
    await app.close().catch(() => undefined)

    for (const proxy of proxies) {
      await proxy.close()
    }

    await provider.close()
    sandbox.cleanup()
  }
})
