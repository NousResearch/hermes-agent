/**
 * NF2: after a lost connection Desktop shows the turn's REAL state, and a
 * follow-up never hangs behind a turn whose outcome is unknown (#134889 NF2).
 *
 * Real Electron + real gateway; only the LLM is scripted. Three faults:
 *
 *  1. Socket drop mid-turn (the primary WebSocket is routed through a loopback
 *     TCP proxy the test resets). The turn finishes on the gateway while the
 *     renderer is disconnected. After the reconnect the completed reply is
 *     rendered exactly once and nothing still spins.
 *  2. Socket drop mid-turn, then a follow-up sent right after the reconnect
 *     while the first turn is still running (held by the provider). The
 *     follow-up is admitted, runs after the first turn, and both replies land.
 *  3. SIGKILL of the gateway mid-turn. Desktop's supervisor brings a new owner
 *     up; owner recovery fences the lost turn `unknown`. The chat must offer the
 *     named lost-turn card (Discard), never a spinner. A follow-up sent now is
 *     either queued visibly behind the lost turn or refused by name; Discard
 *     then lets it run to completion.
 *
 * Every wait is on a DOM state, a provider record or a /proc census with a
 * deadline; the "no spinner" check is a bounded observation after the settle.
 */

import fs from 'node:fs'
import net from 'node:net'
import path from 'node:path'

import { expect, type Page, test } from '@playwright/test'

import {
  backendProcesses,
  coreAppEnv,
  createCoreSandbox,
  currentSessionId,
  launchCoreApp,
  recordWebSockets,
  routePrimaryWebSocket,
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

const words = (marker: string, ...rest: string[]) => [
  `${marker} `,
  ...rest.map((w, i) => (i === rest.length - 1 ? w : `${w} `))
]

const viewport = (page: Page) => page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
const stopButton = (page: Page) => page.locator('[data-slot="composer-root"] button[aria-label="Stop"]')
const lostCard = (page: Page) => page.locator('[data-slot="queue-lost-note"]')
const lostDiscard = (page: Page) => page.locator('[data-slot="queue-lost-discard"]')
const queueHeader = (page: Page) => page.locator('.status-section-trigger').filter({ hasText: /Queued/ })

const OUT = process.env.HERMES_E2E_NF2_OUT

async function dump(page: Page, name: string, extra: Record<string, unknown> = {}) {
  if (!OUT) {
    return
  }

  fs.mkdirSync(OUT, { recursive: true })
  const text = await page.evaluate(() => document.body.innerText).catch(e => `evaluate failed: ${String(e)}`)
  fs.writeFileSync(path.join(OUT, `${name}.txt`), `${text}\n\n${JSON.stringify(extra, null, 1)}`)
  await page.screenshot({ path: path.join(OUT, `${name}.png`) }).catch(() => undefined)
}

/** Loopback proxy for the primary socket: `block()` resets live connections AND refuses new ones. */
function blockableProxy(targetPort: number) {
  const live = new Set<net.Socket>()
  let blocked = false
  let accepted = 0

  const server = net.createServer(client => {
    if (blocked) {
      client.resetAndDestroy()

      return
    }

    accepted++
    const upstream = net.connect(targetPort, '127.0.0.1')
    live.add(client)

    const forget = () => {
      live.delete(client)
      client.destroy()
      upstream.destroy()
    }

    for (const sock of [client, upstream]) {
      sock.on('error', forget)
      sock.on('close', forget)
    }

    client.pipe(upstream)
    upstream.pipe(client)
  })

  return new Promise<{
    port: number
    accepted: () => number
    connections: () => number
    block: () => void
    unblock: () => void
    close: () => Promise<void>
  }>((resolve, reject) => {
    server.on('error', reject)
    server.listen(0, '127.0.0.1', () =>
      resolve({
        port: (server.address() as net.AddressInfo).port,
        accepted: () => accepted,
        connections: () => live.size,
        block: () => {
          blocked = true

          for (const sock of [...live]) {
            sock.resetAndDestroy()
          }
        },
        unblock: () => {
          blocked = false
        },
        close: () =>
          new Promise<void>(done => {
            for (const sock of [...live]) {
              sock.destroy()
            }

            server.close(() => done())
          })
      })
    )
  })
}

/**
 * Failure evidence for the settle checks. This flake has only ever reproduced on CI, so the artifact
 * must carry its own story: every non-delta gateway frame (events with seq/running, RPC results with
 * running/replay) and every Stop-button transition, on one clock.
 */
interface Timeline {
  t0: number
  lines: string[]
}

function summarizeFrame(text: string): string {
  try {
    const msg = JSON.parse(text)

    if (msg?.method === 'event') {
      const p = msg.params ?? {}

      return /delta$/.test(String(p.type)) ? '' : `event ${p.type} seq=${p.seq} running=${p.payload?.running}`
    }

    if (msg?.method) {
      return `${msg.method} id=${msg.id}`
    }

    if (msg?.result && typeof msg.result === 'object') {
      const r = msg.result
      const running = 'running' in r ? ` running=${r.running}` : ''

      const replay = Array.isArray(r.events)
        ? ` replay=[${r.events
            .filter((e: { type?: string }) => !/delta$/.test(String(e.type)))
            .map((e: { seq?: number; type?: string; payload?: { running?: unknown } }) =>
              `${e.seq}:${e.type}${e.payload && 'running' in e.payload ? `(running=${e.payload.running})` : ''}`
            )
            .join(' ')}]`
        : ''

      return `result id=${msg.id}${running}${replay}`
    }
  } catch {
    /* binary frames carry nothing for the timeline */
  }

  return ''
}

function recordTimeline(page: Page): Timeline {
  const timeline: Timeline = { t0: Date.now(), lines: [] }
  const note = (line: string) => timeline.lines.push(`${Date.now() - timeline.t0} ${line}`)

  page.on('websocket', ws => {
    if (!ws.url().includes('/api/ws')) {
      return
    }

    note('ws open')
    ws.on('close', () => note('ws close'))

    const noteFrame = (dir: 'rx' | 'tx', payload: string | Buffer) => {
      const line = summarizeFrame(String(payload))

      if (line) {
        note(`${dir} ${line}`)
      }
    }

    ws.on('framereceived', frame => noteFrame('rx', frame.payload))
    ws.on('framesent', frame => noteFrame('tx', frame.payload))
  })

  return timeline
}

/** In-page Stop transitions (MutationObserver, so no polling load); re-run after every reload. */
async function watchStop(page: Page): Promise<void> {
  await page.evaluate(() => {
    const w = window as unknown as { __stopLog?: string[]; __stopObs?: MutationObserver }
    const query = () => document.querySelectorAll('[data-slot="composer-root"] button[aria-label="Stop"]').length
    let last = query()
    w.__stopLog = [`${Date.now()} stop=${last}`]
    w.__stopObs?.disconnect()
    w.__stopObs = new MutationObserver(() => {
      const now = query()

      if (now !== last) {
        last = now
        w.__stopLog!.push(`${Date.now()} stop=${now}`)
      }
    })
    w.__stopObs.observe(document.body, { attributes: true, childList: true, subtree: true })
  })
}

async function attachTimeline(page: Page, timeline: Timeline, name: string): Promise<void> {
  const stops = await page
    .evaluate(() => (window as unknown as { __stopLog?: string[] }).__stopLog ?? [])
    .catch(() => [] as string[])

  const merged = [
    ...timeline.lines,
    ...stops.map(line => {
      const [at, rest] = [Number(line.split(' ')[0]), line.slice(line.indexOf(' ') + 1)]

      return `${at - timeline.t0} DOM ${rest}`
    })
  ].sort((a, b) => Number(a.split(' ')[0]) - Number(b.split(' ')[0]))

  // A file in the test's output dir: CI uploads test-results/core on failure.
  const file = test.info().outputPath(`${name}.txt`)
  fs.writeFileSync(file, merged.join('\n'))
  await test.info().attach(name, { path: file, contentType: 'text/plain' })
}

/** Nothing spins for `ms`: no Stop button in the composer at any sample. */
async function settledFor(page: Page, timeline: Timeline, ms = 3_000): Promise<void> {
  const end = Date.now() + ms

  while (Date.now() < end) {
    const count = await stopButton(page).count()

    if (count !== 0) {
      // Let the transition finish so the evidence shows how long the indicator stayed up.
      await page.waitForTimeout(2_000)
      await attachTimeline(page, timeline, 'stop-timeline')
    }

    expect(count, 'a turn indicator (Stop) is still showing after the turn settled').toBe(0)
    await page.waitForTimeout(250)
  }
}

function setup(label: string) {
  return async () => {
    const provider = await startScriptedProvider()
    const sandbox = createCoreSandbox(label)
    writeProviderHome(sandbox.hermesHome, provider.url)
    const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
    const ws = recordWebSockets(page)
    const timeline = recordTimeline(page)

    const finished = (marker: string, step = 0, timeout = 120_000) =>
      expect
        .poll(() => provider.completions.some(c => c.marker === marker && c.step === step && c.finished), {
          timeout,
          message: `provider finished ${marker} step ${step}`
        })
        .toBe(true)

    return { provider, sandbox, app, page, ws, finished, timeline }
  }
}

test('socket drop mid-turn: the finished reply shows once on reconnect, and a follow-up right after runs', async () => {
  const { provider, sandbox, app, page, ws, finished, timeline } = await setup('nf2-drop')()
  const proxies: { close: () => Promise<void> }[] = []
  const session: OracleTarget = { sessionId: '', expectUserMarkers: [] }

  try {
    await waitForInteractive(app, page)
    await installDuplicateSampler(page)

    provider.script(U(1), [{ text: words(A(1), 'first', 'turn') }])
    await send(page, `${U(1)} hello`, 'Enter', ws)
    await finished(U(1))
    await expect.poll(() => currentSessionId(page)).not.toBe('')
    session.sessionId = await currentSessionId(page)
    session.expectUserMarkers.push(U(1))
    await assertTranscriptOracle(page, ws, provider, session, 'baseline')

    const backendPort = Number(new URL(ws.sockets[0]!.url).port)
    const proxy = await blockableProxy(backendPort)
    proxies.push(proxy)
    await routePrimaryWebSocket(app, backendPort, proxy.port)
    await page.reload()
    await waitForInteractive(app, page)
    await installDuplicateSampler(page)
    await watchStop(page)
    await expect.poll(() => proxy.connections(), { timeout: 60_000 }).toBeGreaterThan(0)
    await page.evaluate(id => (window.location.hash = `#/${encodeURIComponent(id)}`), session.sessionId)
    await expect(viewport(page)).toContainText(A(1), { timeout: 60_000 })

    await test.step('(a) turn completes on the gateway while the renderer is disconnected', async () => {
      const hold = gate()
      provider.script(U(2), [{ text: words(A(2), 'finished', 'while', 'away', 'END2'), holdAfterFirstChunk: hold }])
      await send(page, `${U(2)} across a drop`, 'Enter', ws)
      await provider.streamStarted(U(2))
      await expect(viewport(page)).toContainText(A(2))
      // Block the socket: the turn finishes on the gateway while no connection can exist.
      proxy.block()
      await expect.poll(() => proxy.connections()).toBe(0)
      hold.open()
      await finished(U(2))
      await dump(page, 'a-while-disconnected', { stop: await stopButton(page).count() })
      const acceptedBefore = proxy.accepted()
      proxy.unblock()
      await expect.poll(() => proxy.accepted(), { timeout: 90_000, message: 'renderer redialled' }).toBeGreaterThan(acceptedBefore)
      await expect(viewport(page)).toContainText('END2', { timeout: 60_000 })
      await dump(page, 'a-after-reconnect', { stop: await stopButton(page).count() })
      session.expectUserMarkers.push(U(2))
      session.lossyWire = [U(2)]
      await assertTranscriptOracle(page, ws, provider, session, 'drop: completed while away')
      await expect(stopButton(page)).toHaveCount(0, { timeout: 30_000 })
      await settledFor(page, timeline)
    })

    await test.step('(b) follow-up sent right after the reconnect while the prior turn still runs', async () => {
      const hold = gate()
      provider.script(U(3), [{ text: words(A(3), 'slow', 'turn', 'END3'), holdAfterFirstChunk: hold }])
      provider.script(U(4), [{ text: words(A(4), 'follow', 'up', 'END4') }])
      await send(page, `${U(3)} slow`, 'Enter', ws)
      await provider.streamStarted(U(3))
      await expect(viewport(page)).toContainText(A(3))
      const acceptedBefore = proxy.accepted()
      proxy.block()
      proxy.unblock()
      await expect.poll(() => proxy.accepted(), { timeout: 90_000, message: 'renderer redialled' }).toBeGreaterThan(acceptedBefore)
      // Queue (Control+Enter) behind the still-running turn: admitted, not dropped, not hung.
      await send(page, `${U(4)} follow`, 'Control+Enter', ws)
      await dump(page, 'b-queued', { stop: await stopButton(page).count() })
      hold.open()
      await finished(U(3))
      await finished(U(4))
      await expect(viewport(page)).toContainText('END4', { timeout: 60_000 })
      session.expectUserMarkers.push(U(3), U(4))
      session.lossyWire = [U(2), U(3)]
      await assertTranscriptOracle(page, ws, provider, session, 'drop: follow-up after reconnect')
      await expect(stopButton(page)).toHaveCount(0, { timeout: 30_000 })
      await settledFor(page, timeline)
      await dump(page, 'b-done')
    })
  } finally {
    await app.close().catch(() => undefined)

    for (const p of proxies) {
      await p.close().catch(() => undefined)
    }

    await provider.close()
    sandbox.cleanup()
  }
})

test('gateway SIGKILL mid-turn: the lost turn is offered for Discard, and a follow-up never hangs behind it', async () => {
  const { provider, sandbox, app, page, ws, finished, timeline } = await setup('nf2-kill')()
  const session: OracleTarget = { sessionId: '', expectUserMarkers: [] }

  try {
    await waitForInteractive(app, page)
    await installDuplicateSampler(page)
    await watchStop(page)

    provider.script(U(1), [{ text: words(A(1), 'first', 'turn') }])
    await send(page, `${U(1)} hello`, 'Enter', ws)
    await finished(U(1))
    await expect.poll(() => currentSessionId(page)).not.toBe('')
    session.sessionId = await currentSessionId(page)

    const hold = gate()
    provider.script(U(2), [{ text: words(A(2), 'never', 'finishes'), holdAfterFirstChunk: hold }])
    await send(page, `${U(2)} doomed`, 'Enter', ws)
    await provider.streamStarted(U(2))
    await expect(viewport(page)).toContainText(A(2))

    const [victim] = backendProcesses(sandbox)
    expect(victim).toBeTruthy()
    process.kill(victim!.pid, 'SIGKILL')
    hold.open()

    await test.step('(c) owner recovery fences the turn unknown; Desktop offers Discard, no spinner', async () => {
      await expect
        .poll(() => backendProcesses(sandbox).filter(p => p.pid !== victim!.pid).length, {
          timeout: 120_000,
          message: 'a replacement gateway is up'
        })
        .toBe(1)
      await waitForInteractive(app, page)
      await expect(lostDiscard(page)).toBeVisible({ timeout: 90_000 }).catch(async error => {
        await dump(page, 'c-no-lost-card', { stop: await stopButton(page).count(), sent: ws.sent.slice(-30) })
        throw error
      })
      await expect(lostCard(page)).toHaveCount(1)
      await expect(queueHeader(page)).toHaveText(/1 Queued — paused/)
      await dump(page, 'c-lost-card', { stop: await stopButton(page).count() })
      await expect(stopButton(page)).toHaveCount(0, { timeout: 30_000 })
    })

    await test.step('(b+c) follow-up behind the unknown turn: queued under a named pause, runs after Discard', async () => {
      provider.script(U(3), [{ text: words(A(3), 'after', 'discard', 'END3') }])
      await send(page, `${U(3)} next`, 'Enter', ws)
      await dump(page, 'c-followup-sent', { stop: await stopButton(page).count() })
      // The follow-up is admitted behind the lost turn: the queue header counts it and names the
      // pause, and the lost card (the reason + the way out) is still on screen. No spinner.
      await expect(queueHeader(page)).toHaveText(/2 Queued — paused/, { timeout: 30_000 })
      await expect(lostDiscard(page)).toBeVisible()
      await expect(stopButton(page)).toHaveCount(0)
      expect(provider.completions.some(c => c.marker === U(3)), 'the follow-up never ran ahead of the lost turn').toBe(false)
      await dump(page, 'c-followup-queued')

      await lostDiscard(page).click()

      await expect(lostDiscard(page)).toHaveCount(0, { timeout: 60_000 })
      await finished(U(3), 0, 60_000)
      await expect(viewport(page)).toContainText('END3', { timeout: 60_000 })
      await expect(stopButton(page)).toHaveCount(0, { timeout: 30_000 })
      await settledFor(page, timeline)
      session.expectUserMarkers.push(U(1), U(2), U(3))
      session.lossyWire = [U(2)]
      await dump(page, 'c-done')
      await assertTranscriptOracle(page, ws, provider, session, 'after Discard of the lost turn')
    })
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
})
