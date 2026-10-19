/**
 * C2 core: a send from a window that is behind its chat goes through.
 *
 * Two Desktop windows on one chat. Window 2 stops hearing window 1's
 * "turn finished" pings (the cross-window transcript bus is held in that
 * window), so when window 1 runs a turn, window 2 still shows the older
 * transcript. Window 2 then sends. The backend owns the model's context, so
 * that send must go out as typed:
 *
 * - no refusal and no "Chat out of date" notice (#65047's pre-send refusal is
 *   gone: it misfired in single windows and protected nothing the backend
 *   did not already guarantee);
 * - the model's request for window 2's turn carries window 1's turn;
 * - both windows then render the stored transcript once, in stored order.
 */

import { type ElectronApplication, expect, type Page, test } from '@playwright/test'

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
import { startScriptedProvider } from './provider'

const nonce = Math.random()
  .toString(36)
  .slice(2, 8)
  .replace(/[^a-z0-9]/g, 'x')
  .padEnd(4, 'q')

const U = (n: number) => `U${n}-${nonce}`
const A = (n: number) => `A${n}-${nonce}`

const STALE_NOTICE = 'Chat out of date'

function viewport(page: Page) {
  return page.locator('[data-slot="aui_thread-viewport"]').filter({ visible: true }).first()
}

/** One recorder over both windows' sockets; the oracle dedupes events by session + seq. */
function mergedRecorder(...recorders: WsRecorder[]): WsRecorder {
  return {
    get events() {
      return recorders.flatMap(r => r.events)
    },
    get sent() {
      return recorders.flatMap(r => r.sent)
    },
    get sockets() {
      return recorders.flatMap(r => r.sockets)
    }
  }
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

/** Count every appearance of the old refusal notice, from install on. */
async function watchStaleNotice(page: Page) {
  await page.evaluate(text => {
    const w = window as any
    w.__staleNoticeCount = 0
    let showing = false

    new MutationObserver(() => {
      const now = document.body.innerText.includes(text)

      if (now && !showing) {
        w.__staleNoticeCount += 1
      }

      showing = now
    }).observe(document.body, { childList: true, subtree: true, characterData: true })
  }, STALE_NOTICE)
}

const staleNoticeCount = (page: Page) => page.evaluate(() => (window as any).__staleNoticeCount ?? 0)

/**
 * Route every WebSocket opened from here on (the second window's) through the
 * test, so its gateway events can be withheld on demand — the way a window that
 * missed a turn looks. Routed frames are fed into `rec` (the oracle dedupes by
 * session + seq, so attribution across windows does not matter).
 */
async function routeLaterSockets(app: ElectronApplication, rec: WsRecorder) {
  const ctl = { holdEvents: false }

  const parse = (raw: string | Buffer) => {
    try {
      return JSON.parse(String(raw))
    } catch {
      return null
    }
  }

  await app.context().routeWebSocket(/\/api\/ws/, client => {
    const server = client.connectToServer()
    const socket = rec.sockets.push({ id: rec.sockets.length, url: client.url(), closed: false }) - 1

    client.onMessage(raw => {
      const msg = parse(raw)

      if (typeof msg?.method === 'string') {
        rec.sent.push({ socket, method: msg.method, params: msg.params })
      }

      server.send(raw)
    })
    server.onMessage(raw => {
      const msg = parse(raw)
      const isEvent = msg?.method === 'event' && msg.params

      if (isEvent) {
        rec.events.push({
          socket,
          type: String(msg.params.type ?? ''),
          sessionId: String(msg.params.session_id ?? ''),
          seq: typeof msg.params.seq === 'number' ? msg.params.seq : null,
          payload: msg.params.payload
        })
      }

      if (isEvent && ctl.holdEvents) {
        return
      }

      client.send(raw)
    })
  })

  return ctl
}

/**
 * Let a window drop the cross-window transcript bus on demand: set
 * `window.__holdTranscriptBus = true` and its `hermes:transcript` listeners
 * stop firing. Installed on the app context so the second window gets it.
 */
async function installTranscriptBusHold(app: ElectronApplication) {
  await app.context().addInitScript(() => {
    const Native = window.BroadcastChannel

    if (!Native) {
      return
    }

    window.BroadcastChannel = class extends Native {
      override addEventListener(type: string, listener: any, options?: any) {
        if (this.name === 'hermes:transcript' && type === 'message') {
          const held = (event: Event) => {
            if (!(window as any).__holdTranscriptBus) {
              listener.call(this, event)
            }
          }

          return super.addEventListener(type, held, options)
        }

        return super.addEventListener(type, listener, options)
      }
    } as typeof BroadcastChannel
  })
}

test('a window behind its chat still sends, and the model sees the other window’s turn', async () => {
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('behind-send')
  writeProviderHome(sandbox.hermesHome, provider.url)
  const { app, page } = await launchCoreApp(coreAppEnv(sandbox))
  const ws1 = recordWebSockets(page)

  try {
    await installTranscriptBusHold(app)
    await waitForInteractive(app, page)
    await watchStaleNotice(page)

    provider.script(U(1), [{ text: [`${A(1)} `, 'opened ', 'here'] }])
    await send(page, `${U(1)} start the chat`, 'Enter', ws1)
    await settled(page, ws1, A(1))
    const session: OracleTarget = { sessionId: await currentSessionId(page), expectUserMarkers: [U(1)] }

    // Window 2's sockets are routed, so its event stream can be withheld.
    const ws2: WsRecorder = { events: [], sent: [], sockets: [] }
    const window2Wire = await routeLaterSockets(app, ws2)
    const opened = app.waitForEvent('window')
    await page.evaluate(id => (window as any).hermesDesktop.openSessionWindow(id), session.sessionId)
    const page2 = await opened
    const ws = mergedRecorder(ws1, ws2)
    await waitForInteractive(app, page2)
    await expect(viewport(page2)).toContainText(A(1), { timeout: 60_000 })
    await watchStaleNotice(page2)
    await installDuplicateSampler(page)
    await installDuplicateSampler(page2)

    await test.step('window 1 advances the chat while window 2 hears nothing', async () => {
      // Both live channels into window 2 go quiet: its gateway events and the
      // cross-window "turn finished" ping.
      await page2.evaluate(() => ((window as any).__holdTranscriptBus = true))
      window2Wire.holdEvents = true
      provider.script(U(2), [{ text: [`${A(2)} `, 'from ', 'window ', 'one'] }])
      await send(page, `${U(2)} window one turn`, 'Enter', ws)
      await settled(page, ws, A(2))
      session.expectUserMarkers.push(U(2))
      window2Wire.holdEvents = false
      // Precondition: window 2 really is behind when it sends.
      await expect(viewport(page2)).not.toContainText(A(2))
    })

    await test.step('window 2 sends from behind; it goes through', async () => {
      provider.script(U(3), [{ text: [`${A(3)} `, 'from ', 'window ', 'two'] }])
      await send(page2, `${U(3)} window two turn`, 'Enter', ws)
      await settled(page2, ws, A(3))
      session.expectUserMarkers.push(U(3))

      // The model's request for window 2's turn carried window 1's turn.
      const request = provider.completions.find(c => c.marker === U(3))
      expect(request, 'window 2 turn reached the model').toBeTruthy()
      expect(JSON.stringify(request!.body.messages), 'model context holds the other window’s turn').toContain(A(2))

      expect(await staleNoticeCount(page2), 'no "Chat out of date" in window 2').toBe(0)
      expect(await staleNoticeCount(page), 'no "Chat out of date" in window 1').toBe(0)
    })

    await test.step('both windows converge on the stored transcript', async () => {
      await page2.evaluate(() => ((window as any).__holdTranscriptBus = false))
      await assertTranscriptOracle(page2, ws, provider, session, 'window 2 after sending from behind')
      await assertTranscriptOracle(page, ws, provider, session, 'window 1 after window 2 sent')
    })
  } finally {
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
})
