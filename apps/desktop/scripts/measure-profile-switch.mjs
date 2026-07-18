// Measure a profile switch end-to-end: click a profile square in the rail,
// then break the wall time into the phases the renderer can observe:
//   - getConnection IPC (Electron: pool backend spawn / reuse + readiness)
//   - gateway WS connect
//   - swap-target clear ($gatewaySwapTarget → sidebar loader gone)
//   - sidebar session rows for the new profile painted
//
// Instruments window.hermesDesktop.getConnection + WebSocket to timestamp the
// phases without touching app code.
//
// Usage:
//   node apps/desktop/scripts/measure-profile-switch.mjs <profileName> [settleTimeoutMs]

const CDP_HTTP = 'http://127.0.0.1:9222'
const PROFILE = process.argv[2]
const SETTLE_TIMEOUT = Number(process.argv[3] || 60000)

if (!PROFILE) {
  console.error('usage: measure-profile-switch.mjs <profileName>')
  process.exit(1)
}

class CDP {
  constructor(ws) { this.ws = ws; this.id = 0; this.pending = new Map() }
  static async open(url) {
    const ws = new WebSocket(url)
    await new Promise((r) => ws.addEventListener('open', r, { once: true }))
    const cdp = new CDP(ws)
    ws.addEventListener('message', (ev) => {
      const m = JSON.parse(ev.data.toString())
      if (m.id != null && cdp.pending.has(m.id)) {
        const { resolve, reject } = cdp.pending.get(m.id)
        cdp.pending.delete(m.id)
        if (m.error) reject(new Error(m.error.message))
        else resolve(m.result)
      }
    })
    ws.addEventListener('close', () => {
      for (const { reject } of cdp.pending.values()) reject(new Error('CDP socket closed'))
      cdp.pending.clear()
    })
    return cdp
  }
  send(method, params) {
    const id = ++this.id
    return new Promise((res, rej) => {
      this.pending.set(id, { resolve: res, reject: rej })
      this.ws.send(JSON.stringify({ id, method, params }))
    })
  }
  async eval(expr) {
    const r = await this.send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true })
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || 'eval failed')
    return r.result.value
  }
  close() { this.ws.close() }
}

async function main() {
  const list = await (await fetch(`${CDP_HTTP}/json`)).json()
  const target = list.find((t) => t.type === 'page' && /5174/.test(t.url))
  if (!target) { console.error('renderer not found on 9222'); process.exit(1) }
  const cdp = await CDP.open(target.webSocketDebuggerUrl)

  // Instrument getConnection + WebSocket once.
  await cdp.eval(`(() => {
    if (window.__PROFILE_SWITCH_OBS__) return 'already'
    const obs = { events: [] }
    const mark = (name, extra) => obs.events.push({ name, t: performance.now(), ...(extra || {}) })
    window.__PROFILE_SWITCH_OBS__ = obs
    window.__psMark = mark

    const desktop = window.hermesDesktop
    if (desktop && desktop.getConnection) {
      const orig = desktop.getConnection.bind(desktop)
      desktop.getConnection = async (profile) => {
        mark('getConnection:start', { profile })
        try {
          const res = await orig(profile)
          mark('getConnection:done', { profile })
          return res
        } catch (e) {
          mark('getConnection:error', { profile, error: String(e).slice(0, 120) })
          throw e
        }
      }
    }

    const OrigWS = window.WebSocket
    window.WebSocket = function (url, ...rest) {
      const ws = new OrigWS(url, ...rest)
      if (String(url).includes('/api/ws')) {
        mark('ws:new', { url: String(url).replace(/token=[^&]+/, 'token=…').slice(0, 90) })
        ws.addEventListener('open', () => mark('ws:open'))
      }
      return ws
    }
    window.WebSocket.prototype = OrigWS.prototype
    Object.assign(window.WebSocket, OrigWS)
    return 'installed'
  })()`)

  const before = await cdp.eval(`(() => {
    const rail = document.querySelector('[data-slot="profile-rail"]')
    return {
      railButtons: rail ? [...rail.querySelectorAll('[role="tab"], button')].map(b => (b.getAttribute('aria-label') || b.title || b.textContent || '').slice(0, 30)) : [],
      sessions: document.querySelectorAll('[data-slot="sidebar-session-row"], [data-session-id]').length
    }
  })()`)
  console.log('rail buttons:', JSON.stringify(before.railButtons))

  const clicked = await cdp.eval(`(() => {
    window.__psMark('click', { profile: ${JSON.stringify(PROFILE)} })
    const rail = document.querySelector('[data-slot="profile-rail"]')
    if (!rail) return 'no-rail'
    const target = [...rail.querySelectorAll('button, [role="tab"]')].find(b =>
      ((b.getAttribute('aria-label') || '') + ' ' + (b.title || '') + ' ' + (b.textContent || '')).toLowerCase().includes(${JSON.stringify(PROFILE.toLowerCase())}))
    if (!target) return 'not-found'
    target.click()
    return 'clicked'
  })()`)
  console.log('click:', clicked)
  if (clicked !== 'clicked') { cdp.close(); process.exit(2) }

  // Poll until the swap settles: loader gone + session rows painted (or empty
  // list settled) + active profile pill shows the target.
  const t0 = Date.now()
  let settled = null
  while (Date.now() - t0 < SETTLE_TIMEOUT) {
    await new Promise((r) => setTimeout(r, 100))
    const s = await cdp.eval(`(() => {
      // The swap overlay stays mounted at opacity-0 after the swap — check the
      // computed opacity of the container that holds the "Waking up …" label.
      const label = [...document.querySelectorAll('div[aria-hidden]')].find(el => /waking up/i.test(el.textContent || ''))
      const overlayVisible = label ? Number(getComputedStyle(label).opacity) > 0.05 : false
      return {
        t: performance.now(),
        overlayVisible,
        sessions: document.querySelectorAll('[data-slot="row-button"]').length
      }
    })()`)
    if (!s.overlayVisible && s.sessions > 0) { settled = s; break }
  }

  await new Promise((r) => setTimeout(r, 400))
  const obs = await cdp.eval('window.__PROFILE_SWITCH_OBS__')
  const events = obs.events
  const click = events.find((e) => e.name === 'click' && e.profile === PROFILE)
  console.log('\n=== PHASES (ms after click) ===')
  for (const e of events) {
    if (e.t < click.t - 5) continue
    console.log(`${(e.t - click.t).toFixed(0).padStart(7)}  ${e.name}${e.profile ? ' [' + e.profile + ']' : ''}${e.error ? ' ' + e.error : ''}${e.url ? ' ' + e.url : ''}`)
  }
  console.log(settled ? `\nsettled (loader gone + rows painted) at ~${Date.now() - t0} ms wall` : '\nTIMEOUT waiting for settle')

  cdp.close()
}

main().catch((e) => { console.error(e); process.exit(1) })
