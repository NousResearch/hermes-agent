// CPU-profile a session switch — outputs a .cpuprofile, a top-self ranking,
// longtask timings, and paint milestones for cold + warm switches.
//
// Drives the real resume path by setting location.hash (same code path as a
// sidebar click: use-route-resume → resumeSession → prefetch + resume RPC).
//
// Usage:
//   node apps/desktop/scripts/profile-session-switch.mjs <sessionA> <sessionB> [rounds]
//   OUT=/tmp/switch.cpuprofile node scripts/profile-session-switch.mjs 2026.. 2026..

import { writeFileSync } from 'node:fs'

const CDP_HTTP = process.env.CDP_HTTP || 'http://127.0.0.1:9222'
const A = process.argv[2]
const B = process.argv[3]
const ROUNDS = Number(process.argv[4] || 2)
const OUT = process.env.OUT || `/tmp/session-switch-${Date.now()}.cpuprofile`
const SETTLE_TIMEOUT = Number(process.env.SETTLE_TIMEOUT || 30000)

if (!A || !B) {
  console.error('usage: profile-session-switch.mjs <sessionA> <sessionB> [rounds]')
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

  // Install observers once: longtasks + rAF frame gaps, tagged per switch.
  await cdp.eval(`(() => {
    if (window.__SWITCH_OBS__) return 'already'
    const obs = { longtasks: [], marks: [] }
    new PerformanceObserver((l) => {
      for (const e of l.getEntries()) obs.longtasks.push({ t: e.startTime, dur: e.duration })
    }).observe({ entryTypes: ['longtask'] })
    window.__SWITCH_OBS__ = obs
    return 'installed'
  })()`)

  const switchTo = async (sid, label) => {
    const t0 = await cdp.eval(`(() => {
      const o = window.__SWITCH_OBS__
      o.marks.push({ label: ${JSON.stringify(label)}, sid: ${JSON.stringify(sid)}, t: performance.now() })
      location.hash = '#/' + ${JSON.stringify(sid)}
      return performance.now()
    })()`)

    // Poll until the transcript for this session has painted and settled:
    // route matches, >0 message roots, and message count stable for 3 polls.
    const deadline = Date.now() + SETTLE_TIMEOUT
    let stable = 0
    let lastCount = -1
    let firstPaintT = null
    while (Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 50))
      const s = await cdp.eval(`({
        t: performance.now(),
        route: location.hash,
        msgs: document.querySelectorAll('[data-slot="aui_message"], [data-slot="aui_assistant-message-root"], [data-slot="aui_user-message-root"]').length,
        parts: document.querySelectorAll('[data-slot="aui_thread-content"] *').length
      })`)
      if (!s.route.includes(sid)) continue
      if (s.msgs > 0 && firstPaintT === null) firstPaintT = s.t
      stable = s.msgs === lastCount && s.msgs > 0 ? stable + 1 : 0
      lastCount = s.msgs
      if (stable >= 3) return { t0, firstPaintT, settledT: s.t, msgs: s.msgs, domNodes: s.parts }
    }
    return { t0, firstPaintT, settledT: null, msgs: lastCount, timedOut: true }
  }

  console.log('starting CPU profile')
  await cdp.send('Profiler.enable')
  await cdp.send('Profiler.setSamplingInterval', { interval: 100 })
  await cdp.send('Profiler.start')

  const results = []
  for (let round = 0; round < ROUNDS; round++) {
    for (const [sid, tag] of [[A, 'A'], [B, 'B']]) {
      const label = `round${round}:${tag}:${round === 0 ? 'cold' : 'warm'}`
      const r = await switchTo(sid, label)
      results.push({ label, sid, ...r })
      const ftp = r.firstPaintT != null ? (r.firstPaintT - r.t0).toFixed(0) : 'n/a'
      const st = r.settledT != null ? (r.settledT - r.t0).toFixed(0) : 'TIMEOUT'
      console.log(`${label.padEnd(18)} first-paint ${String(ftp).padStart(6)} ms   settled ${String(st).padStart(6)} ms   msgs ${r.msgs}  dom ${r.domNodes ?? '?'}`)
      await new Promise((r2) => setTimeout(r2, 800))
    }
  }

  const { profile } = await cdp.send('Profiler.stop')
  writeFileSync(OUT, JSON.stringify(profile))
  console.log('\nwrote', OUT)

  // Longtasks per switch window.
  const obs = await cdp.eval('window.__SWITCH_OBS__')
  console.log('\n=== LONGTASKS (>=50ms main-thread blocks) ===')
  for (let i = 0; i < obs.marks.length; i++) {
    const m = obs.marks[i]
    const end = obs.marks[i + 1]?.t ?? Infinity
    const lts = obs.longtasks.filter((lt) => lt.t >= m.t && lt.t < end)
    const total = lts.reduce((a, b) => a + b.dur, 0)
    console.log(`${m.label.padEnd(18)} ${String(lts.length).padStart(2)} longtasks, ${total.toFixed(0).padStart(5)} ms total  ${lts.map((l) => Math.round(l.dur)).join(', ')}`)
  }

  // Self-time ranking.
  const samples = profile.samples || []
  const timeDeltas = profile.timeDeltas || []
  const nodes = new Map(profile.nodes.map((n) => [n.id, n]))
  const selfTime = new Map()
  for (let i = 0; i < samples.length; i++) {
    selfTime.set(samples[i], (selfTime.get(samples[i]) || 0) + (timeDeltas[i] ?? 0))
  }
  const ranked = [...selfTime.entries()]
    .map(([id, us]) => {
      const cf = nodes.get(id)?.callFrame || {}
      return { ms: us / 1000, name: cf.functionName || '(anonymous)', url: (cf.url || '').slice(-70), line: cf.lineNumber }
    })
    .filter((x) => !/\(root\)|\(idle\)|\(garbage collector\)|\(program\)/.test(x.name))
    .sort((a, b) => b.ms - a.ms)
    .slice(0, 30)

  console.log('\n=== TOP 30 SELF TIME (ms) ACROSS ALL SWITCHES ===')
  for (const r of ranked) {
    console.log(`${r.ms.toFixed(1).padStart(8)}  ${r.name.padEnd(44)}  ${r.url}:${r.line}`)
  }

  cdp.close()
}

main().catch((e) => { console.error(e); process.exit(1) })
