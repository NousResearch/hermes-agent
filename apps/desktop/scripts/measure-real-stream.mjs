// REAL streaming measurement — no React internals.
//
// Measures:
//   1) rAF frame intervals during a verified live stream (long-frame histogram)
//   2) MutationObserver: how often does the live assistant message mutate, what's the budget per mutation
//   3) Text length growth rate (chars/sec)
//   4) PerformanceObserver `longtask` entries (any task > 50ms blocks input)
//
// Detects REAL stream by waiting for assistant-message DOM count to grow past baseline.
// Does NOT cancel — lets the stream run to completion or hits TIMEOUT_MS.

const CDP_HTTP = 'http://127.0.0.1:9222'
const PROMPT = process.env.PROMPT || 'count from 1 to 80, one number per line'
const TIMEOUT_MS = Number(process.env.TIMEOUT_MS || 60000)

async function getTarget() {
  const list = await (await fetch(`${CDP_HTTP}/json`)).json()
  const t = list.find((t) => t.type === 'page' && /5174/.test(t.url))
  if (!t) throw new Error('renderer not found')
  return t
}

class CDP {
  constructor(ws) { this.ws = ws; this.id = 0; this.pending = new Map() }
  static async open(url) {
    const ws = new WebSocket(url)
    await new Promise((r, j) => {
      ws.addEventListener('open', r, { once: true })
      ws.addEventListener('error', (e) => j(e), { once: true })
    })
    const cdp = new CDP(ws)
    ws.addEventListener('message', (event) => {
      const m = JSON.parse(event.data.toString())
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
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || 'eval')
    return r.result.value
  }
  close() { this.ws.close() }
}

async function main() {
  const target = await getTarget()
  const cdp = await CDP.open(target.webSocketDebuggerUrl)

  // Install recorders.
  await cdp.eval(`
    (() => {
      // rAF frame intervals
      window.__FT__ = { times: [], stop: false }
      let last = performance.now()
      const tick = () => {
        if (window.__FT__.stop) return
        const now = performance.now()
        window.__FT__.times.push(now - last)
        last = now
        requestAnimationFrame(tick)
      }
      requestAnimationFrame(tick)

      // longtask observer
      window.__LT__ = { entries: [], stop: false }
      try {
        const po = new PerformanceObserver((list) => {
          if (window.__LT__.stop) return
          for (const e of list.getEntries()) {
            window.__LT__.entries.push({ name: e.name, duration: e.duration, startTime: e.startTime })
          }
        })
        po.observe({ entryTypes: ['longtask'] })
        window.__LT__.po = po
      } catch {}

      // mutation observer on streaming message
      window.__MO__ = { mutations: [], stop: false, currentMsg: null }
      const tryArm = () => {
        const all = document.querySelectorAll('[data-slot="aui_assistant-message-root"]')
        const last = all[all.length - 1]
        if (!last || last === window.__MO__.currentMsg) return
        window.__MO__.currentMsg = last
        if (window.__MO__.obs) window.__MO__.obs.disconnect()
        const obs = new MutationObserver((muts) => {
          if (window.__MO__.stop) return
          const t = performance.now()
          window.__MO__.mutations.push({ t, count: muts.length, len: last.textContent.length })
        })
        obs.observe(last, { childList: true, subtree: true, characterData: true })
        window.__MO__.obs = obs
      }
      window.__MO__.arm = tryArm
      return 'recorders armed'
    })()
  `)

  // Baseline
  const base = JSON.parse(await cdp.eval(`
    JSON.stringify({
      assistantCount: document.querySelectorAll('[data-slot="aui_assistant-message-root"]').length,
      busy: !!document.querySelector('[data-status="running"], [data-busy="true"]'),
      hasComposer: !!document.querySelector('[contenteditable="true"]'),
    })
  `))
  console.log('baseline:', base)
  if (!base.hasComposer) { console.error('no composer'); cdp.close(); return }

  // Type + submit
  await cdp.eval(`
    (() => {
      const ed = document.querySelector('[contenteditable="true"]')
      ed.focus()
      document.execCommand('insertText', false, ${JSON.stringify(PROMPT)})
      return 'typed'
    })()
  `)
  const submitT0 = Date.now()
  await cdp.eval(`
    (() => {
      const ed = document.querySelector('[contenteditable="true"]')
      ed.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true, cancelable: true }))
      return 'submitted'
    })()
  `)

  // Poll for REAL stream (assistant count > baseline). 30 seconds — accommodates
  // slow first-token latencies on big providers.
  let realStreamT = null
  for (let i = 0; i < 600; i++) {
    await new Promise((r) => setTimeout(r, 50))
    const s = JSON.parse(await cdp.eval(`
      JSON.stringify({
        n: document.querySelectorAll('[data-slot="aui_assistant-message-root"]').length,
        busy: !!document.querySelector('[data-status="running"], [data-busy="true"]'),
        text: (() => { const a = document.querySelectorAll('[data-slot="aui_assistant-message-root"]'); return a.length ? a[a.length-1].textContent.length : 0 })()
      })
    `))
    if (s.n > base.assistantCount) {
      realStreamT = Date.now()
      console.log('REAL stream started after', realStreamT - submitT0, 'ms — busy=', s.busy, 'text=', s.text)
      // Arm mutation observer on the new message
      await cdp.eval('window.__MO__.arm()')
      break
    }
  }
  if (!realStreamT) {
    console.error('REAL STREAM NEVER STARTED')
    cdp.close()
    return
  }

  // Sample length growth, wait for completion or timeout
  const samples = []
  const start = Date.now()
  while (Date.now() - start < TIMEOUT_MS) {
    await new Promise((r) => setTimeout(r, 250))
    const s = JSON.parse(await cdp.eval(`
      JSON.stringify({
        t: performance.now(),
        len: (() => { const a = document.querySelectorAll('[data-slot="aui_assistant-message-root"]'); return a.length ? a[a.length-1].textContent.length : 0 })(),
        busy: !!document.querySelector('[data-status="running"], [data-busy="true"]')
      })
    `))
    samples.push(s)
    if (!s.busy && samples.length > 4) {
      await new Promise((r) => setTimeout(r, 300))
      break
    }
  }

  // Pull recordings
  const data = JSON.parse(await cdp.eval(`
    (() => {
      window.__FT__.stop = true
      window.__LT__.stop = true
      window.__MO__.stop = true
      try { window.__LT__.po && window.__LT__.po.disconnect() } catch {}
      try { window.__MO__.obs && window.__MO__.obs.disconnect() } catch {}
      return JSON.stringify({
        frames: window.__FT__.times,
        longtasks: window.__LT__.entries,
        mutations: window.__MO__.mutations,
      })
    })()
  `))

  const { frames, longtasks, mutations } = data

  // Frame histogram (filter to stream window)
  const buckets = { '<=16.7': 0, '16.7-33': 0, '33-50': 0, '50-100': 0, '100-200': 0, '>200': 0 }
  let frameTotal = 0
  let maxFrame = 0
  for (const f of frames) {
    frameTotal += f
    if (f > maxFrame) maxFrame = f
    if (f <= 16.7) buckets['<=16.7']++
    else if (f <= 33) buckets['16.7-33']++
    else if (f <= 50) buckets['33-50']++
    else if (f <= 100) buckets['50-100']++
    else if (f <= 200) buckets['100-200']++
    else buckets['>200']++
  }
  const avgFps = frames.length ? (frames.length / (frameTotal / 1000)).toFixed(1) : 'n/a'
  const slowFrames = frames.filter((f) => f > 33).length
  const veryslowFrames = frames.filter((f) => f > 100).length

  // Longtask summary
  const ltMs = longtasks.reduce((a, b) => a + b.duration, 0)
  const ltMax = longtasks.length ? Math.max(...longtasks.map((e) => e.duration)) : 0

  // Mutation rate
  let mutTotal = mutations.length
  let mutDurs = []
  for (let i = 1; i < mutations.length; i++) {
    mutDurs.push(mutations[i].t - mutations[i - 1].t)
  }
  mutDurs.sort((a, b) => a - b)
  const mutP50 = mutDurs[Math.floor(mutDurs.length * 0.5)] ?? 0
  const mutP95 = mutDurs[Math.floor(mutDurs.length * 0.95)] ?? 0

  // Growth rate
  const firstLen = samples[0]?.len ?? 0
  const lastLen = samples[samples.length - 1]?.len ?? 0
  const elapsedS = samples.length ? (samples[samples.length - 1].t - samples[0].t) / 1000 : 0
  const charsPerSec = elapsedS ? ((lastLen - firstLen) / elapsedS).toFixed(1) : 'n/a'

  console.log('\n=== STREAM RESULTS ===')
  console.log('window:', (frameTotal / 1000).toFixed(1), 's | frames:', frames.length, '| avgFps:', avgFps, '| maxFrame:', maxFrame.toFixed(1), 'ms')
  console.log('frame histogram:', buckets)
  console.log('slow frames (>33ms):', slowFrames, '| very slow (>100ms):', veryslowFrames)
  console.log('longtasks:', longtasks.length, 'total', ltMs.toFixed(0), 'ms — max', ltMax.toFixed(1), 'ms')
  console.log('text grew', firstLen, '→', lastLen, 'chars (', charsPerSec, 'char/s )')
  console.log('mutations on streaming msg:', mutTotal, '| inter-mutation p50:', mutP50.toFixed(1), 'ms', 'p95:', mutP95.toFixed(1), 'ms')

  cdp.close()
}

main().catch((e) => { console.error(e); process.exit(1) })
