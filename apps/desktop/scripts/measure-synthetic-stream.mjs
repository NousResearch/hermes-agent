// Measure render cost of a synthetic stream driven through the live $messages atom.
//
// Why synthetic: the user's LLM credits are depleted; we can't fire a real stream.
// The synthetic stream exercises the exact same React pipeline (assistant-ui runtime →
// repository.addOrUpdateMessage → MessagePrimitive re-render → markdown reflow) as a
// real stream. The only thing it does NOT exercise is the gateway → SSE → optimistic-
// merge path, which is orthogonal to the rendering question.
//
// What we record:
//   1) rAF frame intervals (long-frame histogram; >33ms = perceived jank, >100ms = bad)
//   2) PerformanceObserver `longtask` entries (task >50ms blocks input)
//   3) MutationObserver: per-message mutation count & inter-mutation latency
//   4) Optional: typing latency overlay — typing into composer while streaming
//
// Output is plain text suitable for terminal + a JSON sidecar for diffing across runs.

import { writeFileSync } from 'node:fs'

const CDP_HTTP = 'http://127.0.0.1:9222'
const TOKENS = Number(process.env.TOKENS || 300)
const INTERVAL_MS = Number(process.env.INTERVAL_MS || 16)
// Upstream flush throttle to apply in the synthetic driver. Mirrors what the
// real gateway path does in `use-message-stream.scheduleDeltaFlush`. 0
// disables (worst-case, every token = one React commit).
const FLUSH_MIN_MS = Number(process.env.FLUSH_MIN_MS || 0)
const CHUNK = process.env.CHUNK || 'lorem ipsum '
const TYPE_WHILE_STREAMING = process.env.TYPE_WHILE_STREAMING === '1'
const LABEL = process.env.LABEL || 'baseline'
const OUT = process.env.OUT || `frame-times-${LABEL}.json`

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
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.exception?.description || 'eval')
    return r.result.value
  }
  close() { this.ws.close() }
}

function pct(arr, p) {
  if (!arr.length) return 0
  const i = Math.min(arr.length - 1, Math.floor(arr.length * p))
  return arr[i]
}

async function main() {
  const target = await getTarget()
  const cdp = await CDP.open(target.webSocketDebuggerUrl)

  // Sanity check driver is loaded.
  const probeOk = await cdp.eval('!!window.__PERF_DRIVE__ && !!window.__PERF_DRIVE__.stream')
  if (!probeOk) {
    console.error('__PERF_DRIVE__ not on window — did you reload the renderer after editing perf-probe.tsx?')
    cdp.close()
    process.exit(2)
  }

  // Install recorders.
  await cdp.eval(`
    (() => {
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

      window.__MO__ = { mutations: [], stop: false, currentMsg: null }
      const arm = () => {
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
      window.__MO__.arm = arm

      // Optional: typing observer — fires keystroke timings if asked.
      window.__TYP__ = { times: [], stop: false, lastKey: 0 }
      return 'recorders armed'
    })()
  `)

  // Baseline state.
  const base = JSON.parse(await cdp.eval(`
    JSON.stringify({
      assistantCount: document.querySelectorAll('[data-slot="aui_assistant-message-root"]').length,
      atomCount: window.__PERF_DRIVE__.snapshotMsgs()
    })
  `))
  console.log('baseline:', base)

  // Drive a synthetic stream.
  const streamStart = Date.now()
  await cdp.eval(`window.__PERF_DRIVE__.stream({ chunk: ${JSON.stringify(CHUNK)}, intervalMs: ${INTERVAL_MS}, totalTokens: ${TOKENS}, flushMinMs: ${FLUSH_MIN_MS} })`)

  // After the first paint, arm MO on the new message.
  await new Promise((r) => setTimeout(r, 200))
  await cdp.eval('window.__MO__.arm()')

  // Optional: type while streaming.
  if (TYPE_WHILE_STREAMING) {
    await new Promise((r) => setTimeout(r, 400))
    await cdp.eval(`(() => {
      const ed = document.querySelector('[contenteditable="true"]')
      ed.focus()
      window.__TYP__.startedAt = performance.now()
      const text = 'the quick brown fox jumps over the lazy dog '
      let i = 0
      const tick = () => {
        if (i >= text.length) return
        const t0 = performance.now()
        document.execCommand('insertText', false, text[i])
        // requestAnimationFrame to wait for next paint
        requestAnimationFrame(() => {
          window.__TYP__.times.push(performance.now() - t0)
        })
        i++
        setTimeout(tick, 60)
      }
      tick()
      return 'typing'
    })()`)
  }

  // Wait for stream to complete + small grace.
  const expectedMs = TOKENS * INTERVAL_MS + 1500
  await new Promise((r) => setTimeout(r, expectedMs))

  // Pull recordings.
  const data = JSON.parse(await cdp.eval(`
    (() => {
      window.__FT__.stop = true
      window.__LT__.stop = true
      window.__MO__.stop = true
      window.__TYP__.stop = true
      try { window.__LT__.po && window.__LT__.po.disconnect() } catch {}
      try { window.__MO__.obs && window.__MO__.obs.disconnect() } catch {}
      return JSON.stringify({
        frames: window.__FT__.times,
        longtasks: window.__LT__.entries,
        mutations: window.__MO__.mutations,
        typing: window.__TYP__.times,
        finalText: (() => { const a = document.querySelectorAll('[data-slot="aui_assistant-message-root"]'); return a.length ? a[a.length-1].textContent.length : 0 })()
      })
    })()
  `))

  // Reset DOM back to baseline so we don't accumulate fake messages.
  await cdp.eval('window.__PERF_DRIVE__.reset()')

  // Analysis (trim warm-up: drop frames before first mutation timestamp).
  const firstMut = data.mutations[0]?.t
  const frames = data.frames

  // Sum durations to figure out when each frame happened (relative to recorder start).
  const frameTimeline = []
  let acc = 0
  for (const f of frames) { acc += f; frameTimeline.push(acc) }

  // Mutations are in performance.now() ms; frames started recording when we installed
  // the recorder (before stream). To align: compute total stream window from frames
  // after mutation activity began. Simpler heuristic: drop first 500ms of frames as warm-up.
  const WARMUP_MS = 500
  let dropIdx = 0
  for (let i = 0; i < frames.length; i++) {
    if (frameTimeline[i] >= WARMUP_MS) { dropIdx = i; break }
  }
  const streamFrames = frames.slice(dropIdx)

  const buckets = { '<=16.7': 0, '16.7-33': 0, '33-50': 0, '50-100': 0, '100-200': 0, '>200': 0 }
  let frameTotal = 0
  let maxFrame = 0
  for (const f of streamFrames) {
    frameTotal += f
    if (f > maxFrame) maxFrame = f
    if (f <= 16.7) buckets['<=16.7']++
    else if (f <= 33) buckets['16.7-33']++
    else if (f <= 50) buckets['33-50']++
    else if (f <= 100) buckets['50-100']++
    else if (f <= 200) buckets['100-200']++
    else buckets['>200']++
  }
  const sortedFrames = [...streamFrames].sort((a, b) => a - b)
  const fAvgFps = streamFrames.length ? (streamFrames.length / (frameTotal / 1000)).toFixed(1) : 'n/a'
  const fP50 = pct(sortedFrames, 0.5).toFixed(1)
  const fP95 = pct(sortedFrames, 0.95).toFixed(1)
  const fP99 = pct(sortedFrames, 0.99).toFixed(1)
  const slowFrames = streamFrames.filter((f) => f > 33).length
  const veryslowFrames = streamFrames.filter((f) => f > 100).length

  const ltDur = data.longtasks.map((e) => e.duration).sort((a, b) => a - b)
  const ltMs = ltDur.reduce((a, b) => a + b, 0)
  const ltMax = ltDur.length ? ltDur[ltDur.length - 1] : 0
  const ltP95 = pct(ltDur, 0.95)

  // Mutation cadence.
  const mutDurs = []
  for (let i = 1; i < data.mutations.length; i++) mutDurs.push(data.mutations[i].t - data.mutations[i - 1].t)
  mutDurs.sort((a, b) => a - b)
  const mutP50 = pct(mutDurs, 0.5)
  const mutP95 = pct(mutDurs, 0.95)
  const mutMax = mutDurs.length ? mutDurs[mutDurs.length - 1] : 0

  // Typing latency (optional).
  let typingSummary = null
  if (TYPE_WHILE_STREAMING && data.typing.length) {
    const t = [...data.typing].sort((a, b) => a - b)
    typingSummary = {
      n: t.length,
      p50: pct(t, 0.5).toFixed(1),
      p95: pct(t, 0.95).toFixed(1),
      max: t[t.length - 1].toFixed(1)
    }
  }

  const result = {
    label: LABEL,
    timestamp: new Date().toISOString(),
    config: { TOKENS, INTERVAL_MS, CHUNK, TYPE_WHILE_STREAMING, FLUSH_MIN_MS },
    streamWallMs: Date.now() - streamStart,
    frames: {
      total: streamFrames.length,
      avgFps: fAvgFps,
      windowS: (frameTotal / 1000).toFixed(1),
      p50: fP50,
      p95: fP95,
      p99: fP99,
      max: maxFrame.toFixed(1),
      slow33: slowFrames,
      veryslow100: veryslowFrames,
      histogram: buckets
    },
    longtasks: {
      n: data.longtasks.length,
      totalMs: ltMs.toFixed(0),
      maxMs: ltMax.toFixed(1),
      p95Ms: ltP95.toFixed(1)
    },
    mutations: {
      n: data.mutations.length,
      finalTextLen: data.finalText,
      interMutP50ms: mutP50.toFixed(1),
      interMutP95ms: mutP95.toFixed(1),
      interMutMaxMs: mutMax.toFixed(1)
    },
    typing: typingSummary
  }

  writeFileSync(OUT, JSON.stringify(result, null, 2))

  console.log('\n=== SYNTHETIC STREAM RESULTS ===')
  console.log('label:', LABEL, '| tokens:', TOKENS, '@', INTERVAL_MS, 'ms')
  console.log('streamWallMs:', result.streamWallMs)
  console.log('FRAMES: avgFps', fAvgFps, '| p50', fP50, 'ms | p95', fP95, 'ms | p99', fP99, 'ms | max', maxFrame.toFixed(1), 'ms')
  console.log('FRAMES histogram:', buckets)
  console.log('FRAMES slow(>33):', slowFrames, '/ veryslow(>100):', veryslowFrames, 'of', streamFrames.length)
  console.log('LONGTASKS:', data.longtasks.length, '| total', ltMs.toFixed(0), 'ms | max', ltMax.toFixed(1), 'ms | p95', ltP95.toFixed(1), 'ms')
  console.log('MUTATIONS:', data.mutations.length, '| finalLen', data.finalText, 'chars | inter p50', mutP50.toFixed(1), 'ms | p95', mutP95.toFixed(1), 'ms')
  if (typingSummary) console.log('TYPING-WHILE-STREAMING latency: p50', typingSummary.p50, 'ms | p95', typingSummary.p95, 'ms | n=', typingSummary.n)
  console.log('written to', OUT)

  cdp.close()
}

main().catch((e) => { console.error(e); process.exit(1) })
