// Scroll-up-through-long-history performance. Empirical reproduction of the
// "app gets very slow after prolonged usage, especially when scrolling up in
// long conversation history" complaint — measurement and attribution only, no
// fixes.
//
// What it does:
//   1. For each transcript size (default 200/400/800 turns): load synthetic
//      mixed-markdown turns via __PERF_DRIVE__.loadTranscript, then scroll UP
//      from the bottom in a fixed number of steps, N passes (up-then-jump-back),
//      recording per-pass wall time, per-step frame deltas (dropped frames),
//      and longtask count/total/max. Heap + DOM counters before/after.
//   2. A "prolonged usage" leg: age the SAME renderer without reload (repeated
//      loadTranscript/reset cycles + synthetic streams), then re-measure the
//      400-turn scroll-up and compare against a fresh Page.reload of the same
//      instance — quantifying degradation-after-usage.
//
// CPU attribution: the driver (or run.mjs --cpuprofile) wraps this run; when
// opts.profile is set, every scroll phase is individually wrapped in
// withCpuProfile and the profiles are returned via detail.profiles.

import { SELECTORS, sleep, withCpuProfile } from '../lib/cdp.mjs'
import { frameHistogram, percentile } from '../lib/stats.mjs'

const VIEWPORT = JSON.stringify(SELECTORS.threadViewport)

// One up-scroll pass, evaluated in the page. Jumps to the bottom, settles,
// then steps scrollTop toward 0 in `steps` equal strides (stride derived from
// the live scrollHeight so every pass covers the full distance), one rAF per
// step. Frame pacing is measured as step-to-step deltas — a long task on the
// main thread shows up directly as an inflated delta / dropped frames.
const SCROLL_PASS = cfg => `(async () => {
  const vp = document.querySelector(${VIEWPORT})
  if (!vp) throw new Error('thread viewport not found: ${SELECTORS.threadViewport}')
  const steps = ${cfg.steps}
  vp.scrollTop = vp.scrollHeight
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))
  await new Promise(r => setTimeout(r, ${cfg.settleMs}))
  const lt = []
  let po = null
  try {
    po = new PerformanceObserver(l => { for (const e of l.getEntries()) lt.push({ s: e.startTime, d: e.duration }) })
    po.observe({ entryTypes: ['longtask'] })
  } catch {}
  const deltas = []
  let last = performance.now()
  const evalStart = performance.now()
  const t0 = performance.now()
  for (let i = 0; i < steps && vp.scrollTop > 0; i++) {
    const stride = Math.max(200, Math.ceil(vp.scrollHeight / steps))
    vp.scrollTop = Math.max(0, vp.scrollTop - stride)
    await new Promise(r => requestAnimationFrame(r))
    const now = performance.now()
    deltas.push(now - last)
    last = now
  }
  const wallMs = performance.now() - t0
  const evalWallMs = performance.now() - evalStart
  await new Promise(r => setTimeout(r, 300)) // drain async longtask reports
  const longtasks = lt.concat(po ? po.takeRecords().map(e => ({ s: e.startTime, d: e.duration })) : [])
  try { po && po.disconnect() } catch {}
  return { wallMs, evalWallMs, deltas, longtasks, scrollTopLeft: vp.scrollTop }
})()`

function passMetrics(pass, deltas, longtasks) {
  const ltTotal = longtasks.reduce((a, b) => a + b.d, 0)
  return {
    wall_ms: Math.round(pass.wallMs * 10) / 10,
    eval_wall_ms: Math.round(pass.evalWallMs * 10) / 10,
    longtask_n: longtasks.length,
    longtask_total_ms: Math.round(ltTotal * 10) / 10,
    longtask_max_ms: Math.round((longtasks.length ? Math.max(...longtasks.map(e => e.d)) : 0) * 10) / 10,
    dropped_frames_25ms: deltas.filter(d => d > 25).length,
    dropped_frames_50ms: deltas.filter(d => d > 50).length,
    delta_p95_ms: Math.round(percentile(deltas, 0.95) * 10) / 10,
    delta_max_ms: Math.round(Math.max(...deltas, 0) * 10) / 10,
    // raw samples for post-hoc sanity checks (overlaps, timing anomalies)
    deltas: deltas.map(d => Math.round(d * 10) / 10),
    longtasks: longtasks.map(e => ({ s: Math.round(e.s), d: Math.round(e.d * 10) / 10 }))
  }
}

async function heapSnapshot(cdp) {
  const [usage, dom] = await Promise.all([
    cdp.send('Runtime.getHeapUsage').catch(() => null),
    cdp.send('Memory.getDOMCounters').catch(() => null)
  ])
  return {
    heap_used_mb: usage ? Math.round(Number(usage.usedSize) / 1048576 * 10) / 10 : null,
    dom_nodes: dom?.nodes ?? null,
    dom_documents: dom?.documents ?? null
  }
}

async function runPasses(cdp, cfg) {
  const passes = []
  for (let p = 1; p <= cfg.passes; p++) {
    const r = await cdp.eval(SCROLL_PASS(cfg))
    passes.push({ pass: p, ...passMetrics(r, r.deltas, r.longtasks), histogram: frameHistogram(r.deltas) })
    // Jump back to the bottom and settle before the next pass.
    await cdp.eval(`(() => { const vp = document.querySelector(${VIEWPORT}); if (vp) vp.scrollTop = vp.scrollHeight; return true })()`)
    await sleep(cfg.settleMs)
  }
  return passes
}

async function waitDriver(cdp, timeoutMs = 90000) {
  const deadline = Date.now() + timeoutMs
  while (Date.now() < deadline) {
    try {
      if (await cdp.eval('!!(window.__PERF_DRIVE__ && window.__PERF_DRIVE__.loadTranscript)')) return true
    } catch { /* renderer mid-reload */ }
    await sleep(500)
  }
  return false
}

async function measurePhase(cdp, { turns, passes, steps, settleMs, label, profile, profiles }) {
  const phase = { label, turns }

  const mounted = await cdp.eval(`window.__PERF_DRIVE__.loadTranscript(${turns})`)
  phase.mount_ms = Math.round(mounted * 10) / 10
  const msgs = Number(await cdp.eval('window.__PERF_DRIVE__.snapshotMsgs()'))
  phase.messages = msgs
  await sleep(settleMs)

  phase.heap_before = await heapSnapshot(cdp)
  const cfg = { passes, steps, settleMs }

  const measure = () => runPasses(cdp, cfg)
  const result = profile
    ? await withCpuProfile(cdp, measure).then(r => {
        profiles.push({ label: `${label}-${turns}t`, profile: r.profile })
        return r.result
      })
    : await measure()

  phase.passes = result
  phase.heap_after = await heapSnapshot(cdp)
  await cdp.eval('window.__PERF_DRIVE__.reset()')
  await sleep(1500)
  return phase
}

export default {
  name: 'scroll-up',
  tier: 'report',
  description:
    'Scrolling UP through long conversation history: per-pass longtasks/frame drops, heap/DOM growth, and degradation after prolonged synthetic usage (same renderer) vs a fresh reload.',
  async run(cdp, opts = {}) {
    const turnList = (opts.turnList ?? '200,400,800')
      .toString()
      .split(',')
      .map(s => Number(s.trim()))
      .filter(n => n > 0)
    const passes = Number(opts.passes ?? 4)
    const steps = Number(opts.steps ?? 40)
    const settleMs = Number(opts.settleMs ?? 800)
    const agingCycles = Number(opts.agingCycles ?? 6)
    const agingStreams = Number(opts.agingStreams ?? 4)
    const agingTurns = Number(opts.agingTurns ?? 400)
    const agingStreamTokens = Number(opts.agingStreamTokens ?? 300)
    const profileTurns = Number(opts.profileTurns ?? 400)
    const profile = Boolean(opts.profile)
    const profiles = []
    const detail = { profiles }

    await cdp.send('Runtime.enable')
    if (!(await waitDriver(cdp, 15000))) {
      throw new Error('__PERF_DRIVE__ not available (needs the dev renderer with the perf probe)')
    }

    // ── Phase 1: size sweep on the freshly booted renderer ──────────────────
    detail.sizes = []
    for (const turns of turnList) {
      const cfg = { turns, passes, steps, settleMs, label: 'size', profile: profile && turns === profileTurns, profiles }
      detail.sizes.push(await measurePhase(cdp, cfg))
    }

    // ── Phase 2: prolong the session in the SAME renderer, no reload ────────
    // Alternate transcript churn (mount/reset) with synthetic streams — the
    // shape of hours of real usage compressed to ~a minute.
    const aging = { cycles: [], streams: [] }
    for (let i = 0; i < agingCycles; i++) {
      const ms = await cdp.eval(`window.__PERF_DRIVE__.loadTranscript(${agingTurns})`)
      aging.cycles.push(Math.round(ms * 10) / 10)
      await sleep(500)
      await cdp.eval('window.__PERF_DRIVE__.reset()')
      await sleep(300)
    }
    for (let i = 0; i < agingStreams; i++) {
      const t0 = Date.now()
      await cdp.eval(
        `window.__PERF_DRIVE__.stream({ intervalMs: 16, totalTokens: ${agingStreamTokens}, flushMinMs: 33 })`
      )
      await sleep(agingStreamTokens * 16 + 1500)
      aging.streams.push(Date.now() - t0)
    }
    detail.aging = aging

    // ── Phase 3: re-measure scroll-up AGED, same renderer ───────────────────
    const agedCfg = { turns: profileTurns, passes, steps, settleMs, label: 'aged', profile, profiles }
    detail.aged = await measurePhase(cdp, agedCfg)

    // ── Phase 4: fresh renderer reference — same instance, reloaded ─────────
    await cdp.send('Page.enable')
    await cdp.send('Page.reload')
    if (!(await waitDriver(cdp))) {
      throw new Error('renderer did not come back after Page.reload')
    }
    await sleep(2500)
    const freshCfg = { turns: profileTurns, passes, steps, settleMs, label: 'fresh', profile, profiles }
    detail.fresh = await measurePhase(cdp, freshCfg)

    // ── Aggregate flat metrics (lower is better) ────────────────────────────
    const metrics = {}
    for (const [sz, phase] of [...detail.sizes.map(s => [String(s.turns), s]), ['aged', detail.aged], ['fresh', detail.fresh]]) {
      const first = phase.passes[0]
      const last = phase.passes[phase.passes.length - 1]
      metrics[`pass1_ms_${sz}`] = first.wall_ms
      metrics[`passN_ms_${sz}`] = last.wall_ms
      metrics[`pass1_lt_ms_${sz}`] = first.longtask_total_ms
      metrics[`passN_lt_ms_${sz}`] = last.longtask_total_ms
      metrics[`pass1_dropped_${sz}`] = first.dropped_frames_25ms
      metrics[`passN_dropped_${sz}`] = last.dropped_frames_25ms
      metrics[`heap_grow_mb_${sz}`] =
        first != null && phase.heap_after?.heap_used_mb != null && phase.heap_before?.heap_used_mb != null
          ? Math.round((phase.heap_after.heap_used_mb - phase.heap_before.heap_used_mb) * 10) / 10
          : null
      metrics[`dom_nodes_${sz}`] = phase.heap_after?.dom_nodes ?? null
    }

    // Degradation after prolonged usage: aged vs fresh (same turns, same box).
    for (const key of ['wall_ms', 'longtask_total_ms', 'dropped_frames_25ms']) {
      for (let p = 0; p < passes; p++) {
        const a = detail.aged.passes[p]?.[key]
        const f = detail.fresh.passes[p]?.[key]
        if (typeof a === 'number' && typeof f === 'number' && f > 0) {
          metrics[`aged_fresh_${key}_p${p + 1}`] = Math.round((a / f) * 100) / 100
        }
      }
    }

    return { metrics, detail }
  }
}
