// Streaming render cost. Subsumes measure-synthetic-stream, profile-synth-stream,
// profile-long-stream (synthetic) and measure-real-stream / profile-real-stream
// (via --real). CPU profiling is provided by the runner's --cpuprofile flag.
//
// Metrics (lower is better): longtask count + max, frame p95/p99, slow-frame
// count, inter-mutation p95. These are what "is streaming smooth?" reduces to.

import { SELECTORS, sleep, typeIntoComposer } from '../lib/cdp.mjs'
import { frameHistogram, percentile } from '../lib/stats.mjs'

const RECORDERS = `
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
        for (const e of list.getEntries()) window.__LT__.entries.push({ duration: e.duration, startTime: e.startTime })
      })
      po.observe({ entryTypes: ['longtask'] })
      window.__LT__.po = po
    } catch {}

    window.__MO__ = { mutations: [], stop: false, current: null }
    window.__MO__.arm = () => {
      const all = document.querySelectorAll(${JSON.stringify(SELECTORS.assistantMessage)})
      const last = all[all.length - 1]
      if (!last || last === window.__MO__.current) return
      window.__MO__.current = last
      window.__MO__.obs && window.__MO__.obs.disconnect()
      const obs = new MutationObserver(() => {
        if (window.__MO__.stop) return
        window.__MO__.mutations.push({ t: performance.now(), len: last.textContent.length })
      })
      obs.observe(last, { childList: true, subtree: true, characterData: true })
      window.__MO__.obs = obs
    }
    return 'armed'
  })()
`

const COLLECT = `
  (() => {
    window.__FT__.stop = true
    window.__LT__.stop = true
    window.__MO__.stop = true
    try { window.__LT__.po && window.__LT__.po.disconnect() } catch {}
    try { window.__MO__.obs && window.__MO__.obs.disconnect() } catch {}
    return JSON.stringify({
      frames: window.__FT__.times,
      longtasks: window.__LT__.entries,
      mutations: window.__MO__.mutations
    })
  })()
`

function analyze(data, warmupMs) {
  // Drop warm-up frames (recorder installs before the stream starts).
  const frames = []
  let acc = 0

  for (const f of data.frames) {
    acc += f

    if (acc >= warmupMs) {
      frames.push(f)
    }
  }

  const interMut = []

  for (let i = 1; i < data.mutations.length; i++) {
    interMut.push(data.mutations[i].t - data.mutations[i - 1].t)
  }

  const ltDurations = data.longtasks.map(e => e.duration)
  const windowS = frames.reduce((a, b) => a + b, 0) / 1000

  return {
    metrics: {
      longtasks_n: data.longtasks.length,
      longtask_max_ms: Math.round((ltDurations.length ? Math.max(...ltDurations) : 0) * 10) / 10,
      frame_p95_ms: Math.round(percentile(frames, 0.95) * 10) / 10,
      frame_p99_ms: Math.round(percentile(frames, 0.99) * 10) / 10,
      slow_frames_33: frames.filter(f => f > 33).length,
      intermut_p95_ms: Math.round(percentile(interMut, 0.95) * 10) / 10
    },
    detail: {
      windowS: Math.round(windowS * 10) / 10,
      avgFps: windowS ? Math.round((frames.length / windowS) * 10) / 10 : 0,
      frameHistogram: frameHistogram(frames),
      mutations: data.mutations.length,
      finalLen: data.mutations.at(-1)?.len ?? 0
    }
  }
}

export default {
  name: 'stream',
  tier: 'ci',
  description: 'Assistant-message streaming: longtasks, frame pacing, mutation cadence.',
  async run(cdp, opts = {}) {
    const tokens = Number(opts.tokens ?? 400)
    const intervalMs = Number(opts.intervalMs ?? 16)
    const flushMinMs = Number(opts.flushMinMs ?? 33)
    const chunk = opts.chunk ?? '**word** in _italic_ with `code`, a [link](https://x.dev) and prose. '
    const real = Boolean(opts.real)

    await cdp.send('Runtime.enable')
    await cdp.eval(RECORDERS)

    if (real) {
      // Backend path: fire a real prompt and wait for the stream to appear.
      const baseCount = await cdp.eval(`document.querySelectorAll(${JSON.stringify(SELECTORS.assistantMessage)}).length`)
      await typeIntoComposer(cdp, opts.prompt ?? 'count from 1 to 80, one number per line', { cps: 40 })
      await cdp.eval(`(() => {
        const el = document.querySelector(${JSON.stringify(SELECTORS.composer)})
        el && el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true, cancelable: true }))
      })()`)

      const deadline = Date.now() + Number(opts.timeoutMs ?? 60000)
      let started = false

      while (Date.now() < deadline) {
        await sleep(50)
        const n = await cdp.eval(`document.querySelectorAll(${JSON.stringify(SELECTORS.assistantMessage)}).length`)

        if (n > baseCount) {
          started = true
          break
        }
      }

      if (!started) {
        throw new Error('real stream never started (no LLM credit / backend?)')
      }

      await cdp.eval('window.__MO__.arm()')
      // Let it run to completion or timeout.
      const runDeadline = Date.now() + Number(opts.runMs ?? 30000)

      while (Date.now() < runDeadline) {
        await sleep(250)
        const busy = await cdp.eval(`!!document.querySelector('[data-status="running"], [data-busy="true"]')`)

        if (!busy) {
          break
        }
      }
    } else {
      // Synthetic path: drive $messages directly. No LLM, no credits.
      await cdp.eval(
        `window.__PERF_DRIVE__.stream({ chunk: ${JSON.stringify(chunk)}, intervalMs: ${intervalMs}, totalTokens: ${tokens}, flushMinMs: ${flushMinMs} })`
      )
      await sleep(200)
      await cdp.eval('window.__MO__.arm()')
      await sleep(tokens * intervalMs + 1500)
    }

    const data = JSON.parse(await cdp.eval(COLLECT))

    if (!real) {
      await cdp.eval('window.__PERF_DRIVE__.reset()')
    }

    return analyze(data, real ? 0 : 500)
  }
}
