// Observe the existing workspace without seeding sessions or changing focus.
// Run timing separately from --cpuprofile / render counters: attribution costs
// must not be reported as interaction latency.
import { SELECTORS } from '../lib/cdp.mjs'
import { frameHistogram, summarize } from '../lib/stats.mjs'

export default {
  name: 'live-window',
  tier: 'report',
  description: 'Existing workspace: frame pacing, long tasks, and heap/DOM size; no synthetic sessions.',
  async run(cdp, opts = {}) {
    const seconds = Number(opts.seconds ?? 15)
    if (!Number.isFinite(seconds) || seconds <= 0 || seconds > 60) {
      throw new Error('live-window --seconds must be between 0 and 60')
    }

    const beforeHeap = await cdp.send('Runtime.getHeapUsage')
    const beforeDOM = await cdp.send('Memory.getDOMCounters')
    const sample = await cdp.eval(`(async () => {
      if (window.__RENDER_COUNTS__?.recording() || window.__ATOM_CHURN__?.recording()) {
        throw new Error('Stop render/atom attribution before measuring frame pacing')
      }
      const inventory = () => {
        const threads = [...document.querySelectorAll(${JSON.stringify(SELECTORS.threadViewport)})]
        return {
          threads: threads.length,
          visibleThreads: threads.filter(e => e.checkVisibility({ visibilityProperty: true })).length,
          elements: document.querySelectorAll('*').length,
          focused: document.hasFocus(),
          visibility: document.visibilityState
        }
      }
      const before = inventory()
      if (before.visibility !== 'visible' || !before.focused || !before.visibleThreads) {
        throw new Error('live-window requires a visible workspace with mounted transcripts')
      }
      const frames = [], longTasks = []
      let last, frame
      const observer = new PerformanceObserver(list => {
        longTasks.push(...list.getEntries().map(e => e.duration))
      })
      const tick = time => {
        if (last !== undefined) frames.push(time - last)
        last = time
        frame = requestAnimationFrame(tick)
      }
      const start = performance.now()
      observer.observe({ type: 'longtask' })
      frame = requestAnimationFrame(tick)
      try {
        await new Promise(resolve => setTimeout(resolve, ${seconds * 1000}))
      } finally {
        cancelAnimationFrame(frame)
        longTasks.push(...observer.takeRecords().map(e => e.duration))
        observer.disconnect()
      }
      return { before, after: inventory(), elapsed: performance.now() - start, frames, longTasks }
    })()`)
    const afterHeap = await cdp.send('Runtime.getHeapUsage')
    const afterDOM = await cdp.send('Memory.getDOMCounters')
    const frameMs = summarize(sample.frames)
    const elapsedFrames = sample.frames.reduce((sum, ms) => sum + ms, 0)
    const fps = elapsedFrames ? sample.frames.length * 1000 / elapsedFrames : 0

    return {
      metrics: {
        frame_p95_ms: frameMs.p95,
        frame_p99_ms: frameMs.p99,
        worst_frame_ms: frameMs.max,
        long_tasks: sample.longTasks.length,
        blocking_ms: sample.longTasks.reduce((sum, ms) => sum + Math.max(0, ms - 50), 0)
      },
      detail: {
        ...sample,
        fps,
        stableWorkspace: sample.after.focused && sample.after.visibility === 'visible' &&
          sample.before.threads === sample.after.threads && sample.before.visibleThreads === sample.after.visibleThreads,
        frameMs,
        histogram: frameHistogram(sample.frames),
        beforeHeap,
        afterHeap,
        beforeDOM,
        afterDOM
      }
    }
  }
}
