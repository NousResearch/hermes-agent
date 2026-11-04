// Submit (Enter) latency + scroll stability. Subsumes measure-submit and
// measure-jump. Backend tier: fires a REAL prompt, so run it on a throwaway
// session with a live backend. Report-only (no committed baseline — real
// round-trips are too environment-dependent to gate).

import { SELECTORS, sleep, typeIntoComposer } from '../lib/cdp.mjs'
import { summarize } from '../lib/stats.mjs'

const MEASURE = `
  new Promise((resolve) => {
    const composer = document.querySelector(${JSON.stringify(SELECTORS.composer)})
    const thread = document.querySelector(${JSON.stringify(SELECTORS.threadContent)}) ||
                   document.querySelector(${JSON.stringify(SELECTORS.threadViewport)})
    const viewport = document.querySelector(${JSON.stringify(SELECTORS.threadViewport)})
    const startCount = thread ? thread.querySelectorAll(${JSON.stringify(SELECTORS.turnPair)}).length : 0
    const startScroll = viewport ? viewport.scrollTop : 0
    const m = { start: performance.now(), maxJumpPx: 0 }
    let done = false

    const finish = (reason) => {
      if (done) return
      done = true
      clearTimeout(timer); composerObs.disconnect(); threadObs && threadObs.disconnect()
      m.reason = reason
      resolve(m)
    }

    const composerObs = new MutationObserver(() => {
      if (!m.composerClearedMs && composer && composer.innerText.length === 0) {
        m.composerClearedMs = performance.now() - m.start
      }
    })
    composer && composerObs.observe(composer, { childList: true, subtree: true, characterData: true })

    let threadObs = null
    if (thread) {
      threadObs = new MutationObserver(() => {
        if (viewport) m.maxJumpPx = Math.max(m.maxJumpPx, Math.abs(viewport.scrollTop - startScroll))
        const c = thread.querySelectorAll(${JSON.stringify(SELECTORS.turnPair)}).length
        if (!m.userMsgRenderedMs && c > startCount) {
          m.userMsgRenderedMs = performance.now() - m.start
          requestAnimationFrame(() => { m.userMsgPaintMs = performance.now() - m.start; finish('paint') })
        }
      })
      threadObs.observe(thread, { childList: true, subtree: true })
    }

    const timer = setTimeout(() => finish('timeout'), 5000)
    composer && composer.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true, cancelable: true }))
  })
`

export default {
  name: 'submit',
  tier: 'backend',
  description: 'Enter → composer cleared → user message painted, plus scroll jump.',
  async run(cdp, opts = {}) {
    const rounds = Number(opts.rounds ?? 3)
    await cdp.send('Runtime.enable')

    const clears = []
    const paints = []
    const jumps = []

    for (let i = 0; i < rounds; i++) {
      await typeIntoComposer(cdp, `perf submit round ${i} ${'x'.repeat(30)}`, { cps: 60 })
      await sleep(250)
      const m = await cdp.eval(MEASURE)

      if (typeof m.composerClearedMs === 'number') clears.push(m.composerClearedMs)
      if (typeof m.userMsgPaintMs === 'number') paints.push(m.userMsgPaintMs)
      jumps.push(m.maxJumpPx ?? 0)

      // Let the turn finish before the next round so they don't pile up.
      await sleep(4000)
    }

    return {
      metrics: {
        submit_clear_p95_ms: summarize(clears).p95,
        submit_paint_p95_ms: summarize(paints).p95,
        submit_scroll_jump_max_px: Math.max(0, ...jumps)
      },
      detail: { rounds, clears: summarize(clears), paints: summarize(paints) }
    }
  }
}
