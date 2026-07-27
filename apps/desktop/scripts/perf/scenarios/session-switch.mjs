// Session-switch latency. Subsumes profile-session-switch. Backend tier: needs
// two real stored session ids and a live backend. Report-only.
//
//   node scripts/perf/run.mjs session-switch --a <sidA> --b <sidB> [--rounds 2]

import { SELECTORS, sleep } from '../lib/cdp.mjs'
import { summarize } from '../lib/stats.mjs'

export default {
  name: 'session-switch',
  tier: 'backend',
  description: 'Route to a session and wait for first-paint + settle of its transcript.',
  requiredOpts: ['a', 'b'],
  async run(cdp, opts = {}) {
    const { a, b } = opts
    const rounds = Number(opts.rounds ?? 2)
    const settleTimeoutMs = Number(opts.settleTimeoutMs ?? 30000)

    if (!a || !b) {
      throw new Error('session-switch needs --a <sessionId> --b <sessionId>')
    }

    await cdp.send('Runtime.enable')

    const switchTo = async sid => {
      const t0 = await cdp.eval(`(() => { location.hash = '#/' + ${JSON.stringify(sid)}; return performance.now() })()`)
      const deadline = Date.now() + settleTimeoutMs
      let firstPaint = null
      let stable = 0
      let lastCount = -1

      while (Date.now() < deadline) {
        await sleep(50)
        const s = await cdp.eval(`({
          t: performance.now(),
          route: location.hash,
          msgs: document.querySelectorAll(${JSON.stringify(SELECTORS.assistantMessage)}).length
        })`)

        if (!String(s.route).includes(sid)) {
          continue
        }

        if (s.msgs > 0 && firstPaint === null) {
          firstPaint = s.t - t0
        }

        stable = s.msgs === lastCount && s.msgs > 0 ? stable + 1 : 0
        lastCount = s.msgs

        if (stable >= 3) {
          return { firstPaint, settled: s.t - t0 }
        }
      }

      return { firstPaint, settled: null }
    }

    const firstPaints = []
    const settles = []

    for (let round = 0; round < rounds; round++) {
      for (const sid of [a, b]) {
        const r = await switchTo(sid)

        if (typeof r.firstPaint === 'number') firstPaints.push(r.firstPaint)
        if (typeof r.settled === 'number') settles.push(r.settled)
        await sleep(800)
      }
    }

    return {
      metrics: {
        switch_first_paint_p95_ms: summarize(firstPaints).p95,
        switch_settled_p95_ms: summarize(settles).p95
      },
      detail: { rounds, firstPaint: summarize(firstPaints), settled: summarize(settles) }
    }
  }
}
