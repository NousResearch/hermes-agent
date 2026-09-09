// Session-switch latency. Subsumes profile-session-switch. In ordinary mode it
// needs two real stored session ids and a live backend (report-only). The
// `--verified-cache` / `--delay-runtime` modes use the isolated renderer's
// synthetic, no-network fixture and are mechanically gated.
//
//   node scripts/perf/run.mjs session-switch --a <sidA> --b <sidB> [--rounds 2]

import { SELECTORS, sleep } from '../lib/cdp.mjs'
import { summarize } from '../lib/stats.mjs'

const CONTROLLED_RESULT_SLOT = '__HERMES_PERF_SESSION_SWITCH__'

async function runControlledFixtureRound(cdp, payload, round, timeoutMs) {
  const slot = `${CONTROLLED_RESULT_SLOT}${round}`

  await cdp.eval(`(() => {
    const slot = ${JSON.stringify(slot)}
    const promise = window.__PERF_DRIVE__.sessionSwitch(${JSON.stringify(payload)})
    window[slot] = { state: 'pending', promise }
    promise.then(
      value => { window[slot] = { state: 'fulfilled', value } },
      error => {
        window[slot] = {
          state: 'rejected',
          error: error instanceof Error ? (error.stack || error.message) : String(error)
        }
      }
    )
    return true
  })()`)

  const deadline = Date.now() + timeoutMs

  try {
    while (Date.now() < deadline) {
      const result = await cdp.eval(`window[${JSON.stringify(slot)}] ?? null`)

      if (result?.state === 'fulfilled') {
        return result.value
      }

      if (result?.state === 'rejected') {
        throw new Error(result.error || `session-switch fixture round ${round + 1} rejected`)
      }

      await sleep(10)
    }

    throw new Error(`session-switch fixture round ${round + 1} timed out after ${timeoutMs}ms`)
  } finally {
    await cdp.eval(`delete window[${JSON.stringify(slot)}]`)
  }
}

export default {
  name: 'session-switch',
  tier: 'backend',
  description: 'Route to a session, or run a controlled cache/runtime open fixture.',
  async run(cdp, opts = {}) {
    const { a, b } = opts
    const rounds = Number(opts.rounds ?? 2)
    const settleTimeoutMs = Number(opts.settleTimeoutMs ?? 30000)
    const verifiedCache = opts['verified-cache'] === true
    const hasDelayRuntime = opts['delay-runtime'] !== undefined
    const delayRuntimeMs = hasDelayRuntime ? Number(opts['delay-runtime']) : 0
    const roundSettleMs = Number(opts['round-settle-ms'] ?? 100)
    const controlledFixture = verifiedCache || hasDelayRuntime

    if (controlledFixture) {
      if (!Number.isFinite(delayRuntimeMs) || delayRuntimeMs < 0) {
        throw new Error('--delay-runtime must be a non-negative number of milliseconds')
      }

      if (!Number.isInteger(rounds) || rounds < 1) {
        throw new Error('--rounds must be a positive integer')
      }

      if (!Number.isFinite(roundSettleMs) || roundSettleMs < 0) {
        throw new Error('--round-settle-ms must be a non-negative number')
      }

      if (verifiedCache && rounds < 20) {
        throw new Error('--verified-cache requires at least 20 rounds')
      }

      const hasFixture = await cdp.eval('typeof window.__PERF_DRIVE__?.sessionSwitch === "function"')

      if (!hasFixture) {
        throw new Error('session-switch controlled fixture requires a perf-probe renderer')
      }

      const cacheFirstPaints = []
      const restCommits = []
      const resumeReadies = []
      const agentReadies = []
      let restBeforeResumeCount = 0
      const samples = []

      for (let round = 0; round < rounds; round++) {
        const result = await runControlledFixtureRound(cdp, {
          delayRuntimeMs,
          verifiedCache
        }, round, settleTimeoutMs)
        const marks = result?.marks ?? {}
        const select = marks['hermes.session.select']
        const cache = marks['hermes.session.cache.commit']
        const cachePaintReady = result?.cachePaintReady
        const rest = marks['hermes.session.rest.commit']
        const resume = marks['hermes.session.resume.ready']
        const agent = marks['hermes.session.agent.ready']

        if (![select, resume].every(value => typeof value === 'number')) {
          throw new Error(`session-switch fixture round ${round + 1} omitted a required select/resume mark`)
        }

        if (!verifiedCache && typeof rest !== 'number') {
          throw new Error(`session-switch fixture round ${round + 1} omitted its required REST publish mark`)
        }

        if (verifiedCache) {
          if (typeof cache !== 'number') {
            throw new Error(`session-switch fixture round ${round + 1} did not commit its verified cache`)
          }

          if (typeof cachePaintReady !== 'number' || cachePaintReady < cache) {
            throw new Error(`session-switch fixture round ${round + 1} did not observe a post-cache render`)
          }

          cacheFirstPaints.push(cachePaintReady - select)
        }

        if (typeof rest === 'number') {
          restCommits.push(rest - select)
        }
        resumeReadies.push(resume - select)

        if (typeof agent === 'number') {
          agentReadies.push(agent - select)
        }

        if (typeof rest === 'number' && rest < resume) {
          restBeforeResumeCount += 1
        }

        samples.push({ cache, cachePaintReady, rest, resume, select })

        // Each first-paint sample must begin after the previous round's
        // transition backfill and cleanup have had a chance to settle. The
        // pause is outside every select→paint interval, so it stabilizes the
        // benchmark without improving the measured product latency.
        if (round + 1 < rounds && roundSettleMs > 0) {
          await sleep(roundSettleMs)
        }
      }

      const cacheFirstPaint = summarize(cacheFirstPaints)
      const restCommit = summarize(restCommits)
      const resumeReady = summarize(resumeReadies)
      const agentReady = summarize(agentReadies)

      if (verifiedCache && cacheFirstPaint.p95 > 100) {
        throw new Error(
          `verified cache first-paint p95 ${cacheFirstPaint.p95}ms exceeds 100ms ` +
            `(min=${cacheFirstPaint.min}ms, p50=${cacheFirstPaint.p50}ms, p90=${cacheFirstPaint.p90}ms, ` +
            `p95=${cacheFirstPaint.p95}ms, max=${cacheFirstPaint.max}ms; ` +
            `rounds=${cacheFirstPaints.map(value => Math.round(value * 10) / 10).join(',')})`
        )
      }

      if (hasDelayRuntime && !verifiedCache && restBeforeResumeCount !== rounds) {
        throw new Error(
          `delayed runtime requires REST before resume in every round (${restBeforeResumeCount}/${rounds} observed)`
        )
      }

      return {
        metrics: {
          cache_first_paint_p95_ms: verifiedCache ? cacheFirstPaint.p95 : null,
          rest_commit_p95_ms: restCommits.length ? restCommit.p95 : null,
          resume_ready_p95_ms: resumeReady.p95,
          agent_ready_p95_ms: agentReadies.length ? agentReady.p95 : null,
          rest_before_resume_count: restBeforeResumeCount
        },
        detail: {
          agent_ready: agentReadies.length ? 'reported' : 'unavailable-current-backend-protocol',
          delayedRuntimeMs: hasDelayRuntime ? delayRuntimeMs : null,
          fixture: 'synthetic-local-no-network',
          rounds,
          samples,
          verifiedCache
        }
      }
    }

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
