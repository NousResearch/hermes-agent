// What does a window cost in CPU seconds while nothing is running — and what
// does "a turn is open but silent" cost per process?
//
// #127647's reports ("Hermes is heavy") resolve into per-process facts the
// renderer-side scenarios cannot see: the same composited animation that costs
// ~12% of one core inside the GPU helper on hardware GL burns ~1.5-9 cores
// there when the app is pinned to software rendering (#124255), while the
// renderer itself reads as idle. This scenario diffs CDP
// `SystemInfo.getProcessInfo` CPU-time counters across fixed windows in two
// phases:
//
//   viewed-quiet  window shown, no session content, nothing pending
//   viewed-busy   N busy-but-silent tiles (idle-cost's seeding), nothing arriving
//
// Report-only (tier 'report'): per-process CPU seconds over the recorded wall
// time, median of the runs the runner applies. A healthy parked window scores
// ~0.0 in every row; any sustained deltas ARE the tracker's idle-burn signal.
//
//   node scripts/perf/run.mjs idle-burn --spawn [--tiles 3] [--seconds 10]

import { sampleProcessCpu, diffProcessCpu } from '../lib/processes.mjs'
import { CLEANUP, reveal, seedBusyTiles } from './idle-cost.mjs'

const round = (n, places = 2) => Math.round(n * 10 ** places) / 10 ** places

const sleep = ms => new Promise(r => setTimeout(r, ms))

export default {
  name: 'idle-burn',
  // Report-only until #128134's thresholds land: the scenario establishes the
  // numbers; it does not gate a pass/fail on host variance it has not seen.
  tier: 'report',
  description: 'Per-process CPU deltas (main / gpu / renderer / …) at idle and with busy-but-silent tiles.',
  async run(cdp, opts = {}) {
    const tiles = Math.max(1, Number(opts.tiles ?? 3))
    const seedTurns = Math.max(1, Number(opts.turns ?? 20))
    const seconds = Math.max(2, Number(opts.seconds ?? 10))
    const settleMs = Math.max(0, Number(opts['settle-ms'] ?? 2500))
    const port = opts.perfPort ? Number(opts.perfPort) : undefined

    await cdp.send('Runtime.enable')
    await sleep(settleMs)

    // ── phase 1: viewed-quiet ──────────────────────────────────────────────
    const quietBefore = await sampleProcessCpu(cdp, { port })
    const quietStart = Date.now()
    await sleep(seconds * 1000)
    const quietAfter = await sampleProcessCpu(cdp, { port })
    const quietElapsed = (Date.now() - quietStart) / 1000
    const quietDelta = diffProcessCpu(quietBefore, quietAfter)

    // ── phase 2: viewed-busy (silent streaming state) ──────────────────────
    const ok = await cdp.eval(seedBusyTiles(tiles, seedTurns))

    if (ok !== 'ok') {
      return {
        metrics: { gpu_cpu_s_quiet: round(quietDelta.gpu ?? 0), elapsed_s: round(quietElapsed) },
        detail: { error: `idle-burn: lifecycle setup failed (${ok}); busy phase skipped — quiet phase is valid` }
      }
    }

    for (let n = 1; n <= tiles; n++) {
      await cdp.eval(reveal(`idle-tile-${n}`))
      await sleep(200)
    }

    await sleep(settleMs)

    const busyBefore = await sampleProcessCpu(cdp, { port })
    const busyStart = Date.now()
    await sleep(seconds * 1000)
    const busyAfter = await sampleProcessCpu(cdp, { port })
    const busyElapsed = (Date.now() - busyStart) / 1000
    const busyDelta = diffProcessCpu(busyBefore, busyAfter)

    await cdp.eval(CLEANUP)

    const metrics = {}

    for (const [label, delta] of Object.entries(quietDelta)) {
      metrics[`${label}_cpu_s_quiet`] = round(delta)
    }
    for (const [label, delta] of Object.entries(busyDelta)) {
      metrics[`${label}_cpu_s_busy`] = round(delta)
    }

    return {
      metrics,
      detail: {
        tiles,
        seedTurns,
        seconds,
        quietElapsed: round(quietElapsed),
        busyElapsed: round(busyElapsed),
        // Raw rows, for per-pid questions a regression triage asks next.
        quietProcesses: quietAfter,
        busyProcesses: busyAfter
      }
    }
  }
}
