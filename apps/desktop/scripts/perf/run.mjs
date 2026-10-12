// Desktop perf harness entrypoint.
//
//   node scripts/perf/run.mjs [scenarios...] [flags]
//
// Default (no scenarios): runs the CI suite (stream, keystroke, transcript)
// against a renderer on :9222 and diffs the committed baseline.
//
// Flags:
//   --spawn                launch a fully isolated instance (own user-data-dir +
//                          HERMES_HOME + debug port) instead of attaching
//   --port <n>             CDP port to attach to (default 9222)
//   --dev-port <n>         vite dev-server port to match / spawn (default 5174)
//   --runs <n>             repeat each scenario n times, report the median (default 1)
//   --cpuprofile [dir]     also record a V8 CPU profile per scenario (top-30 self time)
//   --update-baseline      overwrite baseline.json with this run's numbers
//   --json <path>          write the full results JSON here
//   --tier <ci|backend>    run all scenarios of a tier
//   ...scenario opts       e.g. --tokens 600, --turns 400, --real, --a <sid> --b <sid>, --profile <name>
//
// Examples:
//   npm run perf                       # attach to :9222, run CI suite, gate on baseline
//   npm run perf -- --spawn            # isolated instance, no running app needed
//   npm run perf -- stream --cpuprofile --tokens 800
//   npm run perf -- --update-baseline

import { writeFileSync } from 'node:fs'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { withCpuProfile } from './lib/cdp.mjs'
import { compareScenario, loadBaseline, updateBaseline } from './lib/baseline.mjs'
import { attach, buildProdRenderer, coldStartSamples, startIsolatedInstance } from './lib/launch.mjs'
import { cpuProfileTopSelf, median } from './lib/stats.mjs'
import { CI_SCENARIOS, SCENARIOS } from './scenarios/index.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const BASELINE_PATH = join(HERE, 'baseline.json')

function parseArgs(argv) {
  const positional = []
  const flags = {}

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]

    if (arg.startsWith('--')) {
      const [key, inlineValue] = arg.slice(2).split(/=(.*)/s)
      const next = argv[i + 1]

      if (inlineValue !== undefined) {
        flags[key] = inlineValue
      } else if (next === undefined || next.startsWith('--')) {
        flags[key] = true
      } else {
        flags[key] = next
        i++
      }
    } else {
      positional.push(arg)
    }
  }

  return { positional, flags }
}

function medianMetrics(runs) {
  const keys = new Set(runs.flatMap(r => Object.keys(r)))
  const out = {}

  for (const key of keys) {
    const values = runs.map(r => r[key]).filter(v => typeof v === 'number')
    out[key] = values.length ? Math.round(median(values) * 10) / 10 : runs[0][key]
  }

  return out
}

function printMetrics(name, metrics, comparison) {
  console.log(`\n● ${name}`)
  const byMetric = new Map((comparison?.rows ?? []).map(r => [r.metric, r]))

  for (const [metric, value] of Object.entries(metrics)) {
    const row = byMetric.get(metric)

    if (!row || row.baseline === null) {
      console.log(`   ${metric.padEnd(26)} ${String(value).padStart(9)}${row ? '   (new)' : ''}`)
    } else {
      const tag = row.status === 'REGRESSED' ? '  ✗ REGRESSED' : '  ✓'
      const delta = row.deltaPct === null ? '' : ` (${row.deltaPct > 0 ? '+' : ''}${row.deltaPct}%)`
      console.log(
        `   ${metric.padEnd(26)} ${String(value).padStart(9)}  vs ${String(row.baseline).padStart(9)}${delta}${tag}`
      )
    }
  }
}

async function main() {
  const { positional, flags } = parseArgs(process.argv.slice(2))

  let names = positional
  if (!names.length) {
    names = flags.tier ? Object.values(SCENARIOS).filter(s => s.tier === flags.tier).map(s => s.name) : CI_SCENARIOS
  }

  const unknown = names.filter(n => !SCENARIOS[n])
  if (unknown.length) {
    console.error(`unknown scenario(s): ${unknown.join(', ')}\nknown: ${Object.keys(SCENARIOS).join(', ')}`)
    process.exit(2)
  }

  const runs = Number(flags.runs ?? 1)
  const port = Number(flags.port ?? 9222)
  const devPort = Number(flags['dev-port'] ?? 5174)
  const prod = 'prod' in flags
  const cpuProfile = 'cpuprofile' in flags
  const cpuProfileDir = typeof flags.cpuprofile === 'string' ? flags.cpuprofile : HERE

  const coldNames = names.filter(n => SCENARIOS[n].tier === 'cold')
  const liveNames = names.filter(n => SCENARIOS[n].tier !== 'cold')

  // ci + cold metrics are stable enough to gate against the baseline; backend
  // scenarios vary too much with the live environment, so they're report-only.
  const GATED = new Set(['ci', 'cold'])
  const baseline = loadBaseline(BASELINE_PATH)
  const results = []
  let regressed = false

  const record = (name, tier, metrics, detail) => {
    const comparison = GATED.has(tier) ? compareScenario(name, metrics, baseline) : null
    regressed = regressed || Boolean(comparison?.regressed)
    results.push({ name, tier, metrics, detail })
    printMetrics(name, metrics, comparison)
  }

  if (prod) {
    if (!flags.spawn) {
      console.error('--prod requires --spawn (it builds and launches an isolated production renderer)')
      process.exit(2)
    }

    console.log('[perf] building production renderer with the probe (VITE_PERF_PROBE=1)…')
    await buildProdRenderer()
  }

  // Cold start measures the launch itself → a fresh spawn per run.
  if (coldNames.length) {
    if (!flags.spawn) {
      console.error('cold-start requires --spawn (it measures a fresh launch)')
      process.exit(2)
    }

    // Representative WARM-cache samples (see coldStartSamples). Pass --cold-fresh
    // to instead measure the worst-case first-launch (cold code cache).
    const perRun = await coldStartSamples({ runs, port, devPort, prod, warm: !('cold-fresh' in flags) })

    record('cold-start', 'cold', medianMetrics(perRun), { runs, warm: !('cold-fresh' in flags) })
  }

  // Steady-state scenarios share one persistent connection.
  if (liveNames.length) {
    const connection = flags.spawn
      ? await startIsolatedInstance({ port, devPort, prod })
      : await attach({ port, match: prod ? undefined : String(devPort) })

    const { cdp, teardown } = connection

    try {
      for (const name of liveNames) {
        const scenario = SCENARIOS[name]
        const perRun = []
        let detail = null

        for (let i = 0; i < runs; i++) {
          if (cpuProfile && i === 0) {
            const { result, profile } = await withCpuProfile(cdp, () => scenario.run(cdp, flags))
            const out = join(cpuProfileDir, `${name}-${Date.now()}.cpuprofile`)
            writeFileSync(out, JSON.stringify(profile))
            console.log(`\n[cpuprofile] wrote ${out}`)
            console.log('[cpuprofile] top self-time (ms):')
            for (const r of cpuProfileTopSelf(profile, 15)) {
              console.log(`   ${r.ms.toFixed(1).padStart(7)}  ${r.name.padEnd(38)}  ${r.url}:${r.line}`)
            }
            perRun.push(result.metrics)
            detail = result.detail
          } else {
            const result = await scenario.run(cdp, flags)
            perRun.push(result.metrics)
            detail = result.detail
          }
        }

        record(name, scenario.tier, medianMetrics(perRun), detail)
      }
    } finally {
      teardown()
    }
  }

  if (flags.json) {
    writeFileSync(resolve(String(flags.json)), `${JSON.stringify({ timestamp: new Date().toISOString(), results }, null, 2)}\n`)
    console.log(`\nwrote ${flags.json}`)
  }

  if (flags['update-baseline']) {
    updateBaseline(BASELINE_PATH, results.filter(r => GATED.has(r.tier)))
    console.log(`\nupdated ${BASELINE_PATH}`)
    return
  }

  if (regressed) {
    console.error('\n✗ perf regression vs baseline (see REGRESSED rows above)')
    process.exit(1)
  }

  console.log('\n✓ no perf regressions')
}

main().catch(err => {
  console.error('\nperf harness failed:', err.stack ?? err.message)
  process.exit(1)
})
