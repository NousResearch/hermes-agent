// Baseline + regression gate. This is the capability the old one-off scripts
// never had: measured numbers are compared against a committed baseline so a
// PR that regresses streaming/typing/mount cost fails loudly instead of
// silently drifting.
//
// Every tracked metric is "lower is better" (longtask counts, frame/keystroke
// percentiles, mount ms). A metric regresses when it exceeds
// `baseline * (1 + tolFrac) + tolAbs`. tolAbs absorbs sub-millisecond jitter on
// already-fast metrics so they don't false-positive.

import { readFileSync, writeFileSync } from 'node:fs'

const DEFAULT_TOLERANCE = { tolFrac: 0.25, tolAbs: 1 }

export function loadBaseline(path) {
  try {
    return JSON.parse(readFileSync(path, 'utf8'))
  } catch {
    return { _meta: {}, scenarios: {} }
  }
}

/**
 * Compare a scenario's measured metrics against the baseline.
 * @returns {{ rows: Array, regressed: boolean }}
 */
export function compareScenario(name, measured, baseline) {
  const base = baseline.scenarios?.[name]
  const tol = { ...DEFAULT_TOLERANCE, ...(base?.tolerance ?? {}) }
  const rows = []
  let regressed = false

  for (const [metric, value] of Object.entries(measured)) {
    if (typeof value !== 'number') {
      continue
    }

    const baseValue = base?.metrics?.[metric]

    if (typeof baseValue !== 'number') {
      rows.push({ metric, measured: value, baseline: null, limit: null, status: 'new' })

      continue
    }

    const limit = baseValue * (1 + tol.tolFrac) + tol.tolAbs
    const over = value > limit
    regressed = regressed || over

    rows.push({
      metric,
      measured: value,
      baseline: baseValue,
      limit: Math.round(limit * 100) / 100,
      deltaPct: baseValue ? Math.round(((value - baseValue) / baseValue) * 1000) / 10 : null,
      status: over ? 'REGRESSED' : 'ok'
    })
  }

  return { rows, regressed }
}

/** Write measured metrics back as the new baseline for the given scenarios. */
export function updateBaseline(path, results) {
  const baseline = loadBaseline(path)
  baseline.scenarios ??= {}

  for (const { name, metrics } of results) {
    const numeric = Object.fromEntries(Object.entries(metrics).filter(([, v]) => typeof v === 'number'))
    const prev = baseline.scenarios[name] ?? {}
    baseline.scenarios[name] = { ...prev, metrics: numeric }
  }

  baseline._meta = {
    ...baseline._meta,
    updated: new Date().toISOString(),
    platform: `${process.platform}-${process.arch}`,
    node: process.version
  }

  writeFileSync(path, `${JSON.stringify(baseline, null, 2)}\n`)
}

export { DEFAULT_TOLERANCE }
