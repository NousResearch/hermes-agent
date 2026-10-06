// Baseline + regression gate. This is the capability the old one-off scripts
// never had: measured numbers are compared against a committed baseline so a
// PR that regresses streaming/typing/mount cost fails loudly instead of
// silently drifting.
//
// Every tracked metric is "lower is better" (longtask counts, frame/keystroke
// percentiles, mount ms). A metric regresses when it exceeds
// `baseline * (1 + tolFrac) + tolAbs`. tolAbs absorbs sub-millisecond jitter on
// already-fast metrics so they don't false-positive.
//
// PER-PLATFORM. Every number in here is a property of the machine that produced
// it, not of the app alone — spawn, parse and mount costs are CPU- and
// disk-bound, so a committed baseline only means something on the platform that
// captured it. The file therefore holds one bucket per `platform-arch` and the
// gate only ever compares against the CURRENT machine's bucket. The read used
// to ignore the platform entirely, so a Windows or Linux contributor was gated
// against darwin-arm64 numbers, while `--update-baseline` overwrote those same
// numbers (and the `_meta.platform` label) — taking the reference away from the
// machine it was measured on. A machine with no bucket of its own reports its
// metrics and is deliberately NOT gated.

import { readFileSync, writeFileSync } from 'node:fs'

const DEFAULT_TOLERANCE = { tolFrac: 0.25, tolAbs: 1 }

/** The baseline bucket for a machine: `platform-arch` (e.g. `darwin-arm64`). */
export function platformKey(platform = process.platform, arch = process.arch) {
  return `${platform}-${arch}`
}

function readDoc(path) {
  try {
    const parsed = JSON.parse(readFileSync(path, 'utf8'))

    return parsed && typeof parsed === 'object' ? parsed : {}
  } catch {
    return {}
  }
}

/**
 * Every platform bucket in a document, including one migrated from the legacy
 * single-platform shape.
 *
 * The old file was `{ _meta: { platform }, scenarios }` — one platform, no room
 * for a second. It migrates to `{ platforms: { [that platform]: { scenarios } } }`
 * so an existing committed baseline keeps its numbers, attributed to the
 * platform it was actually captured on.
 */
export function baselinePlatforms(doc) {
  if (doc?.platforms && typeof doc.platforms === 'object') {
    return doc.platforms
  }

  const legacyPlatform = doc?._meta?.platform
  const scenarios = doc?.scenarios

  if (legacyPlatform && scenarios && typeof scenarios === 'object' && Object.keys(scenarios).length) {
    return { [legacyPlatform]: { scenarios } }
  }

  return {}
}

/**
 * Load the baseline for `key`.
 *
 * Returns `{ _meta, scenarios, sourcePlatform, gated, availablePlatforms }`.
 * `scenarios` is the bucket a caller compares against; `gated` is false when
 * this machine has no entry, which the runner reports rather than silently
 * treating every metric as an untracked "new" one.
 */
export function loadBaseline(path, key = platformKey()) {
  const doc = readDoc(path)
  const platforms = baselinePlatforms(doc)
  const entry = platforms[key]

  return {
    _meta: doc._meta ?? {},
    scenarios: entry?.scenarios ?? {},
    sourcePlatform: key,
    gated: Boolean(entry),
    availablePlatforms: Object.keys(platforms).sort()
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

/**
 * Write measured metrics into `key`'s bucket as the new baseline for those
 * scenarios, leaving every other platform's bucket untouched.
 *
 * `--update-baseline` is run by whoever is iterating on a machine, so the write
 * must not reach past that machine: the darwin-arm64 numbers a Mac captured are
 * the reference the gate compares against and are not reproducible on the box
 * doing the update.
 *
 * @returns {{ platform: string, scenarios: string[] }}
 */
export function updateBaseline(path, results, key = platformKey()) {
  const doc = readDoc(path)
  const platforms = { ...baselinePlatforms(doc) }
  const entry = platforms[key] ?? {}
  const scenarios = { ...(entry.scenarios ?? {}) }

  for (const { name, metrics } of results) {
    const numeric = Object.fromEntries(Object.entries(metrics).filter(([, v]) => typeof v === 'number'))
    const prev = scenarios[name] ?? {}

    scenarios[name] = { ...prev, metrics: numeric }
  }

  platforms[key] = { ...entry, scenarios }

  // The legacy single-platform `_meta.platform` is superseded by the buckets;
  // leaving it behind would keep claiming one platform owns the whole file.
  const { platform: _legacyPlatform, ...meta } = doc._meta ?? {}

  writeFileSync(
    path,
    `${JSON.stringify(
      {
        _meta: {
          ...meta,
          updated: new Date().toISOString(),
          node: process.version,
          platforms: Object.keys(platforms).sort()
        },
        platforms
      },
      null,
      2
    )}\n`
  )

  return { platform: key, scenarios: Object.keys(scenarios).sort() }
}

export { DEFAULT_TOLERANCE }
