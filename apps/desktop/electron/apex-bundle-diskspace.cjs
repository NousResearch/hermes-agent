'use strict'

/**
 * apex-bundle-diskspace.cjs — hc-472 P1 · C2 (disk watermark + install precheck)
 *
 * WHY (design §4 水位 / §8 预检 — hermes-cloud docs/work-notes/DESIGN-hc472-runtime-bundle.md)
 * -----------------------------------------------------------------------------
 * Two disk guards around the versioned bundle catalog:
 *
 *   1. WATERMARK — versions/ should hold ~two extracted bundles (current +
 *      previous ≈ 2.06 GiB). When usage exceeds a budget (a third version or
 *      bloat crept in) we TIGHTEN: a GC that drops `previous` to reclaim ~1
 *      bundle early, plus a warning. A working `current` beats an instant-rollback
 *      target the disk can't hold (design §4 "不足则先 GC previous").
 *
 *   2. PRECHECK — before pulling the ~0.6 GB archive, refuse if the target volume
 *      can't hold the archive + its extracted staging dir, with a readable error
 *      (design §8 "安装/切换前预检剩余空间"). The archive coexists with the
 *      extracted `.tmp` staging until the post-commit rm, so the requirement is
 *      archive + one extracted bundle + a safety margin, floored at the design's
 *      "single ×2.5".
 *
 * Both thresholds are env-tunable (HERMES_BUNDLE_VERSIONS_BUDGET_BYTES /
 * HERMES_BUNDLE_MIN_FREE_BYTES) with sane defaults; explicit opts always win so
 * every branch is deterministically unit-testable (inject dirSizeOf / freeBytesOf
 * — no real disk needed).
 */

const fs = require('node:fs')
const path = require('node:path')

const layout = require('./apex-bundle-layout.cjs')

const GIB = 1024 * 1024 * 1024
// Accounting anchor: one EXTRACTED bundle ≈ 1.03 GiB (design §2 "1.1–1.4 GB
// 解压"; the ticket pins 1.03 GiB as the retain-2 unit). ≈ 1,105,954,079 bytes.
const SINGLE_BUNDLE_EXTRACTED_BYTES = Math.round(1.03 * GIB)
// versions/ budget: two healthy bundles (current+previous ≈ 2.06 GiB) plus half a
// bundle of slack. Exceeding it → tighten GC (drop previous) + warn. ≈ 2.58 GiB.
const DEFAULT_VERSIONS_BUDGET_BYTES = Math.round(2.5 * SINGLE_BUNDLE_EXTRACTED_BYTES)
// Free-space floor for an install (design §8 "≥ 单份 ×2.5"). When the archive
// size is known we take the LARGER of this floor and (archive + one extracted
// bundle + margin), so the requirement never drops below the design rule. ≈ 2.58 GiB.
const DEFAULT_INSTALL_MIN_FREE_BYTES = Math.round(2.5 * SINGLE_BUNDLE_EXTRACTED_BYTES)
const INSTALL_SAFETY_MARGIN_BYTES = Math.round(0.5 * GIB)

function parseEnvBytes(v) {
  if (v == null) return null
  const n = Number.parseInt(String(v).trim(), 10)
  return Number.isFinite(n) && n > 0 ? n : null
}

function currentPlatform(opts) {
  return (opts && opts.platform) || process.platform
}

/** Recursive byte sum of a dir tree. Best-effort: unreadable entries count 0. */
function dirSize(dir) {
  let total = 0
  let entries
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true })
  } catch {
    return 0
  }
  for (const e of entries) {
    const p = path.join(dir, e.name)
    if (e.isDirectory()) {
      total += dirSize(p)
    } else {
      try {
        total += fs.lstatSync(p).size
      } catch {
        // unreadable / vanished — count 0
      }
    }
  }
  return total
}

/** Per-committed-version + staging usage under versions/. */
function versionsUsage(hermesHome, opts = {}) {
  const { versionsDir } = layout.bundlePaths(hermesHome)
  const sizeOf = typeof opts.dirSizeOf === 'function' ? opts.dirSizeOf : dirSize
  const { versions, staging } = layout.listVersions(hermesHome)
  const byDir = {}
  let total = 0
  for (const name of [...versions, ...staging]) {
    const bytes = sizeOf(path.join(versionsDir, name))
    byDir[name] = bytes
    total += bytes
  }
  return { total, byDir, count: versions.length, staging: staging.length }
}

/** Free bytes on the volume holding `p` (statfs). 0 when unavailable. */
function freeBytesAt(p) {
  try {
    const st = fs.statfsSync(p)
    return st.bavail * st.bsize
  } catch {
    return 0
  }
}

function resolveVersionsBudget(opts = {}) {
  if (Number.isFinite(opts.budgetBytes) && opts.budgetBytes > 0) return opts.budgetBytes
  return parseEnvBytes(process.env.HERMES_BUNDLE_VERSIONS_BUDGET_BYTES) || DEFAULT_VERSIONS_BUDGET_BYTES
}

function resolveInstallMinFree(opts = {}, archiveSize) {
  if (Number.isFinite(opts.minFreeBytes) && opts.minFreeBytes > 0) return opts.minFreeBytes
  const floor = parseEnvBytes(process.env.HERMES_BUNDLE_MIN_FREE_BYTES) || DEFAULT_INSTALL_MIN_FREE_BYTES
  if (Number.isFinite(archiveSize) && archiveSize > 0) {
    return Math.max(floor, archiveSize + SINGLE_BUNDLE_EXTRACTED_BYTES + INSTALL_SAFETY_MARGIN_BYTES)
  }
  return floor
}

/**
 * If versions/ usage exceeds the budget, run a TIGHT GC (drop previous, keep only
 * current) to reclaim ~1 bundle early and surface a warning; otherwise a normal
 * keep-current+previous GC. Returns usage before/after + whether it tightened.
 */
function enforceVersionsWatermark(hermesHome, opts = {}) {
  const budget = resolveVersionsBudget(opts)
  const before = versionsUsage(hermesHome, opts).total
  const overBudget = before > budget
  const gc = layout.garbageCollect(hermesHome, {
    platform: currentPlatform(opts),
    dropPrevious: overBudget,
    isLocked: opts.isLocked
  })
  const after = versionsUsage(hermesHome, opts).total
  const warning = overBudget
    ? `versions/ at ${before} bytes exceeds budget ${budget}; dropped previous to reclaim (now ${after})`
    : null
  return { overBudget, tightened: overBudget, budget, before, after, gc, warning }
}

/**
 * Refuse an install BEFORE the ~0.6 GB download when the target volume can't hold
 * the archive + its extracted staging dir (design §8 预检). Returns a readable
 * message the shell surfaces verbatim. `freeBytesOf` is injected in tests.
 */
function preflightDiskSpace(o = {}) {
  const { hermesHome, archiveSize } = o
  const freeOf = typeof o.freeBytesOf === 'function' ? o.freeBytesOf : freeBytesAt
  const requiredBytes = resolveInstallMinFree(o, archiveSize)
  const freeBytes = freeOf(hermesHome)
  if (freeBytes >= requiredBytes) return { ok: true, freeBytes, requiredBytes }
  const gib = n => (n / GIB).toFixed(2)
  return {
    ok: false,
    reason: 'insufficient_disk',
    freeBytes,
    requiredBytes,
    message:
      `Not enough disk space to install the runtime: need ~${gib(requiredBytes)} GiB free, ` +
      `but only ${gib(freeBytes)} GiB is available on the drive holding ${hermesHome}. ` +
      'Free up space and try again.'
  }
}

module.exports = {
  GIB,
  SINGLE_BUNDLE_EXTRACTED_BYTES,
  DEFAULT_VERSIONS_BUDGET_BYTES,
  DEFAULT_INSTALL_MIN_FREE_BYTES,
  INSTALL_SAFETY_MARGIN_BYTES,
  dirSize,
  versionsUsage,
  freeBytesAt,
  resolveVersionsBudget,
  resolveInstallMinFree,
  enforceVersionsWatermark,
  preflightDiskSpace
}
