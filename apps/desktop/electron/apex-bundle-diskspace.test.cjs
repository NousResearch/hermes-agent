'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const layout = require('./apex-bundle-layout.cjs')
const disk = require('./apex-bundle-diskspace.cjs')

const GIB = disk.GIB

function mkHome() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hb-disk-'))
}
function rm(home) {
  fs.rmSync(home, { recursive: true, force: true })
}
function seedVersion(home, key) {
  const dir = layout.bundlePaths(home).versionDir(key)
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'id.txt'), key)
  return dir
}

// ---------------------------------------------------------------------------
// dirSize + versionsUsage
// ---------------------------------------------------------------------------

test('dirSize: sums nested file bytes; missing dir is 0', () => {
  const home = mkHome()
  try {
    const d = path.join(home, 'tree')
    fs.mkdirSync(path.join(d, 'sub'), { recursive: true })
    fs.writeFileSync(path.join(d, 'a.bin'), Buffer.alloc(1000))
    fs.writeFileSync(path.join(d, 'sub', 'b.bin'), Buffer.alloc(2500))
    assert.equal(disk.dirSize(d), 3500)
    assert.equal(disk.dirSize(path.join(home, 'nope')), 0)
  } finally {
    rm(home)
  }
})

test('versionsUsage: totals committed + staging via injected sizer', () => {
  const home = mkHome()
  try {
    seedVersion(home, 'aaaaaaaaaaaa')
    seedVersion(home, 'bbbbbbbbbbbb')
    fs.mkdirSync(layout.bundlePaths(home).stagingDir('cccccccccccc'), { recursive: true })
    const u = disk.versionsUsage(home, { dirSizeOf: () => 1000 })
    assert.equal(u.total, 3000) // 2 committed + 1 staging
    assert.equal(u.count, 2)
    assert.equal(u.staging, 1)
  } finally {
    rm(home)
  }
})

// ---------------------------------------------------------------------------
// threshold resolution — opts win over env win over default
// ---------------------------------------------------------------------------

test('resolveVersionsBudget: opts > env > default', () => {
  const prev = process.env.HERMES_BUNDLE_VERSIONS_BUDGET_BYTES
  try {
    delete process.env.HERMES_BUNDLE_VERSIONS_BUDGET_BYTES
    assert.equal(disk.resolveVersionsBudget(), disk.DEFAULT_VERSIONS_BUDGET_BYTES)
    process.env.HERMES_BUNDLE_VERSIONS_BUDGET_BYTES = '5000'
    assert.equal(disk.resolveVersionsBudget(), 5000)
    assert.equal(disk.resolveVersionsBudget({ budgetBytes: 9000 }), 9000) // opts wins
  } finally {
    if (prev === undefined) delete process.env.HERMES_BUNDLE_VERSIONS_BUDGET_BYTES
    else process.env.HERMES_BUNDLE_VERSIONS_BUDGET_BYTES = prev
  }
})

test('resolveInstallMinFree: takes the larger of the floor and (archive + extracted + margin)', () => {
  // No archive size → the design floor (single ×2.5).
  assert.equal(disk.resolveInstallMinFree({}), disk.DEFAULT_INSTALL_MIN_FREE_BYTES)
  // A big archive pushes the requirement above the floor.
  const big = 5 * GIB
  const withBig = disk.resolveInstallMinFree({}, big)
  assert.equal(withBig, big + disk.SINGLE_BUNDLE_EXTRACTED_BYTES + disk.INSTALL_SAFETY_MARGIN_BYTES)
  assert.ok(withBig > disk.DEFAULT_INSTALL_MIN_FREE_BYTES)
  // A tiny archive stays at the floor.
  assert.equal(disk.resolveInstallMinFree({}, 1000), disk.DEFAULT_INSTALL_MIN_FREE_BYTES)
})

// ---------------------------------------------------------------------------
// watermark — normal keep-2 vs tighten-drop-previous
// ---------------------------------------------------------------------------

test('enforceVersionsWatermark: under budget keeps current+previous', () => {
  const home = mkHome()
  try {
    seedVersion(home, 'aaaaaaaaaaaa')
    seedVersion(home, 'bbbbbbbbbbbb')
    layout.writePointerAtomic(home, { key: 'bbbbbbbbbbbb', previous: 'aaaaaaaaaaaa' })
    const r = disk.enforceVersionsWatermark(home, { dirSizeOf: () => 1000, budgetBytes: 10_000 })
    assert.equal(r.overBudget, false)
    assert.equal(r.tightened, false)
    assert.equal(r.warning, null)
    // both kept
    assert.equal(fs.existsSync(layout.bundlePaths(home).versionDir('aaaaaaaaaaaa')), true)
    assert.equal(fs.existsSync(layout.bundlePaths(home).versionDir('bbbbbbbbbbbb')), true)
  } finally {
    rm(home)
  }
})

test('enforceVersionsWatermark: over budget drops previous + warns', () => {
  const home = mkHome()
  try {
    seedVersion(home, 'aaaaaaaaaaaa') // previous
    seedVersion(home, 'bbbbbbbbbbbb') // current
    seedVersion(home, 'cccccccccccc') // stale extra
    layout.writePointerAtomic(home, { key: 'bbbbbbbbbbbb', previous: 'aaaaaaaaaaaa' })
    // Report each dir as ~1 GiB so total (3 GiB) exceeds a 2.5 GiB budget.
    const r = disk.enforceVersionsWatermark(home, { dirSizeOf: () => GIB, budgetBytes: Math.round(2.5 * GIB) })
    assert.equal(r.overBudget, true)
    assert.equal(r.tightened, true)
    assert.match(r.warning, /exceeds budget/)
    // Only current survives — previous AND the stale extra are gone.
    assert.equal(fs.existsSync(layout.bundlePaths(home).versionDir('bbbbbbbbbbbb')), true)
    assert.equal(fs.existsSync(layout.bundlePaths(home).versionDir('aaaaaaaaaaaa')), false)
    assert.equal(fs.existsSync(layout.bundlePaths(home).versionDir('cccccccccccc')), false)
  } finally {
    rm(home)
  }
})

// ---------------------------------------------------------------------------
// preflight — the "refuse before download" gate (design §8)
// ---------------------------------------------------------------------------

test('preflightDiskSpace: ok when free >= required', () => {
  const r = disk.preflightDiskSpace({ hermesHome: '/x', archiveSize: 600 * 1024 * 1024, freeBytesOf: () => 64 * GIB })
  assert.equal(r.ok, true)
  assert.ok(r.requiredBytes > 0)
})

test('preflightDiskSpace: refuses with a readable message when free < required', () => {
  const r = disk.preflightDiskSpace({ hermesHome: '/x', archiveSize: 600 * 1024 * 1024, freeBytesOf: () => 200 * 1024 * 1024 })
  assert.equal(r.ok, false)
  assert.equal(r.reason, 'insufficient_disk')
  assert.match(r.message, /Not enough disk space/i)
  assert.match(r.message, /GiB free/)
  assert.ok(r.requiredBytes > r.freeBytes)
})

test('preflightDiskSpace: minFreeBytes override is honoured', () => {
  const r = disk.preflightDiskSpace({ hermesHome: '/x', minFreeBytes: 10 * GIB, freeBytesOf: () => 5 * GIB })
  assert.equal(r.ok, false)
  assert.equal(r.requiredBytes, 10 * GIB)
})
