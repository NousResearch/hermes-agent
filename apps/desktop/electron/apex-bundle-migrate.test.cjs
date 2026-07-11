'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const layout = require('./apex-bundle-layout.cjs')
const migrate = require('./apex-bundle-migrate.cjs')

// The link-creating branches under test are posix symlinks; the Windows junction
// branch is asserted structurally in the module (WIN-VERIFY) and only exercisable
// on a real Windows machine.
const POSIX = process.platform !== 'win32'
const PLAT = { platform: process.platform }

function mkHome() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hb-migrate-'))
}
function rm(home) {
  fs.rmSync(home, { recursive: true, force: true })
}
/** A legacy in-place install: a REAL hermes-agent/ dir with an abs-path venv. */
function seedLegacy(home, extra = {}) {
  const dir = layout.bundlePaths(home).activeLink
  fs.mkdirSync(path.join(dir, 'venv', 'bin'), { recursive: true })
  fs.writeFileSync(path.join(dir, 'venv', 'bin', 'python'), '#!/legacy/abs/venv/bin/python')
  for (const [rel, body] of Object.entries(extra)) {
    fs.mkdirSync(path.dirname(path.join(dir, rel)), { recursive: true })
    fs.writeFileSync(path.join(dir, rel), body)
  }
  return dir
}
/** A committed version dir with a recognizable payload. */
function seedVersion(home, key) {
  const dir = layout.bundlePaths(home).versionDir(key)
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'payload.txt'), `runtime ${key}`)
  return dir
}

// ---------------------------------------------------------------------------
// detection + data-location assertion
// ---------------------------------------------------------------------------

test('detectLegacyInPlace: real dir + no versions is legacy; link or versions is not', () => {
  const home = mkHome()
  try {
    assert.equal(migrate.detectLegacyInPlace(home).legacy, false) // nothing there
    seedLegacy(home)
    assert.equal(migrate.detectLegacyInPlace(home).legacy, true)
    // Once a committed version exists, it is no longer a pristine legacy state.
    seedVersion(home, 'aaaaaaaaaaaa')
    assert.equal(migrate.detectLegacyInPlace(home).legacy, false)
  } finally {
    rm(home)
  }
})

test('assertNoUserDataInLegacy: clean is safe; any user-data marker is not', () => {
  const home = mkHome()
  try {
    const dir = seedLegacy(home)
    assert.equal(migrate.assertNoUserDataInLegacy(dir).safe, true)
    fs.writeFileSync(path.join(dir, '.env'), 'RELAY_KEY=secret')
    const r = migrate.assertNoUserDataInLegacy(dir)
    assert.equal(r.safe, false)
    assert.deepEqual(r.found, ['.env'])
    // An EMPTY marker file/dir is not treated as data (non-empty check).
    const home2 = mkHome()
    const dir2 = seedLegacy(home2)
    fs.mkdirSync(path.join(dir2, 'sessions'))
    assert.equal(migrate.assertNoUserDataInLegacy(dir2).safe, true)
    rm(home2)
  } finally {
    rm(home)
  }
})

// ---------------------------------------------------------------------------
// happy-path migration + rollback
// ---------------------------------------------------------------------------

test('migrateLegacyInPlace: pointer→sentinel, dir aside, link to versions/<key>', { skip: !POSIX }, () => {
  const home = mkHome()
  try {
    seedLegacy(home)
    const key = 'cccccccccccc'
    seedVersion(home, key)
    const r = migrate.migrateLegacyInPlace(home, key, PLAT)
    assert.equal(r.ok, true)
    assert.equal(r.migrated, true)
    assert.equal(r.linkPending, false)
    const p = layout.readPointer(home)
    assert.equal(p.key, key)
    assert.equal(p.previous, migrate.LEGACY_SENTINEL)
    const { activeLink } = layout.bundlePaths(home)
    assert.equal(layout.linkStatus(activeLink).kind, 'link')
    assert.equal(fs.readFileSync(path.join(activeLink, 'payload.txt'), 'utf8'), `runtime ${key}`)
    // legacy content preserved aside, untouched
    const aside = migrate.legacyAsidePath(home)
    assert.equal(fs.readFileSync(path.join(aside, 'venv', 'bin', 'python'), 'utf8'), '#!/legacy/abs/venv/bin/python')
  } finally {
    rm(home)
  }
})

test('migrateLegacyInPlace: refuses (no state change) when user data is inside', () => {
  const home = mkHome()
  try {
    seedLegacy(home, { '.env': 'SECRET=1' })
    const key = 'cccccccccccc'
    seedVersion(home, key)
    const r = migrate.migrateLegacyInPlace(home, key, PLAT)
    assert.equal(r.ok, false)
    assert.equal(r.reason, 'user-data-in-runtime-dir')
    // pointer never advanced, nothing moved
    assert.equal(layout.readPointer(home), null)
    assert.equal(fs.existsSync(migrate.legacyAsidePath(home)), false)
    assert.equal(layout.linkStatus(layout.bundlePaths(home).activeLink).kind, 'dir')
  } finally {
    rm(home)
  }
})

test('rollbackToLegacyInPlace: link points back at the aside; legacy venv resolves through it', { skip: !POSIX }, () => {
  const home = mkHome()
  try {
    seedLegacy(home)
    const key = 'cccccccccccc'
    seedVersion(home, key)
    migrate.migrateLegacyInPlace(home, key, PLAT)
    const r = migrate.rollbackToLegacyInPlace(home, PLAT)
    assert.equal(r.ok, true)
    const p = layout.readPointer(home)
    assert.equal(p.key, migrate.LEGACY_SENTINEL)
    assert.equal(p.previous, key)
    const { activeLink } = layout.bundlePaths(home)
    // The legacy venv's absolute self-reference resolves correctly THROUGH the link.
    assert.equal(fs.readFileSync(path.join(activeLink, 'venv', 'bin', 'python'), 'utf8'), '#!/legacy/abs/venv/bin/python')
  } finally {
    rm(home)
  }
})

test('rollbackToLegacyInPlace: refuses when previous is not the sentinel', () => {
  const home = mkHome()
  try {
    layout.writePointerAtomic(home, { key: 'bbbbbbbbbbbb', previous: 'aaaaaaaaaaaa' })
    const r = migrate.rollbackToLegacyInPlace(home, PLAT)
    assert.equal(r.ok, false)
    assert.equal(r.reason, 'previous-not-legacy')
  } finally {
    rm(home)
  }
})

// ---------------------------------------------------------------------------
// crash-safe reconcile — every interrupted state self-heals to consistency
// ---------------------------------------------------------------------------

test('reconcileMigration: finishes a migration interrupted after pointer-write, before move', { skip: !POSIX }, () => {
  const home = mkHome()
  try {
    seedLegacy(home)
    const key = 'cccccccccccc'
    seedVersion(home, key)
    // Simulate the crash: pointer already advanced, but the dir is STILL in place.
    layout.writePointerAtomic(home, { key, previous: migrate.LEGACY_SENTINEL })
    assert.equal(layout.linkStatus(layout.bundlePaths(home).activeLink).kind, 'dir')

    const rec = migrate.reconcileMigration(home, PLAT)
    assert.equal(rec.reconciled, true)
    assert.equal(rec.action, 'migrate-relink')
    const { activeLink } = layout.bundlePaths(home)
    assert.equal(layout.linkStatus(activeLink).kind, 'link')
    assert.equal(fs.readFileSync(path.join(activeLink, 'payload.txt'), 'utf8'), `runtime ${key}`)
    assert.equal(fs.existsSync(path.join(migrate.legacyAsidePath(home), 'venv')), true)
  } finally {
    rm(home)
  }
})

test('reconcileMigration: finishes a migration interrupted after move, before link', { skip: !POSIX }, () => {
  const home = mkHome()
  try {
    const legacy = seedLegacy(home)
    const key = 'cccccccccccc'
    seedVersion(home, key)
    // Simulate the crash: pointer advanced AND dir moved aside, but no link yet.
    layout.writePointerAtomic(home, { key, previous: migrate.LEGACY_SENTINEL })
    fs.renameSync(legacy, migrate.legacyAsidePath(home))
    assert.equal(layout.linkStatus(layout.bundlePaths(home).activeLink).kind, 'missing')

    const rec = migrate.reconcileMigration(home, PLAT)
    assert.equal(rec.reconciled, true)
    const { activeLink } = layout.bundlePaths(home)
    assert.equal(fs.readFileSync(path.join(activeLink, 'payload.txt'), 'utf8'), `runtime ${key}`)
  } finally {
    rm(home)
  }
})

test('reconcileMigration: heals the link after a rollback to the legacy aside', { skip: !POSIX }, () => {
  const home = mkHome()
  try {
    seedLegacy(home)
    const key = 'cccccccccccc'
    seedVersion(home, key)
    migrate.migrateLegacyInPlace(home, key, PLAT)
    migrate.rollbackToLegacyInPlace(home, PLAT)
    // Drop the link to simulate a crash mid-repoint; pointer.key is the sentinel.
    const { activeLink } = layout.bundlePaths(home)
    layout.removeLinkOnly(activeLink)
    assert.equal(layout.linkStatus(activeLink).kind, 'missing')

    const rec = migrate.reconcileMigration(home, PLAT)
    assert.equal(rec.reconciled, true)
    assert.equal(rec.action, 'relink-legacy')
    assert.equal(fs.readFileSync(path.join(activeLink, 'venv', 'bin', 'python'), 'utf8'), '#!/legacy/abs/venv/bin/python')
  } finally {
    rm(home)
  }
})

// ---------------------------------------------------------------------------
// aside lifecycle — kept while it is a rollback target, reaped once the sentinel
// falls out of the pointer (the second successful update)
// ---------------------------------------------------------------------------

test('gcLegacyAside: kept while sentinel is the fallback; reaped after the next update drops it', { skip: !POSIX }, () => {
  const home = mkHome()
  try {
    seedLegacy(home)
    const first = 'cccccccccccc'
    seedVersion(home, first)
    migrate.migrateLegacyInPlace(home, first, PLAT)
    // Sentinel is pointer.previous → aside is still the rollback target: keep it.
    assert.equal(migrate.gcLegacyAside(home).reason, 'still-rollback-target')
    assert.equal(fs.existsSync(migrate.legacyAsidePath(home)), true)

    // A SECOND update: active path is now a link, so the normal switch applies and
    // sets previous=<first>, dropping the sentinel from the pointer.
    const second = 'dddddddddddd'
    seedVersion(home, second)
    const sw = migrate.switchToVersionOrMigrate(home, second, PLAT)
    assert.equal(sw.ok, true)
    assert.equal(layout.readPointer(home).previous, first)

    // Now the aside is no longer referenced → reaped.
    const gc = migrate.gcLegacyAside(home)
    assert.equal(gc.removed, true)
    assert.equal(fs.existsSync(migrate.legacyAsidePath(home)), false)
  } finally {
    rm(home)
  }
})

test('gcLegacyAside: respects an injected lock (skips, retries next startup)', { skip: !POSIX }, () => {
  const home = mkHome()
  try {
    // A migrated-then-superseded state where the aside would normally be reaped.
    fs.mkdirSync(migrate.legacyAsidePath(home), { recursive: true })
    layout.writePointerAtomic(home, { key: 'dddddddddddd', previous: 'cccccccccccc' })
    const gc = migrate.gcLegacyAside(home, { isLocked: () => true })
    assert.equal(gc.removed, false)
    assert.equal(gc.reason, 'locked')
    assert.equal(fs.existsSync(migrate.legacyAsidePath(home)), true)
  } finally {
    rm(home)
  }
})

test('switchToVersionOrMigrate: a non-legacy (link/missing) active path takes the normal switch', { skip: !POSIX }, () => {
  const home = mkHome()
  try {
    const key = 'cccccccccccc'
    seedVersion(home, key)
    // No hermes-agent dir at all → missing → normal switchToVersion, no sentinel.
    const r = migrate.switchToVersionOrMigrate(home, key, PLAT)
    assert.equal(r.ok, true)
    assert.equal(layout.readPointer(home).previous, null)
    assert.equal(fs.existsSync(migrate.legacyAsidePath(home)), false)
  } finally {
    rm(home)
  }
})

test('migrateLegacyInPlace: idempotent re-run does not create a second aside', { skip: !POSIX }, () => {
  const home = mkHome()
  try {
    seedLegacy(home)
    const key = 'cccccccccccc'
    seedVersion(home, key)
    migrate.migrateLegacyInPlace(home, key, PLAT)
    // Re-running the SAME switch (active path is now a link) is a normal no-op
    // switch, not a second migration — the aside is untouched, no clobber.
    const again = migrate.switchToVersionOrMigrate(home, key, PLAT)
    assert.equal(again.ok, true)
    assert.equal(layout.readPointer(home).key, key)
    assert.equal(fs.existsSync(migrate.legacyAsidePath(home)), true)
  } finally {
    rm(home)
  }
})
