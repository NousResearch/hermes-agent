'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const layout = require('./apex-bundle-layout.cjs')

// ---------------------------------------------------------------------------
// helpers — a throwaway HERMES_HOME per test
// ---------------------------------------------------------------------------

function mkHome() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hb-layout-'))
}
function rm(home) {
  fs.rmSync(home, { recursive: true, force: true })
}
/** Materialize a committed version dir with a recognizable file inside. */
function seedVersion(home, key, marker) {
  const dir = layout.bundlePaths(home).versionDir(key)
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'id.txt'), marker || key)
  return dir
}
function seedStaging(home, key) {
  const dir = layout.bundlePaths(home).stagingDir(key)
  fs.mkdirSync(dir, { recursive: true })
  fs.writeFileSync(path.join(dir, 'half.txt'), 'partial')
  return dir
}
// The active-link branch under test is posix symlinks. The Windows junction
// branch (createActiveLink win32) is asserted structurally in the module and
// flagged WIN-VERIFY; it can only be exercised on a real Windows machine.
const POSIX = process.platform !== 'win32'

// ---------------------------------------------------------------------------
// pointer: read / write / atomicity / schema guard
// ---------------------------------------------------------------------------

test('pointer: write then read round-trips key+previous', () => {
  const home = mkHome()
  try {
    layout.writePointerAtomic(home, { key: 'aaaa', previous: 'bbbb' })
    const p = layout.readPointer(home)
    assert.equal(p.key, 'aaaa')
    assert.equal(p.previous, 'bbbb')
    assert.equal(p.schemaVersion, layout.POINTER_SCHEMA_VERSION)
    assert.ok(p.switchedAt, 'stamps switchedAt')
  } finally {
    rm(home)
  }
})

test('pointer: missing / malformed / wrong-schema all read as null', () => {
  const home = mkHome()
  try {
    assert.equal(layout.readPointer(home), null)
    const { pointerPath } = layout.bundlePaths(home)
    fs.writeFileSync(pointerPath, '{not json')
    assert.equal(layout.readPointer(home), null)
    fs.writeFileSync(pointerPath, JSON.stringify({ schemaVersion: 999, key: 'x' }))
    assert.equal(layout.readPointer(home), null)
    fs.writeFileSync(pointerPath, JSON.stringify({ schemaVersion: 1 })) // no key
    assert.equal(layout.readPointer(home), null)
  } finally {
    rm(home)
  }
})

test('pointer: atomic write leaves no temp turd and last-writer-wins', () => {
  const home = mkHome()
  try {
    layout.writePointerAtomic(home, { key: 'one' })
    layout.writePointerAtomic(home, { key: 'two', previous: 'one' })
    assert.equal(layout.readPointer(home).key, 'two')
    // No `.NNN.tmp` sibling left behind.
    const leftovers = fs.readdirSync(home).filter(n => n.startsWith(layout.POINTER_BASENAME) && n !== layout.POINTER_BASENAME)
    assert.deepEqual(leftovers, [])
  } finally {
    rm(home)
  }
})

// ---------------------------------------------------------------------------
// switch: fresh install + subsequent switch tracks previous, link follows
// ---------------------------------------------------------------------------

test('switchToVersion: fresh switch sets pointer + creates link (posix)', { skip: !POSIX }, () => {
  const home = mkHome()
  try {
    seedVersion(home, 'AAA')
    const res = layout.switchToVersion(home, 'AAA')
    assert.equal(res.ok, true)
    const p = layout.readPointer(home)
    assert.equal(p.key, 'AAA')
    assert.equal(p.previous, null)
    const { activeLink } = layout.bundlePaths(home)
    assert.equal(layout.linkStatus(activeLink).kind, 'link')
    assert.equal(fs.readFileSync(path.join(activeLink, 'id.txt'), 'utf8'), 'AAA')
  } finally {
    rm(home)
  }
})

test('switchToVersion: second switch records previous and repoints link', { skip: !POSIX }, () => {
  const home = mkHome()
  try {
    seedVersion(home, 'AAA')
    seedVersion(home, 'BBB')
    layout.switchToVersion(home, 'AAA')
    const res = layout.switchToVersion(home, 'BBB')
    assert.equal(res.ok, true)
    assert.equal(res.previous, 'AAA')
    const p = layout.readPointer(home)
    assert.equal(p.key, 'BBB')
    assert.equal(p.previous, 'AAA')
    const { activeLink } = layout.bundlePaths(home)
    assert.equal(fs.readFileSync(path.join(activeLink, 'id.txt'), 'utf8'), 'BBB')
  } finally {
    rm(home)
  }
})

test('switchToVersion: missing version dir refuses without writing pointer', () => {
  const home = mkHome()
  try {
    const res = layout.switchToVersion(home, 'GHOST')
    assert.equal(res.ok, false)
    assert.equal(res.reason, 'version-missing')
    assert.equal(layout.readPointer(home), null)
  } finally {
    rm(home)
  }
})

test('switchToVersion: refuses (and does NOT advance pointer) over a legacy real dir', () => {
  const home = mkHome()
  try {
    seedVersion(home, 'AAA')
    // A legacy in-place install occupies hermes-agent as a REAL directory.
    fs.mkdirSync(layout.bundlePaths(home).activeLink, { recursive: true })
    const res = layout.switchToVersion(home, 'AAA')
    assert.equal(res.ok, false)
    assert.equal(res.reason, 'active-path-occupied-by-real-dir')
    assert.equal(layout.readPointer(home), null, 'pointer stays honest — not advanced')
  } finally {
    rm(home)
  }
})

// ---------------------------------------------------------------------------
// rollback: real file-level rollback via pointer swap (the F5 fix)
// ---------------------------------------------------------------------------

test('rollbackToPrevious: swaps key<->previous and repoints link', { skip: !POSIX }, () => {
  const home = mkHome()
  try {
    seedVersion(home, 'AAA')
    seedVersion(home, 'BBB')
    layout.switchToVersion(home, 'AAA')
    layout.switchToVersion(home, 'BBB') // now key=BBB previous=AAA
    const res = layout.rollbackToPrevious(home)
    assert.equal(res.ok, true)
    assert.equal(res.key, 'AAA')
    const p = layout.readPointer(home)
    assert.equal(p.key, 'AAA')
    assert.equal(p.previous, 'BBB', 'the abandoned version becomes redo-able previous')
    const { activeLink } = layout.bundlePaths(home)
    assert.equal(fs.readFileSync(path.join(activeLink, 'id.txt'), 'utf8'), 'AAA')
    // Rolling forward again works (redo).
    const back = layout.rollbackToPrevious(home)
    assert.equal(back.key, 'BBB')
  } finally {
    rm(home)
  }
})

test('rollbackToPrevious: no previous / missing previous dir refuses', () => {
  const home = mkHome()
  try {
    seedVersion(home, 'AAA')
    layout.switchToVersion(home, 'AAA')
    assert.equal(layout.rollbackToPrevious(home).reason, 'no-previous')

    // previous recorded but its dir was GC'd
    layout.writePointerAtomic(home, { key: 'AAA', previous: 'GONE' })
    assert.equal(layout.rollbackToPrevious(home).reason, 'previous-missing')
  } finally {
    rm(home)
  }
})

// ---------------------------------------------------------------------------
// reconcile: heal the link from the pointer after an interrupted switch
// ---------------------------------------------------------------------------

test('reconcileActiveLink: rebuilds a deleted link from the pointer', { skip: !POSIX }, () => {
  const home = mkHome()
  try {
    seedVersion(home, 'AAA')
    layout.switchToVersion(home, 'AAA')
    const { activeLink } = layout.bundlePaths(home)
    layout.removeLinkOnly(activeLink) // simulate crash after pointer, before link
    assert.equal(layout.linkStatus(activeLink).kind, 'missing')
    const res = layout.reconcileActiveLink(home)
    assert.equal(res.reconciled, true)
    assert.equal(res.key, 'AAA')
    assert.equal(fs.readFileSync(path.join(activeLink, 'id.txt'), 'utf8'), 'AAA')
  } finally {
    rm(home)
  }
})

test('reconcileActiveLink: repoints a link aimed at the wrong version', { skip: !POSIX }, () => {
  const home = mkHome()
  try {
    seedVersion(home, 'AAA')
    seedVersion(home, 'BBB')
    layout.switchToVersion(home, 'AAA')
    const { activeLink, versionDir } = layout.bundlePaths(home)
    // Point the link at BBB while the pointer still says AAA (torn switch).
    layout.removeLinkOnly(activeLink)
    fs.symlinkSync(path.relative(path.dirname(activeLink), versionDir('BBB')), activeLink, 'dir')
    const res = layout.reconcileActiveLink(home)
    assert.equal(res.reconciled, true)
    assert.equal(fs.readFileSync(path.join(activeLink, 'id.txt'), 'utf8'), 'AAA')
  } finally {
    rm(home)
  }
})

test('reconcileActiveLink: already-consistent link is a no-op', { skip: !POSIX }, () => {
  const home = mkHome()
  try {
    seedVersion(home, 'AAA')
    layout.switchToVersion(home, 'AAA')
    const res = layout.reconcileActiveLink(home)
    assert.equal(res.reconciled, false)
    assert.equal(res.reason, 'already-consistent')
  } finally {
    rm(home)
  }
})

test('reconcileActiveLink: no pointer / missing version dir are non-fatal', () => {
  const home = mkHome()
  try {
    assert.equal(layout.reconcileActiveLink(home).reason, 'no-pointer')
    layout.writePointerAtomic(home, { key: 'NOPE' })
    assert.equal(layout.reconcileActiveLink(home).reason, 'version-missing')
  } finally {
    rm(home)
  }
})

test('reconcileActiveLink: refuses to touch a legacy real dir', () => {
  const home = mkHome()
  try {
    seedVersion(home, 'AAA')
    layout.writePointerAtomic(home, { key: 'AAA' })
    fs.mkdirSync(layout.bundlePaths(home).activeLink, { recursive: true })
    const res = layout.reconcileActiveLink(home)
    assert.equal(res.reconciled, false)
    assert.equal(res.reason, 'active-path-occupied-by-real-dir')
  } finally {
    rm(home)
  }
})

// ---------------------------------------------------------------------------
// GC: keep current+previous, reap the rest + all `.tmp` half-installs
// ---------------------------------------------------------------------------

test('garbageCollect: keeps current+previous, removes others and staging', () => {
  const home = mkHome()
  try {
    for (const k of ['AAA', 'BBB', 'CCC', 'DDD']) seedVersion(home, k)
    seedStaging(home, 'CCC') // orphan half-install
    seedStaging(home, 'ZZZ')
    layout.writePointerAtomic(home, { key: 'CCC', previous: 'BBB' })

    const r = layout.garbageCollect(home)
    assert.deepEqual(r.kept.sort(), ['BBB', 'CCC'])
    assert.deepEqual(r.removed.sort(), ['AAA', 'DDD'])
    assert.deepEqual(r.orphansRemoved.sort(), ['CCC.tmp', 'ZZZ.tmp'])
    // Disk reflects it.
    const left = layout.listVersions(home)
    assert.deepEqual(left.versions.sort(), ['BBB', 'CCC'])
    assert.deepEqual(left.staging, [])
  } finally {
    rm(home)
  }
})

test('garbageCollect: honors extra keep[] and isLocked skip', () => {
  const home = mkHome()
  try {
    for (const k of ['AAA', 'BBB', 'CCC']) seedVersion(home, k)
    layout.writePointerAtomic(home, { key: 'CCC', previous: null })
    // Keep BBB explicitly; force-skip AAA as if a handle were open.
    const r = layout.garbageCollect(home, { keep: ['BBB'], isLocked: name => name === 'AAA' })
    assert.ok(r.kept.includes('CCC'))
    assert.ok(r.kept.includes('BBB'))
    assert.deepEqual(r.skipped, ['AAA'], 'locked dir skipped, retried next startup')
    assert.ok(fs.existsSync(layout.bundlePaths(home).versionDir('AAA')), 'skip means still on disk')
  } finally {
    rm(home)
  }
})

test('garbageCollect: dropPrevious sheds previous too (C2 disk-pressure path)', () => {
  const home = mkHome()
  try {
    for (const k of ['AAA', 'BBB', 'CCC']) seedVersion(home, k)
    layout.writePointerAtomic(home, { key: 'CCC', previous: 'BBB' })
    const r = layout.garbageCollect(home, { dropPrevious: true })
    assert.equal(r.droppedPrevious, true)
    assert.deepEqual(r.kept, ['CCC'], 'only current survives under pressure')
    assert.deepEqual(r.removed.sort(), ['AAA', 'BBB'])
    assert.equal(fs.existsSync(layout.bundlePaths(home).versionDir('BBB')), false)
    assert.equal(fs.existsSync(layout.bundlePaths(home).versionDir('CCC')), true)
  } finally {
    rm(home)
  }
})

test('garbageCollect: empty / no-versions home is a safe no-op', () => {
  const home = mkHome()
  try {
    const r = layout.garbageCollect(home)
    assert.deepEqual(r, { kept: [], removed: [], skipped: [], orphansRemoved: [], orphansSkipped: [], droppedPrevious: false })
  } finally {
    rm(home)
  }
})

// ---------------------------------------------------------------------------
// listVersions / layoutState
// ---------------------------------------------------------------------------

test('listVersions: splits committed dirs from .tmp staging', () => {
  const home = mkHome()
  try {
    seedVersion(home, 'AAA')
    seedVersion(home, 'BBB')
    seedStaging(home, 'AAA')
    const l = layout.listVersions(home)
    assert.deepEqual(l.versions, ['AAA', 'BBB'])
    assert.deepEqual(l.staging, ['AAA.tmp'])
  } finally {
    rm(home)
  }
})

test('layoutState: reports pointer, versions and active link resolution', { skip: !POSIX }, () => {
  const home = mkHome()
  try {
    seedVersion(home, 'AAA')
    layout.switchToVersion(home, 'AAA')
    const s = layout.layoutState(home)
    assert.equal(s.pointer.key, 'AAA')
    assert.deepEqual(s.versions, ['AAA'])
    assert.equal(s.activeLink.kind, 'link')
    assert.equal(path.basename(s.activeLink.resolvesTo), 'AAA')
  } finally {
    rm(home)
  }
})
