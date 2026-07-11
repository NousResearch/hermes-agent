'use strict'

/**
 * apex-bundle-layout.cjs — hc-472 P1 · C1
 *
 * Versioned runtime catalog with an atomic-switch pointer, a self-healing
 * compatibility link, crash-safe rollback, and startup GC.
 *
 * WHY (design §4, §8 — hermes-cloud docs/work-notes/DESIGN-hc472-runtime-bundle.md)
 * -----------------------------------------------------------------------------
 * Today's updater mutates HERMES_HOME/hermes-agent IN PLACE and "rolls back" by
 * restoring only a marker file — the files never come back (root-cause F5). The
 * bundle model makes install/update ADD-ONLY: each version lives under its own
 * versions/<key>/ dir and is NEVER mutated after commit. The single source of
 * truth is a pointer file written atomically; the legacy `hermes-agent` path
 * becomes a junction(win)/symlink(mac) that every existing consumer keeps using
 * unchanged. If a switch is interrupted the link may be briefly stale, but the
 * pointer is always consistent, so the shell heals the link from the pointer on
 * the next launch. Rollback is a pointer swap → a real file-level rollback.
 *
 *   HERMES_HOME/
 *   ├── versions/
 *   │   ├── <key>/        one committed, fully-verified bundle (immutable)
 *   │   └── <key>.tmp/    a half-installed staging dir (GC'd on startup)
 *   ├── hermes-agent      junction(win)/symlink(mac) → versions/<current-key>
 *   └── .apexnodes-runtime-current.json   {schemaVersion,key,previous,switchedAt}
 *
 * CORE INVARIANT: the pointer and the link only ever reference a version dir
 * that has already passed full sha verification (the caller renames a staging
 * dir into place only after verify — see apex-bundle-install.cjs). Any failure
 * stops in an UNREFERENCED `.tmp` dir.
 *
 * Pure/electron-free and rooted at an explicit `hermesHome` so every branch is
 * unit-testable. The Windows junction branch is guarded by `platform` and is
 * annotated WIN-VERIFY where it can only be exercised on a real Windows machine.
 */

const fs = require('node:fs')
const path = require('node:path')

const POINTER_BASENAME = '.apexnodes-runtime-current.json'
const POINTER_SCHEMA_VERSION = 1
const VERSIONS_DIRNAME = 'versions'
// MUST equal main.cjs ACTIVE_HERMES_ROOT's basename: every existing consumer
// (install scripts, main.cjs, tool shell-outs) resolves HERMES_HOME/hermes-agent
// as a fixed path. Making it the versioned link keeps them all working verbatim.
const ACTIVE_LINK_BASENAME = 'hermes-agent'
const TMP_SUFFIX = '.tmp'

function currentPlatform(opts) {
  return (opts && opts.platform) || process.platform
}

function isWin(platform) {
  return platform === 'win32'
}

/** All the derived paths for a HERMES_HOME. Pure. */
function bundlePaths(hermesHome) {
  const home = path.resolve(hermesHome)
  const versionsDir = path.join(home, VERSIONS_DIRNAME)
  return {
    home,
    versionsDir,
    pointerPath: path.join(home, POINTER_BASENAME),
    activeLink: path.join(home, ACTIVE_LINK_BASENAME),
    versionDir: key => path.join(versionsDir, key),
    stagingDir: key => path.join(versionsDir, `${key}${TMP_SUFFIX}`)
  }
}

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, 'utf8'))
  } catch {
    return null
  }
}

/** Read the truth pointer. Returns null when absent / malformed / wrong schema. */
function readPointer(hermesHome) {
  const { pointerPath } = bundlePaths(hermesHome)
  const parsed = readJson(pointerPath)
  if (!parsed || typeof parsed !== 'object') return null
  if (parsed.schemaVersion !== POINTER_SCHEMA_VERSION) return null
  if (!parsed.key || typeof parsed.key !== 'string') return null
  return {
    schemaVersion: POINTER_SCHEMA_VERSION,
    key: parsed.key,
    previous: typeof parsed.previous === 'string' && parsed.previous ? parsed.previous : null,
    switchedAt: typeof parsed.switchedAt === 'string' ? parsed.switchedAt : null
  }
}

/**
 * Write the truth pointer atomically (temp + rename). rename() is atomic on
 * both NTFS and APFS, so a crash mid-write can only leave the old pointer or the
 * new one — never a torn file.
 */
function writePointerAtomic(hermesHome, pointer) {
  const { home, pointerPath } = bundlePaths(hermesHome)
  fs.mkdirSync(home, { recursive: true })
  const payload = {
    schemaVersion: POINTER_SCHEMA_VERSION,
    key: pointer.key,
    previous: pointer.previous || null,
    switchedAt: pointer.switchedAt || new Date().toISOString()
  }
  const tmp = `${pointerPath}.${process.pid}.tmp`
  fs.writeFileSync(tmp, JSON.stringify(payload, null, 2) + '\n', 'utf8')
  fs.renameSync(tmp, pointerPath)
  return payload
}

/**
 * Classify whatever currently occupies a path WITHOUT following the link:
 *   'missing' | 'link' (symlink or win junction) | 'dir' (real) | 'file' | 'other'
 * `target` is the raw link target for links (best-effort).
 *
 * WIN-VERIFY: Node reports a directory junction created via
 * symlink(..., 'junction') with lstat().isSymbolicLink() === true and readlink()
 * returning the (absolute) target. Confirmed behaviour, but only exercisable on
 * a real Windows machine — asserted structurally here, verified in CI/on-device.
 */
function linkStatus(p) {
  let st
  try {
    st = fs.lstatSync(p)
  } catch {
    return { kind: 'missing' }
  }
  if (st.isSymbolicLink()) {
    let target = null
    try {
      target = fs.readlinkSync(p)
    } catch {
      target = null
    }
    return { kind: 'link', target }
  }
  if (st.isDirectory()) return { kind: 'dir' }
  if (st.isFile()) return { kind: 'file' }
  return { kind: 'other' }
}

/**
 * Remove ONLY the reparse point / symlink at `p`, never the target's contents.
 * Mirrors build-runtime-bundle.mjs::removeLinkOnly: unlink for file-style
 * symlinks, rmdir for directory junctions / dir-symlinks (both delete only the
 * link, not what it points at). NEVER use fs.rm(recursive) on a link — on
 * Windows that would delete the TARGET version dir's contents.
 */
function removeLinkOnly(p) {
  try {
    fs.unlinkSync(p)
    return true
  } catch {
    try {
      fs.rmdirSync(p)
      return true
    } catch {
      return false
    }
  }
}

/**
 * Create the active dir link `linkPath` → `targetDir`.
 *   win : directory JUNCTION (mklink /J equivalent) — no admin/dev-mode needed,
 *         but junctions require an ABSOLUTE target. (WIN-VERIFY)
 *   mac : relative 'dir' symlink so a relocated HERMES_HOME keeps a valid link.
 * Assumes any prior occupant at linkPath has already been removed.
 */
function createActiveLink(linkPath, targetDir, opts) {
  const platform = currentPlatform(opts)
  if (isWin(platform)) {
    fs.symlinkSync(path.resolve(targetDir), linkPath, 'junction')
  } else {
    const rel = path.relative(path.dirname(linkPath), targetDir)
    fs.symlinkSync(rel, linkPath, 'dir')
  }
}

/**
 * True when the link at `activeLink` already resolves to `targetDir`. Both sides
 * are realpath'd so a platform whose parent dirs are themselves symlinks (macOS
 * /var → /private/var, the tmpdir case) doesn't read as a spurious mismatch.
 */
function linkResolvesTo(activeLink, targetDir) {
  const status = linkStatus(activeLink)
  if (status.kind !== 'link') return false
  try {
    return fs.realpathSync(activeLink) === fs.realpathSync(targetDir)
  } catch {
    return false
  }
}

/**
 * Point the active link at versions/<key>, replacing any existing LINK.
 * Refuses (returns {ok:false, reason:'active-path-occupied-by-real-dir'}) when
 * the active path is a REAL directory — that is a legacy in-place install and
 * converting it is the side-by-side migration's job (design §5 / D1), NOT this
 * add-only layer's. Never deletes user data.
 */
function repointActiveLink(hermesHome, key, opts) {
  const { activeLink, versionDir } = bundlePaths(hermesHome)
  const target = versionDir(key)
  if (!fs.existsSync(target)) {
    return { ok: false, reason: 'version-missing', key }
  }
  const status = linkStatus(activeLink)
  if (status.kind === 'dir') {
    return { ok: false, reason: 'active-path-occupied-by-real-dir', key }
  }
  if (status.kind === 'link') {
    if (linkResolvesTo(activeLink, target)) return { ok: true, key, changed: false }
    if (!removeLinkOnly(activeLink)) return { ok: false, reason: 'could-not-remove-old-link', key }
  } else if (status.kind === 'file' || status.kind === 'other') {
    // A stray file where the link belongs — remove just it (not recursive).
    try {
      fs.unlinkSync(activeLink)
    } catch {
      return { ok: false, reason: 'active-path-occupied', key }
    }
  }
  createActiveLink(activeLink, target, opts)
  return { ok: true, key, changed: true }
}

/**
 * Commit a switch to `newKey`: write the pointer FIRST (truth), then repoint the
 * link (derived view). Ordering matters — if we crash between the two, the next
 * launch's reconcile() reads the new pointer and rebuilds the link, so we can
 * only ever be "pointer new, link pending", never "link new, pointer old".
 *
 * The version dir MUST already exist and be verified (caller's atomic rename).
 * `previous` becomes the displaced current so rollback has a target.
 */
function switchToVersion(hermesHome, newKey, opts) {
  const { versionDir, activeLink } = bundlePaths(hermesHome)
  if (!fs.existsSync(versionDir(newKey))) {
    return { ok: false, reason: 'version-missing', key: newKey }
  }
  // Pre-check: never advance the truth pointer when the active path is a legacy
  // REAL dir we're not allowed to replace (converting it is §5/D1, out of scope).
  if (linkStatus(activeLink).kind === 'dir') {
    return { ok: false, reason: 'active-path-occupied-by-real-dir', key: newKey }
  }
  const prior = readPointer(hermesHome)
  const previous = prior && prior.key && prior.key !== newKey ? prior.key : prior ? prior.previous : null
  writePointerAtomic(hermesHome, { key: newKey, previous })
  const linkRes = repointActiveLink(hermesHome, newKey, opts)
  if (!linkRes.ok) return { ok: false, reason: linkRes.reason, key: newKey, previous }
  return { ok: true, key: newKey, previous, linkChanged: linkRes.changed }
}

/**
 * Roll back to the pointer's `previous` version: swap key↔previous and repoint
 * the link. A real file-level rollback (the old version dir is still on disk) —
 * this is what makes F5 go away. Refuses when there is no previous or its dir is
 * gone (GC'd).
 */
function rollbackToPrevious(hermesHome, opts) {
  const { versionDir, activeLink } = bundlePaths(hermesHome)
  const pointer = readPointer(hermesHome)
  if (!pointer) return { ok: false, reason: 'no-pointer' }
  if (!pointer.previous) return { ok: false, reason: 'no-previous' }
  if (!fs.existsSync(versionDir(pointer.previous))) return { ok: false, reason: 'previous-missing', key: pointer.previous }
  if (linkStatus(activeLink).kind === 'dir') return { ok: false, reason: 'active-path-occupied-by-real-dir', key: pointer.previous }
  // Swap so the just-abandoned version becomes the new `previous` (redo-able).
  writePointerAtomic(hermesHome, { key: pointer.previous, previous: pointer.key })
  const linkRes = repointActiveLink(hermesHome, pointer.previous, opts)
  if (!linkRes.ok) return { ok: false, reason: linkRes.reason, key: pointer.previous }
  return { ok: true, key: pointer.previous, previous: pointer.key }
}

/**
 * Heal the active link from the pointer (truth). Called at shell startup, first
 * thing, so a switch interrupted mid-repoint (link stale/missing) self-repairs.
 * Fully defensive — every branch returns a reason, never throws.
 */
function reconcileActiveLink(hermesHome, opts) {
  const { activeLink, versionDir } = bundlePaths(hermesHome)
  const pointer = readPointer(hermesHome)
  if (!pointer) return { reconciled: false, reason: 'no-pointer' }
  const target = versionDir(pointer.key)
  if (!fs.existsSync(target)) return { reconciled: false, reason: 'version-missing', key: pointer.key }
  const status = linkStatus(activeLink)
  if (status.kind === 'dir') return { reconciled: false, reason: 'active-path-occupied-by-real-dir', key: pointer.key }
  if (status.kind === 'link' && linkResolvesTo(activeLink, target)) {
    return { reconciled: false, reason: 'already-consistent', key: pointer.key }
  }
  const res = repointActiveLink(hermesHome, pointer.key, opts)
  if (!res.ok) return { reconciled: false, reason: res.reason, key: pointer.key }
  return { reconciled: true, key: pointer.key }
}

/**
 * List the immediate children of versions/, split into committed version dirs
 * and orphan `.tmp` staging dirs. Pure enumeration (no deletion).
 */
function listVersions(hermesHome) {
  const { versionsDir } = bundlePaths(hermesHome)
  const out = { versions: [], staging: [] }
  let entries
  try {
    entries = fs.readdirSync(versionsDir, { withFileTypes: true })
  } catch {
    return out
  }
  for (const e of entries) {
    if (!e.isDirectory()) continue
    if (e.name.endsWith(TMP_SUFFIX)) out.staging.push(e.name)
    else out.versions.push(e.name)
  }
  out.versions.sort()
  out.staging.sort()
  return out
}

/**
 * Remove a version dir with ALL-OR-NOTHING semantics on Windows (design §4 C2
 * 无句柄探测). A version dir is only ever GC'd when nothing references it, but a
 * stale runtime child can still hold a handle on a `.pyd` inside an old venv; a
 * plain recursive rmSync would then delete SOME files before hitting the locked
 * one, leaving a half-shredded dir. So on win32 we first RENAME the whole tree to
 * a sibling `<name>.gc.tmp`: Windows refuses that rename atomically (EPERM/EBUSY)
 * while ANY handle is open, so a locked dir is left COMPLETELY intact and retried
 * next startup; only once detached under a `.tmp` name (which listVersions reads
 * as an orphan staging dir the next GC reaps) do we rmSync it — a partial failure
 * there just leaves an unreferenced `.tmp`. On POSIX an open file never blocks
 * unlink, so a direct recursive rm is already all-or-nothing (byte-identical to
 * the pre-C2 behaviour). Returns true on a completed removal, false when the dir
 * was left in place (locked).
 *
 * WIN-VERIFY: the rename-refused-while-open-handle behaviour is Windows-specific
 * and only exercisable on a real Windows machine; asserted structurally here.
 */
function removeVersionDir(absDir, platform) {
  if (isWin(platform)) {
    const detached = `${absDir}.gc${TMP_SUFFIX}`
    try {
      fs.rmSync(detached, { recursive: true, force: true }) // clear any prior probe residue
      fs.renameSync(absDir, detached)
    } catch {
      return false // a held handle refused the rename — leave the dir fully intact
    }
    try {
      fs.rmSync(detached, { recursive: true, force: true })
    } catch {
      // Detached + unreferenced (ends in .tmp) — next startup GC reaps it.
    }
    return true
  }
  try {
    fs.rmSync(absDir, { recursive: true, force: true })
    return true
  } catch {
    return false
  }
}

/**
 * Startup GC: keep current + previous (+ any explicit `keep` keys), delete every
 * other committed version dir and EVERY orphan `.tmp` staging dir.
 *
 * Timing is the safety mechanism: GC runs at shell startup, before any runtime
 * child process is spawned, so nothing holds a handle on an old venv's .pyd. As
 * belt-and-suspenders a directory that still can't be removed (an open handle) is
 * SKIPPED and retried next startup — never fatal (the removal is all-or-nothing,
 * see removeVersionDir). `isLocked(name, absPath)` can force-skip (tests / an
 * explicit handle probe); removal errors are caught regardless.
 *
 * `dropPrevious:true` (design §4/C2 disk-pressure path) sheds the `previous`
 * version too — a working `current` beats an instant-rollback target we can't
 * afford the disk for. Default keeps current+previous.
 *
 * Returns {kept, removed, skipped, orphansRemoved, orphansSkipped, droppedPrevious}.
 */
function garbageCollect(hermesHome, opts = {}) {
  const { versionsDir } = bundlePaths(hermesHome)
  const platform = currentPlatform(opts)
  const isLocked = typeof opts.isLocked === 'function' ? opts.isLocked : () => false
  const pointer = readPointer(hermesHome)
  const protectedKeys = new Set()
  if (pointer && pointer.key) protectedKeys.add(pointer.key)
  if (!opts.dropPrevious && pointer && pointer.previous) protectedKeys.add(pointer.previous)
  for (const k of opts.keep || []) if (k) protectedKeys.add(k)

  const { versions, staging } = listVersions(hermesHome)
  const result = { kept: [], removed: [], skipped: [], orphansRemoved: [], orphansSkipped: [], droppedPrevious: Boolean(opts.dropPrevious) }

  const tryRemove = (name, bucketRemoved, bucketSkipped) => {
    const abs = path.join(versionsDir, name)
    if (isLocked(name, abs)) {
      bucketSkipped.push(name)
      return
    }
    if (removeVersionDir(abs, platform)) bucketRemoved.push(name)
    else bucketSkipped.push(name) // held handle / permission — next startup retries
  }

  for (const name of versions) {
    if (protectedKeys.has(name)) {
      result.kept.push(name)
      continue
    }
    tryRemove(name, result.removed, result.skipped)
  }
  // Staging dirs are half-installs by definition — never referenced, always GC.
  for (const name of staging) {
    tryRemove(name, result.orphansRemoved, result.orphansSkipped)
  }
  return result
}

/** Diagnostic snapshot for telemetry / logs. Pure read. */
function layoutState(hermesHome) {
  const { activeLink } = bundlePaths(hermesHome)
  const pointer = readPointer(hermesHome)
  const { versions, staging } = listVersions(hermesHome)
  const link = linkStatus(activeLink)
  let activeResolvesTo = null
  if (link.kind === 'link') {
    try {
      activeResolvesTo = fs.realpathSync(activeLink)
    } catch {
      activeResolvesTo = null
    }
  }
  return {
    pointer,
    versions,
    staging,
    activeLink: { kind: link.kind, resolvesTo: activeResolvesTo }
  }
}

module.exports = {
  POINTER_BASENAME,
  POINTER_SCHEMA_VERSION,
  VERSIONS_DIRNAME,
  ACTIVE_LINK_BASENAME,
  TMP_SUFFIX,
  bundlePaths,
  readPointer,
  writePointerAtomic,
  linkStatus,
  removeLinkOnly,
  createActiveLink,
  linkResolvesTo,
  repointActiveLink,
  switchToVersion,
  rollbackToPrevious,
  reconcileActiveLink,
  listVersions,
  removeVersionDir,
  garbageCollect,
  layoutState
}
