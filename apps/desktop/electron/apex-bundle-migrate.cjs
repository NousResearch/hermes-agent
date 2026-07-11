'use strict'

/**
 * apex-bundle-migrate.cjs — hc-472 P1 · D1 (legacy → versioned side-by-side migration)
 *
 * WHY (design §5 — hermes-cloud docs/work-notes/DESIGN-hc472-runtime-bundle.md)
 * -----------------------------------------------------------------------------
 * A pre-bundle install is a REAL directory at HERMES_HOME/hermes-agent whose venv
 * bakes ABSOLUTE paths into its shebangs / pyvenv.cfg — it cannot be moved and
 * keep working. So the add-only versioned layer (apex-bundle-layout) flatly
 * REFUSES to repoint its link over a real dir (`active-path-occupied-by-real-dir`)
 * and the shell falls back to the legacy install chain forever. D1 upgrades that
 * refusal into a CONTROLLED, side-by-side migration:
 *
 *   1. the new relocatable bundle is staged+committed into versions/<key>/
 *      (caller's job — never touches the legacy dir);
 *   2. we assert NO user data lives inside the legacy runtime dir (design §5:
 *      memory/config/sessions/.env live in HERMES_HOME root + agent home, never
 *      inside hermes-agent/ — this is the tripwire that proves it before we move);
 *   3. the legacy dir is renamed aside to hermes-agent.legacy/ (a one-cycle
 *      rollback fallback), freeing the link path;
 *   4. the active link is pointed at versions/<key>/.
 *
 * The pointer records `previous = LEGACY_SENTINEL` ("legacy-inplace"), so a
 * rollback re-points the link at hermes-agent.legacy — and because a symlink /
 * junction is transparent to absolute-path resolution, the legacy venv's baked
 * `…/hermes-agent/venv/…` paths resolve correctly THROUGH the link, running the
 * old runtime exactly as before (a real file-level rollback, F5 root-fix). Once
 * the sentinel falls out of the pointer (the SECOND successful update) the aside
 * is no longer a rollback target and gcLegacyAside() reaps it.
 *
 * CRASH-SAFETY: like layout.switchToVersion, the pointer is written FIRST (truth)
 * and the move-aside + link are the derived view — reconcileMigration() heals any
 * interrupted state on the next launch. An open handle refusing the move on
 * Windows is NOT fatal: the pointer already names the new version, so the next
 * reconcile (no child holding the old venv) finishes it.
 *
 * Depends only on apex-bundle-layout (+ node fs/path), rooted at an explicit
 * hermesHome so every branch is unit-testable with real tmpdirs.
 */

const fs = require('node:fs')
const path = require('node:path')

const layout = require('./apex-bundle-layout.cjs')

// The renamed-aside legacy in-place runtime dir, kept one cycle as a rollback
// fallback. It sits in HERMES_HOME root (NOT under versions/), so layout GC never
// touches it — its lifecycle is gcLegacyAside() below.
const LEGACY_ASIDE_BASENAME = 'hermes-agent.legacy'

// Sentinel stored as the pointer's `previous` (and, after a rollback, `key`) to
// mean "the fallback IS the renamed-aside legacy dir", not a versions/<key>.
// Matches design §5's `current.json.previous` special value.
const LEGACY_SENTINEL = 'legacy-inplace'

// Basenames whose presence INSIDE the legacy runtime dir means user data lives
// there — migration must NOT move the dir aside (data-loss risk). Design §5 says
// these live OUTSIDE hermes-agent/; this list is the assertion that proves it.
// The clean `git archive` runtime source ships none of them, so on a healthy
// legacy install the assertion always passes; if it ever fires, something is
// unusual and we keep the user on their working in-place runtime.
const DEFAULT_USER_DATA_MARKERS = Object.freeze([
  '.env',
  'sessions',
  'memory',
  'seed_memory',
  'seed_memory.json',
  'config.yaml',
  'config.yml',
  'workspace'
])

function legacyAsidePath(hermesHome) {
  return path.join(path.resolve(hermesHome), LEGACY_ASIDE_BASENAME)
}

/** Non-empty = exists AND (a dir with entries | a file with bytes). */
function existsNonEmpty(p) {
  let st
  try {
    st = fs.statSync(p)
  } catch {
    return false
  }
  if (st.isDirectory()) {
    try {
      return fs.readdirSync(p).length > 0
    } catch {
      return false
    }
  }
  return st.size > 0
}

/**
 * Are we a legacy in-place install with no versioned catalog yet? legacy = the
 * active path is a REAL dir (not a link) AND versions/ has no committed version —
 * exactly the state the add-only layer refuses to repoint over.
 */
function detectLegacyInPlace(hermesHome) {
  const { activeLink } = layout.bundlePaths(hermesHome)
  const activeKind = layout.linkStatus(activeLink).kind
  const hasVersions = layout.listVersions(hermesHome).versions.length > 0
  return { legacy: activeKind === 'dir' && !hasVersions, activeKind, hasVersions }
}

/** Data-location assertion (design §5). Returns {safe, found:[names]}. */
function assertNoUserDataInLegacy(legacyDir, opts = {}) {
  const markers = opts.markers || DEFAULT_USER_DATA_MARKERS
  const found = []
  for (const m of markers) {
    if (existsNonEmpty(path.join(legacyDir, m))) found.push(m)
  }
  return { safe: found.length === 0, found }
}

/** Point the active link at an ARBITRARY target dir, replacing any prior LINK. */
function relinkTo(activeLink, targetDir, opts) {
  const status = layout.linkStatus(activeLink)
  if (status.kind === 'dir') return { ok: false, reason: 'active-path-occupied-by-real-dir' }
  if (status.kind === 'link') {
    if (layout.linkResolvesTo(activeLink, targetDir)) return { ok: true, changed: false }
    if (!layout.removeLinkOnly(activeLink)) return { ok: false, reason: 'could-not-remove-old-link' }
  } else if (status.kind === 'file' || status.kind === 'other') {
    try {
      fs.unlinkSync(activeLink)
    } catch {
      return { ok: false, reason: 'active-path-occupied' }
    }
  }
  layout.createActiveLink(activeLink, targetDir, opts)
  return { ok: true, changed: true }
}

/**
 * Rename the legacy in-place dir aside to hermes-agent.legacy, freeing the active
 * link path. Idempotent (no-op {moved:false} when the active path is not a real
 * dir). REFUSES — never clobbers — when user data is found inside, or an aside
 * already exists. A Windows open handle refusing the rename is reported
 * {ok:false, reason:'aside-move-failed'} so the caller leaves it for reconcile.
 */
function moveLegacyAside(hermesHome, opts = {}) {
  const { activeLink } = layout.bundlePaths(hermesHome)
  const aside = legacyAsidePath(hermesHome)
  if (layout.linkStatus(activeLink).kind !== 'dir') {
    return { ok: true, moved: false, asideDir: aside, reason: 'not-a-real-dir' }
  }
  if (fs.existsSync(aside)) {
    return { ok: false, moved: false, asideDir: aside, reason: 'aside-exists' }
  }
  const data = assertNoUserDataInLegacy(activeLink, opts)
  if (!data.safe) {
    return { ok: false, moved: false, asideDir: aside, reason: 'user-data-in-runtime-dir', found: data.found }
  }
  try {
    fs.renameSync(activeLink, aside)
  } catch (err) {
    return { ok: false, moved: false, asideDir: aside, reason: 'aside-move-failed', error: String((err && err.message) || err) }
  }
  return { ok: true, moved: true, asideDir: aside }
}

/**
 * Migrate a legacy in-place install onto versions/<newKey> (design §5). The
 * version dir MUST already be staged+committed+verified by the caller. Order is
 * crash-safe + reconcilable (pointer first = truth, aside+link = derived view):
 *   1. data-location assertion — refuse cleanly (no state change) if it trips;
 *   2. writePointer {key:newKey, previous:LEGACY_SENTINEL};
 *   3. move hermes-agent -> hermes-agent.legacy;
 *   4. link hermes-agent -> versions/newKey.
 * A crash at any point is healed by reconcileMigration() on the next launch. If
 * the move (3) or link (4) is deferred by an open handle, the pointer already
 * names the new version, so {ok:true, linkPending:true} is returned and reconcile
 * finishes the switch once no child holds the old venv.
 */
function migrateLegacyInPlace(hermesHome, newKey, opts = {}) {
  const { versionDir, activeLink } = layout.bundlePaths(hermesHome)
  const log = typeof opts.log === 'function' ? opts.log : () => {}
  if (!fs.existsSync(versionDir(newKey))) return { ok: false, reason: 'version-missing', key: newKey }
  if (layout.linkStatus(activeLink).kind !== 'dir') {
    // Not a legacy in-place state after all — the normal switch applies.
    return layout.switchToVersion(hermesHome, newKey, opts)
  }
  // 1. Assert BEFORE advancing the pointer so a data-unsafe home changes nothing.
  const data = assertNoUserDataInLegacy(activeLink, opts)
  if (!data.safe) {
    log(`[bundle-migrate] refusing migration: user data in runtime dir (${data.found.join(', ')})`)
    return { ok: false, reason: 'user-data-in-runtime-dir', found: data.found, key: newKey }
  }
  // 2. Truth first.
  layout.writePointerAtomic(hermesHome, { key: newKey, previous: LEGACY_SENTINEL })
  // 3. Move aside.
  const moved = moveLegacyAside(hermesHome, opts)
  if (!moved.ok) {
    // Pointer already advanced; a held handle refused the move. Do NOT roll the
    // pointer back (that would strand a committed version) — the running runtime
    // keeps using the in-place dir until restart, then reconcile finishes it.
    log(`[bundle-migrate] aside deferred (${moved.reason}); reconcile finishes it on next launch`)
    return { ok: true, key: newKey, previous: LEGACY_SENTINEL, linkPending: true, reason: moved.reason }
  }
  // 4. Link (active path is now free).
  const linkRes = layout.repointActiveLink(hermesHome, newKey, opts)
  if (!linkRes.ok) {
    log(`[bundle-migrate] link deferred (${linkRes.reason}); reconcile finishes it on next launch`)
    return { ok: true, key: newKey, previous: LEGACY_SENTINEL, linkPending: true, reason: linkRes.reason }
  }
  log(`[bundle-migrate] migrated legacy in-place -> versions/${newKey} (fallback kept at ${LEGACY_ASIDE_BASENAME})`)
  return { ok: true, key: newKey, previous: LEGACY_SENTINEL, linkPending: false, migrated: true }
}

/**
 * The switch entry point apex-bundle-install uses: a normal layout.switchToVersion
 * when the active path is a link/missing, or the D1 controlled migration when it
 * is a legacy REAL dir. Returns the same {ok, key, previous, ...} shape as
 * switchToVersion so the orchestrator treats both uniformly.
 */
function switchToVersionOrMigrate(hermesHome, newKey, opts = {}) {
  const { activeLink } = layout.bundlePaths(hermesHome)
  if (layout.linkStatus(activeLink).kind === 'dir') {
    return migrateLegacyInPlace(hermesHome, newKey, opts)
  }
  return layout.switchToVersion(hermesHome, newKey, opts)
}

/**
 * Roll back to the legacy in-place dir kept aside by the migration. Valid only
 * while the pointer's `previous` is the LEGACY_SENTINEL (design §5: the first
 * post-migration rollback target). Points the active link at hermes-agent.legacy
 * — the legacy venv's absolute self-references resolve correctly THROUGH the
 * link, so the old runtime runs exactly as before. Pointer first (key:=sentinel)
 * so a crash mid-repoint self-heals.
 */
function rollbackToLegacyInPlace(hermesHome, opts = {}) {
  const { activeLink } = layout.bundlePaths(hermesHome)
  const aside = legacyAsidePath(hermesHome)
  const pointer = layout.readPointer(hermesHome)
  if (!pointer) return { ok: false, reason: 'no-pointer' }
  if (pointer.previous !== LEGACY_SENTINEL) return { ok: false, reason: 'previous-not-legacy' }
  if (!fs.existsSync(aside)) return { ok: false, reason: 'legacy-aside-missing' }
  layout.writePointerAtomic(hermesHome, { key: LEGACY_SENTINEL, previous: pointer.key })
  const res = relinkTo(activeLink, aside, opts)
  if (!res.ok) return { ok: false, reason: res.reason, key: LEGACY_SENTINEL }
  return { ok: true, key: LEGACY_SENTINEL, previous: pointer.key }
}

/**
 * Drive any partial/interrupted migration OR rollback state to consistency at
 * shell startup, before any runtime child spawns. Delegates the plain versioned
 * case to layout.reconcileActiveLink; adds the two D1-specific transitions:
 *   - pointer.key is the LEGACY_SENTINEL (rolled back) → link must resolve to the
 *     aside;
 *   - pointer names a real version but the active path is STILL the legacy real
 *     dir (crash after pointer-write, before move) → finish the move-aside, link.
 * Fully fail-soft; never throws.
 */
function reconcileMigration(hermesHome, opts = {}) {
  const { activeLink, versionDir } = layout.bundlePaths(hermesHome)
  const aside = legacyAsidePath(hermesHome)
  const pointer = layout.readPointer(hermesHome)
  if (!pointer) return { reconciled: false, reason: 'no-pointer' }

  // Rolled back to the legacy in-place dir: the link must resolve to the aside.
  if (pointer.key === LEGACY_SENTINEL) {
    if (!fs.existsSync(aside)) return { reconciled: false, reason: 'legacy-aside-missing' }
    if (layout.linkResolvesTo(activeLink, aside)) return { reconciled: false, reason: 'already-consistent' }
    const res = relinkTo(activeLink, aside, opts)
    if (!res.ok) return { reconciled: false, reason: res.reason }
    return { reconciled: true, action: 'relink-legacy', key: LEGACY_SENTINEL }
  }

  // Forward (versioned) pointer with the legacy real dir still occupying the link
  // path → an interrupted migration; finish its move-aside first.
  if (layout.linkStatus(activeLink).kind === 'dir' && fs.existsSync(versionDir(pointer.key))) {
    const moved = moveLegacyAside(hermesHome, opts)
    if (!moved.ok) return { reconciled: false, reason: moved.reason, key: pointer.key }
  }
  // Then the plain link heal covers create/repair to versions/<key>.
  const rec = layout.reconcileActiveLink(hermesHome, opts)
  if (rec.reconciled) return { reconciled: true, action: 'migrate-relink', key: rec.key }
  return { reconciled: false, reason: rec.reason, key: pointer.key }
}

/**
 * GC the legacy aside once it is no longer a rollback target — i.e. the
 * LEGACY_SENTINEL has fallen out of BOTH pointer.key and pointer.previous, which
 * happens on the SECOND successful update (its switch sets previous = the first
 * bundle key, dropping the sentinel). Design §5: "新版稳定跑通…后 GC 删老
 * hermes-agent.legacy/". Respects an injected isLocked; fail-soft.
 */
function gcLegacyAside(hermesHome, opts = {}) {
  const aside = legacyAsidePath(hermesHome)
  if (!fs.existsSync(aside)) return { removed: false, reason: 'no-aside' }
  const pointer = layout.readPointer(hermesHome)
  if (!pointer) return { removed: false, reason: 'no-pointer' }
  if (pointer.key === LEGACY_SENTINEL || pointer.previous === LEGACY_SENTINEL) {
    return { removed: false, reason: 'still-rollback-target' }
  }
  const isLocked = typeof opts.isLocked === 'function' ? opts.isLocked : () => false
  if (isLocked(LEGACY_ASIDE_BASENAME, aside)) return { removed: false, reason: 'locked' }
  try {
    fs.rmSync(aside, { recursive: true, force: true })
    return { removed: true, path: aside }
  } catch (err) {
    return { removed: false, reason: 'rm-failed', error: String((err && err.message) || err) }
  }
}

/** Diagnostic snapshot for telemetry / logs. Pure read. */
function migrationState(hermesHome) {
  const aside = legacyAsidePath(hermesHome)
  const det = detectLegacyInPlace(hermesHome)
  const pointer = layout.readPointer(hermesHome)
  return {
    legacyInPlace: det.legacy,
    activeKind: det.activeKind,
    asideExists: fs.existsSync(aside),
    sentinelActive: Boolean(pointer && (pointer.key === LEGACY_SENTINEL || pointer.previous === LEGACY_SENTINEL)),
    pointer
  }
}

module.exports = {
  LEGACY_ASIDE_BASENAME,
  LEGACY_SENTINEL,
  DEFAULT_USER_DATA_MARKERS,
  legacyAsidePath,
  detectLegacyInPlace,
  assertNoUserDataInLegacy,
  moveLegacyAside,
  migrateLegacyInPlace,
  switchToVersionOrMigrate,
  rollbackToLegacyInPlace,
  reconcileMigration,
  gcLegacyAside,
  migrationState
}
