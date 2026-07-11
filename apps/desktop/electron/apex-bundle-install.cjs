'use strict'

/**
 * apex-bundle-install.cjs — hc-472 P1 · F2 (+ manifest/version/COS glue)
 *
 * The half-state protection that carries the CORE INVARIANT (design §8):
 * NEVER extract in place. A bundle is staged into versions/<key>.tmp/, stamped
 * for its location + verified per-file against the bundle's own sha index using
 * the bundled tool copy, and only THEN atomically renamed into versions/<key>/.
 * If anything fails (or the process is killed) the damage is confined to an
 * unreferenced `.tmp` dir that startup GC reaps; the pointer/link (C1) only ever
 * name a fully-verified version.
 *
 * Also holds the pure glue the wiring needs:
 *   - parseBundleManifest / checkMinDesktopVersion  (schema + F4 compat gate)
 *   - bundle COS URL derivation (matches .github/workflows/desktop-bundle.yml)
 *   - fixup/verify argv reconstructed structurally from the manifest (robust to
 *     spaces in the user's home path — the pre-rendered manifest.fixup.command
 *     is the human-readable spec of exactly this invocation)
 *   - applyBundleUpdate: the F1→F2→C1 orchestration, every side-effect injected
 *     so the whole flow is unit-testable without electron / tar / the network.
 *
 * Manifest schema authored by scripts/build-runtime-bundle.mjs (sibling
 * manifest.json adds `archive:{name,sha256,size}`).
 */

const fs = require('node:fs')
const path = require('node:path')

const layout = require('./apex-bundle-layout.cjs')
const migrate = require('./apex-bundle-migrate.cjs')
const diskspace = require('./apex-bundle-diskspace.cjs')

const MANIFEST_SCHEMA = 1
const BUNDLE_KIND = 'apexnodes-runtime-bundle'
const FRAMEWORK = 'hermes-agent'
// Public-read COS host root (bucket apexnodes-runtime). Kept in lockstep with
// apexnodes-environment.yaml `cos_base` (minus its /runtime suffix) and
// .github/workflows/desktop-bundle.yml's verify step BASE. Overridable via
// HERMES_RUNTIME_COS_BASE (the same env install.sh's region-detect honours).
const DEFAULT_COS_HOST = 'https://apexnodes-runtime-202606250443-1300912302.cos.ap-guangzhou.myqcloud.com'

class BundleInstallError extends Error {
  constructor(message, code, stage) {
    super(message)
    this.name = 'BundleInstallError'
    this.code = code || 'install_failed'
    if (stage) this.stage = stage
  }
}

// ---------------------------------------------------------------------------
// manifest parsing + compatibility gate
// ---------------------------------------------------------------------------

/**
 * Parse + structurally validate a bundle manifest (string or object). Throws
 * BundleInstallError('bad_manifest') on anything that would make install unsafe.
 * `requireArchive` (default true) also demands the sibling `archive` block that
 * the COS manifest.json carries (the in-bundle .bundle-manifest.json omits it).
 */
function parseBundleManifest(input, { requireArchive = true } = {}) {
  let m = input
  if (typeof input === 'string') {
    try {
      m = JSON.parse(input)
    } catch (err) {
      throw new BundleInstallError(`manifest is not valid JSON: ${err.message}`, 'bad_manifest')
    }
  }
  if (!m || typeof m !== 'object') throw new BundleInstallError('manifest is not an object', 'bad_manifest')
  if (m.schema !== MANIFEST_SCHEMA) throw new BundleInstallError(`unsupported manifest schema ${m.schema} (need ${MANIFEST_SCHEMA})`, 'bad_manifest')
  if (m.kind !== BUNDLE_KIND) throw new BundleInstallError(`unexpected manifest kind ${m.kind}`, 'bad_manifest')
  if (m.framework !== FRAMEWORK) throw new BundleInstallError(`unexpected framework ${m.framework}`, 'bad_manifest')
  if (!m.key || typeof m.key !== 'string') throw new BundleInstallError('manifest missing key', 'bad_manifest')
  if (m.os !== 'win' && m.os !== 'mac') throw new BundleInstallError(`unexpected os ${m.os}`, 'bad_manifest')
  if (!m.arch) throw new BundleInstallError('manifest missing arch', 'bad_manifest')
  if (!m.components || !m.components.node || !m.components.node.path) throw new BundleInstallError('manifest missing components.node.path', 'bad_manifest')
  if (!m.fixup || !m.fixup.script) throw new BundleInstallError('manifest missing fixup.script', 'bad_manifest')
  if (!m.files_index || !m.files_index.sha256) throw new BundleInstallError('manifest missing files_index', 'bad_manifest')
  if (requireArchive) {
    if (!m.archive || !m.archive.name || !m.archive.sha256) {
      throw new BundleInstallError('manifest missing archive{name,sha256}', 'bad_manifest')
    }
  }
  return m
}

/** Parse "1.2.3" (ignoring any -prerelease/+build suffix) → [1,2,3]. */
function parseSemver(v) {
  const core = String(v || '').trim().split(/[-+]/)[0]
  const parts = core.split('.').map(n => Number.parseInt(n, 10))
  if (parts.length === 0 || parts.some(n => Number.isNaN(n))) return null
  while (parts.length < 3) parts.push(0)
  return parts.slice(0, 3)
}

/** a<b → -1, a==b → 0, a>b → 1. Unparseable sorts as "0.0.0". */
function compareSemver(a, b) {
  const pa = parseSemver(a) || [0, 0, 0]
  const pb = parseSemver(b) || [0, 0, 0]
  for (let i = 0; i < 3; i++) {
    if (pa[i] < pb[i]) return -1
    if (pa[i] > pb[i]) return 1
  }
  return 0
}

/**
 * F4 compatibility闸: refuse a bundle whose min_desktop_version is newer than
 * the running shell. Returns {ok, required, current, reason}. A missing/
 * unparseable min_desktop_version is treated as "no constraint" (ok:true) — the
 * build always stamps one, so absence means an old bundle we don't gate.
 */
function checkMinDesktopVersion(manifest, desktopVersion) {
  const required = manifest && manifest.min_desktop_version
  if (!required || !parseSemver(required)) {
    return { ok: true, required: required || null, current: desktopVersion || null, reason: 'no-constraint' }
  }
  if (!desktopVersion || !parseSemver(desktopVersion)) {
    // Can't prove compatibility — fail closed with a clear reason.
    return { ok: false, required, current: desktopVersion || null, reason: 'unknown-desktop-version' }
  }
  const ok = compareSemver(desktopVersion, required) >= 0
  return { ok, required, current: desktopVersion, reason: ok ? 'ok' : 'desktop-too-old' }
}

// ---------------------------------------------------------------------------
// COS URL derivation — mirrors .github/workflows/desktop-bundle.yml exactly:
//   <host>/bundle/hermes-agent/<key>/<os>-<arch>/{manifest.json, <archive>, <archive>.sha256}
// (manifest.json is uploaded under a FIXED name; the archive/sha keep their
//  build names, which the manifest's `archive.name` tells us.)
// ---------------------------------------------------------------------------

function trimTrailingSlash(v) {
  return String(v || '').replace(/\/+$/, '')
}

/** Bucket root from a runtime cos base (strip a trailing `/runtime`). */
function deriveCosHost(cosBase) {
  const clean = trimTrailingSlash(cosBase || '')
  if (!clean) return DEFAULT_COS_HOST
  return clean.replace(/\/runtime$/i, '')
}

/**
 * @returns {{host, prefix, prefixUrl, manifestUrl, objectUrl:(name:string)=>string}}
 */
function bundleCosLayout({ cosBase, key, os, arch, framework = FRAMEWORK }) {
  const host = deriveCosHost(cosBase)
  const prefix = `bundle/${framework}/${key}/${os}-${arch}`
  const prefixUrl = `${host}/${prefix}`
  return {
    host,
    prefix,
    prefixUrl,
    manifestUrl: `${prefixUrl}/manifest.json`,
    objectUrl: name => `${prefixUrl}/${name}`
  }
}

// ---------------------------------------------------------------------------
// fixup/verify invocation — reconstructed from manifest fields (NOT the
// pre-rendered fixup.command string, which naive-splits on spaces and would
// break on a home path containing spaces). The bundled node runs the bundled
// tool copy on the bundle itself: no external binary to keep in sync.
// ---------------------------------------------------------------------------

function bundledNodeExe(root, manifest) {
  const nodeRel = manifest.components.node.path.split('/')
  const tail = manifest.os === 'win' ? ['node.exe'] : ['bin', 'node']
  return path.join(root, ...nodeRel, ...tail)
}

function bundledToolScript(root, manifest) {
  return path.join(root, ...manifest.fixup.script.split('/'))
}

function fixupArgv(root, manifest) {
  return [bundledToolScript(root, manifest), 'fixup', '--root', root]
}

function verifyArgv(root, manifest) {
  return [bundledToolScript(root, manifest), 'verify', '--root', root]
}

// ---------------------------------------------------------------------------
// F2 — stage → fixup → verify → atomic commit
// ---------------------------------------------------------------------------

/**
 * Stage an archive into versions/<key>.tmp, stamp+verify it in place, then
 * atomically rename it to versions/<key>. NEVER touches versions/<key> until the
 * staged tree passes verify.
 *
 * `extract` / `runTool` may be sync or async — both are awaited, so the real
 * wiring can spawn non-blocking child processes (a multi-minute tar extract must
 * not freeze the electron main thread) while tests pass plain sync fakes.
 *
 * @param {object} o
 * @param {string} o.hermesHome
 * @param {string} o.key
 * @param {string} o.archivePath        downloaded, sha-gated archive
 * @param {object} o.manifest           parsed bundle manifest
 * @param {(archivePath:string, destDir:string)=>void|Promise<void>} o.extract   tar -xzf equiv
 * @param {(exe:string, argv:string[], label:string)=>void|Promise<void>} o.runTool   spawn bundled node
 * @param {object} [o.opts]             {platform} for path shaping
 * @param {(msg:string)=>void} [o.log]
 * @returns {Promise<{ok:true, versionDir:string, reused?:boolean}>}
 */
async function stageAndCommitBundle(o) {
  const { hermesHome, key, archivePath, manifest, extract, runTool, log = () => {} } = o
  const paths = layout.bundlePaths(hermesHome)
  const finalDir = paths.versionDir(key)
  const stagingDir = paths.stagingDir(key)

  // An existing committed dir is immutable and was verified before its own
  // commit — reuse it (idempotent re-apply / already-current switch).
  if (fs.existsSync(finalDir)) {
    log(`[bundle-install] versions/${key} already present — reusing`)
    return { ok: true, versionDir: finalDir, reused: true }
  }

  fs.mkdirSync(paths.versionsDir, { recursive: true })
  // Clear any prior half-install for this key (crash residue or a failed run).
  fs.rmSync(stagingDir, { recursive: true, force: true })
  fs.mkdirSync(stagingDir, { recursive: true })

  try {
    log(`[bundle-install] extracting into ${path.basename(stagingDir)}`)
    await extract(archivePath, stagingDir)

    // Confirm the staged tree is actually a bundle for THIS key before trusting.
    const embedded = path.join(stagingDir, '.bundle-manifest.json')
    if (!fs.existsSync(embedded)) throw new BundleInstallError('staged tree has no .bundle-manifest.json', 'extract_incomplete', 'extract')
    const staged = parseBundleManifest(fs.readFileSync(embedded, 'utf8'), { requireArchive: false })
    if (staged.key !== key) throw new BundleInstallError(`staged key ${staged.key} != expected ${key}`, 'key_mismatch', 'extract')

    log('[bundle-install] fixup (stamp for this location)')
    await runTool(bundledNodeExe(stagingDir, manifest), fixupArgv(stagingDir, manifest), 'fixup')

    log('[bundle-install] verify (per-file sha against files.tsv)')
    await runTool(bundledNodeExe(stagingDir, manifest), verifyArgv(stagingDir, manifest), 'verify')

    // Atomic promote. finalDir does not exist (checked above), so rename is a
    // single metadata op — the moment versions/<key> becomes referenceable it is
    // already whole + verified.
    fs.renameSync(stagingDir, finalDir)
    log(`[bundle-install] committed versions/${key}`)
    return { ok: true, versionDir: finalDir }
  } catch (err) {
    // Confine the damage: drop the staging dir so we don't accumulate half
    // trees. (A hard crash instead leaves it for startup GC — same invariant.)
    fs.rmSync(stagingDir, { recursive: true, force: true })
    if (err instanceof BundleInstallError) throw err
    throw new BundleInstallError(String(err && err.message || err), 'stage_failed', 'stage')
  }
}

// ---------------------------------------------------------------------------
// orchestration — F1 (download) → F2 (stage/verify/commit) → C1 (switch) → GC
// ---------------------------------------------------------------------------

/**
 * Full bundle apply. Every effect is injected so the flow is unit-testable end
 * to end with fakes; main.cjs supplies the real download / tar / spawn.
 *
 * @param {object} o
 * @param {string} o.hermesHome
 * @param {string} o.os     'win' | 'mac'
 * @param {string} o.arch   'x64' | 'arm64'
 * @param {string} o.key    runtime bundle key (sha12)
 * @param {string} o.desktopVersion   app.getVersion()
 * @param {string} [o.cosBase]        runtime cos base (host derived from it)
 * @param {(url:string)=>Promise<any>} o.fetchManifest
 * @param {(o:{url,dest,sha256,size})=>Promise<{path:string}>} o.download
 * @param {(archivePath:string, destDir:string)=>void} o.extract
 * @param {(exe:string, argv:string[], label:string)=>void} o.runTool
 * @param {string} [o.downloadDir]     where the archive lands (default versions/.downloads)
 * @param {object} [o.platformOpts]    {platform} forwarded to layout
 * @param {(p:string)=>number} [o.freeBytesOf]   C2 precheck free-space probe (injected in tests)
 * @param {(dir:string)=>number} [o.dirSizeOf]   C2 versions/ usage probe (injected in tests)
 * @param {number} [o.minFreeBytes]    C2 install free-space floor override
 * @param {string[]} [o.migrateMarkers] D1 data-location assertion marker override
 * @param {(msg:string)=>void} [o.log]
 * @returns {Promise<{ok:true, key, versionDir, switched}|{ok:false, error, code, stage}>}
 */
async function applyBundleUpdate(o) {
  const {
    hermesHome,
    os,
    arch,
    key,
    desktopVersion,
    cosBase,
    fetchManifest,
    download,
    extract,
    runTool,
    platformOpts,
    freeBytesOf,
    dirSizeOf,
    minFreeBytes,
    migrateMarkers,
    log = () => {}
  } = o
  const platform = platformOpts && platformOpts.platform
  const paths = layout.bundlePaths(hermesHome)
  const cos = bundleCosLayout({ cosBase, key, os, arch })

  try {
    // 1. Manifest first (small): schema + platform match + compat gate BEFORE we
    //    pull ~0.6 GB.
    const raw = await fetchManifest(cos.manifestUrl)
    const manifest = parseBundleManifest(raw)
    if (manifest.os !== os || manifest.arch !== arch) {
      return { ok: false, code: 'platform_mismatch', stage: 'manifest', error: `manifest is ${manifest.os}-${manifest.arch}, need ${os}-${arch}` }
    }
    if (manifest.key !== key) {
      return { ok: false, code: 'key_mismatch', stage: 'manifest', error: `manifest key ${manifest.key} != requested ${key}` }
    }
    const compat = checkMinDesktopVersion(manifest, desktopVersion)
    if (!compat.ok) {
      return {
        ok: false,
        code: 'min_desktop_version',
        stage: 'manifest',
        error: `runtime bundle needs desktop ≥ ${compat.required} (this shell is ${compat.current || 'unknown'})`,
        required: compat.required,
        current: compat.current
      }
    }

    // 1b. C2 disk precheck: refuse BEFORE pulling ~0.6 GB when the target volume
    //     can't hold the archive + its extracted staging. Under pressure first
    //     force-drop `previous` to reclaim ~1 bundle (design §8 "不足则先 GC
    //     previous"), then re-check; only then refuse with a readable error.
    const archiveSize = manifest.archive.size
    const preflightArgs = { hermesHome, archiveSize, freeBytesOf, dirSizeOf, minFreeBytes, platform }
    let disk = diskspace.preflightDiskSpace(preflightArgs)
    if (!disk.ok) {
      log(`[bundle-install] low disk (free=${disk.freeBytes} < need=${disk.requiredBytes}); dropping previous to reclaim`)
      try {
        layout.garbageCollect(hermesHome, { platform, dropPrevious: true })
      } catch {
        // reclamation is best-effort; the re-check below still gates.
        void 0
      }
      disk = diskspace.preflightDiskSpace(preflightArgs)
    }
    if (!disk.ok) {
      return { ok: false, code: 'insufficient_disk', stage: 'preflight', error: disk.message, freeBytes: disk.freeBytes, requiredBytes: disk.requiredBytes }
    }

    // 2. Download the archive with Range resume, gated on the manifest's sha256.
    const archiveName = manifest.archive.name
    const downloadDir = o.downloadDir || path.join(paths.versionsDir, '.downloads')
    const archivePath = path.join(downloadDir, archiveName)
    log(`[bundle-install] downloading ${archiveName}`)
    await download({
      url: cos.objectUrl(archiveName),
      dest: archivePath,
      sha256: manifest.archive.sha256,
      size: manifest.archive.size
    })

    // 3. Stage → fixup → verify → atomic commit (F2).
    const staged = await stageAndCommitBundle({ hermesHome, key, archivePath, manifest, extract, runTool, opts: platformOpts, log })

    // 4. Atomic switch: pointer (truth) then link (derived) (C1) — upgraded to a
    //    CONTROLLED side-by-side MIGRATION (D1) when the active path is a legacy
    //    in-place dir (turning the add-only layer's real-dir refusal into a
    //    migration + a `legacy-inplace` rollback fallback).
    const sw = migrate.switchToVersionOrMigrate(hermesHome, key, { platform, markers: migrateMarkers, log })
    if (!sw.ok) {
      const code = sw.reason === 'user-data-in-runtime-dir' ? 'migration_refused' : 'switch_failed'
      return { ok: false, code, stage: 'switch', error: `could not activate versions/${key}: ${sw.reason}`, reason: sw.reason, found: sw.found }
    }

    // 5. Best-effort cleanup of the downloaded archive + startup-style GC (keep
    //    current+previous). Never fatal.
    try {
      fs.rmSync(archivePath, { force: true })
    } catch {
      void 0
    }
    let gc = null
    try {
      gc = layout.garbageCollect(hermesHome, { platform: platformOpts && platformOpts.platform })
    } catch {
      gc = null
    }

    return { ok: true, key, versionDir: staged.versionDir, runtimeCommit: manifest.runtime_commit || null, switched: sw, gc }
  } catch (err) {
    const code = (err && err.code) || 'install_failed'
    const stage = (err && err.stage) || 'unknown'
    log(`[bundle-install] FAILED at ${stage}: ${code} — ${err && err.message}`)
    return { ok: false, code, stage, error: (err && err.message) || String(err) }
  }
}

module.exports = {
  MANIFEST_SCHEMA,
  BUNDLE_KIND,
  FRAMEWORK,
  DEFAULT_COS_HOST,
  BundleInstallError,
  parseBundleManifest,
  parseSemver,
  compareSemver,
  checkMinDesktopVersion,
  deriveCosHost,
  bundleCosLayout,
  bundledNodeExe,
  bundledToolScript,
  fixupArgv,
  verifyArgv,
  stageAndCommitBundle,
  applyBundleUpdate
}
