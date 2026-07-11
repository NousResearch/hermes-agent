'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

// hc-473: keep this suite hermetic regardless of how it's invoked (npm run
// test:desktop:platforms already sets this too, but this file must not rely
// on that -- a bare `node --test electron/apex-bundle-install.test.cjs` must
// never let applyBundleUpdate's default sendTelemetry touch the real
// network). Tests that assert on beacon content inject their own fake
// sendTelemetry, which always takes priority over this env var.
process.env.APEXNODES_TELEMETRY = 'off'

const layout = require('./apex-bundle-layout.cjs')
const migrate = require('./apex-bundle-migrate.cjs')
const install = require('./apex-bundle-install.cjs')

// A manifest faithful to scripts/build-runtime-bundle.mjs output (sibling
// manifest.json shape, incl. the `archive` block). The key/os/arch match the
// REAL published win-x64 bundle noted in the ticket:
//   COS bundle/hermes-agent/c2ba29f37c67/win-x64/
const REAL_KEY = 'c2ba29f37c67'
function winManifest(overrides = {}) {
  return {
    schema: 1,
    kind: 'apexnodes-runtime-bundle',
    framework: 'hermes-agent',
    key: REAL_KEY,
    runtime_commit: REAL_KEY + '0'.repeat(28),
    os: 'win',
    arch: 'x64',
    format: 'tar.gz',
    min_desktop_version: '0.17.0',
    components: { node: { path: '.runtime/node' }, python: { path: '.runtime/py/cpython-3.11.9' } },
    fixup: { script: 'scripts/build-runtime-bundle.mjs', command: '...', mutates: [] },
    files_index: { path: '.runtime/files.tsv', sha256: 'f'.repeat(64), count: 1 },
    archive: { name: `runtime-bundle-${REAL_KEY}-win-x64.tar.gz`, sha256: 'a'.repeat(64), size: 379_000_000 },
    ...overrides
  }
}

function mkHome() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hb-install-'))
}
function rm(d) {
  fs.rmSync(d, { recursive: true, force: true })
}

// ---------------------------------------------------------------------------
// parseBundleManifest
// ---------------------------------------------------------------------------

test('parseBundleManifest: accepts the real win-x64 manifest (string + object)', () => {
  const obj = winManifest()
  assert.equal(install.parseBundleManifest(obj).key, REAL_KEY)
  assert.equal(install.parseBundleManifest(JSON.stringify(obj)).os, 'win')
})

test('parseBundleManifest: rejects wrong schema/kind/framework/missing fields', () => {
  const bad = [
    ['not json', install.parseBundleManifest.bind(null, '{oops')],
    ['schema', () => install.parseBundleManifest(winManifest({ schema: 2 }))],
    ['kind', () => install.parseBundleManifest(winManifest({ kind: 'x' }))],
    ['framework', () => install.parseBundleManifest(winManifest({ framework: 'other' }))],
    ['os', () => install.parseBundleManifest(winManifest({ os: 'linux' }))],
    ['no archive', () => install.parseBundleManifest(winManifest({ archive: undefined }))]
  ]
  for (const [label, fn] of bad) {
    assert.throws(fn, err => err.code === 'bad_manifest', `expected bad_manifest for ${label}`)
  }
})

test('parseBundleManifest: embedded (in-bundle) manifest without archive is allowed', () => {
  const m = winManifest({ archive: undefined })
  assert.equal(install.parseBundleManifest(m, { requireArchive: false }).key, REAL_KEY)
})

// ---------------------------------------------------------------------------
// checkMinDesktopVersion (F4 compat gate) — table driven
// ---------------------------------------------------------------------------

test('checkMinDesktopVersion: gates by semver', () => {
  const cases = [
    ['0.17.0', '0.17.0', true],
    ['0.17.0', '0.18.2', true],
    ['0.17.0', '0.16.7', false], // shell too old
    ['0.17.0', '1.0.0', true],
    ['0.17.0', '0.17.0-beta.1', true], // prerelease suffix ignored → core 0.17.0 == required
    [null, '0.1.0', true], // no constraint
    ['0.17.0', null, false], // unknown shell version → fail closed
    ['0.17.0', 'garbage', false]
  ]
  for (const [required, current, wantOk] of cases) {
    const r = install.checkMinDesktopVersion(winManifest({ min_desktop_version: required }), current)
    assert.equal(r.ok, wantOk, `min=${required} desktop=${current} → ${r.reason}`)
  }
})

test('compareSemver: prerelease core compares by numeric triplet (suffix ignored)', () => {
  // 0.17.0-beta.1 has core 0.17.0 == 0.17.0 → NOT less → compatible.
  assert.equal(install.compareSemver('0.17.0-beta.1', '0.17.0'), 0)
  assert.equal(install.compareSemver('0.18.0', '0.17.9'), 1)
  assert.equal(install.compareSemver('0.16.7', '0.17.0'), -1)
})

// ---------------------------------------------------------------------------
// COS URL derivation — must match .github/workflows/desktop-bundle.yml
// ---------------------------------------------------------------------------

test('deriveCosHost: strips a trailing /runtime, honors overrides, defaults', () => {
  assert.equal(
    install.deriveCosHost('https://apexnodes-runtime-202606250443-1300912302.cos.ap-guangzhou.myqcloud.com/runtime'),
    'https://apexnodes-runtime-202606250443-1300912302.cos.ap-guangzhou.myqcloud.com'
  )
  assert.equal(install.deriveCosHost('https://cdn.example.com/'), 'https://cdn.example.com')
  assert.equal(install.deriveCosHost(''), install.DEFAULT_COS_HOST)
})

test('bundleCosLayout: builds the exact real win-x64 COS paths', () => {
  const cos = install.bundleCosLayout({ cosBase: '', key: REAL_KEY, os: 'win', arch: 'x64' })
  assert.equal(cos.prefix, `bundle/hermes-agent/${REAL_KEY}/win-x64`)
  assert.equal(cos.manifestUrl, `${install.DEFAULT_COS_HOST}/bundle/hermes-agent/${REAL_KEY}/win-x64/manifest.json`)
  assert.equal(
    cos.objectUrl(`runtime-bundle-${REAL_KEY}-win-x64.tar.gz`),
    `${install.DEFAULT_COS_HOST}/bundle/hermes-agent/${REAL_KEY}/win-x64/runtime-bundle-${REAL_KEY}-win-x64.tar.gz`
  )
})

// ---------------------------------------------------------------------------
// fixup/verify argv reconstruction (robust to spaces in the home path)
// ---------------------------------------------------------------------------

test('fixup/verify argv: bundled node runs the bundled tool on the root', () => {
  const root = '/Users/some one/.apexnodes/versions/c2ba29f37c67' // space on purpose
  const m = winManifest()
  const node = install.bundledNodeExe(root, m)
  assert.ok(node.endsWith(path.join('.runtime', 'node', 'node.exe')))
  assert.deepEqual(install.fixupArgv(root, m), [path.join(root, 'scripts', 'build-runtime-bundle.mjs'), 'fixup', '--root', root])
  assert.deepEqual(install.verifyArgv(root, m), [path.join(root, 'scripts', 'build-runtime-bundle.mjs'), 'verify', '--root', root])

  const macNode = install.bundledNodeExe(root, winManifest({ os: 'mac' }))
  assert.ok(macNode.endsWith(path.join('.runtime', 'node', 'bin', 'node')))
})

// ---------------------------------------------------------------------------
// F2 — stageAndCommitBundle: NEVER extract in place; commit only after verify
// ---------------------------------------------------------------------------

// A fake extractor that lays down a minimal bundle tree (incl. the embedded
// manifest stageAndCommitBundle re-checks) so the staging→verify→rename path is
// exercised without real tar.
function fakeExtract(key) {
  return (_archivePath, destDir) => {
    fs.mkdirSync(path.join(destDir, '.runtime'), { recursive: true })
    fs.mkdirSync(path.join(destDir, 'scripts'), { recursive: true })
    fs.writeFileSync(path.join(destDir, 'scripts', 'build-runtime-bundle.mjs'), '// tool')
    fs.writeFileSync(path.join(destDir, 'payload.txt'), `runtime ${key}`)
    fs.writeFileSync(
      path.join(destDir, '.bundle-manifest.json'),
      JSON.stringify(winManifest({ key, archive: undefined }))
    )
  }
}

test('stageAndCommitBundle: extracts to .tmp, verifies, atomically commits', async () => {
  const home = mkHome()
  try {
    const calls = []
    const res = await install.stageAndCommitBundle({
      hermesHome: home,
      key: REAL_KEY,
      archivePath: '/nope/archive.tar.gz',
      manifest: winManifest(),
      extract: fakeExtract(REAL_KEY),
      runTool: (_exe, _argv, label) => calls.push(label)
    })
    assert.equal(res.ok, true)
    const finalDir = layout.bundlePaths(home).versionDir(REAL_KEY)
    assert.equal(res.versionDir, finalDir)
    assert.ok(fs.existsSync(path.join(finalDir, 'payload.txt')))
    assert.deepEqual(calls, ['fixup', 'verify'], 'fixup then verify, both on the staging tree')
    // No staging turd left behind.
    assert.equal(fs.existsSync(layout.bundlePaths(home).stagingDir(REAL_KEY)), false)
  } finally {
    rm(home)
  }
})

test('stageAndCommitBundle: verify failure leaves NO committed version, cleans .tmp', async () => {
  const home = mkHome()
  try {
    await assert.rejects(
      install.stageAndCommitBundle({
        hermesHome: home,
        key: REAL_KEY,
        archivePath: '/nope/archive.tar.gz',
        manifest: winManifest(),
        extract: fakeExtract(REAL_KEY),
        runTool: (_exe, _argv, label) => {
          if (label === 'verify') throw new Error('sha mismatch on 3 files')
        }
      }),
      err => err.code === 'stage_failed' || err.stage === 'stage'
    )
    // Core invariant: the committed version dir never appears on a failed verify.
    assert.equal(fs.existsSync(layout.bundlePaths(home).versionDir(REAL_KEY)), false)
    assert.equal(fs.existsSync(layout.bundlePaths(home).stagingDir(REAL_KEY)), false, '.tmp confined + cleaned')
  } finally {
    rm(home)
  }
})

test('stageAndCommitBundle: mismatched embedded key is rejected before commit', async () => {
  const home = mkHome()
  try {
    await assert.rejects(
      install.stageAndCommitBundle({
        hermesHome: home,
        key: REAL_KEY,
        archivePath: '/nope.tar.gz',
        manifest: winManifest(),
        extract: fakeExtract('deadbeefdead'), // extracts a DIFFERENT key
        runTool: () => {}
      }),
      err => err.code === 'key_mismatch'
    )
    assert.equal(fs.existsSync(layout.bundlePaths(home).versionDir(REAL_KEY)), false)
  } finally {
    rm(home)
  }
})

test('stageAndCommitBundle: an already-committed version is reused (idempotent)', async () => {
  const home = mkHome()
  try {
    const finalDir = layout.bundlePaths(home).versionDir(REAL_KEY)
    fs.mkdirSync(finalDir, { recursive: true })
    fs.writeFileSync(path.join(finalDir, 'sentinel'), 'kept')
    let extracted = false
    const res = await install.stageAndCommitBundle({
      hermesHome: home,
      key: REAL_KEY,
      archivePath: '/nope',
      manifest: winManifest(),
      extract: () => {
        extracted = true
      },
      runTool: () => {}
    })
    assert.equal(res.reused, true)
    assert.equal(extracted, false, 'never re-extracts over a committed immutable version')
    assert.ok(fs.existsSync(path.join(finalDir, 'sentinel')))
  } finally {
    rm(home)
  }
})

// ---------------------------------------------------------------------------
// applyBundleUpdate — F1→F2→C1 orchestration (all effects injected)
// ---------------------------------------------------------------------------

function baseDeps(home, key = REAL_KEY, manifest = winManifest()) {
  const seen = { download: 0, extract: 0 }
  return {
    seen,
    hermesHome: home,
    os: 'win',
    arch: 'x64',
    key,
    desktopVersion: '0.18.2',
    cosBase: '',
    platformOpts: { platform: process.platform }, // posix symlink on the mac CI leg
    // C2 probes injected so the disk precheck is deterministic + machine-agnostic:
    // generous free space, and a tiny per-dir size (no real du of the fake tree).
    freeBytesOf: () => 64 * 1024 * 1024 * 1024,
    dirSizeOf: () => 1024,
    fetchManifest: async () => manifest,
    download: async ({ dest }) => {
      seen.download += 1
      fs.mkdirSync(path.dirname(dest), { recursive: true })
      fs.writeFileSync(dest, 'archive-bytes')
      return { path: dest }
    },
    extract: (archivePath, destDir) => {
      seen.extract += 1
      fakeExtract(key)(archivePath, destDir)
    },
    runTool: () => {}
  }
}

test('applyBundleUpdate: full success downloads, commits, switches, GCs', { skip: process.platform === 'win32' }, async () => {
  const home = mkHome()
  try {
    const r = await install.applyBundleUpdate(baseDeps(home))
    assert.equal(r.ok, true)
    assert.equal(r.key, REAL_KEY)
    // pointer + link now point at the new version
    assert.equal(layout.readPointer(home).key, REAL_KEY)
    const { activeLink } = layout.bundlePaths(home)
    assert.equal(fs.readFileSync(path.join(activeLink, 'payload.txt'), 'utf8'), `runtime ${REAL_KEY}`)
    // downloaded archive cleaned up
    assert.equal(fs.existsSync(path.join(layout.bundlePaths(home).versionsDir, '.downloads', winManifest().archive.name)), false)
  } finally {
    rm(home)
  }
})

test('applyBundleUpdate: min_desktop_version too-new rejects BEFORE downloading', async () => {
  const home = mkHome()
  try {
    const deps = baseDeps(home, REAL_KEY, winManifest({ min_desktop_version: '9.9.9' }))
    const r = await install.applyBundleUpdate(deps)
    assert.equal(r.ok, false)
    assert.equal(r.code, 'min_desktop_version')
    assert.equal(r.required, '9.9.9')
    assert.equal(deps.seen.download, 0, 'never pulls ~0.6GB for an incompatible bundle')
    assert.equal(layout.readPointer(home), null)
  } finally {
    rm(home)
  }
})

test('applyBundleUpdate: platform / key mismatch is caught at the manifest gate', async () => {
  const home = mkHome()
  try {
    const wrongOs = await install.applyBundleUpdate(baseDeps(home, REAL_KEY, winManifest({ os: 'mac', arch: 'arm64' })))
    assert.equal(wrongOs.ok, false)
    assert.equal(wrongOs.code, 'platform_mismatch')

    const wrongKey = await install.applyBundleUpdate(baseDeps(home, REAL_KEY, winManifest({ key: 'zzzzzzzzzzzz' })))
    assert.equal(wrongKey.ok, false)
    assert.equal(wrongKey.code, 'key_mismatch')
  } finally {
    rm(home)
  }
})

test('applyBundleUpdate: a verify failure returns {ok:false} and commits nothing', { skip: process.platform === 'win32' }, async () => {
  const home = mkHome()
  try {
    const deps = baseDeps(home)
    deps.runTool = (_e, _a, label) => {
      if (label === 'verify') throw new Error('files.tsv mismatch')
    }
    const r = await install.applyBundleUpdate(deps)
    assert.equal(r.ok, false)
    assert.equal(r.stage, 'stage')
    // Nothing switched — a caller can safely fall back to the legacy chain.
    assert.equal(layout.readPointer(home), null)
    assert.equal(fs.existsSync(layout.bundlePaths(home).versionDir(REAL_KEY)), false)
  } finally {
    rm(home)
  }
})

// ---------------------------------------------------------------------------
// C2 — disk precheck (design §8) inside applyBundleUpdate
// ---------------------------------------------------------------------------

test('applyBundleUpdate: refuses BEFORE downloading when free disk < required', { skip: process.platform === 'win32' }, async () => {
  const home = mkHome()
  try {
    const deps = baseDeps(home)
    deps.freeBytesOf = () => 100 * 1024 * 1024 // 100 MiB — far below a bundle install
    const r = await install.applyBundleUpdate(deps)
    assert.equal(r.ok, false)
    assert.equal(r.code, 'insufficient_disk')
    assert.equal(r.stage, 'preflight')
    assert.match(r.error, /disk space/i)
    assert.equal(deps.seen.download, 0, 'never pulls the archive when the disk cannot hold it')
    assert.equal(layout.readPointer(home), null)
  } finally {
    rm(home)
  }
})

test('applyBundleUpdate: low disk first drops `previous` to reclaim, then proceeds', { skip: process.platform === 'win32' }, async () => {
  const home = mkHome()
  try {
    // Seed a current+previous catalog with a live link at current.
    const older = 'aaaaaaaaaaaa'
    const current = 'bbbbbbbbbbbb'
    for (const k of [older, current]) {
      const dir = layout.bundlePaths(home).versionDir(k)
      fs.mkdirSync(dir, { recursive: true })
      fs.writeFileSync(path.join(dir, 'id.txt'), k)
    }
    layout.writePointerAtomic(home, { key: current, previous: older })
    layout.repointActiveLink(home, current, { platform: process.platform })

    const deps = baseDeps(home)
    let call = 0
    // Below required on the first probe, above it on the re-check after reclaim.
    deps.freeBytesOf = () => (++call === 1 ? 100 * 1024 * 1024 : 64 * 1024 * 1024 * 1024)
    const r = await install.applyBundleUpdate(deps)
    assert.equal(r.ok, true, 'proceeds once the reclaim frees room')
    assert.ok(call >= 2, 're-checks free space after reclaiming')
    // `previous` (older) was dropped to reclaim; new key is now current.
    assert.equal(fs.existsSync(layout.bundlePaths(home).versionDir(older)), false)
    assert.equal(layout.readPointer(home).key, REAL_KEY)
  } finally {
    rm(home)
  }
})

// ---------------------------------------------------------------------------
// D1 — legacy in-place side-by-side migration (design §5) via applyBundleUpdate
// ---------------------------------------------------------------------------

function seedLegacyInPlace(home, extra = {}) {
  const dir = layout.bundlePaths(home).activeLink // HERMES_HOME/hermes-agent, a REAL dir
  fs.mkdirSync(path.join(dir, 'venv', 'bin'), { recursive: true })
  fs.writeFileSync(path.join(dir, 'venv', 'bin', 'python'), '#!/legacy/abs/venv/bin/python')
  fs.writeFileSync(path.join(dir, '.hermes-bootstrap-complete'), '{}')
  for (const [rel, body] of Object.entries(extra)) {
    fs.mkdirSync(path.dirname(path.join(dir, rel)), { recursive: true })
    fs.writeFileSync(path.join(dir, rel), body)
  }
  return dir
}

test('applyBundleUpdate: migrates a legacy in-place install side-by-side', { skip: process.platform === 'win32' }, async () => {
  const home = mkHome()
  try {
    seedLegacyInPlace(home)
    const r = await install.applyBundleUpdate(baseDeps(home))
    assert.equal(r.ok, true)
    assert.equal(r.switched.migrated, true)
    // pointer: new current, previous = the legacy sentinel (rollback fallback)
    const p = layout.readPointer(home)
    assert.equal(p.key, REAL_KEY)
    assert.equal(p.previous, migrate.LEGACY_SENTINEL)
    // active link now resolves to versions/<new>; payload readable through it
    const { activeLink } = layout.bundlePaths(home)
    assert.equal(layout.linkStatus(activeLink).kind, 'link')
    assert.equal(fs.readFileSync(path.join(activeLink, 'payload.txt'), 'utf8'), `runtime ${REAL_KEY}`)
    // legacy dir preserved aside, unmoved-content, as the rollback fallback
    const aside = migrate.legacyAsidePath(home)
    assert.equal(fs.readFileSync(path.join(aside, 'venv', 'bin', 'python'), 'utf8'), '#!/legacy/abs/venv/bin/python')
  } finally {
    rm(home)
  }
})

// ---------------------------------------------------------------------------
// hc-473: anonymous download/verify/switch telemetry
// ---------------------------------------------------------------------------

const TELEMETRY_BASE = { platform: 'win', arch: 'x64', app_version: '0.18.2', runtime_key: REAL_KEY }

test('applyBundleUpdate: full success fires download/verify/switch start+success beacons in order', { skip: process.platform === 'win32' }, async () => {
  const home = mkHome()
  try {
    const telemetryEvents = []
    const deps = baseDeps(home)
    deps.sendTelemetry = ev => telemetryEvents.push(ev)

    const r = await install.applyBundleUpdate(deps)
    assert.equal(r.ok, true)

    assert.deepEqual(telemetryEvents, [
      { ...TELEMETRY_BASE, stage: 'download', status: 'start' },
      { ...TELEMETRY_BASE, stage: 'download', status: 'success' },
      { ...TELEMETRY_BASE, stage: 'verify', status: 'start' },
      { ...TELEMETRY_BASE, stage: 'verify', status: 'success' },
      { ...TELEMETRY_BASE, stage: 'switch', status: 'start' },
      { ...TELEMETRY_BASE, stage: 'switch', status: 'success' }
    ])
  } finally {
    rm(home)
  }
})

test('applyBundleUpdate: refuses migration when user data lives in the runtime dir', { skip: process.platform === 'win32' }, async () => {
  const home = mkHome()
  try {
    seedLegacyInPlace(home, { '.env': 'RELAY_KEY=secret' })
    const r = await install.applyBundleUpdate(baseDeps(home))
    assert.equal(r.ok, false)
    assert.equal(r.code, 'migration_refused')
    assert.equal(r.reason, 'user-data-in-runtime-dir')
    // Nothing moved: no pointer, no aside, hermes-agent still a real dir.
    assert.equal(layout.readPointer(home), null)
    assert.equal(fs.existsSync(migrate.legacyAsidePath(home)), false)
    assert.equal(layout.linkStatus(layout.bundlePaths(home).activeLink).kind, 'dir')
  } finally {
    rm(home)
  }
})

test('applyBundleUpdate: a manifest-gate rejection (min_desktop_version) fires no telemetry at all', async () => {
  const home = mkHome()
  try {
    const telemetryEvents = []
    const deps = baseDeps(home, REAL_KEY, winManifest({ min_desktop_version: '9.9.9' }))
    deps.sendTelemetry = ev => telemetryEvents.push(ev)

    const r = await install.applyBundleUpdate(deps)
    assert.equal(r.ok, false)
    assert.equal(r.code, 'min_desktop_version')
    // Out of this ticket's explicit download/verify/switch scope — the manifest
    // gate itself isn't telemetered (see applyBundleUpdate's own JSDoc).
    assert.deepEqual(telemetryEvents, [])
  } finally {
    rm(home)
  }
})

test('applyBundleUpdate: a C2 disk-preflight refusal fires no telemetry at all', { skip: process.platform === 'win32' }, async () => {
  const home = mkHome()
  try {
    const telemetryEvents = []
    const deps = baseDeps(home)
    deps.freeBytesOf = () => 100 * 1024 * 1024 // far below a bundle install
    deps.sendTelemetry = ev => telemetryEvents.push(ev)

    const r = await install.applyBundleUpdate(deps)
    assert.equal(r.ok, false)
    assert.equal(r.code, 'insufficient_disk')
    // Like the manifest gate, the C2 preflight sits BEFORE the ticket's
    // download/verify/switch scope — no beacons (see applyBundleUpdate JSDoc).
    assert.deepEqual(telemetryEvents, [])
  } finally {
    rm(home)
  }
})

test('applyBundleUpdate: a download failure fires download start+failure, no verify/switch beacons', async () => {
  const home = mkHome()
  try {
    const telemetryEvents = []
    const deps = baseDeps(home)
    deps.download = async () => {
      throw new Error('getaddrinfo ENOTFOUND cos.example.com')
    }
    deps.sendTelemetry = ev => telemetryEvents.push(ev)

    const r = await install.applyBundleUpdate(deps)
    assert.equal(r.ok, false)
    assert.deepEqual(telemetryEvents, [
      { ...TELEMETRY_BASE, stage: 'download', status: 'start' },
      { ...TELEMETRY_BASE, stage: 'download', status: 'failure', error_code: 'download:network' }
    ])
  } finally {
    rm(home)
  }
})

test('applyBundleUpdate: a verify failure fires download success + verify start/failure, no switch beacons', { skip: process.platform === 'win32' }, async () => {
  const home = mkHome()
  try {
    const telemetryEvents = []
    const deps = baseDeps(home)
    deps.runTool = (_e, _a, label) => {
      if (label === 'verify') throw new Error('sha256 mismatch on 3 files')
    }
    deps.sendTelemetry = ev => telemetryEvents.push(ev)

    const r = await install.applyBundleUpdate(deps)
    assert.equal(r.ok, false)
    assert.deepEqual(telemetryEvents, [
      { ...TELEMETRY_BASE, stage: 'download', status: 'start' },
      { ...TELEMETRY_BASE, stage: 'download', status: 'success' },
      { ...TELEMETRY_BASE, stage: 'verify', status: 'start' },
      { ...TELEMETRY_BASE, stage: 'verify', status: 'failure', error_code: 'verify:checksum_mismatch' }
    ])
  } finally {
    rm(home)
  }
})

// hc-472 D1 rebased the switch stage onto switchToVersionOrMigrate: a bare real
// dir at the active path now MIGRATES (success) instead of failing with
// active-path-occupied-by-real-dir. The deterministic switch-stage failure on
// the new baseline is the D1 data-safety refusal (user data inside the legacy
// runtime dir -> migration_refused), so that is what the failure beacon test
// drives; a companion test pins the migration-success case to the same stage.
test('applyBundleUpdate: a switch failure (D1 migration refused) fires switch start+failure after a clean download+verify', { skip: process.platform === 'win32' }, async () => {
  const home = mkHome()
  try {
    seedLegacyInPlace(home, { '.env': 'RELAY_KEY=secret' })
    const telemetryEvents = []
    const deps = baseDeps(home)
    deps.sendTelemetry = ev => telemetryEvents.push(ev)

    const r = await install.applyBundleUpdate(deps)
    assert.equal(r.ok, false)
    assert.equal(r.code, 'migration_refused')
    assert.deepEqual(telemetryEvents, [
      { ...TELEMETRY_BASE, stage: 'download', status: 'start' },
      { ...TELEMETRY_BASE, stage: 'download', status: 'success' },
      { ...TELEMETRY_BASE, stage: 'verify', status: 'start' },
      { ...TELEMETRY_BASE, stage: 'verify', status: 'success' },
      { ...TELEMETRY_BASE, stage: 'switch', status: 'start' },
      { ...TELEMETRY_BASE, stage: 'switch', status: 'failure', error_code: 'switch:user-data-in-runtime-dir' }
    ])
  } finally {
    rm(home)
  }
})

test('applyBundleUpdate: a D1 legacy migration reports as switch success (rides inside the switch stage)', { skip: process.platform === 'win32' }, async () => {
  const home = mkHome()
  try {
    seedLegacyInPlace(home)
    const telemetryEvents = []
    const deps = baseDeps(home)
    deps.sendTelemetry = ev => telemetryEvents.push(ev)

    const r = await install.applyBundleUpdate(deps)
    assert.equal(r.ok, true)
    assert.equal(r.switched.migrated, true)
    const switchEvents = telemetryEvents.filter(ev => ev.stage === 'switch')
    assert.deepEqual(switchEvents, [
      { ...TELEMETRY_BASE, stage: 'switch', status: 'start' },
      { ...TELEMETRY_BASE, stage: 'switch', status: 'success' }
    ])
  } finally {
    rm(home)
  }
})
