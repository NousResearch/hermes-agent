const assert = require('node:assert/strict')
const test = require('node:test')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

// hc-473: keep this suite hermetic regardless of how it's invoked (npm run
// test:desktop:platforms already sets this too, but this file must not rely
// on that -- a bare `node --test electron/bootstrap-runner.test.cjs` must
// never let runBootstrap's default sendTelemetry touch the real network).
// Tests that assert on beacon content inject their own fake sendTelemetry,
// which always takes priority over this env var.
process.env.APEXNODES_TELEMETRY = 'off'

const {
  runBootstrap,
  resolveInstallScript,
  installedAgentInstallScript,
  bundledInstallScript,
  cnInstallEnv,
  cachedScriptPath,
  runtimeKeyFromStamp
} = require('./bootstrap-runner.cjs')
const { normalizeDesktopPlatform } = require('./apexnodes-telemetry.cjs')

const SCRIPT_NAME = process.platform === 'win32' ? 'install.ps1' : 'install.sh'

function mkTmpHome() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-bootstrap-test-'))
}

test('runBootstrap bails immediately when the signal is already aborted', async () => {
  const controller = new AbortController()
  controller.abort()

  const events = []
  const telemetryEvents = []
  const result = await runBootstrap({
    installStamp: null,
    activeRoot: '/tmp/hermes-runner-test',
    sourceRepoRoot: null,
    hermesHome: '/tmp/hermes-runner-test',
    logRoot: '/tmp/hermes-runner-test',
    onEvent: ev => events.push(ev),
    abortSignal: controller.signal,
    sendTelemetry: ev => telemetryEvents.push(ev)
  })

  // Cancelled before any install script is spawned.
  assert.deepEqual(result, { ok: false, cancelled: true })
  assert.ok(
    events.some(ev => ev.type === 'failed' && /cancelled/i.test(ev.error)),
    'should emit a cancelled failure event'
  )
  // hc-473: a cancelled-before-anything-started run still beacons — reported
  // as a bootstrap-level failure (there's no cleaner tri-state slot for
  // "user cancelled", and "did this run reach completion" is what the
  // ops-dashboard funnel actually wants to answer).
  assert.deepEqual(telemetryEvents, [
    {
      platform: normalizeDesktopPlatform(process.platform),
      arch: process.arch,
      app_version: null,
      runtime_key: null,
      stage: 'bootstrap',
      status: 'failure',
      error_code: 'bootstrap:cancelled'
    }
  ])
})

// ---------------------------------------------------------------------------
// hc-473: runtimeKeyFromStamp — commit > branch > version priority
// ---------------------------------------------------------------------------

test('runtimeKeyFromStamp: prefers commit, then branch, then version; null for no stamp', () => {
  assert.equal(runtimeKeyFromStamp(null), null)
  assert.equal(runtimeKeyFromStamp({}), null)
  assert.equal(runtimeKeyFromStamp({ version: '2026.7.1' }), '2026.7.1')
  assert.equal(runtimeKeyFromStamp({ branch: 'v2026.7.1', version: '2026.7.1' }), 'v2026.7.1')
  assert.equal(runtimeKeyFromStamp({ commit: 'a'.repeat(40), branch: 'main', version: '2026.7.1' }), 'a'.repeat(40))
})

test('installedAgentInstallScript resolves the installer in the agent checkout', () => {
  const home = mkTmpHome()
  try {
    assert.equal(installedAgentInstallScript(home), null, 'absent before the checkout exists')

    const scriptsDir = path.join(home, 'hermes-agent', 'scripts')
    fs.mkdirSync(scriptsDir, { recursive: true })
    const scriptPath = path.join(scriptsDir, SCRIPT_NAME)
    fs.writeFileSync(scriptPath, '#!/bin/sh\necho hi\n')

    assert.equal(installedAgentInstallScript(home), scriptPath)
    assert.equal(installedAgentInstallScript(null), null, 'null home -> null')
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('resolveInstallScript prefers a cached script without touching the network', async () => {
  const home = mkTmpHome()
  try {
    const commit = 'a'.repeat(40)
    const cached = cachedScriptPath(home, commit)
    fs.mkdirSync(path.dirname(cached), { recursive: true })
    fs.writeFileSync(cached, '#!/bin/sh\necho cached\n')

    const logs = []
    const result = await resolveInstallScript({
      installStamp: { commit },
      sourceRepoRoot: null,
      hermesHome: home,
      emit: ev => logs.push(ev)
    })

    assert.equal(result.source, 'cache')
    assert.equal(result.path, cached)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('resolveInstallScript falls back to the installed agent checkout on a 404', async () => {
  const home = mkTmpHome()
  try {
    const commit = 'a'.repeat(40)
    // Seed the installed agent checkout so the fallback has something to resolve.
    const scriptsDir = path.join(home, 'hermes-agent', 'scripts')
    fs.mkdirSync(scriptsDir, { recursive: true })
    const installed = path.join(scriptsDir, SCRIPT_NAME)
    fs.writeFileSync(installed, '#!/bin/sh\necho fallback\n')

    const logs = []
    const result = await resolveInstallScript({
      installStamp: { commit },
      sourceRepoRoot: null,
      hermesHome: home,
      emit: ev => logs.push(ev),
      // Simulate GitHub returning a 404 for the pinned commit.
      _download: async () => {
        throw new Error('Failed to download install.sh: HTTP 404')
      }
    })

    assert.equal(result.source, 'installed-agent')
    // It should have copied the installer into the bootstrap cache.
    assert.equal(result.path, cachedScriptPath(home, commit))
    assert.ok(fs.existsSync(result.path), 'fallback script copied into cache')
    assert.ok(
      logs.some(ev => /falling back to installed agent/.test(ev.line || '')),
      'emits a fallback log line'
    )
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('resolveInstallScript rethrows when the 404 fallback is unavailable', async () => {
  const home = mkTmpHome()
  try {
    const commit = 'a'.repeat(40)
    // No installed agent checkout seeded -> nothing to fall back to.
    await assert.rejects(
      resolveInstallScript({
        installStamp: { commit },
        sourceRepoRoot: null,
        hermesHome: home,
        emit: () => {},
        _download: async () => {
          throw new Error('Failed to download install.sh: HTTP 404')
        }
      }),
      /HTTP 404|Failed to download/
    )
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('bundledInstallScript resolves the installer shipped in resourcesPath', () => {
  const dir = mkTmpHome()
  try {
    assert.equal(bundledInstallScript(dir), null, 'absent before the script exists')
    assert.equal(bundledInstallScript(null), null, 'null resourcesPath -> null')

    const scriptPath = path.join(dir, SCRIPT_NAME)
    fs.writeFileSync(scriptPath, '#!/bin/sh\necho bundled\n')
    assert.equal(bundledInstallScript(dir), scriptPath)
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('resolveInstallScript prefers the bundled installer over the GitHub download', async () => {
  const home = mkTmpHome()
  const resources = mkTmpHome()
  try {
    const bundled = path.join(resources, SCRIPT_NAME)
    fs.writeFileSync(bundled, '#!/bin/sh\necho bundled\n')

    let downloadCalled = false
    const logs = []
    const result = await resolveInstallScript({
      installStamp: { commit: 'a'.repeat(40) },
      sourceRepoRoot: null,
      resourcesPath: resources,
      hermesHome: home,
      emit: ev => logs.push(ev),
      _download: async () => {
        downloadCalled = true
        throw new Error('bundled script must short-circuit before any download')
      }
    })

    assert.equal(result.source, 'bundled')
    assert.equal(result.path, bundled)
    assert.equal(downloadCalled, false, 'a bundled script must not trigger a network download')
    assert.ok(
      logs.some(ev => /using bundled/.test(ev.line || '')),
      'emits a "using bundled" log line'
    )
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
    fs.rmSync(resources, { recursive: true, force: true })
  }
})

test('cnInstallEnv decouples the COS base from the mirror flag for auto-detection', () => {
  const savedFlag = process.env.HERMES_CN_MIRRORS
  const savedBase = process.env.HERMES_RUNTIME_COS_BASE
  try {
    delete process.env.HERMES_CN_MIRRORS
    delete process.env.HERMES_RUNTIME_COS_BASE

    // No opt-in, no env, no COS base -> spawn env untouched.
    assert.deepEqual(cnInstallEnv(), {})
    assert.deepEqual(cnInstallEnv({ cnMirrors: false }), {})

    // KEY CHANGE: a packaged build passes runtimeCosBase WITHOUT forcing the
    // mirror flag. The COS base must still be threaded through (so install.sh's
    // own auto-detection can fetch from COS when it picks CN), and the mirror
    // flag must be OMITTED so install.sh runs its region auto-detection.
    assert.deepEqual(cnInstallEnv({ cnMirrors: false, runtimeCosBase: 'https://cos.example/runtime' }), {
      HERMES_RUNTIME_COS_BASE: 'https://cos.example/runtime'
    })

    // Caller forces CN on (e.g. an explicit user/region choice upstream).
    assert.deepEqual(cnInstallEnv({ cnMirrors: true, runtimeCosBase: 'https://cos.example/runtime' }), {
      HERMES_CN_MIRRORS: '1',
      HERMES_RUNTIME_COS_BASE: 'https://cos.example/runtime'
    })

    // Escape hatch: an explicit env flag of '0' is forwarded verbatim (install.sh
    // treats a set flag as authoritative and stays on upstream defaults without
    // probing) even when the caller opts in. The COS base still rides along.
    process.env.HERMES_CN_MIRRORS = '0'
    assert.deepEqual(cnInstallEnv({ cnMirrors: true, runtimeCosBase: 'https://cos.example/runtime' }), {
      HERMES_CN_MIRRORS: '0',
      HERMES_RUNTIME_COS_BASE: 'https://cos.example/runtime'
    })
    // ...and with no COS base, just the forwarded flag.
    delete process.env.HERMES_RUNTIME_COS_BASE
    assert.deepEqual(cnInstallEnv({ cnMirrors: true }), { HERMES_CN_MIRRORS: '0' })

    // Escape hatch: an explicit env flag of '1' forces it on and the env COS
    // base wins over a passed value.
    process.env.HERMES_CN_MIRRORS = '1'
    process.env.HERMES_RUNTIME_COS_BASE = 'https://override.example/r'
    assert.deepEqual(cnInstallEnv({ cnMirrors: false, runtimeCosBase: 'https://passed/r' }), {
      HERMES_CN_MIRRORS: '1',
      HERMES_RUNTIME_COS_BASE: 'https://override.example/r'
    })
  } finally {
    if (savedFlag === undefined) delete process.env.HERMES_CN_MIRRORS
    else process.env.HERMES_CN_MIRRORS = savedFlag
    if (savedBase === undefined) delete process.env.HERMES_RUNTIME_COS_BASE
    else process.env.HERMES_RUNTIME_COS_BASE = savedBase
  }
})

// ---------------------------------------------------------------------------
// hc-452: updateInfo threading. main.cjs resolves {isUpdate, toVersion,
// fromVersion} BEFORE calling runBootstrap (from whether a runtime-pin
// override is pending) and expects it to ride along on the 'manifest' event
// unmodified, so the renderer can show "updating to vX" instead of "one-time
// install" during an opt-in runtime version update. These tests drive
// runBootstrap end-to-end against a minimal fake install.sh stub (posix-only;
// the real install.sh/.ps1 protocol is exercised by the Python test suite
// under tests/test_install_sh_*.py and tests/test_install_ps1_*.py) so the
// threading is verified through the real spawn/parse path, not just a direct
// function call.
//
// Skipped on win32: bootstrap-runner spawns 'bash' unconditionally for a
// posix-kind installer (see spawnBash/resolveInstallScript), which isn't
// necessarily on PATH in a bare Windows CI shell -- this whole file's other
// tests already run cross-platform via runBootstrap's OTHER code paths
// (resolveInstallScript, cnInstallEnv) that don't spawn anything.
const describeManifestFlow = process.platform === 'win32' ? test.skip : test

function writeFakeInstallSh(sourceRepoRoot) {
  // Minimal posix installer stub: understands --manifest (one stage) and
  // --stage complete --non-interactive --json (immediately succeeds). Enough
  // to drive runBootstrap's real fetchManifest -> runStage -> writeMarker
  // path without needing the full 3000+-line real install.sh.
  //
  // Must live at <sourceRepoRoot>/scripts/install.sh -- resolveLocalInstallScript
  // (the "dev shortcut" resolution tier runBootstrap tries first) hardcodes that
  // relative path.
  const scriptsDir = path.join(sourceRepoRoot, 'scripts')
  fs.mkdirSync(scriptsDir, { recursive: true })
  const scriptPath = path.join(scriptsDir, 'install.sh')
  fs.writeFileSync(
    scriptPath,
    [
      '#!/bin/bash',
      'set -e',
      'if [ "$1" = "--manifest" ]; then',
      '  echo \'{"protocol_version":1,"stages":[{"name":"complete","title":"Finish install","category":"runtime","needs_user_input":false}]}\'',
      '  exit 0',
      'fi',
      'if [ "$1" = "--stage" ] && [ "$2" = "complete" ]; then',
      '  echo \'{"ok":true,"stage":"complete","skipped":false}\'',
      '  exit 0',
      'fi',
      'echo "unexpected args: $*" >&2',
      'exit 1',
      ''
    ].join('\n'),
    { mode: 0o755 }
  )
  return scriptPath
}

describeManifestFlow('runBootstrap threads updateInfo through onto the manifest event (update)', async () => {
  const home = mkTmpHome()
  try {
    writeFakeInstallSh(home)
    const events = []
    const result = await runBootstrap({
      installStamp: null,
      activeRoot: path.join(home, 'hermes-agent'),
      sourceRepoRoot: home, // resolveLocalInstallScript looks for scripts/install.sh under here
      hermesHome: home,
      logRoot: home,
      updateInfo: { isUpdate: true, toVersion: '2026.7.2', fromVersion: '2026.7.1' },
      onEvent: ev => events.push(ev),
      writeMarker: payload => payload
    })

    assert.equal(result.ok, true, `bootstrap should succeed against the fake stub: ${JSON.stringify(result)}`)
    const manifestEvents = events.filter(ev => ev.type === 'manifest')
    assert.equal(manifestEvents.length, 1)
    assert.deepEqual(manifestEvents[0].updateInfo, { isUpdate: true, toVersion: '2026.7.2', fromVersion: '2026.7.1' })
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

describeManifestFlow('runBootstrap defaults updateInfo to a first-install shape when the caller omits it', async () => {
  const home = mkTmpHome()
  try {
    writeFakeInstallSh(home)
    const events = []
    const result = await runBootstrap({
      installStamp: null,
      activeRoot: path.join(home, 'hermes-agent'),
      sourceRepoRoot: home,
      hermesHome: home,
      logRoot: home,
      // updateInfo intentionally omitted -- covers a caller (test, dev
      // shortcut, or an older code path) that hasn't been updated to pass it.
      onEvent: ev => events.push(ev),
      writeMarker: payload => payload
    })

    assert.equal(result.ok, true, `bootstrap should succeed against the fake stub: ${JSON.stringify(result)}`)
    const manifestEvents = events.filter(ev => ev.type === 'manifest')
    assert.equal(manifestEvents.length, 1)
    assert.deepEqual(manifestEvents[0].updateInfo, { isUpdate: false, toVersion: null, fromVersion: null })
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

// ---------------------------------------------------------------------------
// hc-473: per-stage + bootstrap-level telemetry, driven end to end against the
// same fake install.sh stub as the updateInfo tests above (posix-only, same
// win32 skip rationale).
// ---------------------------------------------------------------------------

function writeFakeInstallShFailingStage(sourceRepoRoot) {
  // Same one-stage manifest as writeFakeInstallSh, but the stage itself
  // reports ok:false with a reason that should classify as checksum_mismatch
  // (apexnodes-telemetry.cjs classifyErrorCategory) -- exercises the failure
  // beacon path end to end, both the per-stage AND the bootstrap rollup.
  const scriptsDir = path.join(sourceRepoRoot, 'scripts')
  fs.mkdirSync(scriptsDir, { recursive: true })
  const scriptPath = path.join(scriptsDir, 'install.sh')
  fs.writeFileSync(
    scriptPath,
    [
      '#!/bin/bash',
      'set -e',
      'if [ "$1" = "--manifest" ]; then',
      '  echo \'{"protocol_version":1,"stages":[{"name":"complete","title":"Finish install","category":"runtime","needs_user_input":false}]}\'',
      '  exit 0',
      'fi',
      'if [ "$1" = "--stage" ] && [ "$2" = "complete" ]; then',
      '  echo \'{"ok":false,"stage":"complete","reason":"sha256 mismatch on 3 files"}\'',
      '  exit 0',
      'fi',
      'echo "unexpected args: $*" >&2',
      'exit 1',
      ''
    ].join('\n'),
    { mode: 0o755 }
  )
  return scriptPath
}

describeManifestFlow('runBootstrap success: fires bootstrap+stage start/success beacons in order', async () => {
  const home = mkTmpHome()
  try {
    writeFakeInstallSh(home)
    const telemetryEvents = []
    const result = await runBootstrap({
      installStamp: { commit: 'a'.repeat(40), version: '2026.7.1' },
      activeRoot: path.join(home, 'hermes-agent'),
      sourceRepoRoot: home,
      hermesHome: home,
      logRoot: home,
      appVersion: '0.16.7',
      onEvent: () => {},
      writeMarker: payload => payload,
      sendTelemetry: ev => telemetryEvents.push(ev)
    })

    assert.equal(result.ok, true, `bootstrap should succeed against the fake stub: ${JSON.stringify(result)}`)

    const expectedBase = {
      platform: normalizeDesktopPlatform(process.platform),
      arch: process.arch,
      app_version: '0.16.7',
      runtime_key: 'a'.repeat(40)
    }
    assert.deepEqual(telemetryEvents, [
      { ...expectedBase, stage: 'bootstrap', status: 'start' },
      { ...expectedBase, stage: 'complete', status: 'start' },
      { ...expectedBase, stage: 'complete', status: 'success' },
      { ...expectedBase, stage: 'bootstrap', status: 'success' }
    ])
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

describeManifestFlow(
  'runBootstrap failure: stage failure fires BOTH a per-stage AND a bootstrap-rollup failure beacon',
  async () => {
    const home = mkTmpHome()
    try {
      writeFakeInstallShFailingStage(home)
      const telemetryEvents = []
      const result = await runBootstrap({
        installStamp: null,
        activeRoot: path.join(home, 'hermes-agent'),
        sourceRepoRoot: home,
        hermesHome: home,
        logRoot: home,
        onEvent: () => {},
        writeMarker: payload => payload,
        sendTelemetry: ev => telemetryEvents.push(ev)
      })

      assert.equal(result.ok, false)
      assert.equal(result.failedStage, 'complete')

      const stageStart = telemetryEvents.find(ev => ev.stage === 'complete' && ev.status === 'start')
      const stageFailure = telemetryEvents.find(ev => ev.stage === 'complete' && ev.status === 'failure')
      const bootstrapStart = telemetryEvents.find(ev => ev.stage === 'bootstrap' && ev.status === 'start')
      const bootstrapFailure = telemetryEvents.find(ev => ev.stage === 'bootstrap' && ev.status === 'failure')

      assert.ok(stageStart, 'the stage still fires its start beacon')
      assert.ok(stageFailure, 'the stage fires a failure beacon')
      assert.equal(stageFailure.error_code, 'complete:checksum_mismatch')
      assert.ok(bootstrapStart, 'the whole run still fires a start beacon')
      assert.ok(bootstrapFailure, 'the whole run rolls up to a bootstrap-level failure beacon too')
      assert.equal(bootstrapFailure.error_code, 'bootstrap:stage_failed:complete')
      // No bootstrap-level success beacon on a failed run.
      assert.ok(!telemetryEvents.some(ev => ev.stage === 'bootstrap' && ev.status === 'success'))
    } finally {
      fs.rmSync(home, { recursive: true, force: true })
    }
  }
)

describeManifestFlow(
  'runBootstrap: omitting sendTelemetry still succeeds (real default, network suppressed by APEXNODES_TELEMETRY=off)',
  async () => {
    const home = mkTmpHome()
    try {
      writeFakeInstallSh(home)
      const result = await runBootstrap({
        installStamp: null,
        activeRoot: path.join(home, 'hermes-agent'),
        sourceRepoRoot: home,
        hermesHome: home,
        logRoot: home,
        onEvent: () => {},
        writeMarker: payload => payload
        // sendTelemetry intentionally omitted -- must fall back to the real
        // emitter without throwing or otherwise affecting the result. This
        // file sets APEXNODES_TELEMETRY=off at module scope, so the real
        // emitter is a fast no-op here rather than a live network call.
      })
      assert.equal(result.ok, true, `bootstrap should succeed even with no sendTelemetry override: ${JSON.stringify(result)}`)
    } finally {
      fs.rmSync(home, { recursive: true, force: true })
    }
  }
)
