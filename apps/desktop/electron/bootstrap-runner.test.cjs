const assert = require('node:assert/strict')
const test = require('node:test')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const {
  runBootstrap,
  resolveInstallScript,
  installedAgentInstallScript,
  bundledInstallScript,
  cnInstallEnv,
  cachedScriptPath
} = require('./bootstrap-runner.cjs')

const SCRIPT_NAME = process.platform === 'win32' ? 'install.ps1' : 'install.sh'

function mkTmpHome() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-bootstrap-test-'))
}

test('runBootstrap bails immediately when the signal is already aborted', async () => {
  const controller = new AbortController()
  controller.abort()

  const events = []
  const result = await runBootstrap({
    installStamp: null,
    activeRoot: '/tmp/hermes-runner-test',
    sourceRepoRoot: null,
    hermesHome: '/tmp/hermes-runner-test',
    logRoot: '/tmp/hermes-runner-test',
    onEvent: ev => events.push(ev),
    abortSignal: controller.signal
  })

  // Cancelled before any install script is spawned.
  assert.deepEqual(result, { ok: false, cancelled: true })
  assert.ok(
    events.some(ev => ev.type === 'failed' && /cancelled/i.test(ev.error)),
    'should emit a cancelled failure event'
  )
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
