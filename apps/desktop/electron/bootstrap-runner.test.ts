import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { Readable } from 'node:stream'

import { test } from 'vitest'

import {
  buildPinArgs,
  buildPosixPinArgs,
  cachedScriptPath,
  downloadInstallScript,
  hasExistingGitCheckout,
  installedAgentInstallScript,
  installRefForStamp,
  isPinnedCommit,
  resolveInstallScript,
  resolveMarkerPinnedCommit,
  runBootstrap
} from './bootstrap-runner'

const SCRIPT_NAME = process.platform === 'win32' ? 'install.ps1' : 'install.sh'
const ZERO_COMMIT = '0000000000000000000000000000000000000000'

function mkTmpHome() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-bootstrap-test-'))
}

test('fresh packaged bootstrap downloads the pinned installer from the product repository', async () => {
  const home = mkTmpHome()
  const commit = 'a'.repeat(40)
  const metadata = JSON.parse(fs.readFileSync(new URL('../package.json', import.meta.url), 'utf8'))
  const repository = new URL(metadata.repository.url.replace(/^git\+/, '').replace(/\.git$/, '')).pathname
  let requested: URL | undefined
  const payload = 'installer fixture'
  const get = (url, _options, callback) => {
    requested = new URL(url)
    const request = new EventEmitter()
    queueMicrotask(() => {
      const response = Object.assign(Readable.from([payload]), { statusCode: 200, headers: {} })
      callback(response)
    })

    return request
  }

  try {
    const result = await resolveInstallScript({
      installStamp: { commit, branch: 'main' },
      sourceRepoRoot: null,
      hermesHome: home,
      emit: () => {},
      _download: (ref, destination, signal) => downloadInstallScript(ref, destination, signal, get)
    })
    assert.equal(result.source, 'download')
    assert.equal(requested?.host, 'raw.githubusercontent.com')
    assert.equal(requested?.pathname, `${repository}/${commit}/scripts/${SCRIPT_NAME}`)
    assert.equal(fs.readFileSync(result.path, 'utf8'), payload)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

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

test('download deadline and Cancel both abort a stalled HTTPS request', async () => {
  const home = mkTmpHome()

  try {
    for (const cancel of [false, true]) {
      const controller = new AbortController()
      const request = new EventEmitter()
      let requestSignal: AbortSignal | undefined

      const get = (_url, options) => {
        requestSignal = options.signal

        return request
      }

      const pending = downloadInstallScript('main', path.join(home, SCRIPT_NAME), controller.signal, get, 25)

      if (cancel) {controller.abort()}

      await assert.rejects(pending, cancel ? /cancelled/ : /timed out/)
      assert.equal(requestSignal?.aborted, true)
      assert.equal(fs.existsSync(path.join(home, SCRIPT_NAME)), false)
    }

    const installedDir = path.join(home, 'hermes-agent', 'scripts')
    fs.mkdirSync(installedDir, { recursive: true })
    fs.writeFileSync(path.join(installedDir, SCRIPT_NAME), 'installed fallback')
    const controller = new AbortController()
    await assert.rejects(
      resolveInstallScript({
        installStamp: { commit: 'a'.repeat(40) },
        sourceRepoRoot: null,
        hermesHome: home,
        emit: () => {},
        abortSignal: controller.signal,
        _download: async (_ref, _dest, signal) => {
          assert.equal(signal, controller.signal)
          controller.abort()
          throw new Error('download aborted')
        }
      }),
      /download aborted/
    )
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test.skipIf(process.platform === 'win32')('Cancel stops a running manifest and never writes success', async () => {
  const home = mkTmpHome()

  try {
    const scripts = path.join(home, 'scripts')
    fs.mkdirSync(scripts)
    fs.writeFileSync(path.join(scripts, 'install.sh'), 'echo manifest-started\nexec sleep 30\n')
    const controller = new AbortController()
    const events = []
    let wroteMarker = false

    const result = await runBootstrap({
      installStamp: null,
      activeRoot: path.join(home, 'checkout'),
      sourceRepoRoot: home,
      hermesHome: home,
      logRoot: home,
      abortSignal: controller.signal,
      onEvent: ev => {
        events.push(ev)

        if (ev.stage === '__manifest__' && ev.line === 'manifest-started') {controller.abort()}
      },
      writeMarker: () => {
        wroteMarker = true
      }
    })

    assert.deepEqual(result, { ok: false, cancelled: true })
    assert.equal(wroteMarker, false)
    assert.equal(events.some(ev => ev.type === 'complete'), false)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
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

test('existing checkout detection requires git metadata', () => {
  const home = mkTmpHome()

  try {
    const activeRoot = path.join(home, 'hermes-agent')
    assert.equal(hasExistingGitCheckout(activeRoot), false)

    fs.mkdirSync(path.join(activeRoot, '.git'), { recursive: true })
    assert.equal(hasExistingGitCheckout(activeRoot), true)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

test('fresh bootstrap args include the packaged commit pin', () => {
  const installStamp = { commit: 'a'.repeat(40), branch: 'main' }

  assert.deepEqual(buildPinArgs(installStamp), ['-Commit', installStamp.commit, '-Branch', 'main'])
  assert.deepEqual(
    buildPosixPinArgs({
      installStamp,
      activeRoot: '/tmp/hermes-agent',
      hermesHome: '/tmp/hermes'
    }),
    ['--dir', '/tmp/hermes-agent', '--hermes-home', '/tmp/hermes', '--branch', 'main', '--commit', installStamp.commit]
  )
})

test('existing-checkout bootstrap args keep branch but skip the packaged commit pin', () => {
  const installStamp = { commit: 'a'.repeat(40), branch: 'main' }

  assert.deepEqual(buildPinArgs(installStamp, { pinCommit: false }), ['-Branch', 'main'])
  assert.deepEqual(
    buildPosixPinArgs({
      installStamp,
      activeRoot: '/tmp/hermes-agent',
      hermesHome: '/tmp/hermes',
      pinCommit: false
    }),
    ['--dir', '/tmp/hermes-agent', '--hermes-home', '/tmp/hermes', '--branch', 'main']
  )
})

test('fallback install stamps use an unpinned branch ref', () => {
  const stamp = { commit: ZERO_COMMIT, branch: 'main' }

  assert.equal(isPinnedCommit(ZERO_COMMIT), false)
  assert.deepEqual(installRefForStamp(stamp), {
    ref: 'main',
    cacheKey: 'fallback-main',
    pinned: false
  })
  // Must NOT pass -Commit / --commit for the all-zero placeholder.
  assert.deepEqual(buildPinArgs(stamp), ['-Branch', 'main'])
  assert.deepEqual(
    buildPosixPinArgs({
      installStamp: stamp,
      activeRoot: '/tmp/hermes',
      hermesHome: '/tmp/home'
    }),
    ['--dir', '/tmp/hermes', '--hermes-home', '/tmp/home', '--branch', 'main']
  )
})

test('resolveMarkerPinnedCommit prefers real HEAD over fallback stamp zeros', () => {
  const realHead = 'c'.repeat(40)
  assert.equal(
    resolveMarkerPinnedCommit({ commit: ZERO_COMMIT, branch: 'main' }, '/tmp/checkout', {
      resolveHead: () => realHead
    }),
    realHead
  )
  assert.equal(
    resolveMarkerPinnedCommit({ commit: 'd'.repeat(40), branch: 'main' }, '/tmp/checkout', {
      resolveHead: () => realHead
    }),
    'd'.repeat(40),
    'packaged real pin wins over checkout HEAD'
  )
  assert.equal(
    resolveMarkerPinnedCommit({ commit: ZERO_COMMIT, branch: 'main' }, '/tmp/missing', {
      resolveHead: () => null
    }),
    null
  )
})

test('resolveInstallScript downloads fallback stamps by branch instead of zero commit', async () => {
  const home = mkTmpHome()

  try {
    const logs = []
    const refs = []

    const result = await resolveInstallScript({
      installStamp: { commit: ZERO_COMMIT, branch: 'main' },
      sourceRepoRoot: null,
      hermesHome: home,
      emit: ev => logs.push(ev),
      _download: async (ref, destPath) => {
        refs.push(ref)
        fs.mkdirSync(path.dirname(destPath), { recursive: true })
        fs.writeFileSync(destPath, '#!/bin/sh\necho fallback branch\n')

        return destPath
      }
    })

    assert.deepEqual(refs, ['main'])
    assert.equal(result.source, 'download')
    assert.equal(result.commit, null)
    assert.equal(result.path, cachedScriptPath(home, 'fallback-main'))
    assert.ok(
      logs.some(ev => /fallback, unpinned/.test(ev.line || '')),
      'emits an unpinned fallback log line'
    )
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


test('packaged first launch uses its bundled installer without any network request', async () => {
  const home = mkTmpHome()
  const resourcesPath = path.join(home, 'resources')
  fs.mkdirSync(path.join(resourcesPath, 'bootstrap'), { recursive: true })
  const bundled = path.join(resourcesPath, 'bootstrap', SCRIPT_NAME)
  fs.writeFileSync(bundled, 'bundled installer fixture')
  try {
    const result = await resolveInstallScript({
      installStamp: { commit: 'b'.repeat(40) }, sourceRepoRoot: null,
      hermesHome: home, resourcesPath, emit: () => {},
      _download: async () => { throw new Error('HTTP 429: network must not be used') }
    })
    assert.equal(result.path, bundled)
    assert.equal(result.source, 'bundled')
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})
