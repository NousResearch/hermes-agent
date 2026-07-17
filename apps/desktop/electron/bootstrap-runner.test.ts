import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  buildPinArgs,
  buildPosixPinArgs,
  cachedScriptPath,
  hasExistingGitCheckout,
  installedAgentInstallScript,
  resolveInstallScript,
  runBootstrap
} from './bootstrap-runner'

const SCRIPT_NAME = process.platform === 'win32' ? 'install.ps1' : 'install.sh'

function mkTmpHome() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-bootstrap-test-'))
}

function makeInstallerScript(body: string) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-bootstrap-runner-'))
  const scriptsDir = path.join(root, 'scripts')
  fs.mkdirSync(scriptsDir, { recursive: true })
  fs.writeFileSync(path.join(scriptsDir, 'install.sh'), body, { mode: 0o755 })

  return root
}

function makeRunnerDirs() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-bootstrap-home-'))

  return {
    root,
    activeRoot: path.join(root, 'agent'),
    hermesHome: path.join(root, 'home'),
    logRoot: path.join(root, 'logs')
  }
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

test('killed bootstrap stage reports captured stderr context', async () => {
  const sourceRepoRoot = makeInstallerScript(`#!/usr/bin/env bash
if [[ "$*" == *"--manifest"* ]]; then
  echo '{"protocol_version":1,"stages":[{"name":"repository","title":"Repository"}]}'
  exit 0
fi
if [[ "$*" == *"--stage repository"* ]]; then
  echo 'Repository ready'
  echo 'fatal: detached HEAD state' >&2
  exec sleep 2
fi
`)

  const dirs = makeRunnerDirs()
  const controller = new AbortController()
  const events = []

  try {
    const result = await runBootstrap({
      installStamp: null,
      sourceRepoRoot,
      ...dirs,
      onEvent: ev => {
        events.push(ev)

        if (ev.type === 'log' && ev.stage === 'repository' && ev.line.includes('fatal: detached HEAD')) {
          controller.abort()
        }
      },
      abortSignal: controller.signal
    })

    assert.equal(result.ok, false)
    assert.equal(result.failedStage, 'repository')
    assert.match(result.error, /process killed — last output:/)
    assert.match(result.error, /fatal: detached HEAD state/)
    assert.doesNotMatch(result.error, /^cancelled by user$/)
  } finally {
    fs.rmSync(sourceRepoRoot, { recursive: true, force: true })
    fs.rmSync(dirs.root, { recursive: true, force: true })
  }
})

test('killed bootstrap stage without captured output still reports user cancellation', async () => {
  const sourceRepoRoot = makeInstallerScript(`#!/usr/bin/env bash
if [[ "$*" == *"--manifest"* ]]; then
  echo '{"protocol_version":1,"stages":[{"name":"repository","title":"Repository"}]}'
  exit 0
fi
if [[ "$*" == *"--stage repository"* ]]; then
  exec sleep 2
fi
`)

  const dirs = makeRunnerDirs()
  const controller = new AbortController()

  try {
    const result = await runBootstrap({
      installStamp: null,
      sourceRepoRoot,
      ...dirs,
      onEvent: ev => {
        if (ev.type === 'stage' && ev.name === 'repository' && ev.state === 'running') {
          controller.abort()
        }
      },
      abortSignal: controller.signal
    })

    assert.equal(result.ok, false)
    assert.equal(result.failedStage, 'repository')
    assert.equal(result.error, 'cancelled by user')
  } finally {
    fs.rmSync(sourceRepoRoot, { recursive: true, force: true })
    fs.rmSync(dirs.root, { recursive: true, force: true })
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
