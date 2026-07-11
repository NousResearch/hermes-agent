import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import test from 'node:test'

import {
  buildPinArgs,
  buildPosixPinArgs,
  cachedScriptPath,
  hasExistingGitCheckout,
  installedAgentInstallScript,
  installScriptRef,
  installScriptRepositories,
  installerRepoEnv,
  normalizeGitHubRepository,
  resolveInstallScript,
  runBootstrap,
  shouldPinCommit
} from './bootstrap-runner'

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
    [
      '--dir',
      '/tmp/hermes-agent',
      '--hermes-home',
      '/tmp/hermes',
      '--branch',
      'main',
      '--commit',
      installStamp.commit
    ]
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

test('install script resolution honors stamped fork repository metadata', () => {
  const stamp = {
    commit: 'abcdef1234567890',
    repository: 'ForkOwner/hermes-agent',
    repoUrlHttps: 'https://github.com/ForkOwner/hermes-agent.git',
    repoUrlSsh: 'git@github.com:ForkOwner/hermes-agent.git'
  }

  assert.deepEqual(installScriptRepositories(stamp), ['ForkOwner/hermes-agent', 'NousResearch/hermes-agent'])
  assert.deepEqual(installerRepoEnv(stamp), {
    HERMES_INSTALL_REPO_ARCHIVE_BASE: 'https://github.com/ForkOwner/hermes-agent',
    HERMES_INSTALL_REPO_URL_HTTPS: 'https://github.com/ForkOwner/hermes-agent.git',
    HERMES_INSTALL_REPO_URL_SSH: 'git@github.com:ForkOwner/hermes-agent.git'
  })
})

test('install script resolution falls back to bootstrapRef for unpinned local builds', () => {
  const stamp = {
    commit: 'abcdef1234567890',
    bootstrapRef: 'main',
    commitPinned: false,
    repository: 'ForkOwner/hermes-agent'
  }

  assert.equal(installScriptRef(stamp), 'main')
  assert.equal(shouldPinCommit(stamp), false)
  assert.match(cachedScriptPath('/tmp/hermes-home', 'fix/some-branch'), /install-fix_some-branch\.(sh|ps1)$/)
})

test('normalizes GitHub repository values from common remote forms', () => {
  assert.equal(normalizeGitHubRepository('git@github.com:ForkOwner/hermes-agent.git'), 'ForkOwner/hermes-agent')
  assert.equal(normalizeGitHubRepository('https://github.com/NousResearch/hermes-agent.git'), 'NousResearch/hermes-agent')
  assert.equal(normalizeGitHubRepository('NousResearch/hermes-agent'), 'NousResearch/hermes-agent')
  assert.equal(normalizeGitHubRepository('not github'), null)
})
