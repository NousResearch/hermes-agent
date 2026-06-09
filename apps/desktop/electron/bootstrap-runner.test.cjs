const assert = require('node:assert/strict')
const test = require('node:test')

const {
  cachedScriptPath,
  installScriptRef,
  installScriptRepositories,
  installerRepoEnv,
  normalizeGitHubRepository,
  runBootstrap,
  shouldPinCommit
} = require('./bootstrap-runner.cjs')

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

test('install script resolution honors stamped fork repository metadata', () => {
  const stamp = {
    commit: 'abcdef1234567890',
    repository: 'OmarB97/hermes-agent',
    repoUrlHttps: 'https://github.com/OmarB97/hermes-agent.git',
    repoUrlSsh: 'git@github.com:OmarB97/hermes-agent.git'
  }

  assert.deepEqual(installScriptRepositories(stamp), ['OmarB97/hermes-agent', 'NousResearch/hermes-agent'])
  assert.deepEqual(installerRepoEnv(stamp), {
    HERMES_INSTALL_REPO_URL_HTTPS: 'https://github.com/OmarB97/hermes-agent.git',
    HERMES_INSTALL_REPO_URL_SSH: 'git@github.com:OmarB97/hermes-agent.git'
  })
})

test('install script resolution falls back to bootstrapRef for unpinned local builds', () => {
  const stamp = {
    commit: 'abcdef1234567890',
    bootstrapRef: 'main',
    commitPinned: false,
    repository: 'OmarB97/hermes-agent'
  }

  assert.equal(installScriptRef(stamp), 'main')
  assert.equal(shouldPinCommit(stamp), false)
  assert.match(cachedScriptPath('/tmp/hermes-home', 'fix/some-branch'), /install-fix_some-branch\.(sh|ps1)$/)
})

test('normalizes GitHub repository values from common remote forms', () => {
  assert.equal(normalizeGitHubRepository('git@github.com:OmarB97/hermes-agent.git'), 'OmarB97/hermes-agent')
  assert.equal(normalizeGitHubRepository('https://github.com/NousResearch/hermes-agent.git'), 'NousResearch/hermes-agent')
  assert.equal(normalizeGitHubRepository('NousResearch/hermes-agent'), 'NousResearch/hermes-agent')
  assert.equal(normalizeGitHubRepository('not github'), null)
})
