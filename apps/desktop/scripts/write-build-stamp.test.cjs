const assert = require('node:assert/strict')
const test = require('node:test')

const {
  hashText,
  normalizeGitHubRepository,
  remoteBranchName,
  remoteNameFromRef,
  repoUrls
} = require('./write-build-stamp.cjs')

test('normalizes GitHub repository values for stamp metadata', () => {
  assert.equal(normalizeGitHubRepository('git@github.com:OmarB97/hermes-agent.git'), 'OmarB97/hermes-agent')
  assert.equal(normalizeGitHubRepository('https://github.com/NousResearch/hermes-agent.git'), 'NousResearch/hermes-agent')
  assert.equal(normalizeGitHubRepository('NousResearch/hermes-agent'), 'NousResearch/hermes-agent')
  assert.equal(normalizeGitHubRepository('not github'), null)
})

test('splits remote tracking refs without losing slashy branch names', () => {
  assert.equal(remoteNameFromRef('origin/fix/desktop-bootstrap'), 'origin')
  assert.equal(remoteBranchName('origin/fix/desktop-bootstrap'), 'fix/desktop-bootstrap')
  assert.equal(remoteNameFromRef('main'), null)
  assert.equal(remoteBranchName('main'), null)
})

test('derives clone URLs from repository slug', () => {
  assert.deepEqual(repoUrls('OmarB97/hermes-agent'), {
    repoUrlHttps: 'https://github.com/OmarB97/hermes-agent.git',
    repoUrlSsh: 'git@github.com:OmarB97/hermes-agent.git'
  })
})

test('hashText is stable for tracked source diff stamps', () => {
  assert.equal(hashText(''), 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')
  assert.equal(hashText('diff --git a/x b/x\n'), hashText('diff --git a/x b/x\n'))
  assert.notEqual(hashText('diff --git a/x b/x\n'), hashText('diff --git a/y b/y\n'))
})
