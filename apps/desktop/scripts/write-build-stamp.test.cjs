const assert = require('node:assert/strict')
const test = require('node:test')

const {
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
