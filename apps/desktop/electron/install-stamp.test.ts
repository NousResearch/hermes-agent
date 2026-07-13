import assert from 'node:assert/strict'
import test from 'node:test'

import { isLocalProtectedInstallStamp, localBuildUpdateBlock, normalizeInstallStampPayload } from './install-stamp'

test('normalizes full install stamp metadata used by bootstrap and updates', () => {
  const stamp = normalizeInstallStampPayload({
    schemaVersion: 1,
    commit: 'abcdef1234567890',
    branch: 'local/dev',
    repository: 'ForkOwner/hermes-agent',
    bootstrapRef: 'main',
    commitPinned: false,
    repoUrlHttps: 'https://github.com/ForkOwner/hermes-agent.git',
    repoUrlSsh: 'git@github.com:ForkOwner/hermes-agent.git',
    builtAt: '2026-06-05T00:00:00.000Z',
    dirty: false,
    source: 'local'
  })

  assert.equal(stamp.commit, 'abcdef1234567890')
  assert.equal(stamp.repository, 'ForkOwner/hermes-agent')
  assert.equal(stamp.bootstrapRef, 'main')
  assert.equal(stamp.commitPinned, false)
  assert.equal(stamp.repoUrlHttps, 'https://github.com/ForkOwner/hermes-agent.git')
  assert.equal(stamp.source, 'local')
})

test('rejects missing or mismatched stamp schema', () => {
  assert.equal(normalizeInstallStampPayload(null), null)
  assert.equal(normalizeInstallStampPayload({ schemaVersion: 0, commit: 'abcdef1' }), null)
  assert.equal(normalizeInstallStampPayload({ schemaVersion: 1 }), null)
})

test('blocks automatic updates for local unpinned desktop bundles', () => {
  const stamp = normalizeInstallStampPayload({
    schemaVersion: 1,
    commit: 'abcdef1234567890',
    branch: 'local/dev',
    bootstrapRef: 'main',
    commitPinned: false,
    source: 'local'
  })

  assert.equal(isLocalProtectedInstallStamp(stamp), true)
  const block = localBuildUpdateBlock(stamp, {})
  assert.equal(block.manual, true)
  assert.equal(block.reason, 'local-build-install')
  assert.match(block.message, /Automatic self-update is paused/)
})

test('blocks automatic updates for local stamps without pin metadata', () => {
  const stamp = normalizeInstallStampPayload({
    schemaVersion: 1,
    commit: 'abcdef1234567890',
    branch: 'local/dev',
    source: 'local'
  })

  assert.equal(isLocalProtectedInstallStamp(stamp), true)
  const block = localBuildUpdateBlock(stamp, {})
  assert.equal(block.manual, true)
  assert.equal(block.reason, 'local-build-install')
})

test('blocks automatic updates for local fork-pinned desktop bundles', () => {
  const stamp = normalizeInstallStampPayload({
    schemaVersion: 1,
    commit: 'abcdef1234567890',
    branch: 'fix/desktop-bootstrap',
    repository: 'ForkOwner/hermes-agent',
    commitPinned: true,
    source: 'local'
  })

  assert.equal(isLocalProtectedInstallStamp(stamp), true)
  const block = localBuildUpdateBlock(stamp, {})
  assert.equal(block.reason, 'local-build-install')
  assert.match(block.message, /ForkOwner\/hermes-agent@abcdef123456/)
})

test('allows automatic updates for local official pinned desktop bundles', () => {
  const stamp = normalizeInstallStampPayload({
    schemaVersion: 1,
    commit: 'abcdef1234567890',
    branch: 'main',
    repository: 'NousResearch/hermes-agent',
    commitPinned: true,
    source: 'local'
  })

  assert.equal(isLocalProtectedInstallStamp(stamp), false)
  assert.equal(localBuildUpdateBlock(stamp, {}), null)
})

test('allows explicit override for local build update guard', () => {
  const stamp = normalizeInstallStampPayload({
    schemaVersion: 1,
    commit: 'abcdef1234567890',
    commitPinned: false,
    source: 'local'
  })

  assert.equal(localBuildUpdateBlock(stamp, { HERMES_DESKTOP_ALLOW_LOCAL_UNPINNED_UPDATE: '1' }), null)
})
