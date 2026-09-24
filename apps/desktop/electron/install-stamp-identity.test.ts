import assert from 'node:assert/strict'

import { describe, it } from 'vitest'

import { isValidInstallCommit, resolveRunningClientSha } from './install-stamp-identity'

const checkoutSha = 'b'.repeat(40)
const bundleSha = 'a'.repeat(40)

describe('running packaged client identity', () => {
  it('uses the immutable installed commit even when the staging checkout moves', () => {
    assert.equal(resolveRunningClientSha({ checkoutSha, installStamp: { commit: bundleSha }, isPackaged: true }), bundleSha)
    assert.equal(
      resolveRunningClientSha({ checkoutSha, installStamp: { commit: bundleSha.toUpperCase() }, isPackaged: true }),
      bundleSha
    )
  })

  it('uses the checkout for development, missing stamps and invalid packaged commits', () => {
    assert.equal(resolveRunningClientSha({ checkoutSha, installStamp: { commit: bundleSha }, isPackaged: false }), checkoutSha)

    for (const commit of [null, undefined, '', '0'.repeat(40), 'f'.repeat(39), 'g'.repeat(40), 'f'.repeat(41)]) {
      assert.equal(resolveRunningClientSha({ checkoutSha, installStamp: { commit }, isPackaged: true }), checkoutSha)
      assert.equal(isValidInstallCommit(commit), false)
    }

    assert.equal(resolveRunningClientSha({ checkoutSha, installStamp: null, isPackaged: true }), checkoutSha)
  })
})
