'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')

const {
  LOCAL_FILES_DISABLED_CODE,
  LOCAL_FILES_DISABLED_ENV,
  assertLocalFilesAllowed,
  isLocalFilesDisabled,
  localFilesPolicy
} = require('./local-files-policy.cjs')

test('isLocalFilesDisabled is false when the env var is unset or empty', () => {
  assert.equal(isLocalFilesDisabled({}), false)
  assert.equal(isLocalFilesDisabled({ [LOCAL_FILES_DISABLED_ENV]: '' }), false)
  assert.equal(isLocalFilesDisabled({ [LOCAL_FILES_DISABLED_ENV]: '0' }), false)
  assert.equal(isLocalFilesDisabled({ [LOCAL_FILES_DISABLED_ENV]: 'false' }), false)
})

test('isLocalFilesDisabled accepts the common truthy spellings', () => {
  for (const value of ['1', 'true', 'TRUE', 'yes', 'On', '  1  ']) {
    assert.equal(isLocalFilesDisabled({ [LOCAL_FILES_DISABLED_ENV]: value }), true, value)
  }
})

test('localFilesPolicy reports a reason only when disabled', () => {
  assert.deepEqual(localFilesPolicy({}), { disabled: false, reason: null })

  const blocked = localFilesPolicy({ [LOCAL_FILES_DISABLED_ENV]: '1' })
  assert.equal(blocked.disabled, true)
  assert.match(blocked.reason, /Remote-only mode/i)
})

test('assertLocalFilesAllowed throws a coded error when disabled', () => {
  assert.doesNotThrow(() => assertLocalFilesAllowed('Directory read', { env: {} }))

  assert.throws(
    () => assertLocalFilesAllowed('Directory read', { env: { [LOCAL_FILES_DISABLED_ENV]: '1' } }),
    error => {
      assert.equal(error.code, LOCAL_FILES_DISABLED_CODE)
      assert.match(error.message, /Directory read blocked/)
      return true
    }
  )
})
