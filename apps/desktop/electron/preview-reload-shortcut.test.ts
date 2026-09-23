import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import { shouldClaimReloadShortcut } from './preview-reload-shortcut'

describe('shouldClaimReloadShortcut', () => {
  test('claims Ctrl/Cmd+R only for a live webview guest', () => {
    assert.equal(
      shouldClaimReloadShortcut({
        isDestroyed: () => false,
        getType: () => 'webview'
      }),
      true
    )
  })

  test('lets the key through when focus is the host window', () => {
    assert.equal(
      shouldClaimReloadShortcut({
        isDestroyed: () => false,
        getType: () => 'window'
      }),
      false
    )
  })

  test('lets the key through when nothing is focused', () => {
    assert.equal(shouldClaimReloadShortcut(null), false)
    assert.equal(shouldClaimReloadShortcut(undefined), false)
  })

  test('lets the key through when the focused contents are destroyed', () => {
    assert.equal(
      shouldClaimReloadShortcut({
        isDestroyed: () => true,
        getType: () => 'webview'
      }),
      false
    )
  })
})
