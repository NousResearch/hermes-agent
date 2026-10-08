import assert from 'node:assert/strict'

import { test, vi } from 'vitest'

import { resolvePickerStartPath } from './path-picker'

test('resolvePickerStartPath uses Downloads only when the caller requests the fallback', () => {
  const downloadsPath = vi.fn(() => '/Users/test/Downloads')

  assert.equal(resolvePickerStartPath({ fallbackToDownloads: true }, downloadsPath), '/Users/test/Downloads')
  assert.equal(resolvePickerStartPath({}, downloadsPath), undefined)
  assert.equal(downloadsPath.mock.calls.length, 1)
})

test('resolvePickerStartPath keeps an explicit default path ahead of Downloads', () => {
  const downloadsPath = vi.fn(() => '/Users/test/Downloads')

  assert.equal(
    resolvePickerStartPath({ defaultPath: '/Users/test/project', fallbackToDownloads: true }, downloadsPath),
    '/Users/test/project'
  )
  assert.equal(downloadsPath.mock.calls.length, 0)
})

test('resolvePickerStartPath leaves native dialog behavior intact when Downloads cannot be resolved', () => {
  assert.equal(
    resolvePickerStartPath({ fallbackToDownloads: true }, () => ''),
    undefined
  )
  assert.equal(
    resolvePickerStartPath({ fallbackToDownloads: true }, () => {
      throw new Error('Downloads unavailable')
    }),
    undefined
  )
})
