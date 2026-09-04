import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isUnsafeRevealPath } from './reveal-path-guard'

test('rejects slash and backslash network/device paths', () => {
  for (const value of [
    '\\\\server\\share\\report.pdf',
    '//server/share/report.pdf',
    '\\\\?\\C:\\secret.txt',
    '\\\\.\\pipe\\name'
  ]) {
    assert.equal(isUnsafeRevealPath(value), true, value)
  }
})

test('allows local paths', () => {
  for (const value of ['C:\\Users\\alex\\report.pdf', '/home/alex/report.pdf', '/mnt/c/Users/alex/report.pdf']) {
    assert.equal(isUnsafeRevealPath(value), false, value)
  }
})
