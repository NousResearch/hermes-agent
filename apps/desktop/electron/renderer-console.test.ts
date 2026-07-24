import assert from 'node:assert/strict'

import { test } from 'vitest'

import { formatRendererConsoleError } from './renderer-console'

test('formats renderer console errors from Electron message details', () => {
  assert.equal(
    formatRendererConsoleError({
      level: 3,
      lineNumber: 17,
      message: 'renderer failed',
      sourceUrl: 'file:///renderer.js'
    }),
    '[renderer console] renderer failed (file:///renderer.js:17)'
  )
})

test('ignores missing renderer console details', () => {
  assert.equal(formatRendererConsoleError(null), null)
  assert.equal(formatRendererConsoleError(undefined), null)
})

test('ignores non-error renderer console messages', () => {
  for (const level of [0, 1, 2]) {
    assert.equal(
      formatRendererConsoleError({
        level,
        lineNumber: 1,
        message: 'not an error',
        sourceUrl: 'file:///renderer.js'
      }),
      null
    )
  }
})
