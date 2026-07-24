import assert from 'node:assert/strict'

import { test } from 'vitest'

import { formatRendererConsoleError } from './renderer-console'

test('formats renderer console errors from Electron webContents details', () => {
  assert.equal(
    formatRendererConsoleError({
      level: 'error',
      lineNumber: 17,
      message: 'renderer failed',
      sourceId: 'file:///renderer.js'
    }),
    '[renderer console] renderer failed (file:///renderer.js:17)'
  )
})

test('ignores missing renderer console details', () => {
  assert.equal(formatRendererConsoleError(null), null)
  assert.equal(formatRendererConsoleError(undefined), null)
})

test('ignores non-error renderer console messages', () => {
  for (const level of ['info', 'warning', 'debug'] as const) {
    assert.equal(
      formatRendererConsoleError({
        level,
        lineNumber: 1,
        message: 'not an error',
        sourceId: 'file:///renderer.js'
      }),
      null
    )
  }
})
