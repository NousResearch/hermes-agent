import assert from 'node:assert/strict'
import test from 'node:test'

import { formatDesktopLogChunk } from './desktop-log-redact'

test('formatDesktopLogChunk redacts session tokens before the local log ring', () => {
  const out = formatDesktopLogChunk('Listening on ws://127.0.0.1:9119/api/ws?token=supersecret')

  assert.match(out, /\?token=<redacted>/)
  assert.ok(!out.includes('supersecret'))
})

test('formatDesktopLogChunk leaves non-secret lines intact', () => {
  assert.equal(formatDesktopLogChunk('Hermes serve ready'), 'Hermes serve ready')
})
