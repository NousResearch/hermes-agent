import { expect, test } from 'vitest'

import { createDesktopLogFormatter, formatDesktopLogChunk } from './desktop-log-redact'

test('formatDesktopLogChunk redacts session tokens before the local log ring', () => {
  const out = formatDesktopLogChunk('Listening on ws://127.0.0.1:9119/api/ws?token=supersecret')

  expect(out).toMatch(/\?token=<redacted>/)
  expect(out).not.toContain('supersecret')
})

test('formatDesktopLogChunk leaves non-secret lines intact', () => {
  expect(formatDesktopLogChunk('Hermes serve ready')).toBe('Hermes serve ready')
})

test('stream formatter redacts prefixes split across chunks', () => {
  const prefixes = [
    '?token=',
    '?ticket=',
    'HERMES_DASHBOARD_SESSION_TOKEN=',
    'X-Hermes-Session-Token: ',
    'Authorization: Bearer '
  ]

  for (const prefix of prefixes) {
    const format = createDesktopLogFormatter()
    expect(format(`stream ${prefix}`)).toBe('')
    const output = format('split-secret\n')
    expect(output).toContain('<redacted>')
    expect(output).not.toContain('split-secret')
  }
})

test('stream formatter redacts a value split across chunks on either stream', () => {
  for (const stream of ['stdout', 'stderr']) {
    const format = createDesktopLogFormatter()
    expect(format(`${stream}: ?token=split`)).toBe('')
    const output = format('-secret\n')
    expect(output).toContain('<redacted>')
    expect(output).not.toContain('split-secret')
  }
})
