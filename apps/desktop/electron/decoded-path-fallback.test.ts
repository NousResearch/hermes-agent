import { describe, expect, it } from 'vitest'

import { decodedPathIfMissing } from './decoded-path-fallback'

describe('decodedPathIfMissing', () => {
  it('returns null when the path has no percent-encoding', () => {
    expect(decodedPathIfMissing('/home/user/My Notes/todo.md')).toBeNull()
  })

  it('decodes %20 sequences so spaced paths can be retried (#102782)', () => {
    expect(decodedPathIfMissing('/home/user/My%20Notes/todo%20with%20spaces.md')).toBe(
      '/home/user/My Notes/todo with spaces.md'
    )
  })

  it('returns null when decoding leaves the string unchanged', () => {
    // Valid hex escape that decodeURIComponent leaves alone only when empty —
    // a path that is already identical after decode is a no-op.
    expect(decodedPathIfMissing('/tmp/plain')).toBeNull()
  })

  it('returns null for malformed percent sequences', () => {
    expect(decodedPathIfMissing('/tmp/%E0%A4%A')).toBeNull()
  })
})
