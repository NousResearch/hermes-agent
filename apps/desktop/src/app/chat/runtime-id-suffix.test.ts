import { describe, expect, it } from 'vitest'

import { stripRuntimeIdSuffix, stripRuntimeIdSuffixNullable } from './runtime-id-suffix'

// The runtime funnel suffixes duplicate message ids (`id#1`, `id#2`) so
// assistant-ui's MessageRepository never sees a colliding render key. Those
// suffixed ids must be stripped back to the real store id before any
// user-action callback (edit / reload / branch / restore) reaches the
// $messages store or the gateway — otherwise the backend lookup misses the
// real message. Real ids are `${timestamp}-${index}-${role}` and never contain
// `#`, so a trailing `#<n>` is unambiguously the dedup marker.
describe('stripRuntimeIdSuffix', () => {
  it('strips a single dedup suffix back to the real id', () => {
    expect(stripRuntimeIdSuffix('1784158733-1-assistant#1')).toBe('1784158733-1-assistant')
    expect(stripRuntimeIdSuffix('1784158733-1-assistant#2')).toBe('1784158733-1-assistant')
  })

  it('leaves an un-suffixed real id untouched (idempotent)', () => {
    expect(stripRuntimeIdSuffix('1784158733-1-assistant')).toBe('1784158733-1-assistant')
    expect(stripRuntimeIdSuffix('u-1')).toBe('u-1')
  })

  it('only strips a trailing #<digits>, not a mid-id hash or non-numeric suffix', () => {
    // real ids never contain '#'; be conservative and only touch a trailing #<n>
    expect(stripRuntimeIdSuffix('weird#id')).toBe('weird#id')
    expect(stripRuntimeIdSuffix('a#1b')).toBe('a#1b')
  })

  it('strips only the final marker (double-suffix collision fallback)', () => {
    // the funnel can produce `id#1#2` only if a literal `id#1` already existed;
    // stripping the last marker still yields a valid parent that exists.
    expect(stripRuntimeIdSuffix('1784158733-1-assistant#1#2')).toBe('1784158733-1-assistant#1')
  })

  it('nullable variant passes null through and strips otherwise', () => {
    expect(stripRuntimeIdSuffixNullable(null)).toBeNull()
    expect(stripRuntimeIdSuffixNullable('1784158733-1-assistant#3')).toBe('1784158733-1-assistant')
    expect(stripRuntimeIdSuffixNullable('u-2')).toBe('u-2')
  })
})
