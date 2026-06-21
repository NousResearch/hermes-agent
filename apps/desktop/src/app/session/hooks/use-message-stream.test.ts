import { describe, expect, it } from 'vitest'

import { isStickySafeProvider } from './use-message-stream'

describe('isStickySafeProvider', () => {
  it('rejects the bare billing-class "custom" that backend pushes for every named providers:/custom_providers: entry', () => {
    expect(isStickySafeProvider('custom')).toBe(false)
  })

  it('rejects "custom:<name>" slugs recovered by _runtime_model_config → canonical_custom_identity', () => {
    expect(isStickySafeProvider('custom:tokenrouter')).toBe(false)
    // The literal value Bruce observed in the broken desktop after a Telegram
    // /model + picker round-trip:
    expect(isStickySafeProvider('custom:minimax m3')).toBe(false)
    expect(isStickySafeProvider('custom: MiniMax M3 Max')).toBe(false)
  })

  it('rejects the rejected shapes regardless of case or surrounding whitespace', () => {
    expect(isStickySafeProvider('Custom')).toBe(false)
    expect(isStickySafeProvider('CUSTOM')).toBe(false)
    expect(isStickySafeProvider('  custom  ')).toBe(false)
    expect(isStickySafeProvider('Custom:Tokenrouter')).toBe(false)
  })

  it('accepts a real, user-facing provider name (the named custom_providers entry)', () => {
    expect(isStickySafeProvider('tokenrouter')).toBe(true)
  })

  it('accepts a built-in provider slug', () => {
    expect(isStickySafeProvider('openrouter')).toBe(true)
    expect(isStickySafeProvider('anthropic')).toBe(true)
    expect(isStickySafeProvider('gemini')).toBe(true)
    expect(isStickySafeProvider('xai-oauth')).toBe(true)
  })

  it('rejects empty / whitespace / null / undefined payloads', () => {
    expect(isStickySafeProvider('')).toBe(false)
    expect(isStickySafeProvider('   ')).toBe(false)
    expect(isStickySafeProvider(null)).toBe(false)
    expect(isStickySafeProvider(undefined)).toBe(false)
  })
})