import { beforeEach, describe, expect, it } from 'vitest'

import type { StaleAuxAssignment } from '@/hermes'

import { dismissStaleAux, readStaleAuxDismissal, staleAuxFingerprint } from './stale-aux-dismissal'

const slots = (entries: Array<[string, string, string, string?]>): StaleAuxAssignment[] =>
  entries.map(([task, provider, model, base_url]) => ({ base_url, task, provider, model }))

describe('staleAuxFingerprint', () => {
  it('binds the acknowledgement to the main provider and the pinned slots', () => {
    const a = staleAuxFingerprint('nous', slots([['vision', 'alibaba', 'qwen3.6-flash']]))
    const same = staleAuxFingerprint('nous', slots([['vision', 'alibaba', 'qwen3.6-flash']]))

    expect(a).toBe(same)
    expect(a).not.toBe(staleAuxFingerprint('openrouter', slots([['vision', 'alibaba', 'qwen3.6-flash']])))
    expect(a).not.toBe(staleAuxFingerprint('nous', slots([['vision', 'alibaba', 'qwen3.6-flash-v2']])))
    expect(a).not.toBe(staleAuxFingerprint('nous', slots([['triage_specifier', 'alibaba', 'qwen3.6-flash']])))
  })

  it('is order-insensitive across slots', () => {
    const first = staleAuxFingerprint('nous', [
      { task: 'vision', provider: 'alibaba', model: 'm1' },
      { task: 'curator', provider: 'kimi', model: 'm2' }
    ])

    const second = staleAuxFingerprint('nous', [
      { task: 'curator', provider: 'kimi', model: 'm2' },
      { task: 'vision', provider: 'alibaba', model: 'm1' }
    ])

    expect(first).toBe(second)
  })

  it('includes the slot endpoint, so a repointed base_url re-arms the banner', () => {
    const pinned = staleAuxFingerprint(
      'nous',
      slots([['vision', 'openai', 'gpt-4o-mini', 'https://api.example.com/v1']])
    )

    // Same task/provider/model on a different endpoint: different billing
    // surface, so a stored acknowledgement must not cover it.
    expect(pinned).not.toBe(
      staleAuxFingerprint('nous', slots([['vision', 'openai', 'gpt-4o-mini', 'https://proxy.example.com/v1']]))
    )
    // Trailing slashes are the same endpoint, not a re-arm.
    expect(pinned).toBe(
      staleAuxFingerprint('nous', slots([['vision', 'openai', 'gpt-4o-mini', 'https://api.example.com/v1/']]))
    )
    // Absent and empty endpoints agree (switch echoes carry no base_url).
    expect(staleAuxFingerprint('nous', slots([['vision', 'openai', 'gpt-4o-mini']]))).toBe(
      staleAuxFingerprint('nous', slots([['vision', 'openai', 'gpt-4o-mini', '']]))
    )
  })

  it('normalizes the main provider casing and surrounding whitespace', () => {
    expect(staleAuxFingerprint('  Nous ', slots([]))).toBe(staleAuxFingerprint('nous', slots([])))
  })
})

describe('stale-aux dismissal persistence', () => {
  beforeEach(() => {
    window.localStorage.clear()
  })

  it('persists per profile and re-arms when the pin configuration changes', () => {
    dismissStaleAux('research', 'nous', slots([['vision', 'alibaba', 'qwen3.6-flash']]))

    expect(readStaleAuxDismissal('research')).toBe(
      staleAuxFingerprint('nous', slots([['vision', 'alibaba', 'qwen3.6-flash']]))
    )
    // A different profile never sees the acknowledgement.
    expect(readStaleAuxDismissal('default')).toBeNull()
    // A different pin configuration is not the acknowledged one.
    expect(readStaleAuxDismissal('research')).not.toBe(
      staleAuxFingerprint('nous', slots([['vision', 'alibaba', 'qwen3.6-flash-2']]))
    )
  })

  it('keeps same-named profile acknowledgements isolated by gateway', () => {
    const first = { connectionId: 'gateway-a', profile: 'default' }
    const second = { connectionId: 'gateway-b', profile: 'default' }

    dismissStaleAux(first, 'nous', slots([['vision', 'alibaba', 'qwen3.6-flash']]))

    expect(readStaleAuxDismissal(first)).toBe(
      staleAuxFingerprint('nous', slots([['vision', 'alibaba', 'qwen3.6-flash']]))
    )
    expect(readStaleAuxDismissal(second)).toBeNull()
  })

  it('does not persist a dismissal across same-ID descriptor replacement', () => {
    const firstOwner = { baseUrl: 'https://gateway-a.example', mode: 'remote' as const, token: 'synthetic-a' }
    const replacementOwner = { baseUrl: 'https://gateway-b.example', mode: 'remote' as const, token: 'synthetic-b' }
    const first = { connectionId: 'shared-id', connectionOwner: firstOwner, profile: 'default' }
    const replacement = { connectionId: 'shared-id', connectionOwner: replacementOwner, profile: 'default' }

    dismissStaleAux(first, 'nous', slots([['vision', 'alibaba', 'qwen3.6-flash']]))

    expect(readStaleAuxDismissal(first)).toBe(
      staleAuxFingerprint('nous', slots([['vision', 'alibaba', 'qwen3.6-flash']]))
    )
    expect(readStaleAuxDismissal(replacement)).toBeNull()
    expect(window.localStorage.length).toBe(1)
  })

  it('does not persist a dismissal across legacy descriptor replacement', () => {
    const first = {
      connectionId: null,
      legacyConnection: { baseUrl: 'https://legacy-a.example', mode: 'remote' as const, token: 'synthetic-a' },
      profile: 'default'
    }

    const replacement = {
      connectionId: null,
      legacyConnection: { baseUrl: 'https://legacy-b.example', mode: 'remote' as const, token: 'synthetic-b' },
      profile: 'default'
    }

    dismissStaleAux(first, 'nous', slots([['vision', 'alibaba', 'qwen3.6-flash']]))

    expect(readStaleAuxDismissal(first)).not.toBeNull()
    expect(readStaleAuxDismissal(replacement)).toBeNull()
    expect(window.localStorage.length).toBe(1)
  })

  it('persists across a recreated descriptor for the same authority without keying credentials', () => {
    const first = {
      connectionId: 'gateway-a',
      connectionOwner: { baseUrl: 'https://gateway.example/', mode: 'remote' as const, token: 'old-secret' },
      profile: 'default'
    }

    const refreshed = {
      connectionId: 'gateway-a',
      connectionOwner: { baseUrl: 'https://gateway.example', mode: 'remote' as const, token: 'new-secret' },
      profile: 'default'
    }

    dismissStaleAux(first, 'nous', slots([['vision', 'alibaba', 'qwen3.6-flash']]))

    expect(readStaleAuxDismissal(refreshed)).not.toBeNull()
    expect(Object.keys(window.localStorage)[0]).not.toContain('secret')
  })
})
