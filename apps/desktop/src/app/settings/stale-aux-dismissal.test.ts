import { beforeEach, describe, expect, it } from 'vitest'

import type { StaleAuxAssignment } from '@/hermes'

import { dismissStaleAux, readStaleAuxDismissal, staleAuxFingerprint } from './stale-aux-dismissal'

const slots = (entries: Array<[string, string, string]>): StaleAuxAssignment[] =>
  entries.map(([task, provider, model]) => ({ task, provider, model }))

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

    expect(readStaleAuxDismissal('research')).toBe(staleAuxFingerprint('nous', slots([['vision', 'alibaba', 'qwen3.6-flash']])))
    // A different profile never sees the acknowledgement.
    expect(readStaleAuxDismissal('default')).toBeNull()
    // A different pin configuration is not the acknowledged one.
    expect(readStaleAuxDismissal('research')).not.toBe(
      staleAuxFingerprint('nous', slots([['vision', 'alibaba', 'qwen3.6-flash-2']]))
    )
  })
})
