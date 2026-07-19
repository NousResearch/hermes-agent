import { afterEach, beforeEach, describe, expect, expectTypeOf, it, vi } from 'vitest'

import {
  $petSignals,
  clearPetSignal,
  clearPetSignals,
  derivePetSignalState,
  type PetSignal,
  selectPetStateSignal,
  upsertPetSignal
} from './pet-signals'

let defaultSignalSource = 'provider-a-0'
let testSourceSequence = 0

beforeEach(() => {
  defaultSignalSource = `provider-a-${++testSourceSequence}`
})

const signal = (overrides: Partial<PetSignal> = {}): PetSignal => ({
  createdAt: 1_000,
  id: 'agent-1',
  priority: 0,
  source: defaultSignalSource,
  state: 'working',
  ...overrides
})

const malformedOrderingSignals = [
  ['NaN priority', { priority: Number.NaN }],
  ['infinite priority', { priority: Number.POSITIVE_INFINITY }],
  ['NaN creation time', { createdAt: Number.NaN }],
  ['infinite creation time', { createdAt: Number.POSITIVE_INFINITY }]
] satisfies readonly (readonly [string, Partial<PetSignal>])[]

const clearAllPetSignals = () => {
  for (const source of new Set($petSignals.get().map(signal => signal.source))) {
    clearPetSignals(source)
  }
}

afterEach(clearAllPetSignals)

describe('pet signal arbitration', () => {
  it.each([
    ['working', 'run'],
    ['thinking', 'review'],
    ['blocked', 'waiting'],
    ['done', 'wave'],
    ['failed', 'failed']
  ] as const)('maps %s to the %s animation', (state, expected) => {
    expect(derivePetSignalState([signal({ state })], 1_000)).toBe(expected)
  })

  it('ignores a signal at its expiry boundary', () => {
    expect(derivePetSignalState([signal({ expiresAt: 1_000, state: 'failed' })], 1_000)).toBe('idle')
  })

  it.each([Number.NaN, Number.POSITIVE_INFINITY])('ignores a non-finite expiry timestamp', expiresAt => {
    expect(derivePetSignalState([signal({ expiresAt, state: 'failed' })], 1_000)).toBe('idle')
  })

  it('selects the signal with the highest explicit priority', () => {
    const lower = signal({ createdAt: 900, priority: 1, state: 'failed' })
    const higher = signal({ createdAt: 800, id: 'agent-2', priority: 2, state: 'working' })

    expect(selectPetStateSignal([lower, higher], 1_000)).toBe(higher)
  })

  it('selects the newest signal when priorities tie', () => {
    const older = signal({ createdAt: 900, priority: 1, state: 'failed' })
    const newer = signal({ createdAt: 950, id: 'agent-2', priority: 1, state: 'done' })

    expect(selectPetStateSignal([older, newer], 1_000)).toBe(newer)
  })

  it('uses source as a stable tie-breaker', () => {
    const lexicalLast = signal({ id: 'agent-z', priority: 1, source: 'z-provider', state: 'failed' })
    const lexicalFirst = signal({ id: 'agent-a', priority: 1, source: 'a-provider', state: 'working' })

    expect(selectPetStateSignal([lexicalLast, lexicalFirst], 1_000)).toBe(lexicalFirst)
    expect(selectPetStateSignal([lexicalFirst, lexicalLast], 1_000)).toBe(lexicalFirst)
  })

  it('uses id after source ties', () => {
    const lexicalLast = signal({ id: 'agent-z', priority: 1, state: 'failed' })
    const lexicalFirst = signal({ id: 'agent-a', priority: 1, state: 'working' })

    expect(selectPetStateSignal([lexicalLast, lexicalFirst], 1_000)).toBe(lexicalFirst)
    expect(selectPetStateSignal([lexicalFirst, lexicalLast], 1_000)).toBe(lexicalFirst)
  })

  it.each(malformedOrderingSignals)('ignores a signal with %s', (_label, overrides) => {
    const malformed = signal({ priority: 100, state: 'failed', ...overrides })
    const valid = signal({ id: 'valid', state: 'working' })

    expect(selectPetStateSignal([malformed, valid], 1_000)).toBe(valid)
  })

  it('ignores an unknown runtime state', () => {
    const malformed = signal({ priority: 100, state: 'unknown' as PetSignal['state'] })
    const valid = signal({ id: 'valid', state: 'working' })

    expect(selectPetStateSignal([malformed, valid], 1_000)).toBe(valid)
  })
})

describe('pet signal store', () => {
  it('exposes signal state as read-only', () => {
    expectTypeOf($petSignals).not.toHaveProperty('set')
    expectTypeOf($petSignals.get()).toEqualTypeOf<readonly PetSignal[]>()
  })

  it('inserts a signal', () => {
    const next = signal()

    upsertPetSignal(next)

    expect($petSignals.get()).toEqual([next])
  })

  it('replaces a signal with the same source and id', () => {
    const first = signal()
    const replacement = signal({ createdAt: 2_000, state: 'thinking' })

    upsertPetSignal(first)
    upsertPetSignal(replacement)

    expect($petSignals.get()).toEqual([replacement])
  })

  it('does not replace a newer same-key signal with a stale update', () => {
    const newer = signal({ createdAt: 2_000, state: 'done' })
    const stale = signal({ createdAt: 1_000, state: 'failed' })

    upsertPetSignal(newer)
    upsertPetSignal(stale)

    expect($petSignals.get()).toEqual([newer])
  })

  it('does not replace a same-key signal with an equal-timestamp retry', () => {
    const first = signal({ createdAt: 1_000, state: 'working' })
    const retry = signal({ createdAt: 1_000, state: 'failed' })

    upsertPetSignal(first)
    upsertPetSignal(retry)

    expect($petSignals.get()).toEqual([first])
  })

  it('removes a signal at its expiry time', () => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000)

    try {
      upsertPetSignal(signal({ expiresAt: 1_100 }))

      vi.advanceTimersByTime(99)
      expect($petSignals.get()).toHaveLength(1)

      vi.advanceTimersByTime(1)
      expect($petSignals.get()).toEqual([])
    } finally {
      clearAllPetSignals()
      vi.useRealTimers()
    }
  })

  it('snapshots input so caller mutation cannot prevent expiry', () => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000)
    const input = signal({ expiresAt: 1_100 })

    try {
      upsertPetSignal(input)
      ;(input as { expiresAt?: number }).expiresAt = undefined

      vi.advanceTimersByTime(100)
      expect($petSignals.get()).toEqual([])
    } finally {
      clearAllPetSignals()
      vi.useRealTimers()
    }
  })

  it('rearms cleanup for the next signal expiry', () => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000)
    const later = signal({ expiresAt: 1_200, id: 'agent-2' })

    try {
      upsertPetSignal(signal({ expiresAt: 1_100 }))
      upsertPetSignal(later)

      vi.advanceTimersByTime(100)
      expect($petSignals.get()).toEqual([later])

      vi.advanceTimersByTime(100)
      expect($petSignals.get()).toEqual([])
    } finally {
      clearAllPetSignals()
      vi.useRealTimers()
    }
  })

  it('rearms cleanup when a replacement moves the nearest expiry', () => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000)
    const replacement = signal({ expiresAt: 1_200 })

    try {
      upsertPetSignal(signal({ createdAt: 900, expiresAt: 1_100 }))
      upsertPetSignal(replacement)

      vi.advanceTimersByTime(100)
      expect($petSignals.get()).toEqual([replacement])

      vi.advanceTimersByTime(100)
      expect($petSignals.get()).toEqual([])
    } finally {
      clearAllPetSignals()
      vi.useRealTimers()
    }
  })

  it('rearms cleanup when the nearest-expiring signal is cleared', () => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000)
    const later = signal({ expiresAt: 1_200, id: 'agent-2' })

    try {
      upsertPetSignal(signal({ expiresAt: 1_100 }))
      upsertPetSignal(later)
      clearPetSignal(later.source, 'agent-1')

      vi.advanceTimersByTime(100)
      expect($petSignals.get()).toEqual([later])

      vi.advanceTimersByTime(100)
      expect($petSignals.get()).toEqual([])
    } finally {
      clearAllPetSignals()
      vi.useRealTimers()
    }
  })

  it('does not retain a signal that is already expired', () => {
    vi.useFakeTimers()
    vi.setSystemTime(1_100)

    try {
      upsertPetSignal(signal({ expiresAt: 1_100 }))

      expect($petSignals.get()).toEqual([])
    } finally {
      clearAllPetSignals()
      vi.useRealTimers()
    }
  })

  it.each(malformedOrderingSignals)('rejects a signal with %s', (_label, overrides) => {
    upsertPetSignal(signal(overrides))

    expect($petSignals.get()).toEqual([])
  })

  it('rejects an unknown runtime state', () => {
    upsertPetSignal(signal({ state: 'unknown' as PetSignal['state'] }))

    expect($petSignals.get()).toEqual([])
  })

  it.each([Number.NaN, Number.POSITIVE_INFINITY])('rejects a non-finite expiry timestamp', expiresAt => {
    upsertPetSignal(signal({ expiresAt }))

    expect($petSignals.get()).toEqual([])
  })

  it('accepts a signal that is owned by explicit clear instead of a timer', () => {
    const persistent = signal({ expiresAt: undefined })

    upsertPetSignal(persistent)

    expect($petSignals.get()).toEqual([persistent])
    clearPetSignal(persistent.source, persistent.id)
  })

  it('clears only signals owned by one source', () => {
    const remaining = signal({ id: 'build-1', source: `${defaultSignalSource}-b` })

    const owned = signal()

    upsertPetSignal(owned)
    upsertPetSignal(signal({ id: 'agent-2' }))
    upsertPetSignal(remaining)
    clearPetSignals(owned.source)

    expect($petSignals.get()).toEqual([remaining])
  })

  it('does not let a guarded stale clear erase a newer replacement', () => {
    const previous = signal({ createdAt: 1_000 })
    const newer = signal({ createdAt: 2_000 })

    upsertPetSignal(previous)
    upsertPetSignal(newer)
    clearPetSignal(newer.source, newer.id, previous.createdAt)
    expect($petSignals.get()).toEqual([newer])

    clearPetSignal(newer.source, newer.id, newer.createdAt)
    expect($petSignals.get()).toEqual([])
  })

  it.each([
    ['equal-timestamp', 1_000],
    ['older', 900]
  ] as const)('does not resurrect an %s retry after guarded clear', (_label, createdAt) => {
    const first = signal({ createdAt: 1_000 })
    const retry = signal({ createdAt, state: 'failed' })

    upsertPetSignal(first)
    clearPetSignal(first.source, first.id, first.createdAt)
    upsertPetSignal(retry)

    expect($petSignals.get()).toEqual([])

    const newer = signal({ createdAt: 1_100, state: 'done' })
    upsertPetSignal(newer)
    expect($petSignals.get()).toEqual([newer])
  })

  it('does not resurrect a stale retry after source-wide clear', () => {
    const first = signal({ createdAt: 1_000 })

    upsertPetSignal(first)
    clearPetSignals(first.source)
    upsertPetSignal(signal({ createdAt: 1_000, state: 'failed' }))

    expect($petSignals.get()).toEqual([])

    const newer = signal({ createdAt: 1_100, state: 'done' })
    upsertPetSignal(newer)
    expect($petSignals.get()).toEqual([newer])
  })

  it.each([
    ['equal-timestamp', 1_000],
    ['older', 900]
  ] as const)('does not resurrect an %s retry after timer expiry', (_label, createdAt) => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000)

    try {
      upsertPetSignal(signal({ createdAt: 1_000, expiresAt: 1_100 }))
      vi.advanceTimersByTime(100)
      upsertPetSignal(signal({ createdAt, expiresAt: 1_200, state: 'failed' }))

      expect($petSignals.get()).toEqual([])

      const newer = signal({ createdAt: 1_100, expiresAt: 1_200, state: 'done' })
      upsertPetSignal(newer)
      expect($petSignals.get()).toEqual([newer])
    } finally {
      clearAllPetSignals()
      vi.useRealTimers()
    }
  })

  it.each([
    ['guarded', 'equal-timestamp', 1_000],
    ['guarded', 'older', 900],
    ['unguarded', 'equal-timestamp', 1_000],
    ['unguarded', 'older', 900],
    ['source-wide', 'equal-timestamp', 1_000],
    ['source-wide', 'older', 900]
  ] as const)('blocks a reentrant %s clear followed by an %s retry', (clearMode, _label, createdAt) => {
    const first = signal({ createdAt: 1_000 })
    const retry = signal({ createdAt, state: 'failed' })
    let reacted = false

    const unlisten = $petSignals.listen(signals => {
      if (reacted || signals.length !== 1) {
        return
      }

      reacted = true

      if (clearMode === 'source-wide') {
        clearPetSignals(first.source)
      } else {
        clearPetSignal(first.source, first.id, clearMode === 'guarded' ? first.createdAt : undefined)
      }

      upsertPetSignal(retry)
    })

    try {
      upsertPetSignal(first)
      expect($petSignals.get()).toEqual([])
    } finally {
      unlisten()
    }
  })

  it('blocks intermediate work reentered during replacement publication', () => {
    const first = signal({ createdAt: 1_000 })
    const replacement = signal({ createdAt: 2_000, state: 'thinking' })
    const intermediate = signal({ createdAt: 1_500, state: 'failed' })

    upsertPetSignal(first)

    const unlisten = $petSignals.listen(signals => {
      if (signals[0]?.createdAt !== replacement.createdAt) {
        return
      }

      clearPetSignal(replacement.source, replacement.id, replacement.createdAt)
      upsertPetSignal(intermediate)
    })

    try {
      upsertPetSignal(replacement)
      expect($petSignals.get()).toEqual([])
    } finally {
      unlisten()
    }
  })

  it('keeps a strictly newer generation published by a nested listener', () => {
    const first = signal({ createdAt: 1_000 })
    const replacement = signal({ createdAt: 2_000, state: 'thinking' })
    const newest = signal({ createdAt: 3_000, state: 'done' })

    upsertPetSignal(first)

    const unlisten = $petSignals.listen(signals => {
      if (signals[0]?.createdAt === replacement.createdAt) {
        upsertPetSignal(newest)
      }
    })

    try {
      upsertPetSignal(replacement)
      expect($petSignals.get()).toEqual([newest])
    } finally {
      unlisten()
    }
  })

  it('rejects a queued intermediate generation after a nested newer publication', () => {
    const first = signal({ createdAt: 1_000 })
    const replacement = signal({ createdAt: 2_000, state: 'thinking' })
    const intermediate = signal({ createdAt: 2_500, state: 'failed' })
    const newest = signal({ createdAt: 3_000, state: 'done' })

    upsertPetSignal(first)

    const unlisten = $petSignals.listen(signals => {
      if (signals[0]?.createdAt === replacement.createdAt) {
        upsertPetSignal(newest)
      }
    })

    try {
      upsertPetSignal(replacement)
      upsertPetSignal(intermediate)
      expect($petSignals.get()).toEqual([newest])
    } finally {
      unlisten()
    }
  })

  it('clears one signal without erasing its siblings', () => {
    const sibling = signal({ id: 'agent-2' })
    const otherSource = signal({ id: 'build-1', source: `${defaultSignalSource}-b` })

    upsertPetSignal(signal())
    upsertPetSignal(sibling)
    upsertPetSignal(otherSource)
    clearPetSignal(sibling.source, 'agent-1')

    expect($petSignals.get()).toEqual([sibling, otherSource])
  })
})
