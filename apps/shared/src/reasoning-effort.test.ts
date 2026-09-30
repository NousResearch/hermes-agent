import { describe, expect, it } from 'vitest'

import {
  DEFAULT_REASONING_EFFORT,
  filterReasoningEfforts,
  isReasoningEffort,
  REASONING_EFFORT_VALUES,
  REASONING_EFFORTS
} from './reasoning-effort'

describe('reasoning-effort', () => {
  it('is one duplicate-free value set with `none` as the only non-level', () => {
    expect(new Set(REASONING_EFFORT_VALUES).size).toBe(REASONING_EFFORT_VALUES.length)
    expect(REASONING_EFFORT_VALUES.filter(v => !isReasoningEffort(v))).toEqual(['none'])
  })

  it('defaults to a real level and recognizes it case-insensitively', () => {
    expect(REASONING_EFFORTS).toContain(DEFAULT_REASONING_EFFORT)
    expect(isReasoningEffort(DEFAULT_REASONING_EFFORT.toUpperCase())).toBe(true)
  })
})

describe('filterReasoningEfforts', () => {
  it('offers only the supported levels, in ladder order', () => {
    expect(filterReasoningEfforts(['high', 'low', 'max'])).toEqual(['low', 'high', 'max'])
  })

  it('keeps the full ladder when the route vocabulary is unknown or empty', () => {
    expect(filterReasoningEfforts(undefined)).toBe(REASONING_EFFORTS)
    expect(filterReasoningEfforts(null)).toBe(REASONING_EFFORTS)
    expect(filterReasoningEfforts([])).toBe(REASONING_EFFORTS)
  })

  it('keeps the full ladder when no supported entry is a ladder level', () => {
    // A bespoke (non-ladder) vocabulary is not a narrowing signal.
    expect(filterReasoningEfforts(['thinking', 'off'])).toBe(REASONING_EFFORTS)
  })

  it('keeps the surface current pick listed even when the route drops it', () => {
    // A Hermes-internal step such as `ultra` is clamped server-side; hiding the
    // saved pick would blank a Select / deselect the radio showing it.
    expect(filterReasoningEfforts(['low', 'medium', 'high'], 'ultra')).toEqual([
      'low',
      'medium',
      'high',
      'ultra'
    ])
    expect(filterReasoningEfforts(['low', 'medium', 'high'], 'none')).toEqual(['low', 'medium', 'high'])
  })
})
