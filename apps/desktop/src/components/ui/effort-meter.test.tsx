import { describe, expect, it } from 'vitest'

import { REASONING_EFFORTS } from '@/lib/reasoning-effort'

import { resolveEffortMeter } from './effort-meter'

describe('resolveEffortMeter', () => {
  it('fills one segment per scale step', () => {
    expect(resolveEffortMeter('minimal', 'medium')).toEqual({ fill: 1, label: 'Min' })
    expect(resolveEffortMeter('medium', 'medium')).toEqual({ fill: 3, label: 'Med' })
    expect(resolveEffortMeter('ultra', 'medium')).toEqual({ fill: REASONING_EFFORTS.length, label: 'Ultra' })
  })

  it('inherits the fallback when unset and empties on thinking-off', () => {
    expect(resolveEffortMeter('', 'high')).toEqual({ fill: 4, label: 'High' })
    expect(resolveEffortMeter('none', 'high')).toEqual({ fill: 0, label: 'Off' })
  })
})
