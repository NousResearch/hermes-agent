import { afterEach, describe, expect, it } from 'vitest'

import { $currentUsage, setCurrentUsage } from '@/store/session'

import type { ClientSessionState } from '../../../types'

import { publishWarmSessionRuntime } from './session-usage'
import { applyRuntimeInfo } from './utils'

// The statusbar's cache-hit / t/s readouts belong to the session on screen: switching away from a
// session must never leave its numbers painted over the one switched to.
const previous = { avg_tps: 42, cache_hit_pct: 87, calls: 2, compressions: 4, input: 10, output: 5, total: 15 }

const readouts = () => {
  const { avg_tps, cache_hit_pct, compressions } = $currentUsage.get()

  return { avg_tps, cache_hit_pct, compressions }
}

describe('statusbar usage follows the session switched to', () => {
  afterEach(() => setCurrentUsage({ calls: 0, input: 0, output: 0, total: 0 }))

  it('paints a warm session its own last reading, and none of the previous one when it has none', () => {
    const own = { avg_tps: 12, cache_hit_pct: 55, calls: 3, input: 30, output: 9, total: 39 }

    setCurrentUsage(previous)
    publishWarmSessionRuntime({ branch: '', runtimeStartedAt: null, usage: own } as unknown as ClientSessionState)
    expect($currentUsage.get()).toEqual(own)

    setCurrentUsage(previous)
    publishWarmSessionRuntime({ branch: '', runtimeStartedAt: null, usage: null } as unknown as ClientSessionState)
    expect(readouts()).toEqual({})

    // Agentless reopen (no reading since a gateway restart): the resolved row's totals, never zeros, and
    // whichever list the row came from (a cron or messaging row is not in Recents).
    publishWarmSessionRuntime({ branch: '', runtimeStartedAt: null, usage: null } as unknown as ClientSessionState, {
      input_tokens: 700,
      output_tokens: 80
    })
    expect($currentUsage.get()).toEqual({ calls: 0, input: 700, output: 80, total: 780 })
  })

  it('clears the previous readouts when the resume snapshot omits them', () => {
    setCurrentUsage(previous)
    applyRuntimeInfo({ usage: { calls: 0, input: 0, output: 0, total: 0 } })
    expect(readouts()).toEqual({})
  })
})
