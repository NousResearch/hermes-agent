import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { backendTransitionFor, beginBackendTransition, resetBackendTransitionsForTests } from './backend-transition'
import { $activeGatewayProfile } from './profile'
import { $gatewayState } from './session'
import { endSwitchTransition, SWITCH_SETTLE_MAX_MS } from './switch-transition'

beforeEach(() => {
  vi.useFakeTimers()
  $activeGatewayProfile.set('work')
  $gatewayState.set('open')
  beginBackendTransition('work', 'starting', { source: 'switch' })
})

afterEach(() => {
  resetBackendTransitionsForTests()
  $activeGatewayProfile.set('default')
  $gatewayState.set('idle')
  vi.useRealTimers()
})

describe("the switch's one state lasts until the window is on the new route", () => {
  it('covers the reconnect on the shown profile and ends when the new socket opens', async () => {
    endSwitchTransition('work', true)
    expect(backendTransitionFor('work')?.kind).toBe('starting')

    $gatewayState.set('connecting')
    await vi.advanceTimersByTimeAsync(5_000)
    expect(backendTransitionFor('work')?.kind).toBe('starting')

    $gatewayState.set('open')
    expect(backendTransitionFor('work')).toBeNull()
  })

  it('ends after a short wait when the socket never visibly dropped', async () => {
    endSwitchTransition('work', true)
    await vi.advanceTimersByTimeAsync(1_999)
    expect(backendTransitionFor('work')).not.toBeNull()
    await vi.advanceTimersByTimeAsync(1)
    expect(backendTransitionFor('work')).toBeNull()
  })

  it('gives up at its deadline so a backend that never answers shows as it is', async () => {
    endSwitchTransition('work', true)
    $gatewayState.set('closed')
    await vi.advanceTimersByTimeAsync(SWITCH_SETTLE_MAX_MS)
    expect(backendTransitionFor('work')).toBeNull()
  })

  it('ends at once when the switch did not apply or the profile is in the background', () => {
    endSwitchTransition('work', false)
    expect(backendTransitionFor('work')).toBeNull()

    beginBackendTransition('notes', 'stopping', { source: 'switch' })
    endSwitchTransition('notes', true)
    expect(backendTransitionFor('notes')).toBeNull()
  })

  it('leaves a newer switch of the same profile alone: only that switch ends its own state', async () => {
    endSwitchTransition('work', true)
    $gatewayState.set('connecting')
    // Switched off again before the first switch's socket opened.
    beginBackendTransition('work', 'stopping', { source: 'switch' })

    $gatewayState.set('open')
    expect(backendTransitionFor('work')).toMatchObject({ kind: 'stopping', source: 'switch' })
    await vi.advanceTimersByTimeAsync(SWITCH_SETTLE_MAX_MS)
    expect(backendTransitionFor('work')).toMatchObject({ kind: 'stopping', source: 'switch' })

    endSwitchTransition('work', false)
    expect(backendTransitionFor('work')).toBeNull()
  })
})
