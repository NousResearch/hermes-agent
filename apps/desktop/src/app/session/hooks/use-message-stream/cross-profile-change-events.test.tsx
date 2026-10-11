import type { GatewayEvent } from '@hermes/shared'
import { QueryClient } from '@tanstack/react-query'
import { act, cleanup } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $cronChangeTick, $sessionsChangeTick } from '@/store/live-sync'
import { $activeGatewayProfile } from '@/store/profile'
import { $sessionStates } from '@/store/session-states'

import { type MessageStreamHarness, renderMessageStream } from './test-harness'

// `sessions.changed` / `cron.changed` from a BACKGROUND profile's socket must
// still refresh the lists: the change watcher broadcasts on any served
// profile's state.db write, and the all-profiles sidebar shows every profile's
// rows (#85302 — foreign liveness rides the same refresh cadence). Foreground-
// scoped broadcasts (pet) stay gated to the active source.

const ACTIVE_SID = 'session-active'
const ACTIVE_PROFILE = 'default'
let stream: MessageStreamHarness
let queryClient: QueryClient

function mountStream() {
  stream = renderMessageStream(ACTIVE_SID, { activeGatewayProfile: ACTIVE_PROFILE, queryClient })
}

const changedEvent = (type: string, profile: string) =>
  act(() =>
    stream.handleEvent({
      payload: {},
      profile,
      session_id: '',
      type
    } as GatewayEvent)
  )

beforeEach(() => {
  queryClient = new QueryClient()
  $sessionStates.set({})
  $activeGatewayProfile.set(ACTIVE_PROFILE)
})

afterEach(() => {
  cleanup()
  $sessionStates.set({})
  vi.restoreAllMocks()
})

describe('background-profile change broadcasts', () => {
  it('sessions.changed from another profile bumps the list-refresh tick', () => {
    mountStream()
    const before = $sessionsChangeTick.get()
    changedEvent('sessions.changed', 'saf-auditor')
    expect($sessionsChangeTick.get()).toBe(before + 1)
  })

  it('cron.changed from another profile bumps the cron-refresh tick', () => {
    mountStream()
    const before = $cronChangeTick.get()
    changedEvent('cron.changed', 'saf-auditor')
    expect($cronChangeTick.get()).toBe(before + 1)
  })

  it('keeps foreground-scoped broadcasts (pet) gated to the active profile', () => {
    mountStream()
    // A pet.changed from a foreign profile must not repaint the active pet;
    // the sessions tick staying put is the observable no-cross-talk proof.
    const before = $sessionsChangeTick.get()
    changedEvent('pet.changed', 'saf-auditor')
    expect($sessionsChangeTick.get()).toBe(before)
    expect($activeGatewayProfile.get()).toBe(ACTIVE_PROFILE)
  })
})
