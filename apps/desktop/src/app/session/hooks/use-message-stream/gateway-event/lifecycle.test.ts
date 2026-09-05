import { beforeEach, describe, expect, it, vi } from 'vitest'

const ingestBackendSkin = vi.fn()
const resetCronCompletionNotifications = vi.fn()
const setChangeEventsAvailable = vi.fn()

vi.mock('@/store/cron-completion-notifications', () => ({
  resetCronCompletionNotifications: () => resetCronCompletionNotifications()
}))

vi.mock('@/store/live-sync', () => ({
  notifyCronChanged: vi.fn(),
  notifyPairingChanged: vi.fn(),
  notifyPetChanged: vi.fn(),
  notifyPlatformsChanged: vi.fn(),
  notifySessionsChanged: vi.fn(),
  setChangeEventsAvailable: (available: boolean) => setChangeEventsAvailable(available)
}))

vi.mock('@/store/session-states', () => ({
  dropSessionState: vi.fn(),
  unbindTileRuntime: vi.fn()
}))

vi.mock('@/themes/backend-sync', () => ({
  ingestBackendSkin: (...args: unknown[]) => ingestBackendSkin(...args)
}))

import { handleLifecycleEvent } from './lifecycle'

function readyContext(fromActiveSource: boolean) {
  return {
    deps: {},
    event: { payload: { change_events: true }, type: 'gateway.ready' },
    fromActiveSource: () => fromActiveSource,
    payload: { change_events: true }
  } as never
}

describe('gateway.ready cron completion baseline', () => {
  beforeEach(() => {
    ingestBackendSkin.mockClear()
    resetCronCompletionNotifications.mockClear()
    setChangeEventsAvailable.mockClear()
  })

  it('resets the completion baseline for the active source reconnect', () => {
    expect(handleLifecycleEvent(readyContext(true))).toBe(true)
    expect(resetCronCompletionNotifications).toHaveBeenCalledOnce()
  })

  it('does not reset the active baseline for a background source reconnect', () => {
    expect(handleLifecycleEvent(readyContext(false))).toBe(true)
    expect(resetCronCompletionNotifications).not.toHaveBeenCalled()
  })
})
