import { renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeSessionId } from '@/store/session'

import { useBackgroundSync } from './use-background-sync'

vi.mock('@/store/profile', () => ({ refreshActiveProfile: vi.fn() }))

function setup(overrides: Partial<Parameters<typeof useBackgroundSync>[0]> = {}) {
  const params: Parameters<typeof useBackgroundSync>[0] = {
    activeIsMessaging: false,
    activeSessionId: null,
    freshDraftReady: false,
    gatewayState: 'open',
    refreshActiveMessagingTranscript: vi.fn(),
    refreshCronJobs: vi.fn(),
    refreshCurrentModel: vi.fn(),
    refreshHermesConfig: vi.fn(),
    refreshMessagingSessions: vi.fn(),
    refreshSessions: vi.fn(),
    requestGateway: vi.fn(),
    ...overrides
  }

  renderHook(() => useBackgroundSync(params))

  return params
}

describe('useBackgroundSync model reseeding', () => {
  beforeEach(() => {
    $activeSessionId.set(null)
  })

  it('force-reseeds the composer when booting onto a fresh draft', () => {
    const { refreshCurrentModel } = setup()

    expect(refreshCurrentModel).toHaveBeenCalledWith(true)
  })

  it('does not force-reseed when booting with a live session', () => {
    $activeSessionId.set('runtime-1')
    const { refreshCurrentModel } = setup({ activeSessionId: 'runtime-1' })

    expect(refreshCurrentModel).toHaveBeenCalledTimes(1)
    expect(refreshCurrentModel).toHaveBeenCalledWith(false)
  })

  it('force-reseeds when a new-session draft becomes ready', () => {
    const { refreshCurrentModel, refreshHermesConfig } = setup({ freshDraftReady: true })

    expect(refreshCurrentModel).toHaveBeenCalledWith(true)
    expect(refreshHermesConfig).toHaveBeenCalledOnce()
  })
})
