import { renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { useBackgroundSync } from './use-background-sync'

describe('useBackgroundSync', () => {
  it('force-reseeds the profile default for a fresh new-session draft', () => {
    const refreshCurrentModel = vi.fn()

    renderHook(() =>
      useBackgroundSync({
        activeIsMessaging: false,
        activeSessionId: null,
        freshDraftReady: true,
        gatewayState: 'open',
        refreshActiveMessagingTranscript: vi.fn(),
        refreshCronJobs: vi.fn(),
        refreshCurrentModel,
        refreshHermesConfig: vi.fn(),
        refreshMessagingSessions: vi.fn(),
        refreshSessions: vi.fn(),
        requestGateway: vi.fn()
      })
    )

    expect(refreshCurrentModel).toHaveBeenCalledWith(true)
  })

  it('does not reseed the profile default while a live session is active', () => {
    const refreshCurrentModel = vi.fn()

    renderHook(() =>
      useBackgroundSync({
        activeIsMessaging: false,
        activeSessionId: 'runtime-1',
        freshDraftReady: true,
        gatewayState: 'open',
        refreshActiveMessagingTranscript: vi.fn(),
        refreshCronJobs: vi.fn(),
        refreshCurrentModel,
        refreshHermesConfig: vi.fn(),
        refreshMessagingSessions: vi.fn(),
        refreshSessions: vi.fn(),
        requestGateway: vi.fn()
      })
    )

    expect(refreshCurrentModel).not.toHaveBeenCalledWith(true)
  })
})
