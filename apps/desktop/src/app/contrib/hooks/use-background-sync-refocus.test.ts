import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'

import { useBackgroundSync } from './use-background-sync'

type GatewayState = string

interface HarnessOverrides {
  gatewayState?: GatewayState
}

interface BackgroundSyncHarness {
  refreshSessions: ReturnType<typeof vi.fn>
  refreshMessagingSessions: ReturnType<typeof vi.fn>
}

/** Render useBackgroundSync with every callback stubbed. The gateway-open
 *  mount effect itself performs one refreshSessions() reseed, so assertions
 *  compare call counts against a mount baseline rather than zero. */
function renderBackgroundSync(overrides: HarnessOverrides = {}): BackgroundSyncHarness {
  const refreshSessions = vi.fn().mockResolvedValue(undefined)
  const refreshMessagingSessions = vi.fn().mockResolvedValue(undefined)

  renderHook(() =>
    useBackgroundSync({
      activeConnectionId: null,
      activeGatewayProfile: 'default',
      activeIsMessaging: false,
      activeSessionId: null,
      activeStoredSessionId: null,
      freshDraftReady: false,
      gatewayState: overrides.gatewayState ?? 'open',
      refreshActiveTranscript: vi.fn().mockResolvedValue(undefined),
      refreshCronJobs: vi.fn().mockResolvedValue(undefined),
      refreshCurrentModel: vi.fn().mockResolvedValue(undefined),
      refreshHermesConfig: vi.fn().mockResolvedValue(undefined),
      refreshMessagingSessions,
      refreshSessions,
      requestGateway: vi.fn().mockResolvedValue({ sessions: [] }),
      updateSessionState: vi.fn((sessionId, updater, storedSessionId) =>
        updater(createClientSessionState(storedSessionId ?? sessionId))
      )
    })
  )

  return { refreshMessagingSessions, refreshSessions }
}

describe('useBackgroundSync window-refocus refresh', () => {
  afterEach(() => {
    vi.clearAllTimers()
    vi.useRealTimers()
  })

  it('fires the sidebar session-list refresh action when the window refocuses', () => {
    const { refreshSessions } = renderBackgroundSync()
    const mountBaseline = refreshSessions.mock.calls.length

    act(() => {
      window.dispatchEvent(new Event('focus'))
    })

    expect(refreshSessions.mock.calls.length).toBe(mountBaseline + 1)
  })

  it('coalesces a focus + visibilitychange burst into a single refresh', () => {
    const { refreshMessagingSessions, refreshSessions } = renderBackgroundSync()
    const mountBaseline = refreshSessions.mock.calls.length

    // A refocus commonly fires `focus` and `visibilitychange` back-to-back;
    // only the first may hit the list endpoint within the 2s window.
    act(() => {
      window.dispatchEvent(new Event('focus'))
      document.dispatchEvent(new Event('visibilitychange'))
      window.dispatchEvent(new Event('focus'))
    })

    expect(refreshSessions.mock.calls.length).toBe(mountBaseline + 1)
    // The refocus path must NOT call the separate messaging refresh: the
    // batched sidebar request in refreshSessions() already carries the
    // messaging slices, and the standalone callback has no request-generation
    // guard, so a late duplicate response could overwrite newer rows.
    expect(refreshMessagingSessions).not.toHaveBeenCalled()
  })

  it('refreshes again once the 2s coalesce window has elapsed', () => {
    vi.useFakeTimers()

    const { refreshSessions } = renderBackgroundSync()
    const mountBaseline = refreshSessions.mock.calls.length

    act(() => {
      window.dispatchEvent(new Event('focus'))
    })
    expect(refreshSessions.mock.calls.length).toBe(mountBaseline + 1)

    act(() => {
      vi.advanceTimersByTime(2_000)
    })
    act(() => {
      window.dispatchEvent(new Event('focus'))
    })

    expect(refreshSessions.mock.calls.length).toBe(mountBaseline + 2)
  })

  it('does not refresh on refocus while the gateway is closed', () => {
    const { refreshMessagingSessions, refreshSessions } = renderBackgroundSync({ gatewayState: 'closed' })

    act(() => {
      window.dispatchEvent(new Event('focus'))
      document.dispatchEvent(new Event('visibilitychange'))
    })

    expect(refreshSessions).not.toHaveBeenCalled()
    expect(refreshMessagingSessions).not.toHaveBeenCalled()
  })
})