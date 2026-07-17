import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useBackgroundSync } from './use-background-sync'

const noopAsync = vi.fn().mockResolvedValue(undefined)

function Harness({ refreshMessagingSessions }: { refreshMessagingSessions: () => void }) {
  useBackgroundSync({
    activeIsMessaging: false,
    activeSessionId: null,
    freshDraftReady: false,
    gatewayState: 'open',
    refreshActiveMessagingTranscript: noopAsync,
    refreshCronJobs: noopAsync,
    refreshCurrentModel: noopAsync,
    refreshHermesConfig: noopAsync,
    refreshMessagingSessions,
    refreshSessions: noopAsync,
    requestGateway: noopAsync
  })

  return null
}

describe('useBackgroundSync messaging refresh', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.spyOn(document, 'visibilityState', 'get').mockReturnValue('visible')
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('refreshes after 15 seconds and whenever the window regains focus', () => {
    const refreshMessagingSessions = vi.fn()

    render(<Harness refreshMessagingSessions={refreshMessagingSessions} />)

    act(() => vi.advanceTimersByTime(14_999))
    expect(refreshMessagingSessions).not.toHaveBeenCalled()

    act(() => vi.advanceTimersByTime(1))
    expect(refreshMessagingSessions).toHaveBeenCalledTimes(1)

    act(() => window.dispatchEvent(new Event('focus')))
    expect(refreshMessagingSessions).toHaveBeenCalledTimes(2)
  })
})
