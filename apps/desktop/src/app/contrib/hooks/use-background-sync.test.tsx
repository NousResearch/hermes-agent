import { act, cleanup, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useBackgroundSync } from './use-background-sync'
import type { GatewayRequester } from '../types'

const noopRequestGateway: GatewayRequester = (() => Promise.resolve({})) as GatewayRequester

function Harness({
  activeIsMessaging = false,
  activeSessionId = null,
  freshDraftReady = false,
  gatewayState = 'open',
  refreshActiveMessagingTranscript = vi.fn(),
  refreshCronJobs = vi.fn(),
  refreshCurrentModel = vi.fn(),
  refreshHermesConfig = vi.fn(),
  refreshMessagingSessions = vi.fn(),
  refreshSessions = vi.fn(),
  requestGateway = noopRequestGateway
}: Partial<Parameters<typeof useBackgroundSync>[0]> = {}) {
  useBackgroundSync({
    activeIsMessaging,
    activeSessionId,
    freshDraftReady,
    gatewayState,
    refreshActiveMessagingTranscript,
    refreshCronJobs,
    refreshCurrentModel,
    refreshHermesConfig,
    refreshMessagingSessions,
    refreshSessions,
    requestGateway
  })

  return null
}

describe('useBackgroundSync — session list poll', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    // jsdom defaults to 'visible'
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      value: 'visible'
    })
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    vi.useRealTimers()
  })

  it('polls refreshSessions on an interval while the gateway is open and the tab is visible', () => {
    const refreshSessions = vi.fn(async () => undefined)

    render(<Harness gatewayState="open" refreshSessions={refreshSessions} />)

    // The initial gateway-open effect calls refreshSessions once immediately
    expect(refreshSessions).toHaveBeenCalledTimes(1)

    // Advance past the poll interval (15s)
    act(() => {
      vi.advanceTimersByTime(15_000)
    })

    expect(refreshSessions).toHaveBeenCalledTimes(2)

    act(() => {
      vi.advanceTimersByTime(15_000)
    })

    expect(refreshSessions).toHaveBeenCalledTimes(3)
  })

  it('does not poll when the gateway is not open', () => {
    const refreshSessions = vi.fn(async () => undefined)

    render(<Harness gatewayState="connecting" refreshSessions={refreshSessions} />)

    act(() => {
      vi.advanceTimersByTime(60_000)
    })

    // The gateway-open reseed effect doesn't fire, and the poll doesn't start
    expect(refreshSessions).not.toHaveBeenCalled()
  })

  it('stops polling after unmount', () => {
    const refreshSessions = vi.fn(async () => undefined)

    const { unmount } = render(<Harness gatewayState="open" refreshSessions={refreshSessions} />)

    // Initial reseed
    expect(refreshSessions).toHaveBeenCalledTimes(1)

    unmount()

    act(() => {
      vi.advanceTimersByTime(60_000)
    })

    expect(refreshSessions).toHaveBeenCalledTimes(1)
  })

  it('does not call refreshSessions when the tab is hidden', () => {
    const refreshSessions = vi.fn(async () => undefined)

    render(<Harness gatewayState="open" refreshSessions={refreshSessions} />)

    // Initial reseed from the gateway-open effect
    expect(refreshSessions).toHaveBeenCalledTimes(1)

    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      value: 'hidden'
    })

    act(() => {
      vi.advanceTimersByTime(30_000)
    })

    // The poll interval fires but the visibility gate suppresses the call
    expect(refreshSessions).toHaveBeenCalledTimes(1)
  })

  it('swallows rejections from refreshSessions without surfacing unhandled promise rejections', async () => {
    const refreshSessions = vi.fn(async () => {
      throw new Error('transient')
    })

    const handler = vi.fn()
    window.addEventListener('unhandledrejection', handler)

    render(<Harness gatewayState="open" refreshSessions={refreshSessions} />)

    await act(async () => {
      vi.advanceTimersByTime(15_000)
      await Promise.resolve()
    })

    expect(refreshSessions).toHaveBeenCalledTimes(2)
    expect(handler).not.toHaveBeenCalled()

    window.removeEventListener('unhandledrejection', handler)
  })
})
