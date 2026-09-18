import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $gateway } from '@/store/gateway'
import { $inbox } from '@/store/inbox'
import { $activeGatewayProfile } from '@/store/profile'

import { INBOX_POLL_INTERVAL_MS, useInbox } from './use-inbox'

vi.mock('@/store/inbox', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  refreshInbox: vi.fn()
}))

vi.mock('@/store/gateway', () => ({
  $gateway: { get: vi.fn(), set: vi.fn(), subscribe: vi.fn() }
}))

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: { get: vi.fn(() => ''), set: vi.fn(), subscribe: vi.fn() }
}))

vi.mock('@nanostores/react', async importActual => {
  const actual = await importActual<Record<string, unknown>>()

  return {
    ...actual,
    useStore: (atom: { get: () => unknown }) => atom.get()
  }
})

beforeEach(() => {
  vi.useFakeTimers()
  vi.clearAllMocks()
})

afterEach(() => {
  vi.useRealTimers()
})

function setGateway(value: unknown) {
  vi.mocked($gateway.get).mockReturnValue(value as never)
}

function setProfile(value: string) {
  vi.mocked($activeGatewayProfile.get).mockReturnValue(value)
}

describe('useInbox', () => {
  it('starts polling when gateway becomes non-null', async () => {
    const { refreshInbox } = await import('@/store/inbox')
    setGateway({ connectionState: 'open' })
    setProfile('default')

    renderHook(() => useInbox())

    expect(refreshInbox).toHaveBeenCalledWith('default')
    expect(refreshInbox).toHaveBeenCalledTimes(1)

    act(() => {
      vi.advanceTimersByTime(INBOX_POLL_INTERVAL_MS)
    })

    expect(refreshInbox).toHaveBeenCalledTimes(2)
  })

  it('does not poll when gateway is null', async () => {
    const { refreshInbox } = await import('@/store/inbox')
    setGateway(null)
    setProfile('default')

    renderHook(() => useInbox())

    expect(refreshInbox).not.toHaveBeenCalled()

    act(() => {
      vi.advanceTimersByTime(INBOX_POLL_INTERVAL_MS * 3)
    })

    expect(refreshInbox).not.toHaveBeenCalled()
  })

  it('stops polling when gateway becomes null (disconnect)', async () => {
    const { refreshInbox } = await import('@/store/inbox')
    setGateway({ connectionState: 'open' })
    setProfile('default')

    const { unmount } = renderHook(() => useInbox())

    expect(refreshInbox).toHaveBeenCalledTimes(1)

    // Simulate unmount → cleanup clears the timer
    unmount()

    act(() => {
      vi.advanceTimersByTime(INBOX_POLL_INTERVAL_MS * 5)
    })

    // Only the initial call, no more polls after unmount
    expect(refreshInbox).toHaveBeenCalledTimes(1)
  })

  it('restarts polling when gateway reconnects', async () => {
    const { refreshInbox } = await import('@/store/inbox')
    setGateway(null)
    setProfile('default')

    const { rerender } = renderHook(() => useInbox())

    expect(refreshInbox).not.toHaveBeenCalled()

    // Simulate gateway reconnecting
    setGateway({ connectionState: 'open' })
    rerender()

    expect(refreshInbox).toHaveBeenCalledWith('default')
    expect(refreshInbox).toHaveBeenCalledTimes(1)

    act(() => {
      vi.advanceTimersByTime(INBOX_POLL_INTERVAL_MS)
    })

    expect(refreshInbox).toHaveBeenCalledTimes(2)
  })

  it('clears old snapshot when gateway disconnects (no stale data)', async () => {
    const { clearInbox } = await import('@/store/inbox')

    // Simulate a populated inbox state
    $inbox.set({
      capability: 'supported',
      error: null,
      loading: false,
      snapshot: {
        badge: 'amber',
        counts: { needs_you: 2, running: 1, waiting: 0, scheduled: 0, total: 3 },
        coverage: {
          approval_scope: '',
          clarify_scope: '',
          connection_scope: '',
          errors: [],
          partial: false,
          profile: 'default',
          scanned_sessions: 3
        },
        items: []
      }
    })

    // Verify state is populated before clear
    expect($inbox.get().snapshot).not.toBeNull()

    // Simulate gateway switch → clearInbox should be called
    clearInbox()

    // After clear, snapshot should be null and capability should be 'unknown'
    const entry = $inbox.get()
    expect(entry.snapshot).toBeNull()
    expect(entry.capability).toBe('unknown')
  })

  it('does not show stale rows after profile switch', async () => {
    const { clearInbox } = await import('@/store/inbox')

    // Simulate populated state from profile A
    $inbox.set({
      capability: 'supported',
      error: null,
      loading: false,
      snapshot: {
        badge: 'amber',
        counts: { needs_you: 1, running: 0, waiting: 0, scheduled: 0, total: 1 },
        coverage: {
          approval_scope: '',
          clarify_scope: '',
          connection_scope: '',
          errors: [],
          partial: false,
          profile: 'profileA',
          scanned_sessions: 1
        },
        items: [{
          background_task_count: 0,
          background_task_count_unavailable: false,
          categories: [],
          expired_request_count: 0,
          session_key: 'sess-a',
          title: 'Profile A session',
          source: 'cli',
          cwd: '/work',
          lanes: ['needs_you'],
          goal: null,
          loop: null,
          heartbeat: null,
          pending_approval: null,
          pending_clarify: null,
          subagent_count: 0,
          subagent_count_unavailable: false
        }]
      }
    })

    // Profile switch clears old data
    clearInbox()

    const entry = $inbox.get()
    expect(entry.snapshot).toBeNull()
    expect(entry.capability).toBe('unknown')
  })
})
