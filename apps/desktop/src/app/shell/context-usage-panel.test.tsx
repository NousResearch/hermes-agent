import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ContextBreakdown, UsageStats } from '@/types/hermes'

import { ContextUsagePanel } from './context-usage-panel'
import { useContextBreakdown } from './hooks/use-context-breakdown'
import {
  loadContextUsageSnapshot,
  saveContextUsageSnapshot,
  type ContextUsageVersion
} from '@/store/context-usage-cache'

const usage: UsageStats = {
  calls: 1,
  context_max: 272_000,
  context_percent: 47,
  context_used: 128_200,
  input: 0,
  output: 0,
  total: 0
}

const breakdown: ContextBreakdown = {
  categories: [{ color: 'teal', id: 'conversation', label: 'Conversation', tokens: 241_400 }],
  context_max: 272_000,
  context_percent: 89,
  context_used: 241_400,
  estimated_total: 286_600,
  model: 'test-model'
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  window.localStorage.clear()
})

describe('useContextBreakdown', () => {
  it('fetches for a session that has not run a turn yet', async () => {
    const requestGateway = vi.fn().mockResolvedValue(breakdown)

    const { result } = renderHook(() =>
      useContextBreakdown({ busy: false, enabled: true, requestGateway, sessionId: 'runtime-1' })
    )

    await waitFor(() => expect(result.current.breakdown).toEqual(breakdown))
    expect(requestGateway).toHaveBeenCalledWith('session.context_breakdown', { session_id: 'runtime-1' })
  })

  it('does not fetch while the gauge is hidden, and fetches once it is shown', async () => {
    const requestGateway = vi.fn().mockResolvedValue(breakdown)

    const { rerender } = renderHook(
      ({ enabled }) => useContextBreakdown({ busy: false, enabled, requestGateway, sessionId: 'runtime-1' }),
      { initialProps: { enabled: false } }
    )

    expect(requestGateway).not.toHaveBeenCalled()

    rerender({ enabled: true })

    await waitFor(() => expect(requestGateway).toHaveBeenCalledTimes(1))
  })

  it('skips the estimate mid-turn — the gateway streams measured usage then', () => {
    const requestGateway = vi.fn().mockResolvedValue(breakdown)

    renderHook(() => useContextBreakdown({ busy: true, enabled: true, requestGateway, sessionId: 'runtime-1' }))

    expect(requestGateway).not.toHaveBeenCalled()
  })

  it('refetches on a session switch and never reports the previous session numbers', async () => {
    const requestGateway = vi.fn().mockResolvedValue(breakdown)

    const { rerender, result } = renderHook(
      ({ sessionId }) => useContextBreakdown({ busy: false, enabled: true, requestGateway, sessionId }),
      { initialProps: { sessionId: 'runtime-1' } }
    )

    await waitFor(() => expect(result.current.breakdown).toEqual(breakdown))

    // Switching sessions must drop the numbers immediately — painting them
    // under the new session's name would be a lie until its own fetch lands.
    requestGateway.mockImplementation(() => new Promise(() => undefined))
    rerender({ sessionId: 'runtime-2' })

    expect(result.current.breakdown).toBeNull()
    expect(requestGateway).toHaveBeenLastCalledWith('session.context_breakdown', { session_id: 'runtime-2' })
  })

  it('reports the measured occupancy the backend sends, not just the estimate', async () => {
    // `context_used` on the payload is already the measured figure once a turn
    // has run — the estimate is the backend's own fallback, not a second value
    // the client has to choose between.
    const measured: ContextBreakdown = { ...breakdown, context_used: 12_000 }
    const requestGateway = vi.fn().mockResolvedValue(measured)

    const { result } = renderHook(() =>
      useContextBreakdown({ busy: false, enabled: true, requestGateway, sessionId: 'runtime-1' })
    )

    await waitFor(() => expect(result.current.breakdown?.context_used).toBe(12_000))
  })

  it('zeroes the occupancy for an empty conversation (baseline estimate is not usage)', async () => {
    const empty: ContextBreakdown = {
      categories: [{ color: 'gray', id: 'system_prompt', label: 'System prompt', tokens: 22_200 }],
      context_max: 2_000_000,
      context_percent: 1,
      context_source: 'local_estimate',
      context_used: 22_200,
      estimated_total: 22_200,
      model: 'test-model'
    }
    const requestGateway = vi.fn().mockResolvedValue(empty)

    const { result } = renderHook(() =>
      useContextBreakdown({ busy: false, enabled: true, requestGateway, sessionId: 'runtime-1' })
    )

    await waitFor(() => expect(result.current.breakdown?.context_max).toBe(2_000_000))
    expect(result.current.breakdown?.context_used).toBe(0)
    expect(result.current.breakdown?.context_percent).toBe(0)
    expect(result.current.breakdown?.categories).toEqual([])
  })

  it('paints the durable last-known read while the live fetch is pending', async () => {
    const version: ContextUsageVersion = { input_tokens: 100, message_count: 4, model: 'm', output_tokens: 50 }
    saveContextUsageSnapshot(
      'stored-1',
      { context_max: 200_000, context_percent: 34, context_used: 68_000, model: 'm' },
      { profile: 'coder' },
      version
    )
    const requestGateway = vi.fn().mockReturnValue(new Promise(() => {}))

    const { result } = renderHook(() =>
      useContextBreakdown({
        busy: false,
        enabled: true,
        persist: { scope: { profile: 'coder' }, storedSessionId: 'stored-1', version },
        requestGateway,
        sessionId: 'runtime-1'
      })
    )

    await waitFor(() => expect(result.current.loading).toBe(true))
    expect(result.current.breakdown?.context_source).toBe('restored')
    expect(result.current.breakdown?.context_used).toBe(68_000)
    expect(result.current.breakdown?.context_max).toBe(200_000)
  })

  it('keeps the restored read when the backend reports a no-agent payload', async () => {
    const version: ContextUsageVersion = { input_tokens: 100, message_count: 4, model: 'm', output_tokens: 50 }
    saveContextUsageSnapshot(
      'stored-1',
      { context_max: 200_000, context_percent: 34, context_used: 68_000, model: 'm' },
      { profile: 'coder' },
      version
    )
    const noData: ContextBreakdown = {
      categories: [],
      context_max: 0,
      context_percent: 0,
      context_used: 0,
      estimated_total: 0,
      model: ''
    }
    const requestGateway = vi.fn().mockResolvedValue(noData)

    const { result } = renderHook(() =>
      useContextBreakdown({
        busy: false,
        enabled: true,
        persist: { scope: { profile: 'coder' }, storedSessionId: 'stored-1', version },
        requestGateway,
        sessionId: 'runtime-1'
      })
    )

    await waitFor(() => expect(requestGateway).toHaveBeenCalled())
    await waitFor(() => expect(result.current.loading).toBe(false))
    expect(result.current.breakdown?.context_source).toBe('restored')
    expect(result.current.breakdown?.context_used).toBe(68_000)
  })

  it('replaces the restored read with live data and banks the new read', async () => {
    const version: ContextUsageVersion = { input_tokens: 100, message_count: 4, model: 'm', output_tokens: 50 }
    saveContextUsageSnapshot(
      'stored-1',
      { context_max: 200_000, context_percent: 34, context_used: 68_000, model: 'm' },
      { profile: 'coder' },
      version
    )
    const requestGateway = vi.fn().mockResolvedValue(breakdown)

    const { result } = renderHook(() =>
      useContextBreakdown({
        busy: false,
        enabled: true,
        persist: { scope: { profile: 'coder' }, storedSessionId: 'stored-1', version },
        requestGateway,
        sessionId: 'runtime-1'
      })
    )

    await waitFor(() => expect(result.current.breakdown).toEqual(breakdown))
    expect(loadContextUsageSnapshot('stored-1', { profile: 'coder' }, version)?.context_used).toBe(241_400)
  })

  it('ignores a restored read whose version drifted', async () => {
    saveContextUsageSnapshot(
      'stored-1',
      { context_max: 200_000, context_percent: 34, context_used: 68_000, model: 'm' },
      { profile: 'coder' },
      { input_tokens: 100, message_count: 4, model: 'm', output_tokens: 50 }
    )
    const requestGateway = vi.fn().mockReturnValue(new Promise(() => {}))

    const { result } = renderHook(() =>
      useContextBreakdown({
        busy: false,
        enabled: true,
        persist: {
          scope: { profile: 'coder' },
          storedSessionId: 'stored-1',
          version: { input_tokens: 100, message_count: 5, model: 'm', output_tokens: 50 }
        },
        requestGateway,
        sessionId: 'runtime-1'
      })
    )

    await waitFor(() => expect(result.current.loading).toBe(true))
    expect(result.current.breakdown).toBeNull()
  })
})
describe('ContextUsagePanel', () => {
  it('marks estimates but preserves the provider-usage header', () => {
    for (const estimated of [true, false]) {
      const { container, unmount } = render(
        <ContextUsagePanel breakdown={breakdown} loading={false} usage={{ ...usage, context_estimated: estimated }} />
      )

      const header = container.querySelector('[data-slot="context-usage-panel"] > div')?.textContent ?? ''

      expect(header.includes('~')).toBe(estimated)
      expect(container.querySelector('li')?.textContent).toContain('~')
      unmount()
    }
  })

  it('renders the usage it is handed, so the popover matches the bar', () => {
    render(<ContextUsagePanel breakdown={breakdown} loading={false} usage={usage} />)

    expect(screen.getByText('47% Full')).toBeTruthy()
    expect(screen.getByText('Conversation')).toBeTruthy()
  })

  it('says so when there is no breakdown rather than painting an empty bar', () => {
    render(<ContextUsagePanel breakdown={null} loading={false} usage={usage} />)

    expect(screen.getByText('No context data yet')).toBeTruthy()
  })
})
