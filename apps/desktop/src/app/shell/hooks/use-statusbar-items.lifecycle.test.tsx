import { act, cleanup, render, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { renderMessageStream } from '@/app/session/hooks/use-message-stream/test-harness'
import { $activeSessionId, $busy, $currentUsage, $selectedStoredSessionId } from '@/store/session'
import { $statusbarHiddenIds } from '@/store/statusbar-prefs'
import type { ContextBreakdown, UsageStats } from '@/types/hermes'

import { useStatusbarItems } from './use-statusbar-items'

function usage(tokens: number): UsageStats {
  return {
    calls: 1,
    context_estimated: false,
    context_source: 'measured',
    context_max: 100_000,
    context_percent: tokens / 1_000,
    context_used: tokens,
    input: 100,
    output: 10,
    total: 110
  }
}

function breakdown(tokens: number): ContextBreakdown {
  return {
    ...usage(tokens),
    categories: [],
    context_max: 100_000,
    context_percent: tokens / 1_000,
    context_used: tokens,
    estimated_total: tokens,
    model: 'test-model'
  }
}

function mountStatusbar() {
  const pending: Array<{ resolve: (value: ContextBreakdown) => void; reject: (error: Error) => void }> = []

  function requestGateway<T = unknown>(method: string): Promise<T> {
    if (method !== 'session.context_breakdown') {
      return Promise.resolve({} as T)
    }

    return new Promise<ContextBreakdown>((resolve, reject) => pending.push({ resolve, reject })) as Promise<T>
  }

  const options = {
    agentsOpen: false,
    chatOpen: true,
    commandCenterOpen: false,
    extraLeftItems: [],
    extraRightItems: [],
    freshDraftReady: false,
    gatewayState: 'open',
    inferenceStatus: null,
    openAgents: vi.fn(),
    openCommandCenterSection: vi.fn(),
    requestGateway,
    statusSnapshot: null,
    toggleCommandCenter: vi.fn()
  }

  const hook = renderHook(() => useStatusbarItems(options))
  const meter = () => hook.result.current.statusbarItems.find(item => item.id === 'context-usage')!

  return { ...hook, meter, pending, requestGateway }
}

describe('statusbar context usage lifecycle', () => {
  const hidden = $statusbarHiddenIds.get()

  beforeEach(() => {
    $statusbarHiddenIds.set([])
    $activeSessionId.set('runtime-a')
    $selectedStoredSessionId.set('stored-a')
    $busy.set(false)
    $currentUsage.set(usage(10_000))
  })

  afterEach(() => {
    cleanup()
    $statusbarHiddenIds.set(hidden)
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
    $busy.set(false)
    $currentUsage.set({ calls: 0, input: 0, output: 0, total: 0 })
  })

  it('keeps live usage at turn end until a fresh idle breakdown arrives', async () => {
    const { meter, pending } = mountStatusbar()
    const stream = renderMessageStream('runtime-a')

    const tick = (tokens: number, sessionId = 'runtime-a') =>
      stream.handleEvent({ type: 'session.usage', session_id: sessionId, payload: { usage: usage(tokens) } })

    await act(async () => pending[0].resolve(breakdown(20_000)))
    expect(meter().label).toBe('20k/100k')

    act(() => {
      $busy.set(true)
      tick(30_000)
    })
    expect(meter().label).toBe('30k/100k')

    act(() => tick(40_000))
    expect(meter().label).toBe('40k/100k')
    expect(pending).toHaveLength(1)
    const content = meter().menuContent
    const panel = render(typeof content === 'function' ? content(vi.fn()) : content)
    expect(panel.container.textContent).toContain('40k')
    expect(panel.container.textContent).toContain('40%')

    act(() => tick(90_000, 'background-runtime'))
    expect(meter().label).toBe('40k/100k')

    act(() => $busy.set(false))
    expect(meter().label).toBe('40k/100k')
    expect(pending).toHaveLength(2)

    await act(async () => pending[1].resolve(breakdown(45_000)))
    expect(meter().label).toBe('45k/100k')
  })

  it('does not restore the pre-turn snapshot when the idle refresh fails', async () => {
    const { meter, pending } = mountStatusbar()
    await act(async () => pending[0].resolve(breakdown(20_000)))
    act(() => {
      $busy.set(true)
      $currentUsage.set(usage(40_000))
    })
    act(() => $busy.set(false))
    await act(async () => pending[1].reject(new Error('disconnected')))
    expect(meter().label).toBe('40k/100k')
  })

  it('ignores an idle RPC that resolves after a turn has started', async () => {
    const { meter, pending } = mountStatusbar()
    act(() => {
      $busy.set(true)
      $currentUsage.set(usage(40_000))
    })
    await act(async () => pending[0].resolve(breakdown(20_000)))
    expect(meter().label).toBe('40k/100k')
    act(() => $busy.set(false))
    expect(meter().label).toBe('40k/100k')
  })

  it('does not accept the previous session RPC after switching sessions', async () => {
    const { meter, pending } = mountStatusbar()
    act(() => {
      $activeSessionId.set('runtime-b')
      $selectedStoredSessionId.set('stored-b')
      $currentUsage.set(usage(5_000))
    })
    expect(meter().label).toBe('5k/100k')
    await act(async () => pending[1].resolve(breakdown(8_000)))
    await act(async () => pending[0].resolve(breakdown(20_000)))
    expect(meter().label).toBe('8k/100k')
  })

  it('waits to fetch until the gauge is enabled and skips fetching while busy', async () => {
    $statusbarHiddenIds.set(['context-usage'])
    const { meter, pending } = mountStatusbar()
    expect(pending).toHaveLength(0)
    act(() => {
      $busy.set(true)
      $statusbarHiddenIds.set([])
      $currentUsage.set(usage(30_000))
    })
    expect(meter().label).toBe('30k/100k')
    expect(pending).toHaveLength(0)
    act(() => $busy.set(false))
    expect(pending).toHaveLength(1)
    await act(async () => pending[0].resolve(breakdown(35_000)))
    expect(meter().label).toBe('35k/100k')
  })
})
