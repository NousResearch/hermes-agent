import { act, cleanup, fireEvent, render, renderHook, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import {
  bindDesktopMetrics,
  captureDesktopMetricsGate,
  DESKTOP_METRICS_STATE_KEY,
  resetDesktopMetricsForTests
} from '@/store/desktop-metrics'
import { immediateMetricsLocks, setMetricsGate, settleMetrics, storedMetricsAggregate } from '@/store/desktop-metrics-test-locks'
import { $fleetRoster } from '@/store/fleet-roster'
import * as profile from '@/store/profile'
import { $shellChoice } from '@/store/shell-variant'

import type { AppView } from '../../routes'

import { SimplifiedNewSession } from './simplified-new-session'
import { useIaNavigationMetrics } from './use-ia-navigation-metrics'
import { useIaSearchMetrics } from './use-ia-search-metrics'

// Aggregate writes land after a short coordination phase: settle (on either clock) before reading.
const settle = () => (vi.isFakeTimers() ? vi.advanceTimersByTimeAsync(0) : settleMetrics())

async function actions(): Promise<Record<string, number>> {
  await settle()
  const stored = storedMetricsAggregate()

  return stored ? JSON.parse(stored).today.actions : {}
}

beforeEach(async () => {
  immediateMetricsLocks()
  // The picker's listbox scrolls its option into view while the test settles metric writes.
  Object.defineProperty(Element.prototype, 'scrollIntoView', { configurable: true, value: vi.fn() })
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $shellChoice.set(true)
  $apiRequestScope.set({ connectionId: null, profile: 'default' })
  bindDesktopMetrics(null, '|default')
  await setMetricsGate(captureDesktopMetricsGate, 'on')
  $fleetRoster.set(null)
  profile.$profiles.set([{ name: 'default', is_default: true }, { name: 'research' }] as typeof profile.$profiles.value)
  vi.spyOn(profile, 'newSessionInProfile').mockImplementation(() => {})
  vi.spyOn(profile, 'newSessionInAgent').mockImplementation(() => {})
})
afterEach(() => {
  vi.unstubAllGlobals()
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
  $shellChoice.set(false)
  resetDesktopMetricsForTests()
})

it('records visible transitions and fast backtracks, excluding hidden time and pre-consent visits', async () => {
  vi.useFakeTimers()
  vi.setSystemTime(1_000_000)

  const { rerender } = renderHook(({ view }: { view: AppView }) => useIaNavigationMetrics(view), {
    initialProps: { view: 'chat' as AppView }
  })

  act(() => {
    vi.advanceTimersByTime(6000)
  })
  rerender({ view: 'capabilities' })
  act(() => {
    vi.advanceTimersByTime(2000)
  })
  rerender({ view: 'chat' })
  expect((await actions())['ia.transition.chat_to_capabilities|state']).toBe(1)
  expect((await actions())['ia.dwell.chat_5s_to30s|state']).toBe(1)
  expect((await actions())['ia.backtrack.chat|state']).toBe(1)
  vi.spyOn(window.document, 'visibilityState', 'get').mockReturnValue('hidden')
  fireEvent(window.document, new Event('visibilitychange'))
  act(() => {
    vi.advanceTimersByTime(600_000)
  })
  vi.spyOn(window.document, 'visibilityState', 'get').mockReturnValue('visible')
  fireEvent(window.document, new Event('visibilitychange'))
  expect((await actions())['ia.dwell.chat_gte5m|state']).toBeUndefined()
  await act(async () => {
    await setMetricsGate(captureDesktopMetricsGate, 'off')
  })
  rerender({ view: 'artifacts' })
  await act(async () => {
    await setMetricsGate(captureDesktopMetricsGate, 'on')
  })
  expect((await actions())['ia.transition.chat_to_artifacts|state']).toBeUndefined()
})

it('counts settled search outcomes once, excludes stale searches and query text', async () => {
  vi.useFakeTimers()

  const { rerender } = renderHook(({ query, pending, count }) => useIaSearchMetrics(query, pending, false, count), {
    initialProps: { query: 'private first', pending: false, count: 0 }
  })

  rerender({ query: 'private second', pending: true, count: 0 })
  act(() => {
    vi.advanceTimersByTime(1000)
  })
  expect((await actions())['ia.search.used|state']).toBeUndefined()
  rerender({ query: 'private second', pending: false, count: 3 })
  act(() => {
    vi.advanceTimersByTime(500)
  })
  rerender({ query: 'private second', pending: false, count: 4 })
  act(() => {
    vi.advanceTimersByTime(500)
  })
  expect((await actions())['ia.search.results|state']).toBe(1)
  expect((await actions())['ia.search.early|state']).toBe(1)
  expect(JSON.stringify(await actions())).not.toContain('private')
})

it('does not attribute a newly settled search to the previous profile while metrics rebind', async () => {
  vi.useFakeTimers()

  const { rerender } = renderHook(({ query }) => useIaSearchMetrics(query, false, false, 3), {
    initialProps: { query: 'alpha search' }
  })

  act(() => {
    vi.advanceTimersByTime(500)
  })
  expect((await actions())['ia.search.results|state']).toBe(1)

  act(() => {
    $apiRequestScope.set({ connectionId: null, profile: 'beta' })
  })
  rerender({ query: 'beta search before binding' })
  act(() => {
    vi.advanceTimersByTime(500)
  })
  expect((await actions())['ia.search.results|state']).toBe(1)

  await act(async () => {
    bindDesktopMetrics(null, '|beta')
    await setMetricsGate(captureDesktopMetricsGate, 'off')
  })
  rerender({ query: 'beta opted out' })
  act(() => {
    vi.advanceTimersByTime(500)
  })
  expect((await actions())['ia.search.results|state']).toBe(1)

  await act(async () => {
    await setMetricsGate(captureDesktopMetricsGate, 'on')
  })
  rerender({ query: 'beta consenting search' })
  act(() => {
    vi.advanceTimersByTime(500)
  })

  await settle()

  const records = Array.from({ length: window.localStorage.length }, (_, i) => window.localStorage.key(i) ?? '')
    .filter(key => key.startsWith(`${DESKTOP_METRICS_STATE_KEY}:`) && !key.slice(DESKTOP_METRICS_STATE_KEY.length + 1).includes(':'))
    .map(key => JSON.parse(window.localStorage.getItem(key)!))
    .filter(record => record !== null)

  expect(records.map(record => record.today.actions['ia.search.results|state']).sort()).toEqual([1, 1])
})

it('distinguishes a default selection from dismissing the new-session picker', async () => {
  render(<SimplifiedNewSession onFallback={vi.fn()} sessions={[]} />)
  fireEvent.keyDown(screen.getByRole('combobox'), { key: 'Enter' })
  // A pick is the action: the blank chat starts at once and counts as a default start.
  fireEvent.click(screen.getByRole('option', { name: /Start blank chat/ }))
  expect((await actions())['ia.new_session.default|click']).toBe(1)
  expect(profile.newSessionInProfile).toHaveBeenCalledWith('default')
  expect((await actions())['ia.new_session.dismissed|click']).toBeUndefined()
  fireEvent.keyDown(screen.getByRole('combobox'), { key: 'Enter' })
  fireEvent.keyDown(screen.getByRole('listbox'), { key: 'Escape' })
  expect((await actions())['ia.new_session.dismissed|click']).toBe(1)
  expect((await actions())['ia.new_session.picker_opened|click']).toBe(2)
  expect(profile.newSessionInProfile).toHaveBeenCalledTimes(1)
})

it('counts opening the picker from the sidebar with the keyboard as a sidebar entry', async () => {
  render(<SimplifiedNewSession onFallback={vi.fn()} sessions={[]} />)
  fireEvent.keyDown(screen.getByRole('combobox'), { key: 'Enter' })
  expect((await actions())['ia.new_session.sidebar|click']).toBe(1)
  expect((await actions())['ia.new_session.command|click']).toBeUndefined()
})
