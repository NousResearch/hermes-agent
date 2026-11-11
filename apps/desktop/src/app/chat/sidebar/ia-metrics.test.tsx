import { act, cleanup, fireEvent, render, renderHook, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import {
  bindDesktopMetrics,
  DESKTOP_METRICS_STATE_KEY,
  resetDesktopMetricsForTests,
  setDesktopMetricsGate
} from '@/store/desktop-metrics'
import { $fleetRoster } from '@/store/fleet-roster'
import * as profile from '@/store/profile'
import { $shellChoice } from '@/store/shell-variant'
import { requestSimplifiedNewSession } from '@/store/simplified-new-session'

import type { AppView } from '../../routes'

import { SimplifiedNewSession } from './simplified-new-session'
import { useIaNavigationMetrics } from './use-ia-navigation-metrics'
import { useIaSearchMetrics } from './use-ia-search-metrics'

function actions(): Record<string, number> {
  const key = Array.from({ length: window.localStorage.length }, (_, i) => window.localStorage.key(i) ?? '').find(key =>
    key.startsWith(DESKTOP_METRICS_STATE_KEY)
  )

  return key ? JSON.parse(window.localStorage.getItem(key)!).today.actions : {}
}

beforeEach(() => {
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $shellChoice.set(true)
  $apiRequestScope.set({ connectionId: null, profile: 'default' })
  bindDesktopMetrics(null, '|default')
  setDesktopMetricsGate('on')
  $fleetRoster.set(null)
  profile.$profiles.set([{ name: 'default', is_default: true }, { name: 'research' }] as typeof profile.$profiles.value)
  vi.spyOn(profile, 'newSessionInProfile').mockImplementation(() => {})
  vi.spyOn(profile, 'newSessionInAgent').mockImplementation(() => {})
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.restoreAllMocks()
  $shellChoice.set(false)
  resetDesktopMetricsForTests()
})

it('records visible transitions and fast backtracks, excluding hidden time and pre-consent visits', () => {
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
  expect(actions()['ia.transition.chat_to_capabilities|state']).toBe(1)
  expect(actions()['ia.dwell.chat_5s_to30s|state']).toBe(1)
  expect(actions()['ia.backtrack.chat|state']).toBe(1)
  vi.spyOn(window.document, 'visibilityState', 'get').mockReturnValue('hidden')
  fireEvent(window.document, new Event('visibilitychange'))
  act(() => {
    vi.advanceTimersByTime(600_000)
  })
  vi.spyOn(window.document, 'visibilityState', 'get').mockReturnValue('visible')
  fireEvent(window.document, new Event('visibilitychange'))
  expect(actions()['ia.dwell.chat_gte5m|state']).toBeUndefined()
  act(() => {
    setDesktopMetricsGate('off')
  })
  rerender({ view: 'artifacts' })
  act(() => {
    setDesktopMetricsGate('on')
  })
  expect(actions()['ia.transition.chat_to_artifacts|state']).toBeUndefined()
})

it('counts settled search outcomes once, excludes stale searches and query text', () => {
  vi.useFakeTimers()

  const { rerender } = renderHook(({ query, pending, count }) => useIaSearchMetrics(query, pending, false, count), {
    initialProps: { query: 'private first', pending: false, count: 0 }
  })

  rerender({ query: 'private second', pending: true, count: 0 })
  act(() => {
    vi.advanceTimersByTime(1000)
  })
  expect(actions()['ia.search.used|state']).toBeUndefined()
  rerender({ query: 'private second', pending: false, count: 3 })
  act(() => {
    vi.advanceTimersByTime(500)
  })
  rerender({ query: 'private second', pending: false, count: 4 })
  act(() => {
    vi.advanceTimersByTime(500)
  })
  expect(actions()['ia.search.results|state']).toBe(1)
  expect(actions()['ia.search.early|state']).toBe(1)
  expect(JSON.stringify(actions())).not.toContain('private')
})

it('does not attribute a newly settled search to the previous profile while metrics rebind', () => {
  vi.useFakeTimers()
  const { rerender } = renderHook(({ query }) => useIaSearchMetrics(query, false, false, 3), {
    initialProps: { query: 'alpha search' }
  })

  act(() => {
    vi.advanceTimersByTime(500)
  })
  expect(actions()['ia.search.results|state']).toBe(1)

  act(() => {
    $apiRequestScope.set({ connectionId: null, profile: 'beta' })
  })
  rerender({ query: 'beta search before binding' })
  act(() => {
    vi.advanceTimersByTime(500)
  })
  expect(actions()['ia.search.results|state']).toBe(1)

  act(() => {
    bindDesktopMetrics(null, '|beta')
    setDesktopMetricsGate('off')
  })
  rerender({ query: 'beta opted out' })
  act(() => {
    vi.advanceTimersByTime(500)
  })
  expect(actions()['ia.search.results|state']).toBe(1)

  act(() => {
    setDesktopMetricsGate('on')
  })
  rerender({ query: 'beta consenting search' })
  act(() => {
    vi.advanceTimersByTime(500)
  })
  const records = Array.from({ length: window.localStorage.length }, (_, i) => window.localStorage.key(i) ?? '')
    .filter(key => key.startsWith(DESKTOP_METRICS_STATE_KEY))
    .map(key => JSON.parse(window.localStorage.getItem(key)!))
  expect(records.map(record => record.today.actions['ia.search.results|state']).sort()).toEqual([1, 1])
})

it('distinguishes a default selection from dismissing the new-session picker', () => {
  render(<SimplifiedNewSession onFallback={vi.fn()} sessions={[]} />)
  fireEvent.keyDown(screen.getByRole('combobox'), { key: 'Enter' })
  fireEvent.click(screen.getByRole('option', { name: /Start blank session/ }))
  expect(actions()['ia.new_session.default|click']).toBeUndefined()
  expect(profile.newSessionInProfile).not.toHaveBeenCalled()
  act(() => {
    expect(requestSimplifiedNewSession()).toBe(true)
  })
  expect(actions()['ia.new_session.default|click']).toBe(1)
  expect(profile.newSessionInProfile).toHaveBeenCalledWith('default')
  expect(actions()['ia.new_session.dismissed|click']).toBeUndefined()
  fireEvent.keyDown(screen.getByRole('combobox'), { key: 'Enter' })
  fireEvent.keyDown(screen.getByRole('listbox'), { key: 'Escape' })
  expect(actions()['ia.new_session.dismissed|click']).toBe(1)
  expect(actions()['ia.new_session.command|click']).toBe(1)
  expect(actions()['ia.new_session.picker_opened|click']).toBe(2)
  expect(profile.newSessionInProfile).toHaveBeenCalledTimes(1)
})

it('counts opening the picker from the sidebar with the keyboard as a sidebar entry', () => {
  render(<SimplifiedNewSession onFallback={vi.fn()} sessions={[]} />)
  fireEvent.keyDown(screen.getByRole('combobox'), { key: 'Enter' })
  expect(actions()['ia.new_session.sidebar|click']).toBe(1)
  expect(actions()['ia.new_session.command|click']).toBeUndefined()
})
