import { renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { $apiRequestScope } from '@/api/client'
import { en } from '@/i18n/en'
import {
  bindDesktopMetrics,
  captureDesktopMetricsGate,
  DESKTOP_METRICS_STATE_KEY,
  resetDesktopMetricsForTests
} from '@/store/desktop-metrics'
import { immediateMetricsLocks, setMetricsGate } from '@/store/desktop-metrics-test-locks'
import { $shellChoice } from '@/store/shell-variant'

import { useOldNameRows } from './old-name-rows'

const go = (path: string) => () => void path

function recorded(): string {
  return Array.from({ length: window.localStorage.length }, (_, i) => window.localStorage.key(i) ?? '')
    .filter(key => key.startsWith(DESKTOP_METRICS_STATE_KEY))
    .map(key => window.localStorage.getItem(key))
    .join('')
}

beforeEach(() => {
  immediateMetricsLocks()
  window.localStorage.clear()
  resetDesktopMetricsForTests()
  $apiRequestScope.set({ connectionId: 'gateway', profile: 'default' })
  bindDesktopMetrics(null, 'gateway|default')
  $shellChoice.set(true)
})

afterEach(() => {
  resetDesktopMetricsForTests()
  vi.unstubAllGlobals()
})

it('offers the renamed place for a classic word, named as the rail names it', () => {
  const { result } = renderHook(() => useOldNameRows('sessions', true, go))

  expect(result.current[0]?.items.map(({ description, label }) => [label, description])).toEqual([
    [en.simplifiedAgents.chats, 'Formerly Sessions']
  ])
})

it('offers nothing outside the redesign or for an ordinary query', () => {
  expect(renderHook(() => useOldNameRows('sessions', false, go)).result.current).toEqual([])
  expect(renderHook(() => useOldNameRows('quarterly', true, go)).result.current).toEqual([])
})

it('counts a whole classic word once per opening, by its code only, and only with consent', async () => {
  renderHook(() => useOldNameRows('profiles', true, go))
  expect(recorded()).toBe('')

  await setMetricsGate(captureDesktopMetricsGate, 'on')

  const { rerender } = renderHook(({ query }) => useOldNameRows(query, true, go), {
    initialProps: { query: 'cro' }
  })

  rerender({ query: 'cron jobs' })
  rerender({ query: 'cron' })

  expect(recorded()).toContain('"ia.alias_search.cron|palette":1')
  expect(recorded()).not.toContain('jobs')
})
