// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { CronJob } from '@/hermes'
import { en } from '@/i18n/en'
import { queryClient } from '@/lib/query-client'
import { setCronJobs } from '@/store/cron'
import { setSidebarOpen } from '@/store/layout'
import { $shellChoice } from '@/store/shell-variant'

import { $lastViewedSchedule } from './landing'

import { CronView } from './index'

const getCronJobs = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getAutomationBlueprints: async () => ({ blueprints: [] }),
  getCronJobRuns: async () => [],
  getCronJobs: (profile?: string) => getCronJobs(profile)
}))

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

const schedule = (id: string, name: string): CronJob =>
  ({
    deliver: 'local',
    enabled: true,
    id,
    name,
    prompt: `${name} prompt`,
    schedule_display: 'Daily',
    state: 'scheduled'
  }) as CronJob

const jobs = [schedule('a', 'Alpha digest'), schedule('b', 'Bravo report'), schedule('c', 'Charlie sweep')]

async function renderSchedules() {
  await act(async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <CronView onClose={vi.fn()} />
      </QueryClientProvider>
    )
  })

  return screen.getByRole('complementary', { name: en.cron.title })
}

const heading = () => screen.getByRole('heading', { level: 1 }).textContent

beforeAll(() => {
  HTMLElement.prototype.scrollIntoView ??= () => undefined
  vi.stubGlobal('CSS', { escape: (value: string) => value })
})

beforeEach(() => {
  $shellChoice.set(true)
  setSidebarOpen(true)
  setCronJobs(jobs)
  getCronJobs.mockResolvedValue(jobs)
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  setCronJobs([])
  $lastViewedSchedule.set(null)
  $shellChoice.set(null)
})

describe('Schedules in the page sidebar panel', () => {
  it('lists every schedule under its count, the first one open on the stage', async () => {
    const panel = await renderSchedules()

    expect(within(panel).getByText(en.cron.count(3))).toBeTruthy()
    expect(within(panel).getByRole('button', { current: 'page', name: 'Alpha digest' })).toBeTruthy()
    expect(heading()).toBe('Alpha digest')
  })

  it('opens a clicked schedule on the stage and reopens it when the page comes back', async () => {
    const panel = await renderSchedules()

    fireEvent.click(within(panel).getByRole('button', { name: 'Charlie sweep' }))
    expect(heading()).toBe('Charlie sweep')
    expect(within(panel).getByRole('button', { current: 'page' }).textContent).toContain('Charlie sweep')

    cleanup()
    await renderSchedules()
    expect(heading()).toBe('Charlie sweep')
  })

  it('walks the rows with the arrow keys', async () => {
    const panel = await renderSchedules()
    const first = within(panel).getByRole('button', { name: 'Alpha digest' })

    first.focus()
    fireEvent.keyDown(first, { key: 'ArrowDown' })
    expect(window.document.activeElement?.textContent).toContain('Bravo report')
    fireEvent.keyDown(window.document.activeElement!, { key: 'End' })
    expect(window.document.activeElement?.textContent).toContain('Charlie sweep')
  })

  it('keeps the open schedule on the stage when the panel is folded away', async () => {
    await renderSchedules()
    act(() => setSidebarOpen(false))

    expect(screen.queryByRole('complementary', { name: en.cron.title })).toBeNull()
    expect(heading()).toBe('Alpha digest')
  })

  it('offers the empty card with its one action when there are no schedules', async () => {
    setCronJobs([])
    getCronJobs.mockResolvedValue([])
    await renderSchedules()

    expect(heading()).toBe(en.cron.title)
    expect(screen.getByText(en.cron.emptyTitleNew)).toBeTruthy()
  })
})
