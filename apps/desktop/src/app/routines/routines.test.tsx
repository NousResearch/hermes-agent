// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { MemoryRouter, useLocation } from 'react-router'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { AutomationBlueprint, CronJob, ProfileInfo } from '@/hermes'
import { en } from '@/i18n/en'
import { queryClient } from '@/lib/query-client'
import { agentSidebarKey } from '@/store/agent-sidebar-preferences'
import { $cronJobs, setCronFocusJobId, setCronJobs } from '@/store/cron'
import { setInterfaceMode } from '@/store/interface-mode'
import { notify } from '@/store/notifications'
import { $profiles } from '@/store/profile'
import { $shellChoice } from '@/store/shell-variant'

import { SimplifiedGlobalNav } from '../chat/sidebar/simplified-global-nav'

import { $heldRoutine } from './routine-held-delete'
import { RUNS_BATCH } from './routine-runs'
import { routinesByAgent } from './routines-by-agent'
import { routinesCopy } from './routines-copy'
import { RoutinesPage } from './routines-page'
import { $routineSelection, $routineSheet, closeRoutineSheet, openRoutineSheet, selectRoutine } from './routines-store'

const api = vi.hoisted(() => ({
  createCronJob: vi.fn(),
  deleteCronJob: vi.fn(),
  getAutomationBlueprints: vi.fn(),
  getCronJobRuns: vi.fn(),
  getCronJobs: vi.fn(),
  getOwnerCronDeliveryTargets: vi.fn(),
  pauseCronJob: vi.fn(),
  resumeCronJob: vi.fn(),
  triggerCronJob: vi.fn(),
  updateCronJob: vi.fn()
}))

vi.mock('@/hermes', async importOriginal => ({ ...(await importOriginal<typeof HermesApi>()), ...api }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))
vi.mock('@/lib/model-options', () => ({ requestModelOptions: vi.fn(async () => ({ providers: [] })) }))

const copy = routinesCopy('en')

const brief = {
  deliver: 'local',
  enabled: true,
  id: 'job-brief',
  name: 'Daily brief',
  next_run_at: '2099-01-01T07:00:00Z',
  profile: 'default',
  prompt: 'Brief me',
  schedule: { expr: '0 7 * * *', kind: 'cron' },
  state: 'scheduled'
} as CronJob

const review = { ...brief, id: 'job-review', name: 'Weekly review' } as CronJob

const failing = {
  ...brief,
  failure_streak: 1,
  id: 'job-flights',
  last_error: 'RuntimeError: The request timed out. Retry later.',
  last_run_at: new Date(Date.now() - 86_400_000).toISOString(),
  latest_execution: { id: 'e1', status: 'failed' },
  name: 'Flight price monitor',
  profile: 'ops'
} as CronJob

const paused = { ...brief, enabled: false, id: 'job-follow', name: 'Candidate follow-ups', state: 'paused' } as CronJob

const workday: AutomationBlueprint = {
  appUrl: '',
  category: 'daily',
  command: '/blueprint workday-start',
  description: 'A weekday nudge with your agenda.',
  fields: [],
  key: 'workday-start',
  prefill: {
    deliver: 'origin',
    name: 'Workday start reminder',
    prompt: 'Give the user a brief weekday start-of-day nudge.',
    schedule: '0 9 * * 1-5',
    skills: ['google-workspace']
  },
  tags: [],
  title: 'Workday start reminder'
}

const inbox: AutomationBlueprint = {
  ...workday,
  category: 'email',
  description: 'Ping only about mail that needs attention.',
  key: 'important-mail',
  prefill: null,
  title: 'Important-mail monitor'
}

function LocationProbe() {
  const location = useLocation()

  return <span data-testid="location">{`${location.pathname}${location.search}`}</span>
}

function renderPage(path = '/cron') {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[path]}>
        <RoutinesPage onClose={vi.fn()} />
        <LocationProbe />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

const panel = () => screen.getByRole('complementary')
const heading = () => screen.getByRole('heading', { level: 1 })
const sheet = () => screen.getByRole('dialog')

async function openNewRoutine() {
  fireEvent.click(within(panel()).getByRole('button', { name: en.cron.newCron }))

  return screen.findByRole('dialog', { name: copy.createTitle })
}

/** Focus `element` as Tab does: keyboard focus is what opens a tooltip (jsdom has no :focus-visible). */
function keyboardFocus(element: HTMLElement) {
  const matches = Element.prototype.matches

  vi.spyOn(Element.prototype, 'matches').mockImplementation(function (this: Element, selector: string) {
    return selector === ':focus-visible' ? this === element : matches.call(this, selector)
  })
  act(() => element.focus())
  vi.mocked(Element.prototype.matches).mockRestore()
}

/** IntersectionObserver for the run history's "load more" edge: a test reaches it by hand. */
let reachEnd: () => void = () => undefined

beforeAll(() => {
  HTMLElement.prototype.scrollIntoView ??= () => undefined
  HTMLElement.prototype.hasPointerCapture ??= () => false
  HTMLElement.prototype.releasePointerCapture ??= () => undefined
  vi.stubGlobal(
    'IntersectionObserver',
    class {
      constructor(callback: IntersectionObserverCallback) {
        reachEnd = () => callback([{ isIntersecting: true } as IntersectionObserverEntry], this as never)
      }

      observe() {}

      disconnect() {}
    }
  )
})

beforeEach(() => {
  localStorage.clear()
  $shellChoice.set(true)
  $profiles.set([{ name: 'default' }, { name: 'ops' }] as ProfileInfo[])
  setCronJobs([brief, failing, paused])
  api.getCronJobs.mockResolvedValue([brief, failing, paused])
  api.getCronJobRuns.mockResolvedValue([])
  api.getAutomationBlueprints.mockResolvedValue({ blueprints: [workday, inbox] })
  api.getOwnerCronDeliveryTargets.mockResolvedValue([
    { home_env_var: null, home_target_set: true, id: 'local', name: 'Local (save only)' },
    { home_env_var: null, home_target_set: true, id: 'bot-chat:default', name: 'Bot Chat (default)' }
  ])
  api.pauseCronJob.mockResolvedValue(brief)
  api.resumeCronJob.mockResolvedValue(brief)
  api.deleteCronJob.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  closeRoutineSheet()
  selectRoutine(null)
  $heldRoutine.set(null)
  setCronFocusJobId(null)
  setCronJobs([])
  setInterfaceMode('advanced')
  $shellChoice.set(null)
  vi.clearAllMocks()
})

describe('the Routines page', () => {
  it('lists every routine by state in the panel, and opens the first one with nothing selected', async () => {
    renderPage()

    expect(within(panel()).getByText(`${copy.groups.attention} · 1`)).toBeTruthy()
    expect(within(panel()).getByText(`${copy.groups.scheduled} · 1`)).toBeTruthy()
    expect(within(panel()).getByText(`${copy.groups.paused} · 1`)).toBeTruthy()
    // No "All routines" row and no table of every routine: the panel is the list.
    expect(screen.queryByText('All routines')).toBeNull()
    expect(screen.queryByRole('table', { name: /routines/i })).toBeNull()
    // The first in the list (Needs attention) is on the stage, marked current in the panel.
    expect(heading().textContent).toContain('Flight price monitor')
    expect(within(panel()).getByRole('button', { name: /Flight price monitor/ }).getAttribute('aria-current')).toBe('page')
  })

  it('says there are no routines yet, with New routine, when there are none', async () => {
    setCronJobs([])
    api.getCronJobs.mockResolvedValue([])
    renderPage()

    expect(await screen.findByText(en.cron.emptyTitleNew)).toBeTruthy()
    expect(screen.getByText(copy.emptyDescription)).toBeTruthy()
  })

  it('offers Run now and Pause on a routine that is on, Run routine and Resume on a paused one', async () => {
    selectRoutine({ id: brief.id, profile: 'default' })
    const { unmount } = renderPage()

    expect(screen.getByRole('button', { name: copy.actions.runNow })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: en.cron.pause }))
    await waitFor(() => expect(api.pauseCronJob).toHaveBeenCalledWith(brief.id, { profile: 'default' }))
    unmount()

    selectRoutine({ id: paused.id, profile: 'default' })
    renderPage()

    expect(screen.getByRole('button', { name: copy.actions.runRoutine })).toBeTruthy()
    expect(screen.queryByRole('button', { name: en.cron.pause })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: en.cron.resume }))
    await waitFor(() => expect(api.resumeCronJob).toHaveBeenCalledWith(paused.id, { profile: 'default' }))
  })

  it('offers neither Pause nor Resume on a routine that has finished its runs', () => {
    const finished = { ...brief, enabled: false, id: 'job-done', name: 'One-off report', state: 'completed' } as CronJob
    setCronJobs([brief, finished])
    selectRoutine({ id: finished.id, profile: 'default' })
    renderPage()

    expect(screen.getByRole('button', { name: en.cron.edit })).toBeTruthy()
    expect(screen.queryByRole('button', { name: en.cron.pause })).toBeNull()
    expect(screen.queryByRole('button', { name: en.cron.resume })).toBeNull()
  })

  // A one-off whose run failed has no run left either, and the backend refuses to pause it; a
  // recurring routine in the same state still has its schedule.
  it('offers Pause on a failed routine only while it still has runs to come', () => {
    const failed = { ...brief, id: 'job-failed', name: 'One-off import', state: 'error' } as CronJob
    setCronJobs([
      { ...failed, schedule: { kind: 'once' } },
      { ...failed, id: 'job-stuck', schedule: { kind: 'cron' } }
    ])
    selectRoutine({ id: 'job-failed', profile: 'default' })
    const view = renderPage()
    expect(screen.queryByRole('button', { name: en.cron.pause })).toBeNull()
    view.unmount()

    selectRoutine({ id: 'job-stuck', profile: 'default' })
    renderPage()
    expect(screen.getByRole('button', { name: en.cron.pause })).toBeTruthy()
  })

  it('explains each header action in a tooltip that keyboard focus opens too', async () => {
    selectRoutine({ id: brief.id, profile: 'default' })
    renderPage()

    const tips: [string, string][] = [
      [copy.actions.runNow, copy.actions.runNowHint],
      [en.cron.pause, en.cron.pause],
      [en.cron.edit, en.cron.edit]
    ]

    for (const [name, tip] of tips) {
      keyboardFocus(screen.getByRole('button', { name }))
      expect((await screen.findByRole('tooltip')).textContent).toBe(tip)
      act(() => screen.getByRole('button', { name }).blur())
      await waitFor(() => expect(screen.queryByRole('tooltip')).toBeNull())
    }
  })

  it('says which run failed and why, and retries it', async () => {
    api.triggerCronJob.mockResolvedValue(failing)
    renderPage()

    expect(screen.getByText(copy.banner.failed.yesterday(''))).toBeTruthy()
    expect(screen.getByText(/^The request timed out\./)).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: copy.actions.retry }))
    await waitFor(() => expect(api.triggerCronJob).toHaveBeenCalledWith(failing.id, { profile: 'ops' }))
  })

  it('shows what the routine does beside when it runs and where results go', async () => {
    selectRoutine({ id: brief.id, profile: 'default' })
    renderPage()
    const summary = screen.getByRole('region', { name: 'Daily brief' })

    expect(within(summary).getByText('Brief me')).toBeTruthy()
    expect(within(summary).getByText(copy.frequency)).toBeTruthy()
    expect(within(summary).getByText(copy.lastRan)).toBeTruthy()
    expect(within(summary).getByText(copy.notRunYet)).toBeTruthy()
    expect(within(summary).getByText(copy.keptInRuns)).toBeTruthy()
  })
})

describe('the run history', () => {
  const run = (id: string, patch: Record<string, unknown> = {}) => ({
    ended_at: 1_700_000_052,
    id,
    source: 'cron',
    started_at: 1_700_000_000,
    ...patch
  })

  it('offers each run’s View in the row, and says what went wrong in place for a run with no chat', async () => {
    api.getCronJobRuns.mockResolvedValue([
      run('cron_job-brief_1'),
      { id: 'cron_output:job-brief:1', source: 'cron_output', started_at: 1_699_990_000, title: 'FAILED · Script exited 1' }
    ])
    selectRoutine({ id: brief.id, profile: 'default' })
    render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter>
          <RoutinesPage onClose={vi.fn()} onOpenSession={vi.fn()} />
        </MemoryRouter>
      </QueryClientProvider>
    )
    const table = await screen.findByRole('table')
    const [, first, second] = within(table).getAllByRole('row')

    expect(within(first).getByText('52s')).toBeTruthy()
    expect(within(first).getByRole('button', { name: copy.actions.view })).toBeTruthy()
    expect(within(second).getByText(copy.runChips.failed)).toBeTruthy()
    fireEvent.click(within(second).getByRole('button', { name: copy.actions.viewError }))
    expect(screen.getByText('Script exited 1')).toBeTruthy()
  })

  it('loads the next batch as the end of the history scrolls into view', async () => {
    api.getCronJobRuns.mockImplementation(async (_id: string, limit: number) =>
      Array.from({ length: limit }, (_, index) => run(`cron_job-brief_${index}`, { started_at: 1_700_000_000 - index }))
    )
    selectRoutine({ id: brief.id, profile: 'default' })
    renderPage()
    await screen.findByRole('table')

    expect(api.getCronJobRuns).toHaveBeenLastCalledWith(brief.id, RUNS_BATCH, { profile: 'default' })
    act(() => reachEnd())
    await waitFor(() => expect(api.getCronJobRuns).toHaveBeenLastCalledWith(brief.id, 2 * RUNS_BATCH, { profile: 'default' }))
    await waitFor(() => expect(within(screen.getByRole('table')).getAllByRole('row')).toHaveLength(2 * RUNS_BATCH + 1))
  })
})

describe('creating a routine', () => {
  it('shows nothing but where to start until a start is picked', async () => {
    renderPage()
    const created = await openNewRoutine()

    expect(within(created).getByLabelText(copy.agent)).toBeTruthy()
    expect(within(created).getByRole('radio', { name: new RegExp(copy.blank) })).toBeTruthy()
    expect(within(created).queryByLabelText(copy.instruction)).toBeNull()
    expect(within(created).queryByText(/^First run/)).toBeNull()
  })

  it('offers Create only once the new routine has a start', async () => {
    renderPage()
    const created = await openNewRoutine()
    const create = () => screen.getByRole<HTMLButtonElement>('button', { name: en.cron.createAction })

    // Nothing to create yet: a submit could only fail on fields the sheet has not shown.
    expect(create().disabled).toBe(true)

    fireEvent.click(within(created).getByRole('radio', { name: new RegExp(copy.blank) }))
    expect(create().disabled).toBe(false)
  })

  it('needs no agent choice with only one agent', async () => {
    $profiles.set([{ name: 'default' }] as ProfileInfo[])
    renderPage()
    const created = await openNewRoutine()

    expect(within(created).queryByLabelText(copy.agent)).toBeNull()
  })

  it('asks for a frequency, then says when it first runs, and creates it in its agent', async () => {
    api.createCronJob.mockResolvedValue({ ...brief, id: 'job-new', name: 'Inbox sweep' })
    renderPage()
    const created = await openNewRoutine()
    fireEvent.click(within(created).getByRole('radio', { name: new RegExp(copy.blank) }))
    fireEvent.change(within(created).getByLabelText(new RegExp(`^${copy.name}`)), { target: { value: 'Inbox sweep' } })
    fireEvent.change(within(created).getByLabelText(copy.instruction), { target: { value: 'Summarize my inbox' } })

    // No frequency yet: no read-back and no first run, and Create asks for one.
    expect(screen.queryByText(/^First run/)).toBeNull()
    await act(async () => fireEvent.click(screen.getByRole('button', { name: en.cron.createAction })))
    expect(within(created).getByText(copy.scheduleRequired)).toBeTruthy()
    expect(api.createCronJob).not.toHaveBeenCalled()

    fireEvent.click(within(created).getByRole('combobox', { name: copy.frequency }))
    fireEvent.click(await screen.findByRole('option', { name: copy.frequencies.daily }))
    expect(within(created).getByText('Every day at 9:00 AM')).toBeTruthy()
    expect(screen.getByText(/^First run: (today|tomorrow), 9:00 AM$/)).toBeTruthy()

    await act(async () => fireEvent.click(screen.getByRole('button', { name: en.cron.createAction })))

    expect(api.createCronJob).toHaveBeenCalledWith(
      { deliver: 'local', name: 'Inbox sweep', prompt: 'Summarize my inbox', schedule: '0 9 * * *' },
      { profile: 'default' }
    )
    expect(vi.mocked(notify)).toHaveBeenCalledWith(expect.objectContaining({ kind: 'success', title: en.cron.created }))
    await waitFor(() => expect(screen.queryByRole('dialog')).toBeNull())
    expect($routineSelection.get()).toEqual({ id: 'job-new', profile: 'default' })
  })

  it('starts from a template: searched by category, everything it can fill already filled', async () => {
    api.createCronJob.mockResolvedValue({ ...brief, id: 'job-workday', name: 'Workday start reminder' })
    renderPage()
    const created = await openNewRoutine()
    fireEvent.click(within(created).getByRole('radio', { name: new RegExp(copy.template) }))

    expect(await within(created).findByRole('region', { name: copy.templateCategories.daily })).toBeTruthy()
    expect(within(created).getByRole('region', { name: copy.templateCategories.email })).toBeTruthy()
    fireEvent.change(within(created).getByPlaceholderText(copy.searchTemplates), { target: { value: 'weekday' } })
    expect(within(created).queryByRole('region', { name: copy.templateCategories.email })).toBeNull()
    fireEvent.click(within(created).getByRole('button', { name: /Workday start reminder/ }))

    expect((within(created).getByLabelText(new RegExp(`^${copy.name}`)) as HTMLInputElement).value).toBe('Workday start reminder')
    expect((within(created).getByLabelText(copy.instruction) as HTMLTextAreaElement).value).toBe(
      'Give the user a brief weekday start-of-day nudge.'
    )
    expect(within(created).getByText('Monday to Friday at 9:00 AM')).toBeTruthy()
    expect(screen.getByText(/^First run: /)).toBeTruthy()

    await act(async () => fireEvent.click(screen.getByRole('button', { name: en.cron.createAction })))

    expect(api.createCronJob).toHaveBeenCalledWith(
      expect.objectContaining({ deliver: 'local', schedule: '0 9 * * 1-5', skills: ['google-workspace'] }),
      { profile: 'default' }
    )
  })

  it('opens for the agent an agent’s page asked for', async () => {
    openRoutineSheet({ mode: 'create', profile: 'ops' })
    renderPage()
    const created = await screen.findByRole('dialog', { name: copy.createTitle })

    expect(within(created).getByRole('combobox', { name: copy.agent }).textContent).toBe('ops')
  })
})

describe('editing a routine', () => {
  it('opens its sheet from the pencil, keeps the agent fixed, and saves to its owner', async () => {
    api.updateCronJob.mockResolvedValue({ ...failing, name: 'Flights' })
    renderPage()
    fireEvent.click(screen.getByRole('button', { name: en.cron.edit }))
    const edit = await screen.findByRole('dialog', { name: copy.editTitle('Flight price monitor') })

    const agent = within(edit).getByLabelText(copy.agent) as HTMLInputElement
    expect(agent.readOnly && agent.getAttribute('aria-disabled')).toBe('true')
    keyboardFocus(agent)
    // The sheet is modal, so everything outside it (the tooltip's portal too) is aria-hidden.
    expect((await screen.findByRole('tooltip', { hidden: true })).textContent).toBe(copy.agentLocked)

    fireEvent.change(within(edit).getByLabelText(new RegExp(`^${copy.name}`)), { target: { value: 'Flights' } })
    await act(async () => fireEvent.click(screen.getByRole('button', { name: copy.saveRoutine })))

    expect(api.updateCronJob).toHaveBeenCalledWith(failing.id, expect.objectContaining({ name: 'Flights' }), {
      profile: 'ops'
    })
    expect(vi.mocked(notify)).toHaveBeenCalledWith(expect.objectContaining({ title: en.cron.updated }))
  })
})

describe('deleting a routine', () => {
  async function confirmDelete(name: string) {
    fireEvent.click(screen.getByRole('button', { name: en.cron.edit }))
    const edit = await screen.findByRole('dialog', { name: copy.editTitle(name) })
    fireEvent.click(within(edit).getByRole('button', { name: copy.deleteRoutine }))

    return edit
  }

  it('asks in the footer first, and Keep it changes nothing', async () => {
    selectRoutine({ id: brief.id, profile: 'default' })
    renderPage()
    const edit = await confirmDelete('Daily brief')

    expect(within(edit).getByText(copy.deleteConfirm)).toBeTruthy()
    expect(within(edit).getByText(copy.deleteConfirmHint)).toBeTruthy()
    expect(edit.ownerDocument.activeElement).toBe(within(edit).getByRole('button', { name: copy.keepIt }))
    fireEvent.click(within(edit).getByRole('button', { name: copy.keepIt }))
    expect(within(edit).getByRole('button', { name: copy.deleteRoutine })).toBeTruthy()
    expect(api.pauseCronJob).not.toHaveBeenCalled()
  })

  it('holds it paused and out of the list, lands on the next routine, and Undo brings it back as it was', async () => {
    setCronJobs([brief, review, paused])
    api.getCronJobs.mockResolvedValue([brief, review, paused])
    selectRoutine({ id: brief.id, profile: 'default' })
    renderPage()
    const edit = await confirmDelete('Daily brief')
    await act(async () => fireEvent.click(within(edit).getByRole('button', { name: copy.actions.delete })))

    // Every hold call reaches the routine's owner: its agent, on this window's machine.
    expect(api.pauseCronJob).toHaveBeenCalledWith(brief.id, { connectionId: null, profile: 'default' })
    expect(api.deleteCronJob).not.toHaveBeenCalled()
    expect(within(panel()).queryByRole('button', { name: /Daily brief/ })).toBeNull()
    expect(heading().textContent).toContain('Weekly review')
    expect(screen.getByText(copy.deleted('Daily brief'))).toBeTruthy()

    await act(async () => fireEvent.click(screen.getByRole('button', { name: copy.actions.undo })))

    expect(api.resumeCronJob).toHaveBeenCalledWith(brief.id, { connectionId: null, profile: 'default' })
    expect(api.deleteCronJob).not.toHaveBeenCalled()
    expect(heading().textContent).toContain('Daily brief')
    expect(screen.queryByText(copy.deleted('Daily brief'))).toBeNull()
  })

  it('makes it final when the banner is dismissed, or the page left', async () => {
    selectRoutine({ id: paused.id, profile: 'default' })
    const { unmount } = renderPage()
    let edit = await confirmDelete('Candidate follow-ups')
    await act(async () => fireEvent.click(within(edit).getByRole('button', { name: copy.actions.delete })))

    // It was already paused: nothing to pause, and nothing to resume later.
    expect(api.pauseCronJob).not.toHaveBeenCalled()
    await act(async () => fireEvent.click(screen.getByRole('button', { name: en.common.close })))
    expect(api.deleteCronJob).toHaveBeenCalledWith(paused.id, { connectionId: null, profile: 'default' })

    selectRoutine({ id: brief.id, profile: 'default' })
    edit = await confirmDelete('Daily brief')
    await act(async () => fireEvent.click(within(edit).getByRole('button', { name: copy.actions.delete })))
    await act(async () => unmount())

    expect(api.deleteCronJob).toHaveBeenLastCalledWith(brief.id, { connectionId: null, profile: 'default' })
  })
})

describe('an agent’s routines for its chats', () => {
  it('groups the snapshot’s routines by agent in the page’s order, and says nothing without a scope', () => {
    const byAgent = routinesByAgent([failing, brief, { ...failing, id: 'x', name: 'Alpha' }], '\u0000all')

    expect(byAgent.get(agentSidebarKey({ connectionId: null, profile: 'default' }))?.map(job => job.id)).toEqual([
      brief.id
    ])
    expect(byAgent.get(agentSidebarKey({ connectionId: null, profile: 'ops' }))?.map(job => job.id)).toEqual([
      'x',
      'job-flights'
    ])
    expect(routinesByAgent([brief], '').size).toBe(0)
  })
})

describe('the rail’s Routines destination', () => {
  function renderRail() {
    return render(
      <MemoryRouter>
        <SimplifiedGlobalNav onNavigate={vi.fn()} />
      </MemoryRouter>
    )
  }

  it('carries a failure dot while a routine needs attention, and nothing when all is well', () => {
    $cronJobs.set([brief, failing])
    const { container, unmount } = renderRail()
    const button = screen.getByRole('button', { name: en.cron.title })

    expect(container.querySelector('[data-slot="rail-alert"]')).toBeTruthy()
    expect(button.getAttribute('aria-description')).toBe(copy.railAttention(1))
    unmount()

    $cronJobs.set([brief, paused])
    const quiet = renderRail()

    expect(quiet.container.querySelector('[data-slot="rail-alert"]')).toBeNull()
  })
})

it('keeps the sheet closed until something opens it', () => {
  expect($routineSheet.get()).toEqual({ mode: 'closed' })
})
