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
import { $cronJobs, setCronJobs } from '@/store/cron'
import { $interfaceMode } from '@/store/interface-mode'
import { $profiles } from '@/store/profile'
import { $shellChoice } from '@/store/shell-variant'

import { SimplifiedGlobalNav } from '../chat/sidebar/simplified-global-nav'

import { AgentRoutines } from './agent-routines'
import { AgentRoutinesChip, agentRoutineSummaries } from './agent-routines-chip'
import { routinesCopy } from './routines-copy'
import { RoutinesPage } from './routines-page'
import { $routineOverviewQuery, $routineRename, closeRoutineEditor, selectRoutine } from './routines-store'

const api = vi.hoisted(() => ({
  createCronJob: vi.fn(),
  getAgentRoutines: vi.fn(),
  getAutomationBlueprints: vi.fn(),
  getCronJobRuns: vi.fn(),
  getCronJobs: vi.fn(),
  getOwnerCronDeliveryTargets: vi.fn(),
  instantiateAutomationBlueprint: vi.fn(),
  updateCronJob: vi.fn()
}))

vi.mock('@/hermes', async importOriginal => ({ ...(await importOriginal<typeof HermesApi>()), ...api }))
vi.mock('../agent-settings/data', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  getAgentRoutines: () => api.getAgentRoutines()
}))
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

const failing = {
  ...brief,
  failure_streak: 3,
  id: 'job-flights',
  last_error: 'RuntimeError: The request timed out. Retry later.',
  latest_execution: { id: 'e1', status: 'failed' },
  name: 'Flight price monitor',
  profile: 'ops'
} as CronJob

const paused = { ...brief, enabled: false, id: 'job-follow', name: 'Candidate follow-ups', state: 'paused' } as CronJob

const blueprint: AutomationBlueprint = {
  appUrl: '',
  category: 'daily',
  command: '/blueprint morning-brief',
  description: 'A short daily briefing.',
  fields: [
    { default: '08:00', help: '', label: 'What time?', name: 'time', optional: false, options: [], type: 'time' },
    { default: 'origin', help: '', label: 'Deliver', name: 'deliver', optional: false, options: [], type: 'text' }
  ],
  key: 'morning-brief',
  tags: [],
  title: 'Morning briefing'
}

function LocationProbe() {
  const location = useLocation()

  return <span data-testid="location">{`${location.pathname}${location.search}`}</span>
}

function renderPage() {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <RoutinesPage onClose={vi.fn()} />
        <LocationProbe />
      </MemoryRouter>
    </QueryClientProvider>
  )
}

async function openNewRoutine() {
  fireEvent.click(screen.getAllByRole('button', { name: en.cron.newCron })[0])
  await screen.findByText(copy.pickDescription)
}

beforeAll(() => {
  HTMLElement.prototype.scrollIntoView ??= () => undefined
  HTMLElement.prototype.hasPointerCapture ??= () => false
})

beforeEach(() => {
  $shellChoice.set(true)
  $profiles.set([{ name: 'default' }, { name: 'ops' }] as ProfileInfo[])
  setCronJobs([brief, failing, paused])
  api.getCronJobs.mockResolvedValue([brief, failing, paused])
  api.getCronJobRuns.mockResolvedValue([])
  api.getAutomationBlueprints.mockResolvedValue({ blueprints: [blueprint] })
  api.getOwnerCronDeliveryTargets.mockResolvedValue([
    { home_env_var: null, home_target_set: true, id: 'local', name: 'Local (save only)' },
    { home_env_var: null, home_target_set: true, id: 'bot-chat:default', name: 'Bot Chat (default)' }
  ])
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  closeRoutineEditor()
  selectRoutine(null)
  $routineRename.set(null)
  $routineOverviewQuery.set('')
  setCronJobs([])
  $interfaceMode.set('advanced')
  $shellChoice.set(null)
  vi.clearAllMocks()
})

describe('the Routines page', () => {
  it('groups every agent’s routines by status in the panel, with the problem on top of the overview', async () => {
    renderPage()
    const panel = await screen.findByRole('complementary')

    expect(within(panel).getByText(`${copy.groups.attention} · 1`)).toBeTruthy()
    expect(within(panel).getByText(`${copy.groups.scheduled} · 1`)).toBeTruthy()
    expect(within(panel).getByText(`${copy.groups.paused} · 1`)).toBeTruthy()
    expect(within(panel).queryByText(new RegExp(copy.groups.finished))).toBeNull()
    // The banner names the problem in plain words, with the first sentence of the error and Retry.
    expect(screen.getByText(copy.banner.failedStreak('Flight price monitor', 3))).toBeTruthy()
    expect(screen.getByText(/^The request timed out\. ·/)).toBeTruthy()
    expect(screen.getByRole('button', { name: copy.actions.retry })).toBeTruthy()
  })

  it('creates a blank routine in the chosen agent through the one editor', async () => {
    const created = { ...brief, id: 'job-new', name: 'Morning brief' }
    api.createCronJob.mockResolvedValue(created)
    renderPage()
    await openNewRoutine()
    fireEvent.click(screen.getByRole('button', { name: copy.startBlank }))
    fireEvent.change(await screen.findByLabelText(copy.name), { target: { value: 'Morning brief' } })
    fireEvent.change(screen.getByLabelText(copy.instruction), { target: { value: 'Summarize my inbox' } })
    // The read-back says the default schedule in words.
    expect(screen.getByText('Every day at 9:00 AM')).toBeTruthy()

    await act(async () => fireEvent.click(screen.getByRole('button', { name: en.cron.createAction })))

    expect(api.createCronJob).toHaveBeenCalledWith(
      { deliver: 'local', name: 'Morning brief', prompt: 'Summarize my inbox', schedule: '0 9 * * *' },
      { profile: 'default' }
    )
  })

  it('keeps the rail page to quick actions: editing goes to the routine in its agent’s settings', async () => {
    selectRoutine({ id: failing.id, profile: 'ops' })
    renderPage()
    await screen.findAllByText(copy.whatItDoes)

    expect(screen.queryByRole('button', { name: copy.actions.change })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: copy.editInAgent('ops') }))

    const target = new URLSearchParams(screen.getByTestId('location').textContent!.split('?')[1])

    expect(screen.getByTestId('location').textContent).toMatch(/^\/settings\?/)
    expect(Object.fromEntries(target)).toMatchObject({
      agentProfile: 'ops',
      edit: '1',
      panel: 'settings',
      routine: 'job-flights',
      tab: 'agent:schedules'
    })
  })

  it('lists a refused run under Runs as soon as the routine records it, in one plain sentence', async () => {
    const raw =
      "GatewayUnavailableError: Gateway inaccessible: profile_parked (profile(s) 'ops' still run their own gateway; fold them with `hermes gateway migrate --multiplex`)"

    const quiet = { ...brief, id: 'job-watch', name: 'Competitor news watch', profile: 'ops' } as CronJob
    setCronJobs([brief, quiet, paused])
    api.getCronJobs.mockResolvedValue([brief, quiet, paused])
    selectRoutine({ id: quiet.id, profile: 'ops' })
    renderPage()
    await screen.findByText(copy.noRuns)

    // Run now: the scheduler refuses the attempt up front and writes its output, with no cron.changed.
    api.getCronJobRuns.mockResolvedValue([
      { id: 'cron_output:job-watch:1', source: 'cron_output', started_at: 1_700_000_000, title: `FAILED · ${raw}` }
    ])
    act(() =>
      setCronJobs([
        brief,
        { ...quiet, last_error: raw, latest_execution: { error: raw, id: 'attempt-1', status: 'failed' } },
        paused
      ])
    )

    const plain = copy.banner.agentNotRunning('ops')
    // The run is listed with its status, and the banner says what happened in words.
    await waitFor(() => expect(screen.queryByText(copy.noRuns)).toBeNull())
    expect(screen.getByText(plain)).toBeTruthy()
    expect(screen.getByText(new RegExp(`^${plain} ·`))).toBeTruthy()
    // The backend's own text stays behind Details, never in the banner's sentence.
    const details = screen.getByText(copy.banner.details).closest('details')!
    expect(details.open).toBe(false)
    expect(within(details).getByText(raw)).toBeTruthy()
    expect(screen.getAllByText(/migrate --multiplex/)).toHaveLength(1)
  })

  it('renames a routine as a rail quick action, in its own agent', async () => {
    api.updateCronJob.mockResolvedValue({ ...brief, name: 'Morning brief' })
    $routineRename.set(brief)
    renderPage()
    const name = await screen.findByLabelText(copy.name)

    fireEvent.change(name, { target: { value: 'Morning brief' } })
    await act(async () => fireEvent.click(screen.getByRole('button', { name: copy.rename })))

    expect(api.updateCronJob).toHaveBeenCalledWith('job-brief', { name: 'Morning brief' }, { profile: 'default' })
  })

  it('starts from a ready-made routine: its own fields, seeded, created by the backend', async () => {
    api.instantiateAutomationBlueprint.mockResolvedValue({ ...brief, id: 'job-bp', name: 'Morning briefing' })
    renderPage()
    await openNewRoutine()
    fireEvent.click(await screen.findByRole('button', { name: `${copy.use}: Morning briefing` }))

    const time = (await screen.findByLabelText('What time?')) as HTMLInputElement

    expect(time.value).toBe('08:00')
    expect(screen.getByRole('heading', { name: 'Morning briefing' })).toBeTruthy()
    await act(async () => fireEvent.click(screen.getByRole('button', { name: en.cron.createAction })))

    expect(api.instantiateAutomationBlueprint).toHaveBeenCalledWith(
      { blueprint: 'morning-brief', values: { deliver: 'local', time: '08:00' } },
      'default',
      null
    )
    expect(api.createCronJob).not.toHaveBeenCalled()
  })

})

describe('Agent › Settings › Routines', () => {
  function renderAgent(search: string) {
    return render(
      <QueryClientProvider client={queryClient}>
        <MemoryRouter initialEntries={[`/settings${search}`]}>
          <AgentRoutines owner={{ profile: 'ops' }} />
        </MemoryRouter>
      </QueryClientProvider>
    )
  }

  it('opens the linked routine in the one editor and saves it in that agent', async () => {
    api.getAgentRoutines.mockResolvedValue([failing])
    api.updateCronJob.mockResolvedValue({ ...failing, name: 'Fares' })
    renderAgent('?agentProfile=ops&panel=settings&tab=agent%3Aschedules&routine=job-flights&edit=1')
    const name = (await screen.findByLabelText(copy.name)) as HTMLInputElement

    expect(name.value).toBe('Flight price monitor')
    fireEvent.change(name, { target: { value: 'Fares' } })
    await act(async () => fireEvent.click(screen.getByRole('button', { name: copy.saveChanges })))

    expect(api.updateCronJob).toHaveBeenCalledWith(
      'job-flights',
      expect.objectContaining({ name: 'Fares', schedule: '0 7 * * *' }),
      { connectionId: null, profile: 'ops' }
    )
    expect(api.updateCronJob.mock.calls[0][1]).not.toHaveProperty('repeat')
  })

  it('shows the cron expression and routine id only in developer mode', async () => {
    api.getAgentRoutines.mockResolvedValue([{ ...brief, profile: 'ops' }])
    $interfaceMode.set('simple')
    const { unmount } = renderAgent('?routine=job-brief')
    await screen.findAllByText(copy.whatItDoes)

    expect(screen.queryByText('job-brief')).toBeNull()
    expect(screen.queryByText(copy.expression)).toBeNull()
    expect(screen.getAllByText(/Every day at 7:00 AM/).length).toBeGreaterThan(0)
    unmount()

    $interfaceMode.set('advanced')
    renderAgent('?routine=job-brief')
    await screen.findAllByText(copy.whatItDoes)

    expect(screen.getByText('job-brief')).toBeTruthy()
    expect(screen.getByText('0 7 * * *')).toBeTruthy()
  })
})

describe('the chat sidebar’s Routines chip', () => {
  it('counts each agent’s routines from the snapshot, flags the failing ones, and says nothing without a scope', () => {
    const summaries = agentRoutineSummaries([brief, failing, { ...failing, id: 'x', latest_execution: null, last_error: null }], '\u0000all')

    expect(summaries.get(agentSidebarKey({ connectionId: null, profile: 'default' }))).toEqual({ count: 1, failing: [] })
    expect(summaries.get(agentSidebarKey({ connectionId: null, profile: 'ops' }))?.count).toBe(2)
    expect(summaries.get(agentSidebarKey({ connectionId: null, profile: 'ops' }))?.failing.map(job => job.id)).toEqual([
      'job-flights'
    ])
    expect(agentRoutineSummaries([brief], '').size).toBe(0)
  })

  it('opens Routines on that agent’s routines, and is absent for an agent with none', () => {
    const { container, unmount } = render(
      <MemoryRouter>
        <AgentRoutinesChip agentLabel="ops" summary={{ count: 2, failing: [failing] }} />
        <LocationProbe />
      </MemoryRouter>
    )

    expect(container.querySelector('.portal-agent-routines-dot')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: new RegExp(copy.chipTooltip(2, 'ops')) }))
    expect(screen.getByTestId('location').textContent).toBe('/cron')
    expect($routineOverviewQuery.get()).toBe('ops')
    unmount()

    const empty = render(
      <MemoryRouter>
        <AgentRoutinesChip agentLabel="ops" summary={undefined} />
      </MemoryRouter>
    )

    expect(empty.container.querySelector('.portal-agent-routines')).toBeNull()
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
