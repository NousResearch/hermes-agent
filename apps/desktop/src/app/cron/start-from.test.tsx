// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { AutomationBlueprint, CronJob } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { setCronJobs } from '@/store/cron'

const hermes = vi.hoisted(() => ({
  createCronJob: vi.fn(),
  getAutomationBlueprints: vi.fn(),
  getCronJobs: vi.fn(),
  renderAutomationBlueprint: vi.fn()
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  createCronJob: hermes.createCronJob,
  getAutomationBlueprints: hermes.getAutomationBlueprints,
  getCronDeliveryTargets: async () => [{ home_env_var: null, home_target_set: true, id: 'local', name: 'Local' }],
  getCronJobRuns: async () => [],
  getCronJobs: hermes.getCronJobs,
  renderAutomationBlueprint: hermes.renderAutomationBlueprint
}))

vi.mock('@/lib/model-options', () => ({ requestModelOptions: async () => ({ providers: [] }) }))

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

const sourceJob = {
  deliver: 'local',
  enabled: true,
  id: 'job-1',
  name: 'Inbox digest',
  prompt: 'Summarize my unread mail',
  schedule: { expr: '30 7 * * 1-5', kind: 'cron' },
  skills: ['google-workspace'],
  state: 'scheduled',
  workdir: '/work/inbox'
} as CronJob

const morningBrief: AutomationBlueprint = {
  appUrl: 'hermes://blueprint/morning-brief',
  category: 'daily',
  command: '/blueprint morning-brief',
  description: 'A short daily briefing.',
  fields: [
    {
      default: '08:00',
      help: '',
      label: 'What time?',
      name: 'time',
      optional: false,
      options: [],
      strict: true,
      type: 'time'
    }
  ],
  key: 'morning-brief',
  plugin: null,
  source: 'builtin',
  tags: [],
  title: 'Morning briefing'
}

async function renderCron() {
  const { CronView } = await import('./index')

  await act(async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <CronView onClose={vi.fn()} />
      </QueryClientProvider>
    )
  })
}

beforeAll(() => {
  HTMLElement.prototype.scrollIntoView ??= () => undefined
  vi.stubGlobal('CSS', { escape: (value: string) => value })
})

beforeEach(() => {
  setCronJobs([sourceJob])
  hermes.getCronJobs.mockResolvedValue([sourceJob])
  hermes.getAutomationBlueprints.mockResolvedValue({ blueprints: [morningBrief] })
  hermes.createCronJob.mockResolvedValue({ ...sourceJob, id: 'job-new' })
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  setCronJobs([])
  vi.clearAllMocks()
})

describe('cron editor "Start from"', () => {
  it('copies a job into the editor and creates it with the settings the editor does not show', async () => {
    await renderCron()
    await act(async () => screen.getByRole('button', { name: 'New cron' }).click())

    fireEvent.click(await screen.findByRole('combobox', { name: 'Start from' }))
    fireEvent.click(await screen.findByRole('option', { name: 'Inbox digest' }))

    expect((screen.getByLabelText('Prompt') as HTMLTextAreaElement).value).toBe('Summarize my unread mail')
    expect((screen.getByLabelText(/^Name/) as HTMLInputElement).value).toBe('Inbox digest (copy)')
    expect((screen.getByLabelText('Time') as HTMLInputElement).value).toBe('07:30')

    await act(async () => screen.getByRole('button', { name: 'Create cron' }).click())

    await waitFor(() => expect(hermes.createCronJob).toHaveBeenCalled())
    expect(hermes.createCronJob.mock.calls[0][0]).toMatchObject({
      name: 'Inbox digest (copy)',
      prompt: 'Summarize my unread mail',
      schedule: '30 7 * * 1-5',
      skills: ['google-workspace'],
      workdir: '/work/inbox'
    })
  })

  it('customizes a recipe, as filled, into the full editor and keeps its skills', async () => {
    hermes.renderAutomationBlueprint.mockResolvedValue({
      deliver: 'origin',
      name: 'Morning briefing',
      prompt: 'Produce a concise morning briefing',
      schedule: '45 6 * * *',
      skills: ['google-workspace']
    })

    await renderCron()
    await act(async () => (await screen.findByText('Morning briefing')).closest('button')?.click())
    fireEvent.change(await screen.findByLabelText('What time?'), { target: { value: '06:45' } })
    await act(async () => screen.getByRole('button', { name: 'Customize prompt' }).click())

    expect(hermes.renderAutomationBlueprint).toHaveBeenCalledWith(
      { blueprint: 'morning-brief', values: { time: '06:45' } },
      expect.any(String)
    )
    const prompt = (await screen.findByLabelText('Prompt')) as HTMLTextAreaElement

    expect(prompt.value).toBe('Produce a concise morning briefing')
    expect((screen.getByLabelText('Time') as HTMLInputElement).value).toBe('06:45')
    expect(screen.getByText('Runs with skills: google-workspace')).toBeTruthy()

    fireEvent.change(prompt, { target: { value: 'Produce a concise morning briefing, weather first' } })
    await act(async () => screen.getByRole('button', { name: 'Create cron' }).click())

    await waitFor(() => expect(hermes.createCronJob).toHaveBeenCalled())
    expect(hermes.createCronJob.mock.calls[0][0]).toMatchObject({
      deliver: 'local',
      prompt: 'Produce a concise morning briefing, weather first',
      schedule: '45 6 * * *',
      skills: ['google-workspace']
    })
  })
})
