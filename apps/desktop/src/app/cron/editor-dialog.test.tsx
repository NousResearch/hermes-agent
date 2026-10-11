// @vitest-environment jsdom
import { QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import type { CronJob } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { setCronJobs } from '@/store/cron'

const createCronJob = vi.fn()
const getCronJobs = vi.fn()

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  createCronJob: (body: unknown) => createCronJob(body),
  getAutomationBlueprints: async () => ({ blueprints: [] }),
  getCronDeliveryTargets: async () => [{ home_env_var: null, home_target_set: true, id: 'local', name: 'Local' }],
  getCronJobRuns: async () => [],
  getCronJobs: (profile?: string) => getCronJobs(profile)
}))

vi.mock('@/lib/model-options', () => ({ requestModelOptions: async () => ({ providers: [] }) }))

vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

const created = {
  deliver: 'local',
  enabled: true,
  id: 'job-new',
  name: 'Inbox digest',
  prompt: 'Summarize my inbox',
  schedule: { expr: '30 17 * * *', kind: 'cron' },
  state: 'scheduled'
} as CronJob

async function renderCron(props: { onTestPrompt?: (prompt: string) => void } = {}) {
  const { CronView } = await import('./index')

  await act(async () => {
    render(
      <QueryClientProvider client={queryClient}>
        <CronView onClose={vi.fn()} {...props} />
      </QueryClientProvider>
    )
  })
}

async function openNewJobEditor() {
  await act(async () => {
    screen.getByRole('button', { name: 'New cron' }).click()
  })

  return screen.findByLabelText('Prompt')
}

beforeAll(() => {
  HTMLElement.prototype.scrollIntoView ??= () => undefined
  vi.stubGlobal('CSS', { escape: (value: string) => value })
})

beforeEach(() => {
  setCronJobs([])
  getCronJobs.mockResolvedValue([])
  createCronJob.mockResolvedValue(created)
})

afterEach(() => {
  cleanup()
  queryClient.clear()
  setCronJobs([])
  vi.clearAllMocks()
})

describe('cron editor dialog', () => {
  it('saves the time the user picks for a preset instead of the 9:00 AM seed', async () => {
    await renderCron()
    fireEvent.change(await openNewJobEditor(), { target: { value: 'Summarize my inbox' } })
    fireEvent.change(screen.getByLabelText('Time'), { target: { value: '17:30' } })

    await act(async () => {
      screen.getByRole('button', { name: 'Create cron' }).click()
    })

    await waitFor(() => expect(createCronJob).toHaveBeenCalled())
    expect(createCronJob.mock.calls[0][0]).toMatchObject({ prompt: 'Summarize my inbox', schedule: '30 17 * * *' })
  })

  it('hands the prompt to a new chat and reopens the unsaved draft on the next visit', async () => {
    const onTestPrompt = vi.fn()

    await renderCron({ onTestPrompt })
    fireEvent.change(await openNewJobEditor(), { target: { value: 'Summarize my inbox' } })
    fireEvent.change(screen.getByLabelText('Time'), { target: { value: '07:15' } })

    await act(async () => {
      screen.getByRole('button', { name: 'Test in new chat' }).click()
    })

    expect(onTestPrompt).toHaveBeenCalledWith('Summarize my inbox')
    expect(createCronJob).not.toHaveBeenCalled()

    // The shell navigates away (unmounting the overlay); coming back restores the form.
    cleanup()
    await renderCron({ onTestPrompt })

    expect(((await screen.findByLabelText('Prompt')) as HTMLTextAreaElement).value).toBe('Summarize my inbox')
    expect((screen.getByLabelText('Time') as HTMLInputElement).value).toBe('07:15')
  })
})
