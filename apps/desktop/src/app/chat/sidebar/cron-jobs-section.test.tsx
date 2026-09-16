import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import type * as HermesModule from '@/hermes'

const { run } = vi.hoisted(() => ({
  run: { id: 'run-1', last_active: 1, profile: 'worker', connection_id: 'source-b' } as SessionInfo
}))

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesModule>()),
  getCronJobRuns: vi.fn().mockResolvedValue([run])
}))
vi.mock('@/components/pane-shell/pane-visibility', () => ({ usePaneVisible: () => true }))
vi.mock('@/i18n', () => ({ useI18n: () => ({ t: { cron: { loading: 'Loading', noRuns: 'No runs' } } }) }))

import { CronJobSidebarRuns } from './cron-jobs-section'

afterEach(cleanup)

it('hands the clicked cron run row and its exact owner to session opening', async () => {
  const onOpenRun = vi.fn()
  render(<CronJobSidebarRuns jobId="job-1" onOpenRun={onOpenRun} />)

  await waitFor(() => expect(screen.getByRole('button')).toBeTruthy())
  fireEvent.click(screen.getByRole('button'))

  expect(onOpenRun).toHaveBeenCalledWith('run-1', run)
})
