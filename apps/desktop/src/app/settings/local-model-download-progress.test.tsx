import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type * as RuntimeJobs from '@/store/local-runtime-jobs'
import type * as Notifications from '@/store/notifications'
import type { LocalRuntimeJob } from '@/types/hermes'

vi.mock('@/hermes', () => ({
  pauseLocalDownload: vi.fn(),
  resumeLocalDownload: vi.fn()
}))

vi.mock('@/store/notifications', async importOriginal => ({
  ...(await importOriginal<typeof Notifications>()),
  notifyError: vi.fn()
}))

vi.mock('@/store/local-runtime-jobs', async importOriginal => ({
  ...(await importOriginal<typeof RuntimeJobs>()),
  watchLocalRuntimeJobs: vi.fn()
}))

import { pauseLocalDownload, resumeLocalDownload } from '@/hermes'

import { isDownloadPhase, LocalModelDownloadActions } from './local-model-download-progress'

function job(overrides: Partial<LocalRuntimeJob>): LocalRuntimeJob {
  return {
    detail: '',
    done_bytes: 40,
    error: null,
    job_id: 'j1',
    kind: 'model-download',
    model_id: 'm1',
    phase: 'downloading',
    status: 'running',
    target: 'Qwen3.6 27B',
    total_bytes: 100,
    ...overrides
  }
}

function renderActions(current: LocalRuntimeJob) {
  act(() => {
    render(
      <I18nProvider configClient={null}>
        <LocalModelDownloadActions job={current} />
      </I18nProvider>
    )
  })
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(pauseLocalDownload).mockResolvedValue({ ok: true, paused: true })
  vi.mocked(resumeLocalDownload).mockResolvedValue({ ok: true, resumed: true })
})

afterEach(() => {
  cleanup()
})

describe('LocalModelDownloadActions', () => {
  it('running download with can_pause:true renders Pause; clicking sends pauseLocalDownload exactly once with the job id', async () => {
    renderActions(job({ can_pause: true }))

    const pause = screen.getByRole('button', { name: /pause/i })
    fireEvent.click(pause)

    await waitFor(() => {
      expect(pauseLocalDownload).toHaveBeenCalledTimes(1)
    })
    expect(pauseLocalDownload).toHaveBeenCalledWith('j1')
    expect(resumeLocalDownload).not.toHaveBeenCalled()
  })

  it('running download without explicit can_pause shows NO control (no guessing about old backends)', () => {
    renderActions(job({}))

    expect(screen.queryByRole('button', { name: /pause/i })).toBeNull()
  })

  it('can_pause:false hides the control (server start / default assignment)', () => {
    renderActions(job({ can_pause: false }))

    expect(screen.queryByRole('button', { name: /pause/i })).toBeNull()
  })

  it('pause_requested renders a disabled pending control, not a vanished one', () => {
    renderActions(job({ can_pause: false, pause_requested: true }))

    const pending = screen.getByRole('button', { name: /pause/i })
    expect((pending as HTMLButtonElement).disabled).toBe(true)
  })

  it('paused job renders the Paused label + Resume (can_resume:true); clicking sends resumeLocalDownload with the job id', async () => {
    renderActions(job({ can_resume: true, status: 'paused' }))

    expect(screen.getByText(/paused/i)).toBeTruthy()

    const resume = screen.getByRole('button', { name: /resume/i })
    fireEvent.click(resume)

    await waitFor(() => {
      expect(resumeLocalDownload).toHaveBeenCalledTimes(1)
    })
    expect(resumeLocalDownload).toHaveBeenCalledWith('j1')
    expect(pauseLocalDownload).not.toHaveBeenCalled()
  })

  it('paused hides Resume when can_resume is not explicitly true', () => {
    renderActions(job({ status: 'paused' }))

    expect(screen.getByText(/paused/i)).toBeTruthy()
    expect(screen.queryByRole('button', { name: /resume/i })).toBeNull()
  })

  it('a paused:false response is a benign race — truth is re-kicked, no false failure toast', async () => {
    const { notifyError } = await import('@/store/notifications')
    const { watchLocalRuntimeJobs } = await import('@/store/local-runtime-jobs')

    // The job may have settled between render and click; the backend says
    // nothing paused. Refresh the authoritative snapshot; do NOT toast a
    // download failure the user never saw.
    vi.mocked(pauseLocalDownload).mockResolvedValue({ ok: true, paused: false })

    renderActions(job({ can_pause: true }))
    fireEvent.click(screen.getByRole('button', { name: /pause/i }))

    await waitFor(() => {
      expect(watchLocalRuntimeJobs).toHaveBeenCalled()
    })
    expect(notifyError).not.toHaveBeenCalled()
  })

  it('a rejected pause propagates to the error path — never a silent success', async () => {
    const { notifyError } = await import('@/store/notifications')

    vi.mocked(pauseLocalDownload).mockRejectedValue(new Error('unknown download job'))

    renderActions(job({ can_pause: true }))
    fireEvent.click(screen.getByRole('button', { name: /pause/i }))

    await waitFor(() => {
      expect(notifyError).toHaveBeenCalledTimes(1)
    })
  })

  it('no control on settled jobs', () => {
    renderActions(job({ can_pause: true, status: 'done' }))

    expect(screen.queryByRole('button', { name: /pause|resume/i })).toBeNull()
  })
})

describe('isDownloadPhase', () => {
  it('download phases cover every job kind that fetches bytes; finalization is not one', () => {
    expect(isDownloadPhase(job({ kind: 'model-download' }))).toBe(true)
    expect(isDownloadPhase(job({ kind: 'runtime-install', phase: 'downloading-runtime' }))).toBe(true)
    expect(isDownloadPhase(job({ kind: 'quickstart', phase: 'downloading' }))).toBe(true)
    expect(isDownloadPhase(job({ kind: 'quickstart', phase: 'setting-default' }))).toBe(false)
    expect(isDownloadPhase(job({ kind: 'model-activate', phase: 'loading' }))).toBe(false)
  })
})
