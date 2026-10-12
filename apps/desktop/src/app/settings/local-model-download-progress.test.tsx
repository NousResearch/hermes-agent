import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
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

vi.mock('@/store/local-runtime-jobs', (): object => ({
  watchLocalRuntimeJobs: vi.fn()
}))

import { pauseLocalDownload, resumeLocalDownload } from '@/hermes'

import {
  downloadStatusText,
  formatEta,
  formatSpeed,
  isDownloadPhase,
  LocalModelDownloadActions
} from './local-model-download-progress'

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

  it('pause and resume failures name the action that failed, not a generic download failure', async () => {
    const { notifyError } = await import('@/store/notifications')

    vi.mocked(pauseLocalDownload).mockRejectedValue(new Error('offline'))
    renderActions(job({ can_pause: true }))
    fireEvent.click(screen.getByRole('button', { name: /pause/i }))
    await waitFor(() => expect(notifyError).toHaveBeenCalledTimes(1))
    const pauseTitle: string = vi.mocked(notifyError).mock.calls[0]?.[1] as string
    cleanup()

    vi.mocked(resumeLocalDownload).mockRejectedValue(new Error('offline'))
    renderActions(job({ can_resume: true, status: 'paused' }))
    fireEvent.click(screen.getByRole('button', { name: /resume/i }))
    await waitFor(() => expect(notifyError).toHaveBeenCalledTimes(2))
    const resumeTitle: string = vi.mocked(notifyError).mock.calls[1]?.[1] as string

    expect(pauseTitle).toMatch(/pause/i)
    expect(resumeTitle).toMatch(/resume/i)
    expect(pauseTitle).not.toBe(resumeTitle)
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

// The status copy as a locale supplies it; the composer only assembles it.
const statusCopy = {
  downloadEta: (time: string) => `~${time} left`,
  downloadEtaHours: (hours: number, minutes: number) => (minutes ? `${hours} h ${minutes} min` : `${hours} h`),
  downloadEtaMinutes: (count: number) => `${count} min`,
  downloadEtaSeconds: (count: number) => `${count} sec`,
  downloadPausedLabel: 'Paused',
  downloadProgress: (done: string, total: string) => `${done} of ${total}`,
  downloadSpeed: (rate: string) => `${rate}`,
  downloadStatusRunning: 'Downloading'
}

describe('downloadStatusText', () => {
  it('composes state, bytes, speed and ETA while bytes move', () => {
    const text = downloadStatusText(
      job({ bytes_per_sec: 24 * (1 << 20), done_bytes: 1 << 30, eta_seconds: 120, total_bytes: 4 * (1 << 30) }),
      statusCopy
    )

    expect(text).toBe('Downloading · 1.0 GB of 4.0 GB · 24 MB/s · ~2 min left')
  })

  it('drops speed and ETA rather than guessing when the backend reports none', () => {
    // 4 * (1 << 30): a plain shift overflows 32-bit and would silently read 0.
    const text = downloadStatusText(job({ done_bytes: 1 << 30, total_bytes: 4 * (1 << 30) }), statusCopy)

    expect(text).toBe('Downloading · 1.0 GB of 4.0 GB')
  })

  it('a parked job keeps its frozen counter but shows no live rate', () => {
    const text = downloadStatusText(
      job({
        bytes_per_sec: 24 * (1 << 20),
        done_bytes: 1 << 30,
        eta_seconds: 120,
        status: 'paused',
        total_bytes: 4 * (1 << 30)
      }),
      statusCopy
    )

    expect(text).toBe('Paused · 1.0 GB of 4.0 GB')
  })

  it('a non-download phase shows the phase detail, not a byte counter', () => {
    const text = downloadStatusText(
      job({ detail: 'Starting the local server', kind: 'quickstart', phase: 'starting-server' }),
      statusCopy
    )

    expect(text).toBe('Starting the local server')
  })
})

describe('speed and ETA formatting', () => {
  it('reports nothing for an unknown or stalled rate', () => {
    expect(formatSpeed(undefined)).toBe('')
    expect(formatSpeed(0)).toBe('')
    expect(formatEta(null, statusCopy)).toBe('')
    expect(formatEta(0.4, statusCopy)).toBe('')
  })

  it('scales the rate unit and the ETA magnitude', () => {
    expect(formatSpeed(24 * (1 << 20))).toBe('24 MB/s')
    expect(formatSpeed(512 * (1 << 10))).toBe('0.5 MB/s')
    expect(formatSpeed(2 * (1 << 30))).toBe('2.0 GB/s')
    expect(formatEta(45, statusCopy)).toBe('45 sec')
    expect(formatEta(120, statusCopy)).toBe('2 min')
    expect(formatEta(3_900, statusCopy)).toBe('1 h 5 min')
    expect(formatEta(7_200, statusCopy)).toBe('2 h')
  })

  it('spells the ETA in the locale’s words, not a hardcoded unit', () => {
    const ja = {
      downloadEtaHours: (hours: number, minutes: number) => `${hours}時間${minutes ? `${minutes}分` : ''}`,
      downloadEtaMinutes: (count: number) => `${count}分`,
      downloadEtaSeconds: (count: number) => `${count}秒`
    }

    expect(formatEta(45, ja)).toBe('45秒')
    expect(formatEta(3_900, ja)).toBe('1時間5分')
  })
})
