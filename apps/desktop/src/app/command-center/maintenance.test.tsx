import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { ActionResponse, ActionStatusResponse } from '@/types/hermes'

import { MaintenancePanel } from './maintenance'

// Mirrors the constants in maintenance.tsx — the poll cadence and the ceiling
// after which the panel stops tailing a still-running action.
const ACTION_POLL_MS = 1200
const ACTION_POLL_LIMIT = 240

// Every spawn-backed op button is gated on the last observed action status.
const OP_BUTTONS = ['Run doctor', 'Security audit', 'Create backup', 'Run now']

const getActionStatus = vi.fn<(name: string, lines?: number) => Promise<ActionStatusResponse>>()
const getCuratorStatus = vi.fn()
const getMemoryStatus = vi.fn()
const startAction = vi.fn<() => Promise<ActionResponse>>()

vi.mock('@/hermes', () => ({
  getActionStatus: (name: string, lines?: number) => getActionStatus(name, lines),
  getCuratorStatus: () => getCuratorStatus(),
  getMemoryStatus: () => getMemoryStatus(),
  resetMemory: () => Promise.resolve({ deleted: [] }),
  runBackup: () => startAction(),
  runCurator: () => startAction(),
  runDebugShare: () => Promise.resolve({ urls: {} }),
  runDoctor: () => startAction(),
  runSecurityAudit: () => startAction(),
  setCuratorPaused: () => Promise.resolve()
}))

vi.mock('@/store/activity', () => ({ upsertDesktopActionTask: () => {} }))
vi.mock('@/store/notifications', () => ({ notify: () => {}, notifyError: () => {} }))

const TAILED_LINE = 'doctor: checking providers'

// The "we stopped following this action" notices, keyed off the en locale.
const DEGRADED_NOTICE = /the status endpoint stopped responding/i
const EXHAUSTED_NOTICE = /still running when the log tail timed out/i

function actionStatus(overrides: Partial<ActionStatusResponse> = {}): ActionStatusResponse {
  return { exit_code: null, lines: [TAILED_LINE], name: 'doctor', pid: 4242, running: true, ...overrides }
}

function opButton(name: string): HTMLButtonElement {
  return screen.getByRole('button', { name }) as HTMLButtonElement
}

function opButtonsDisabled(): boolean[] {
  return OP_BUTTONS.map(name => opButton(name).disabled)
}

async function advance(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms)
  })
}

/** Press the explicit re-verify affordance offered alongside the stop notice. */
async function clickRecheck() {
  await act(async () => {
    fireEvent.click(opButton('Check status'))
  })
}

/** Render the panel and press "Run doctor", settling on the first polled status. */
async function launchDoctor() {
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <MaintenancePanel />
    </I18nProvider>
  )

  await waitFor(() => expect(opButton('Run now')).toBeTruthy())

  fireEvent.click(opButton('Run doctor'))

  await waitFor(() => expect(opButtonsDisabled()).toEqual([true, true, true, true]))
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  startAction.mockResolvedValue({ name: 'doctor', ok: true, pid: 4242 })
  getCuratorStatus.mockResolvedValue({
    archive_after_days: null,
    enabled: true,
    interval_hours: 6,
    last_run_at: null,
    min_idle_hours: null,
    paused: false,
    stale_after_days: null
  })
  getMemoryStatus.mockResolvedValue({ active: '', builtin_files: { memory: 0, user: 0 }, providers: [] })
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.resetAllMocks()
})

describe('MaintenancePanel action gate', () => {
  it('keeps the gate closed when the status endpoint starts failing mid-tail, and releases it only on a verified completion', async () => {
    getActionStatus.mockResolvedValueOnce(actionStatus()).mockRejectedValueOnce(new Error('status endpoint unreachable'))

    await launchDoctor()

    await advance(ACTION_POLL_MS)

    // The tail stopped, but the backend never said the action finished, so the
    // gate must stay closed and the stored status must still read `running`.
    await waitFor(() => expect(screen.getByText(DEGRADED_NOTICE)).toBeTruthy())
    expect(opButtonsDisabled()).toEqual([true, true, true, true])
    expect(screen.getByText(TAILED_LINE)).toBeTruthy()
    expect(screen.getByText('Running...')).toBeTruthy()

    // Re-checking while the action is genuinely still running must NOT release
    // the gate — the backend value stays authoritative.
    getActionStatus.mockResolvedValueOnce(actionStatus())
    await clickRecheck()

    expect(opButtonsDisabled()).toEqual([true, true, true, true])
    expect(screen.getByText(EXHAUSTED_NOTICE)).toBeTruthy()

    // Only once the backend itself reports the action done does the gate open.
    getActionStatus.mockResolvedValueOnce(actionStatus({ exit_code: 0, lines: [TAILED_LINE, 'doctor: ok'], running: false }))
    await clickRecheck()

    await waitFor(() => expect(opButtonsDisabled()).toEqual([false, false, false, false]))
    expect(screen.queryByText(EXHAUSTED_NOTICE)).toBeNull()
    expect(screen.queryByText(DEGRADED_NOTICE)).toBeNull()
    expect(screen.getByText(/doctor: ok/)).toBeTruthy()
  })

  it('keeps the gate closed when the poll ceiling is reached on a long-running action, and releases it only on a verified completion', async () => {
    getActionStatus.mockResolvedValue(actionStatus())

    await launchDoctor()

    // One poll short of the ceiling the action is still being tailed, so no
    // notice is shown yet.
    await advance((ACTION_POLL_LIMIT - 2) * ACTION_POLL_MS)

    expect(opButtonsDisabled()).toEqual([true, true, true, true])
    expect(screen.queryByText(EXHAUSTED_NOTICE)).toBeNull()

    await advance(2 * ACTION_POLL_MS)

    await waitFor(() => expect(screen.getByText(EXHAUSTED_NOTICE)).toBeTruthy())
    expect(opButtonsDisabled()).toEqual([true, true, true, true])
    expect(screen.getByText(TAILED_LINE)).toBeTruthy()

    // Exhausting the ceiling stops the tail — it must not keep polling.
    const pollsAtCeiling = getActionStatus.mock.calls.length
    await advance(10 * ACTION_POLL_MS)
    expect(getActionStatus).toHaveBeenCalledTimes(pollsAtCeiling)

    getActionStatus.mockResolvedValue(actionStatus({ exit_code: 0, lines: [TAILED_LINE, 'doctor: ok'], running: false }))
    await clickRecheck()

    await waitFor(() => expect(opButtonsDisabled()).toEqual([false, false, false, false]))
    expect(screen.queryByText(EXHAUSTED_NOTICE)).toBeNull()
  })

  it('leaves the normal completion path untouched', async () => {
    getActionStatus
      .mockResolvedValueOnce(actionStatus())
      .mockResolvedValue(actionStatus({ exit_code: 0, lines: [TAILED_LINE, 'doctor: ok'], running: false }))

    await launchDoctor()

    await advance(ACTION_POLL_MS)

    await waitFor(() => expect(opButtonsDisabled()).toEqual([false, false, false, false]))
    expect(screen.getByText(/doctor: ok/)).toBeTruthy()

    // A finished action stops the tail: no further polls are scheduled.
    expect(getActionStatus).toHaveBeenCalledTimes(2)

    await advance(10 * ACTION_POLL_MS)

    expect(getActionStatus).toHaveBeenCalledTimes(2)
  })
})
