import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { getActionStatus, runDebugShare, setApiRequestConnection, setApiRequestProfile } from '@/hermes'
import { $desktopActionTasks } from '@/store/activity'
import { $confirmRequest, settleConfirm } from '@/store/confirm'

import { MaintenancePanel } from './maintenance'

// The backend spawns each op under one fixed action name ('doctor', 'security-audit',
// 'backup', 'curator-run'), a re-spawn replaces the record under that name, and
// /api/actions/<name>/status reports the latest run. The fake keeps that contract.
const runs: Record<string, number> = {}
const running: Record<string, boolean> = {}
let nextRunStaysRunning = false

function spawn(name: string) {
  runs[name] = (runs[name] ?? 0) + 1
  running[name] = nextRunStaysRunning

  return Promise.resolve({ name, ok: true, pid: 1000 + runs[name] })
}

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getActionStatus: vi.fn(async (name: string) => ({
    exit_code: running[name] ? null : 0,
    lines: [`${name} run ${runs[name]} output`],
    name,
    pid: 1000 + runs[name],
    running: running[name]
  })),
  getCuratorStatus: vi.fn(() => new Promise(() => {})),
  getMemoryStatus: vi.fn(() => new Promise(() => {})),
  runDebugShare: vi.fn(async () => ({ auto_delete_seconds: 21600, failures: [], ok: true, redacted: true, urls: {} })),
  runDoctor: vi.fn(() => spawn('doctor')),
  runSecurityAudit: vi.fn(() => spawn('security-audit'))
}))

beforeEach(() => {
  for (const key of Object.keys(runs)) {
    delete runs[key]
    delete running[key]
  }

  nextRunStaysRunning = false
  $desktopActionTasks.set({})
  vi.mocked(getActionStatus).mockClear()
  vi.mocked(runDebugShare).mockClear()
})

afterEach(cleanup)

const button = (name: string) => screen.getByRole('button', { name }) as HTMLButtonElement

describe('MaintenancePanel action tail', () => {
  it('tails a second run of the same op', async () => {
    render(<MaintenancePanel />)

    await act(async () => void fireEvent.click(button('Run doctor')))
    await screen.findByText('doctor run 1 output')

    nextRunStaysRunning = true
    await act(async () => void fireEvent.click(button('Run doctor')))

    await screen.findByText('doctor run 2 output')
    expect(vi.mocked(getActionStatus)).toHaveBeenCalledTimes(2)
    expect(screen.getByText('Running...')).toBeTruthy()
    expect(button('Run doctor').disabled).toBe(true)
    expect($desktopActionTasks.get().doctor?.status).toMatchObject({ pid: 1002, running: true })
  })

  it('tails a different op launched after the first', async () => {
    render(<MaintenancePanel />)

    await act(async () => void fireEvent.click(button('Run doctor')))
    await screen.findByText('doctor run 1 output')

    nextRunStaysRunning = true
    await act(async () => void fireEvent.click(button('Security audit')))

    await screen.findByText('security-audit run 1 output')
    expect(vi.mocked(getActionStatus)).toHaveBeenLastCalledWith('security-audit', 200)
    expect(button('Security audit').disabled).toBe(true)
  })
})

describe('MaintenancePanel debug share', () => {
  // The backend refuses an unconfirmed upload (#94876); the panel must ask first.
  it('uploads nothing when the confirmation is cancelled', async () => {
    render(<MaintenancePanel />)

    await act(async () => void fireEvent.click(button('Debug share')))
    expect($confirmRequest.get()?.destructive).toBe(true)
    await act(async () => settleConfirm(false))

    expect(vi.mocked(runDebugShare)).not.toHaveBeenCalled()
  })

  it('uploads only after the user confirms', async () => {
    render(<MaintenancePanel />)

    await act(async () => void fireEvent.click(button('Debug share')))
    expect(vi.mocked(runDebugShare)).not.toHaveBeenCalled()
    await act(async () => settleConfirm(true))

    expect(vi.mocked(runDebugShare)).toHaveBeenCalledTimes(1)
  })

  // Profile shortcuts stay live behind the modal: the upload must publish the
  // logs of the profile the user confirmed for, not whichever is active later.
  it('uploads the profile that was active when the confirmation opened', async () => {
    const actual = await vi.importActual<typeof HermesApi>('@/hermes')
    const api = vi.fn(async (_request: { connectionId?: string; path: string; profile?: string }) => ({
      auto_delete_seconds: 21600,
      failures: [],
      ok: true,
      redacted: true,
      urls: {}
    }))
    vi.stubGlobal('hermesDesktop', { api })
    vi.mocked(runDebugShare).mockImplementationOnce(actual.runDebugShare)
    setApiRequestConnection('gw-a')
    setApiRequestProfile('alpha')

    try {
      render(<MaintenancePanel />)

      await act(async () => void fireEvent.click(button('Debug share')))
      setApiRequestConnection('gw-b')
      setApiRequestProfile('beta')
      await act(async () => settleConfirm(true))

      expect(api).toHaveBeenCalledTimes(1)
      expect(api.mock.calls[0][0]).toMatchObject({
        connectionId: 'gw-a',
        path: '/api/ops/debug-share',
        profile: 'alpha'
      })
    } finally {
      setApiRequestConnection(null)
      setApiRequestProfile(null)
      vi.unstubAllGlobals()
    }
  })
})
