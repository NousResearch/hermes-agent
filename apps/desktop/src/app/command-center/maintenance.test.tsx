import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { getActionStatus } from '@/hermes'
import { $desktopActionTasks } from '@/store/activity'

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
