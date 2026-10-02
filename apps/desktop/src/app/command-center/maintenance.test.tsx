import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { getActionStatus, getLogs, getStatus, updateHermes } from '@/hermes'
import { $desktopActionTasks } from '@/store/activity'

import { MaintenancePanel } from './maintenance'

import { CommandCenterView } from './index'

const memoryOracle = vi.hoisted(() => ({
  downloadGatewayMediaFile: vi.fn(),
  getCuratorStatus: vi.fn(),
  getMemoryStatus: vi.fn(),
  isRemoteGateway: vi.fn(),
  openExternal: vi.fn()
}))

vi.mock('@/lib/media', () => ({
  downloadGatewayMediaFile: memoryOracle.downloadGatewayMediaFile,
  isRemoteGateway: memoryOracle.isRemoteGateway
}))
vi.mock('@/store/system-actions', () => ({ confirmSharedGatewayRestart: vi.fn().mockResolvedValue(null) }))

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
  getCuratorStatus: memoryOracle.getCuratorStatus,
  getMemoryStatus: memoryOracle.getMemoryStatus,
  getStatus: vi.fn(),
  getLogs: vi.fn(),
  restartGateway: vi.fn(),
  updateHermes: vi.fn(),
  runDoctor: vi.fn(() => spawn('doctor')),
  runSecurityAudit: vi.fn(() => spawn('security-audit'))
}))

beforeEach(() => {
  for (const key of Object.keys(runs)) {
    delete runs[key]
    delete running[key]
  }

  nextRunStaysRunning = false
  memoryOracle.getCuratorStatus.mockReturnValue(new Promise(() => {}))
  memoryOracle.getMemoryStatus.mockReturnValue(new Promise(() => {}))
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

describe('fork memory file routing', () => {
  const remoteMemoryPath = '/home/remote/.hermes/memories/MEMORY.md'
  const localDownloadedPath = 'C:\\Users\\vip\\Downloads\\MEMORY.md'
  beforeEach(() => {
    memoryOracle.getCuratorStatus.mockResolvedValue({
      archive_after_days: null,
      enabled: false,
      interval_hours: null,
      last_run_at: null,
      min_idle_hours: null,
      paused: false,
      stale_after_days: null
    })
    memoryOracle.getMemoryStatus.mockResolvedValue({
      active: 'builtin',
      builtin_files: { memory: 42, user: 0 },
      builtin_paths: {
        memory: remoteMemoryPath,
        user: '/home/remote/.hermes/memories/USER.md'
      },
      providers: []
    })
    memoryOracle.isRemoteGateway.mockReturnValue(true)
    memoryOracle.downloadGatewayMediaFile.mockResolvedValue({ path: localDownloadedPath, saved: true })
    memoryOracle.openExternal.mockResolvedValue(undefined)

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        openExternal: memoryOracle.openExternal,
        writeClipboard: vi.fn()
      }
    })
  })
  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })
  describe('MaintenancePanel memory files', () => {
    it('downloads a remote memory file before opening the local copy', async () => {
      const { MaintenancePanel } = await import('./maintenance')

      render(<MaintenancePanel />)
      expect(await screen.findByText('Agent memory (MEMORY.md)')).toBeTruthy()

      const [openMemory] = screen.getAllByRole('button', { name: 'Open file' })
      fireEvent.click(openMemory!)

      await waitFor(() => expect(memoryOracle.downloadGatewayMediaFile).toHaveBeenCalledWith(remoteMemoryPath))
      expect(memoryOracle.openExternal).toHaveBeenCalledWith('file:///C%3A/Users/vip/Downloads/MEMORY.md')
      expect(memoryOracle.openExternal).not.toHaveBeenCalledWith(expect.stringContaining('/home/remote/'))
    })

    it('opens a local memory path directly without downloading it', async () => {
      const localMemoryPath = 'C:\\Users\\vip\\AppData\\Local\\hermes\\memories\\MEMORY.md'
      memoryOracle.isRemoteGateway.mockReturnValue(false)
      memoryOracle.getMemoryStatus.mockResolvedValue({
        active: 'builtin',
        builtin_files: { memory: 42, user: 0 },
        builtin_paths: { memory: localMemoryPath },
        providers: []
      })
      const { MaintenancePanel } = await import('./maintenance')

      render(<MaintenancePanel />)
      expect(await screen.findByText('Agent memory (MEMORY.md)')).toBeTruthy()
      fireEvent.click(screen.getAllByRole('button', { name: 'Open file' })[0]!)

      await waitFor(() =>
        expect(memoryOracle.openExternal).toHaveBeenCalledWith(
          'file:///C%3A/Users/vip/AppData/Local/hermes/memories/MEMORY.md'
        )
      )
      expect(memoryOracle.downloadGatewayMediaFile).not.toHaveBeenCalled()
    })

    it('does not open a file when a remote download is cancelled', async () => {
      memoryOracle.downloadGatewayMediaFile.mockResolvedValue({ canceled: true, saved: false })
      const { MaintenancePanel } = await import('./maintenance')

      render(<MaintenancePanel />)
      expect(await screen.findByText('Agent memory (MEMORY.md)')).toBeTruthy()
      fireEvent.click(screen.getAllByRole('button', { name: 'Open file' })[0]!)

      await waitFor(() => expect(memoryOracle.downloadGatewayMediaFile).toHaveBeenCalledWith(remoteMemoryPath))
      expect(memoryOracle.openExternal).not.toHaveBeenCalled()
    })

    it('disables opening when the backend reports bytes without a file path', async () => {
      memoryOracle.isRemoteGateway.mockReturnValue(false)
      memoryOracle.getMemoryStatus.mockResolvedValue({
        active: 'builtin',
        builtin_files: { memory: 42, user: 0 },
        providers: [],
        builtin_paths: {}
      })
      const { MaintenancePanel } = await import('./maintenance')

      render(<MaintenancePanel />)
      expect(await screen.findByText('Agent memory (MEMORY.md)')).toBeTruthy()

      const [openMemory] = screen.getAllByRole('button', { name: 'Open file' })
      expect((openMemory as HTMLButtonElement).disabled).toBe(true)
      fireEvent.click(openMemory!)
      expect(memoryOracle.openExternal).not.toHaveBeenCalled()
    })
  })
})

describe('fork durable system actions', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.mocked(getStatus).mockResolvedValue({ version: '0.21.0', gateway_running: true, active_sessions: 0 } as Awaited<
      ReturnType<typeof getStatus>
    >)
    vi.mocked(getLogs).mockResolvedValue({ file: 'agent', lines: [] })
    vi.mocked(updateHermes).mockReset()
    vi.mocked(updateHermes).mockResolvedValue({ name: 'update', ok: true, pid: 123 })
    vi.mocked(getActionStatus).mockReset()
    vi.mocked(getActionStatus).mockResolvedValue({ name: 'update', running: false, exit_code: 0, lines: [], pid: 123 })
  })
  afterEach(() => {
    cleanup()
    vi.useRealTimers()
  })

  const mount = async () => {
    await act(async () => {
      render(
        <MemoryRouter>
          <CommandCenterView
            initialSection="system"
            onClose={() => {}}
            onDeleteSession={vi.fn()}
            onOpenSession={() => {}}
          />
        </MemoryRouter>
      )
    })
  }

  it('blocks duplicate starts and releases ownership for a second completed action', async () => {
    await mount()
    await act(async () => {
      fireEvent.click(button('Update Hermes'))
      fireEvent.click(button('Update Hermes'))
    })
    expect(updateHermes).toHaveBeenCalledTimes(1)
    expect(button('Update Hermes').disabled).toBe(true)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1200)
    })
    expect(button('Update Hermes').disabled).toBe(false)
    await act(async () => {
      fireEvent.click(button('Update Hermes'))
    })
    expect(updateHermes).toHaveBeenCalledTimes(2)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1200)
    })
  })

  it('releases failed startup ownership so a retry can run', async () => {
    vi.mocked(updateHermes).mockRejectedValueOnce(new Error('startup failed'))
    await mount()
    await act(async () => {
      fireEvent.click(button('Update Hermes'))
    })
    expect(button('Update Hermes').disabled).toBe(false)
    expect(screen.getByText('startup failed')).toBeTruthy()
    await act(async () => {
      fireEvent.click(button('Update Hermes'))
    })
    expect(updateHermes).toHaveBeenCalledTimes(2)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1200)
    })
  })

  it('follows a durable action beyond 22 seconds and survives a transient backend restart', async () => {
    const startedAt = Date.now()
    vi.mocked(getActionStatus).mockRejectedValueOnce(new Error('backend restarting'))
    vi.mocked(getActionStatus).mockImplementation(async () => ({
      name: 'update',
      running: Date.now() - startedAt < 30_000,
      exit_code: Date.now() - startedAt < 30_000 ? null : 0,
      lines: [],
      pid: 123
    }))
    await mount()
    await act(async () => {
      fireEvent.click(button('Update Hermes'))
      await vi.advanceTimersByTimeAsync(40_000)
    })
    expect(getActionStatus).toHaveBeenCalledTimes(25)
    expect(getActionStatus).toHaveBeenLastCalledWith('update', 180, undefined, 5000)
    expect(button('Update Hermes').disabled).toBe(false)
    expect($desktopActionTasks.get().update?.status.running).toBe(false)
  })
  it('retains the five-minute deadline message after the automatic status refresh', async () => {
    vi.mocked(getActionStatus).mockResolvedValue({
      name: 'update',
      running: true,
      exit_code: null,
      lines: [],
      pid: 123
    })
    await mount()
    await act(async () => {
      fireEvent.click(button('Update Hermes'))
      await vi.advanceTimersByTimeAsync(299_999)
    })
    expect(screen.queryByText('Action is still running; check recent logs for the final status.')).toBeNull()
    expect(button('Update Hermes').disabled).toBe(true)
    const pollsBeforeDeadline = vi.mocked(getActionStatus).mock.calls.length
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1)
    })
    expect(screen.getByText('Action is still running; check recent logs for the final status.')).toBeTruthy()
    expect($desktopActionTasks.get().update?.status.running).toBe(true)
    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000)
    })
    expect(getActionStatus).toHaveBeenCalledTimes(pollsBeforeDeadline)
    expect(getStatus).toHaveBeenCalled()
  })
})
