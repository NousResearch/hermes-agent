import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({
  downloadGatewayMediaFile: vi.fn(),
  getCuratorStatus: vi.fn(),
  getMemoryStatus: vi.fn(),
  isRemoteGateway: vi.fn(),
  openExternal: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getActionStatus: vi.fn(),
  getCuratorStatus: mocks.getCuratorStatus,
  getMemoryStatus: mocks.getMemoryStatus,
  resetMemory: vi.fn(),
  runBackup: vi.fn(),
  runCurator: vi.fn(),
  runDebugShare: vi.fn(),
  runDoctor: vi.fn(),
  runSecurityAudit: vi.fn(),
  setCuratorPaused: vi.fn()
}))

vi.mock('@/lib/media', () => ({
  downloadGatewayMediaFile: mocks.downloadGatewayMediaFile,
  isRemoteGateway: mocks.isRemoteGateway
}))

vi.mock('@/store/activity', () => ({ upsertDesktopActionTask: vi.fn() }))
vi.mock('@/store/confirm', () => ({ confirm: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

const remoteMemoryPath = '/home/remote/.hermes/memories/MEMORY.md'
const localDownloadedPath = 'C:\\Users\\vip\\Downloads\\MEMORY.md'

beforeEach(() => {
  mocks.getCuratorStatus.mockResolvedValue({
    archive_after_days: null,
    enabled: false,
    interval_hours: null,
    last_run_at: null,
    min_idle_hours: null,
    paused: false,
    stale_after_days: null
  })
  mocks.getMemoryStatus.mockResolvedValue({
    active: 'builtin',
    builtin_files: { memory: 42, user: 0 },
    builtin_paths: {
      memory: remoteMemoryPath,
      user: '/home/remote/.hermes/memories/USER.md'
    },
    providers: []
  })
  mocks.isRemoteGateway.mockReturnValue(true)
  mocks.downloadGatewayMediaFile.mockResolvedValue({ path: localDownloadedPath, saved: true })
  mocks.openExternal.mockResolvedValue(undefined)

  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      openExternal: mocks.openExternal,
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

    await waitFor(() => expect(mocks.downloadGatewayMediaFile).toHaveBeenCalledWith(remoteMemoryPath))
    expect(mocks.openExternal).toHaveBeenCalledWith('file:///C%3A/Users/vip/Downloads/MEMORY.md')
    expect(mocks.openExternal).not.toHaveBeenCalledWith(expect.stringContaining('/home/remote/'))
  })


  it('opens a local memory path directly without downloading it', async () => {
    const localMemoryPath = 'C:\\Users\\vip\\AppData\\Local\\hermes\\memories\\MEMORY.md'
    mocks.isRemoteGateway.mockReturnValue(false)
    mocks.getMemoryStatus.mockResolvedValue({
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
      expect(mocks.openExternal).toHaveBeenCalledWith(
        'file:///C%3A/Users/vip/AppData/Local/hermes/memories/MEMORY.md'
      )
    )
    expect(mocks.downloadGatewayMediaFile).not.toHaveBeenCalled()
  })

  it('does not open a file when a remote download is cancelled', async () => {
    mocks.downloadGatewayMediaFile.mockResolvedValue({ canceled: true, saved: false })
    const { MaintenancePanel } = await import('./maintenance')

    render(<MaintenancePanel />)
    expect(await screen.findByText('Agent memory (MEMORY.md)')).toBeTruthy()
    fireEvent.click(screen.getAllByRole('button', { name: 'Open file' })[0]!)

    await waitFor(() => expect(mocks.downloadGatewayMediaFile).toHaveBeenCalledWith(remoteMemoryPath))
    expect(mocks.openExternal).not.toHaveBeenCalled()
  })

  it('disables opening when the backend reports bytes without a file path', async () => {
    mocks.isRemoteGateway.mockReturnValue(false)
    mocks.getMemoryStatus.mockResolvedValue({
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
    expect(mocks.openExternal).not.toHaveBeenCalled()
  })
})
