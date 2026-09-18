import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileScope } from '@/hermes'

import { ComputerUsePanel } from './computer-use-panel'

const getComputerUseStatus = vi.fn()
const grantComputerUsePermissions = vi.fn()
const saveHermesConfigRecord = vi.fn()
const notifyError = vi.fn()

vi.mock('@/hermes', () => ({
  getActionStatus: vi.fn(),
  getComputerUseStatus: (profile?: ProfileScope) => getComputerUseStatus(profile),
  grantComputerUsePermissions: (profile?: ProfileScope) => grantComputerUsePermissions(profile),
  saveHermesConfigRecord: (config: unknown, profile?: ProfileScope) => saveHermesConfigRecord(config, profile)
}))

vi.mock('@/store/activity', () => ({ upsertDesktopActionTask: vi.fn() }))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: (...args: unknown[]) => notifyError(...args) }))

const remoteScope = { connectionId: 'lab-gateway', profile: 'operator' }

function status(overrides: Record<string, unknown> = {}) {
  return {
    platform: 'linux',
    platform_supported: true,
    installed: true,
    version: 'cua-driver 0.5.1',
    ready: true,
    can_grant: false,
    checks: [],
    accessibility: null,
    screen_recording: null,
    screen_recording_capturable: null,
    source: null,
    error: null,
    target: 'auto',
    is_wsl: true,
    driver_platform: 'windows',
    driver_command: '/mnt/c/Users/Alice/cua-driver.exe',
    target_error: null,
    ...overrides
  }
}

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(done => {
    resolve = done
  })

  return { promise, resolve }
}

beforeEach(() => {
  getComputerUseStatus.mockReset()
  saveHermesConfigRecord.mockReset()
  getComputerUseStatus.mockResolvedValue(status())
  saveHermesConfigRecord.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('ComputerUsePanel WSL target selection', () => {
  it('persists a sparse target update to the same remote capability scope and refreshes authoritative status', async () => {
    getComputerUseStatus
      .mockResolvedValueOnce(status({ target: 'auto' }))
      .mockResolvedValueOnce(status({ target: 'windows' }))

    render(<ComputerUsePanel profile={remoteScope} />)
    fireEvent.click(await screen.findByRole('button', { name: 'Windows host' }))

    await waitFor(() =>
      expect(saveHermesConfigRecord).toHaveBeenCalledWith({ computer_use: { target: 'windows' } }, remoteScope)
    )
    await waitFor(() => expect(getComputerUseStatus).toHaveBeenLastCalledWith(remoteScope))
    expect(getComputerUseStatus).toHaveBeenCalledTimes(2)
    expect(screen.getByRole('button', { name: 'Windows host' }).getAttribute('aria-pressed')).toBe('true')
  })

  it('reflects the stored target after remounting', async () => {
    getComputerUseStatus.mockResolvedValue(status({ target: 'linux', driver_platform: 'linux' }))

    const first = render(<ComputerUsePanel profile={remoteScope} />)
    expect((await screen.findByRole('button', { name: 'Linux guest' })).getAttribute('aria-pressed')).toBe('true')
    first.unmount()

    render(<ComputerUsePanel profile={remoteScope} />)
    expect((await screen.findByRole('button', { name: 'Linux guest' })).getAttribute('aria-pressed')).toBe('true')
    expect(getComputerUseStatus).toHaveBeenLastCalledWith(remoteScope)
  })

  it('renders the selector before the selected driver is installed', async () => {
    getComputerUseStatus.mockResolvedValue(status({ installed: false, ready: null, driver_command: null }))

    render(<ComputerUsePanel profile={remoteScope} />)

    expect(await screen.findByRole('button', { name: 'Automatic' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Windows host' })).toBeTruthy()
    expect(screen.getByText(/Install the cua-driver backend below/)).toBeTruthy()
  })

  it('does not render the target selector outside WSL', async () => {
    getComputerUseStatus.mockResolvedValue(status({ is_wsl: false, target: undefined }))

    render(<ComputerUsePanel profile={remoteScope} />)

    await screen.findByText('Driver health')
    expect(screen.queryByRole('button', { name: 'Automatic' })).toBeNull()
  })

  it('keeps the persisted selection when a save is rejected', async () => {
    saveHermesConfigRecord.mockRejectedValue(new Error('denied'))

    render(<ComputerUsePanel profile={remoteScope} />)
    const automatic = await screen.findByRole('button', { name: 'Automatic' })
    fireEvent.click(screen.getByRole('button', { name: 'Linux guest' }))

    await waitFor(() => expect(notifyError).toHaveBeenCalled())
    expect(automatic.getAttribute('aria-pressed')).toBe('true')
    expect(screen.getByRole('button', { name: 'Linux guest' }).getAttribute('aria-pressed')).toBe('false')
    expect(getComputerUseStatus).toHaveBeenCalledTimes(1)
  })

  it('ignores a slow status response from the previous scope', async () => {
    const oldStatus = deferred<ReturnType<typeof status>>()
    getComputerUseStatus
      .mockReturnValueOnce(oldStatus.promise)
      .mockResolvedValueOnce(status({ target: 'linux', driver_platform: 'linux' }))

    const { rerender } = render(<ComputerUsePanel profile={remoteScope} />)
    const localScope = { connectionId: 'local', profile: 'default' }
    rerender(<ComputerUsePanel profile={localScope} />)

    expect((await screen.findByRole('button', { name: 'Linux guest' })).getAttribute('aria-pressed')).toBe('true')
    oldStatus.resolve(status({ target: 'windows' }))

    await waitFor(() => expect(getComputerUseStatus).toHaveBeenLastCalledWith(localScope))
    expect(screen.getByRole('button', { name: 'Linux guest' }).getAttribute('aria-pressed')).toBe('true')
  })

  it('isolates a slow save from a newly selected scope and disables the selector while saving', async () => {
    const oldSave = deferred<{ ok: boolean }>()
    getComputerUseStatus
      .mockResolvedValueOnce(status({ target: 'auto' }))
      .mockResolvedValueOnce(status({ target: 'linux', driver_platform: 'linux' }))
    saveHermesConfigRecord.mockReturnValueOnce(oldSave.promise)

    const { rerender } = render(<ComputerUsePanel profile={remoteScope} />)
    fireEvent.click(await screen.findByRole('button', { name: 'Windows host' }))
    expect(screen.getByRole('button', { name: 'Automatic' }).hasAttribute('disabled')).toBe(true)

    const localScope = { connectionId: 'local', profile: 'default' }
    rerender(<ComputerUsePanel profile={localScope} />)
    expect((await screen.findByRole('button', { name: 'Linux guest' })).getAttribute('aria-pressed')).toBe('true')

    oldSave.resolve({ ok: true })
    await waitFor(() =>
      expect(saveHermesConfigRecord).toHaveBeenCalledWith({ computer_use: { target: 'windows' } }, remoteScope)
    )
    expect(getComputerUseStatus).toHaveBeenCalledTimes(2)
    expect(screen.getByRole('button', { name: 'Linux guest' }).getAttribute('aria-pressed')).toBe('true')
  })

  it('shows the effective driver platform, command, and backend target conflict', async () => {
    getComputerUseStatus.mockResolvedValue(
      status({ target_error: 'HERMES_COMPUTER_USE_TARGET forces Linux while config selects Windows.' })
    )

    render(<ComputerUsePanel profile={remoteScope} />)

    expect(await screen.findByText(/Effective driver: Windows/)).toBeTruthy()
    expect(screen.getByText(/\/mnt\/c\/Users\/Alice\/cua-driver\.exe/)).toBeTruthy()
    expect(screen.getByText(/forces Linux while config selects Windows/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Automatic' }).getAttribute('aria-pressed')).toBe('true')
  })
})

it.each(['macos', 'darwin'])('formats the effective macOS driver platform %s', async driverPlatform => {
  getComputerUseStatus.mockResolvedValue(status({ platform: 'darwin', is_wsl: false, driver_platform: driverPlatform }))
  render(<ComputerUsePanel />)
  expect(await screen.findByText('Effective driver: macOS')).toBeTruthy()
})
