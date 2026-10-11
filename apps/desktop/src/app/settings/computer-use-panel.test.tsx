// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ComputerUseStatus, ComputerUseTarget } from '@/types/hermes'

import { ComputerUsePanel } from './computer-use-panel'

const { getActionStatus, getComputerUseStatus, grantComputerUsePermissions } = vi.hoisted(() => ({
  getActionStatus: vi.fn(),
  getComputerUseStatus: vi.fn<(target?: ComputerUseTarget) => Promise<ComputerUseStatus>>(),
  grantComputerUsePermissions: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getActionStatus,
  getComputerUseStatus,
  grantComputerUsePermissions
}))
vi.mock('@/i18n', () => ({
  useI18n: () => ({ t: { settings: { computerUse: { driverHealth: 'Driver health' } } } })
}))
vi.mock('@/store/notifications', () => ({ notify: vi.fn(), notifyError: vi.fn() }))

const linuxGuest: ComputerUseStatus = {
  platform: 'linux', platform_supported: true, installed: true, version: 'cua-driver 1.0', ready: false,
  can_grant: false, checks: [], accessibility: null, screen_recording: null, screen_recording_capturable: null,
  source: null, error: null
}

const windowsHost: ComputerUseStatus = { ...linuxGuest, platform: 'win32', ready: true }

function deferred<T>() {
  let resolve!: (value: T) => void

  const promise = new Promise<T>(nextResolve => {
    resolve = nextResolve
  })

  return { promise, resolve }
}

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
  vi.useRealTimers()
})

describe('ComputerUsePanel', () => {
  it('offers a Windows-host target for a remote WSL guest and routes its status through the local bridge', async () => {
    getComputerUseStatus.mockImplementation(target => Promise.resolve(target === 'windows-host' ? windowsHost : linuxGuest))

    function PanelHarness() {
      const [target, setTarget] = React.useState<ComputerUseTarget>('guest')

      return <ComputerUsePanel onTargetChange={setTarget} target={target} />
    }

    render(<PanelHarness />)

    expect(await screen.findByRole('button', { name: 'Windows host' })).toBeTruthy()
    await act(async () => fireEvent.click(screen.getByRole('button', { name: 'Windows host' })))

    expect(getComputerUseStatus).toHaveBeenLastCalledWith('windows-host')
    expect(screen.getByText(/Windows desktop \(this device\)/)).toBeTruthy()
    expect(screen.getByText('Ready')).toBeTruthy()
  })

  it('keeps the Linux guest status visible when the optional Windows-host probe is unavailable', async () => {
    getComputerUseStatus.mockImplementation(target =>
      target === 'windows-host' ? Promise.reject(new Error('local bridge unavailable')) : Promise.resolve(linuxGuest)
    )

    render(<ComputerUsePanel onTargetChange={vi.fn()} target="guest" />)

    expect(await screen.findByText('Not ready')).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Windows host' })).toBeNull()
  })

  it('polls a Windows-host permission action through its local owner', async () => {
    const localOwner = { connectionId: 'local' }
    const permissionStatus = { ...windowsHost, can_grant: true, ready: false }
    getComputerUseStatus.mockImplementation(target => Promise.resolve(target === 'windows-host' ? permissionStatus : linuxGuest))
    grantComputerUsePermissions.mockResolvedValue({ ok: true, name: 'computer-use-permissions' })
    getActionStatus.mockResolvedValue({ name: 'computer-use-permissions', running: false, lines: [], exit_code: 0 })

    render(<ComputerUsePanel onTargetChange={vi.fn()} target="windows-host" />)

    const grantButton = await screen.findByRole('button', { name: 'Grant permissions' })
    vi.useFakeTimers()
    fireEvent.click(grantButton)
    await act(async () => Promise.resolve())
    await act(async () => vi.advanceTimersByTimeAsync(1500))

    expect(getActionStatus).toHaveBeenCalledWith('computer-use-permissions', 200, localOwner)
  })

  it('ignores a stale guest refresh after the target changes to Windows host', async () => {
    const firstGuest = deferred<ComputerUseStatus>()
    let guestRequests = 0
    getComputerUseStatus.mockImplementation(target => {
      if (target === 'windows-host') {
        return Promise.resolve(windowsHost)
      }

      guestRequests += 1

      return guestRequests === 1 ? firstGuest.promise : Promise.resolve(linuxGuest)
    })

    const { rerender } = render(<ComputerUsePanel onTargetChange={vi.fn()} target="guest" />)
    rerender(<ComputerUsePanel onTargetChange={vi.fn()} target="windows-host" />)

    expect(await screen.findByText('Ready')).toBeTruthy()

    await act(async () => firstGuest.resolve(linuxGuest))

    expect(screen.getByText('Ready')).toBeTruthy()
  })

  it('does not publish an old permission action refresh after the target changes', async () => {
    const permissionStatus = { ...linuxGuest, can_grant: true }
    getComputerUseStatus.mockImplementation(target => Promise.resolve(target === 'windows-host' ? windowsHost : permissionStatus))
    grantComputerUsePermissions.mockResolvedValue({ ok: true, name: 'computer-use-permissions' })
    getActionStatus.mockResolvedValue({ name: 'computer-use-permissions', running: false, lines: [], exit_code: 0 })

    const { rerender } = render(<ComputerUsePanel onTargetChange={vi.fn()} target="guest" />)
    const grantButton = await screen.findByRole('button', { name: 'Grant permissions' })
    vi.useFakeTimers()
    fireEvent.click(grantButton)
    await act(async () => Promise.resolve())

    await act(async () => {
      rerender(<ComputerUsePanel onTargetChange={vi.fn()} target="windows-host" />)
      await Promise.resolve()
    })
    expect(screen.getByText('Ready')).toBeTruthy()

    await act(async () => vi.advanceTimersByTimeAsync(1500))

    expect(screen.getByText('Ready')).toBeTruthy()
  })
})
