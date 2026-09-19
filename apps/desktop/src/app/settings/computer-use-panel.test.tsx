// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import * as React from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { ComputerUseStatus, ComputerUseTarget } from '@/types/hermes'

import { ComputerUsePanel } from './computer-use-panel'

const { getComputerUseStatus, grantComputerUsePermissions } = vi.hoisted(() => ({
  getComputerUseStatus: vi.fn<(target?: ComputerUseTarget) => Promise<ComputerUseStatus>>(),
  grantComputerUsePermissions: vi.fn()
}))

vi.mock('@/hermes', () => ({
  getActionStatus: vi.fn(),
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

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
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
})
