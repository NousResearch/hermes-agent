import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopUpdateStatus, DesktopVersionInfo, HermesConnection } from '@/global'
import { en } from '@/i18n/en'
import { $connection } from '@/store/session'
import {
  $backendUpdateStatus,
  $desktopVersion,
  $updateStatus,
  checkBackendUpdates,
  checkUpdates,
  refreshDesktopVersion,
  startActiveUpdate,
  type UpdateApplyState
} from '@/store/updates'

import { AboutSettings } from './about-settings'

vi.mock('@/store/session', async (): Promise<Record<string, unknown>> => {
  const { atom } = await import('nanostores')

  return { $connection: atom<HermesConnection | null>(null) }
})

vi.mock('@/store/updates', async (): Promise<Record<string, unknown>> => {
  const { atom } = await import('nanostores')

  const idle: UpdateApplyState = {
    applying: false,
    stage: 'idle',
    message: '',
    percent: null,
    error: null,
    command: null,
    log: []
  }

  return {
    $desktopVersion: atom<DesktopVersionInfo | null>(null),
    $backendUpdateApply: atom<UpdateApplyState>(idle),
    $backendUpdateChecking: atom<boolean>(false),
    $backendUpdateStatus: atom<DesktopUpdateStatus | null>(null),
    $updateApply: atom<UpdateApplyState>(idle),
    $updateChecking: atom<boolean>(false),
    $updateStatus: atom<DesktopUpdateStatus | null>(null),
    checkBackendUpdates: vi.fn<() => Promise<DesktopUpdateStatus | null>>().mockResolvedValue(null),
    checkUpdates: vi.fn<() => Promise<DesktopUpdateStatus | null>>().mockResolvedValue(null),
    refreshDesktopVersion: vi.fn<() => Promise<DesktopVersionInfo | null>>().mockResolvedValue(null),
    openUpdateOverlayFor: vi.fn(),
    openUpdatesWindow: vi.fn(),
    startActiveUpdate: vi.fn()
  }
})

function connection(mode: 'local' | 'remote'): HermesConnection {
  return {
    baseUrl: 'http://gateway:9119',
    isFullscreen: false,
    mode,
    nativeOverlayWidth: 0,
    token: 'test-token',
    wsUrl: 'ws://gateway:9119',
    logs: [],
    windowButtonPosition: null
  }
}

describe('AboutSettings', (): void => {
  beforeEach((): void => {
    vi.clearAllMocks()
    $connection.set(connection('local'))
    $desktopVersion.set({
      appVersion: '1.2.3',
      electronVersion: '40',
      nodeVersion: '26',
      platform: 'linux',
      hermesRoot: '/test/hermes',
      installId: 'test-install'
    })
    $updateStatus.set({ supported: true, behind: 0 })
    $backendUpdateStatus.set({ supported: true, behind: 0 })
  })

  afterEach((): void => {
    cleanup()
  })

  it('shows shared version details and refreshes each remote gateway independently', (): void => {
    const { rerender } = render(<AboutSettings />)

    expect(screen.getByRole('heading', { name: en.updates.appName })).toBeTruthy()
    expect(screen.getByText(en.updates.versionDetailsInstallId)).toBeTruthy()
    expect(screen.getByText(en.updates.latestBody)).toBeTruthy()
    expect(screen.queryByText(en.updates.latestBodyBackend)).toBeNull()
    expect(refreshDesktopVersion).toHaveBeenCalledTimes(1)
    expect(checkBackendUpdates).not.toHaveBeenCalled()

    $connection.set(connection('remote'))
    rerender(<AboutSettings />)

    expect(screen.getByText(en.updates.latestBodyBackend)).toBeTruthy()
    expect(checkBackendUpdates).toHaveBeenCalledTimes(1)
    expect(screen.getAllByRole('link', { name: en.updates.releaseNotes })).toHaveLength(1)

    $connection.set({ ...connection('remote'), baseUrl: 'http://other-gateway:9119' })
    rerender(<AboutSettings />)

    expect(refreshDesktopVersion).toHaveBeenCalledTimes(3)
    expect(checkBackendUpdates).toHaveBeenCalledTimes(2)
  })

  it('shows the fixed package channel with no selector', (): void => {
    $desktopVersion.set({ ...$desktopVersion.get()!, channel: 'canary' })
    render(<AboutSettings />)
    expect(screen.getByText(`${en.updates.version('1.2.3')} · canary`)).toBeTruthy()
    expect(screen.queryByRole('combobox')).toBeNull()
  })

  it('a commit client refuses updates without hiding an unrelated backend update', (): void => {
    const message: string = "This build doesn't get updates. Ask the developer who gave it to you for a new build."
    $connection.set(connection('remote'))
    $desktopVersion.set({ ...$desktopVersion.get()!, source: 'commit-build' })
    $updateStatus.set({ supported: false, mechanism: 'external', reason: 'commit-build', message })
    $backendUpdateStatus.set({ supported: true, behind: 3 })
    render(<AboutSettings />)
    expect(screen.getByText(message)).toBeTruthy()
    expect(screen.getAllByRole('button', { name: en.updates.checkNow })).toHaveLength(1)
    fireEvent.click(screen.getByRole('button', { name: en.updates.updateNow }))
    expect(startActiveUpdate).toHaveBeenCalledWith('backend')
  })

  it('forces a fresh check only for the card the user checks', async (): Promise<void> => {
    $connection.set(connection('remote'))
    render(<AboutSettings />)
    vi.mocked(checkBackendUpdates).mockClear()

    const buttons = screen.getAllByRole('button', { name: en.updates.checkNow })
    fireEvent.click(buttons[0])
    await waitFor((): void => {
      expect(checkUpdates).toHaveBeenCalledWith({ force: true })
    })
    expect(checkBackendUpdates).not.toHaveBeenCalled()

    fireEvent.click(buttons[1])
    await waitFor((): void => {
      expect(checkBackendUpdates).toHaveBeenCalledWith({ force: true })
    })
    expect(checkUpdates).toHaveBeenCalledTimes(1)
  })
})
