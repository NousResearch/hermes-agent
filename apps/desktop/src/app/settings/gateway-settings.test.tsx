import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileInfo } from '@/types/hermes'

const getConnectionConfig = vi.fn()
const saveConnectionConfig = vi.fn()
const restartCurrentBackend = vi.fn()
const profiles = atom<ProfileInfo[]>([])

vi.mock('@/store/profile', () => ({
  $profiles: profiles,
  refreshActiveProfile: vi.fn()
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn(),
  readableError: (_error: unknown, fallback: string) => ({ message: fallback })
}))

const localConnection = {
  cloudOrg: '',
  envOverride: false,
  mode: 'local',
  remoteAuthMode: 'token',
  remoteOauthConnected: false,
  remoteTokenPreview: null,
  remoteTokenSet: false,
  remoteUrl: ''
}

beforeEach(() => {
  getConnectionConfig.mockResolvedValue(localConnection)
  saveConnectionConfig.mockResolvedValue(localConnection)
  restartCurrentBackend.mockResolvedValue({ ok: true, mode: 'local' })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { getConnectionConfig, saveConnectionConfig, restartCurrentBackend }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('GatewaySettings', () => {
  it('loads the machine-level connection config (no profile scoping)', async () => {
    const { GatewaySettings } = await import('./gateway-settings')

    render(<GatewaySettings />)
    expect(await screen.findByText('Local gateway')).toBeTruthy()
    expect(
      screen.getByText('Start a private Hermes backend on localhost. This is the default and works offline.')
    ).toBeTruthy()

    // The page manages the machine's gateway connections; it must load the
    // global config, never a per-profile override.
    await waitFor(() => expect(getConnectionConfig).toHaveBeenCalledWith(null))
    expect(getConnectionConfig).not.toHaveBeenCalledWith(expect.any(String))

    // The legacy per-profile scope switcher must not render.
    expect(screen.queryByText('Applies to')).toBeNull()
    expect(screen.queryByText('All profiles')).toBeNull()
    expect(screen.queryByText('Use default gateway')).toBeNull()
  })

  it('disables duplicate current-backend restart clicks while request is in flight', async () => {
    let resolveRestart!: (value: { ok: true; mode: 'local' }) => void
    restartCurrentBackend.mockReturnValue(
      new Promise(resolve => {
        resolveRestart = resolve as (value: { ok: true; mode: 'local' }) => void
      })
    )
    const { GatewaySettings } = await import('./gateway-settings')

    render(<GatewaySettings />)
    const button = await screen.findByRole('button', { name: 'Restart current backend' })

    fireEvent.click(button)
    await waitFor(() => expect((button as HTMLButtonElement).disabled).toBe(true))
    fireEvent.click(button)
    expect(restartCurrentBackend).toHaveBeenCalledTimes(1)

    resolveRestart({ ok: true, mode: 'local' })
    await waitFor(() => expect((button as HTMLButtonElement).disabled).toBe(false))
  })

  it('surfaces the not-ready reason as notification detail on restart failure', async () => {
    const { notify } = await import('@/store/notifications')
    vi.mocked(notify).mockClear()
    restartCurrentBackend.mockResolvedValue({
      ok: false,
      reason: 'not-ready',
      message: 'Current SSH backend has no served session token.'
    })
    const { GatewaySettings } = await import('./gateway-settings')

    render(<GatewaySettings />)
    const button = await screen.findByRole('button', { name: 'Restart current backend' })

    fireEvent.click(button)

    await waitFor(() =>
      expect(notify).toHaveBeenCalledWith(
        expect.objectContaining({
          detail: 'Current SSH backend has no served session token.',
          kind: 'error'
        })
      )
    )
  })
})
