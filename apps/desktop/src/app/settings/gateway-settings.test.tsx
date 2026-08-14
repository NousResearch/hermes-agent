import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileInfo } from '@/types/hermes'

const getConnectionConfig = vi.fn()
const applyConnectionConfig = vi.fn()
const saveConnectionConfig = vi.fn()
const rehomeSecondaryGateway = vi.fn(async () => undefined)
const profiles = atom<ProfileInfo[]>([])

vi.mock('@/store/profile', () => ({
  $profiles: profiles,
  refreshActiveProfile: vi.fn()
}))

vi.mock('@/store/gateway', () => ({ rehomeSecondaryGateway }))

const localConnection = {
  cloudOrg: '',
  envOverride: false,
  inherited: false,
  mode: 'local',
  profileOverride: false,
  remoteAuthMode: 'token',
  remoteOauthConnected: false,
  remoteTokenPreview: null,
  remoteTokenSet: false,
  remoteUrl: '',
  secureTokenStorage: true,
  sshHost: '',
  sshKeyPath: '',
  sshPort: null,
  sshRemoteHermesPath: '',
  sshRemoteProfile: '',
  sshUser: ''
}

beforeEach(() => {
  profiles.set([
    {
      has_env: false,
      is_default: true,
      model: null,
      name: 'default',
      path: '/tmp/hermes',
      provider: null,
      skill_count: 0
    },
    {
      has_env: false,
      is_default: false,
      model: null,
      name: 'work',
      path: '/tmp/hermes/profiles/work',
      provider: null,
      skill_count: 0
    }
  ])
  getConnectionConfig.mockResolvedValue(localConnection)
  applyConnectionConfig.mockResolvedValue(localConnection)
  saveConnectionConfig.mockResolvedValue(localConnection)
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { applyConnectionConfig, getConnectionConfig, saveConnectionConfig }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('GatewaySettings', () => {
  it('renders Local and Inherit as distinct named-profile choices', async () => {
    getConnectionConfig.mockImplementation(async profile =>
      profile === 'work'
        ? { ...localConnection, inherited: true, profile: 'work', profileOverride: false }
        : localConnection
    )
    const { GatewaySettings } = await import('./gateway-settings')

    render(<GatewaySettings />)
    expect(await screen.findByText('Local gateway')).toBeTruthy()
    expect(
      screen.getByText('Start a private Hermes backend on localhost. This is the default and works offline.')
    ).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'work' }))

    await waitFor(() => expect(getConnectionConfig).toHaveBeenLastCalledWith('work'))
    expect(await screen.findByText('Use default gateway')).toBeTruthy()
    expect(screen.getByText("Remove this profile's override and use the default connection.")).toBeTruthy()
    expect(screen.getByText('Local gateway')).toBeTruthy()
    expect(screen.getByText('Start a private Hermes backend on localhost. This is the default and works offline.')).toBeTruthy()

    const inheritCard = screen.getByRole('button', { name: /Use default gateway/ })

    fireEvent.click(inheritCard)
    await waitFor(() => expect(inheritCard.className).toContain('border-primary'))
    fireEvent.click(screen.getByRole('button', { name: 'Save for next restart' }))

    await waitFor(() =>
      expect(saveConnectionConfig).toHaveBeenCalledWith(expect.objectContaining({ inherit: true, profile: 'work' }))
    )
  })

  it('shows and clears an SSH remote-profile mapping for a named Desktop profile', async () => {
    getConnectionConfig.mockImplementation(async profile =>
      profile === 'work'
        ? {
            ...localConnection,
            mode: 'ssh',
            profile: 'work',
            sshHost: 'remote-box',
            sshUser: 'alice',
            sshPort: 22,
            sshKeyPath: '',
            sshRemoteHermesPath: '/opt/hermes/bin/hermes',
            sshRemoteProfile: 'default'
          }
        : localConnection
    )
    saveConnectionConfig.mockReturnValue(new Promise(() => {}))
    const { GatewaySettings } = await import('./gateway-settings')

    render(<GatewaySettings />)
    fireEvent.click(await screen.findByRole('button', { name: 'work' }))

    await waitFor(() => expect(getConnectionConfig).toHaveBeenLastCalledWith('work'))
    expect(await screen.findByText('Remote profile (optional)')).toBeTruthy()

    const input = screen.getByPlaceholderText('work')

    expect((input as HTMLInputElement).value).toBe('default')
    fireEvent.change(input, { target: { value: '' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save for next restart' }))

    await waitFor(() =>
      expect(saveConnectionConfig).toHaveBeenCalledWith(
        expect.objectContaining({
          profile: 'work',
          sshRemoteProfile: ''
        })
      )
    )
  })

  it('applies and rehomes only the selected profile socket', async () => {
    getConnectionConfig.mockImplementation(async profile =>
      profile === 'work'
        ? { ...localConnection, inherited: true, profile: 'work', profileOverride: false }
        : localConnection
    )
    applyConnectionConfig.mockResolvedValue({
      ...localConnection,
      inherited: false,
      profile: 'work',
      profileOverride: true
    })
    const { GatewaySettings } = await import('./gateway-settings')

    render(<GatewaySettings />)
    fireEvent.click(await screen.findByRole('button', { name: 'work' }))
    await waitFor(() => expect(getConnectionConfig).toHaveBeenLastCalledWith('work'))
    const localCard = await screen.findByRole('button', { name: /Local gateway/ })

    fireEvent.click(localCard)
    await waitFor(() => expect(localCard.className).toContain('border-primary'))
    fireEvent.click(screen.getByRole('button', { name: 'Save and reconnect' }))

    await waitFor(() =>
      expect(applyConnectionConfig).toHaveBeenCalledWith(
        expect.objectContaining({ mode: 'local', profile: 'work' })
      )
    )
    expect(applyConnectionConfig.mock.calls[0]?.[0]).not.toHaveProperty('inherit')
    await waitFor(() => expect(rehomeSecondaryGateway).toHaveBeenCalledWith('work'))
  })
})
