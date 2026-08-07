import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileInfo } from '@/types/hermes'

// Radix Select calls scrollIntoView / pointer-capture APIs jsdom lacks.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const getConnectionConfig = vi.fn()
const saveConnectionConfig = vi.fn()
const profiles = atom<ProfileInfo[]>([])

vi.mock('@/store/profile', () => ({
  $profiles: profiles,
  refreshActiveProfile: vi.fn()
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
  saveConnectionConfig.mockResolvedValue(localConnection)
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { getConnectionConfig, saveConnectionConfig }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('GatewaySettings', () => {
  it('labels local mode as default inheritance for a named profile', async () => {
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
    expect(
      screen.queryByText('Start a private Hermes backend on localhost. This is the default and works offline.')
    ).toBeNull()
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

  it('focuses the custom SSH host input on the first "Custom" selection', async () => {
    getConnectionConfig.mockResolvedValue({
      ...localConnection,
      mode: 'ssh',
      sshHost: '',
      sshUser: '',
      sshPort: 22,
      sshKeyPath: '',
      sshRemoteHermesPath: '',
      sshRemoteProfile: ''
    })
    const sshConfigHosts = vi.fn().mockResolvedValue({ hosts: ['github.com'] })
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { getConnectionConfig, saveConnectionConfig, sshConfigHosts }
    })
    const { GatewaySettings } = await import('./gateway-settings')

    render(<GatewaySettings />)

    // With ~/.ssh/config aliases available the host field is a dropdown.
    const trigger = await screen.findByRole('combobox')
    fireEvent.click(trigger)
    fireEvent.click(screen.getByRole('option', { name: 'Custom (enter manually)…' }))

    // The dropdown is swapped for a free-text input that must be focused and
    // immediately typeable on the FIRST selection (no round-trip through
    // another option) — Radix's deferred focus restoration must not steal it.
    const hostRow = screen.getByText('Host').closest('.grid') as HTMLElement
    const input = within(hostRow).getByRole('textbox') as HTMLInputElement

    await waitFor(() => expect(document.activeElement).toBe(input))
  })
})
