import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const getConnectionConfig = vi.fn()
const saveConnectionConfig = vi.fn()

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

  it('switches to SSH mode when an ssh:// URL is pasted into Remote URL', async () => {
    getConnectionConfig.mockResolvedValue({
      ...localConnection,
      mode: 'remote',
      remoteUrl: 'https://gateway.example.com/hermes'
    })
    const { GatewaySettings } = await import('./gateway-settings')

    render(<GatewaySettings />)
    expect(await screen.findByText('Remote URL')).toBeTruthy()

    fireEvent.change(screen.getByPlaceholderText('https://gateway.example.com/hermes'), {
      target: { value: 'ssh://alice@botnet:2222?profile=brawn' }
    })

    expect(screen.queryByText('Remote URL')).toBeNull()
    expect(screen.getByDisplayValue('botnet')).toBeTruthy()
    expect(screen.getByDisplayValue('alice')).toBeTruthy()
    expect(screen.getByDisplayValue('2222')).toBeTruthy()

  })


})
