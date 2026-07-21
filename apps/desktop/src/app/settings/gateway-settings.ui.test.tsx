import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopCloudStatus, DesktopConnectionConfig } from '@/global'

import { GatewaySettings } from './gateway-settings'

const getConnectionConfig = vi.fn()
const selectConnectionConfig = vi.fn()

function config(overrides: Partial<DesktopConnectionConfig> = {}): DesktopConnectionConfig {
  return {
    cloudOrg: '',
    connections: [],
    envOverride: false,
    mode: 'local',
    profile: null,
    remoteAuthMode: 'token',
    remoteOauthConnected: false,
    remoteTokenPreview: null,
    remoteTokenSet: false,
    remoteUrl: '',
    selectedConnectionId: null,
    selectedConnectionName: '',
    ...overrides
  }
}

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
})

beforeEach(() => {
  const initial = config()

  getConnectionConfig.mockResolvedValue(initial)
  selectConnectionConfig.mockResolvedValue(initial)
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      api: vi.fn().mockResolvedValue({ current: 'default', profiles: [] }),
      applyConnectionConfig: vi.fn(),
      cloud: {
        agentSignIn: vi.fn(),
        discover: vi.fn(),
        login: vi.fn(),
        logout: vi.fn(),
        status: vi.fn().mockResolvedValue({ signedIn: false })
      },
      deleteConnectionConfig: vi.fn(),
      getConnectionConfig,
      oauthLoginConnectionConfig: vi.fn(),
      oauthLogoutConnectionConfig: vi.fn(),
      probeConnectionConfig: vi.fn(),
      revealLogs: vi.fn(),
      saveConnectionConfig: vi.fn(),
      selectConnectionConfig,
      testConnectionConfig: vi.fn()
    }
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('GatewaySettings connection picker', () => {
  it('opens and focuses a blank remote editor from Local', async () => {
    render(<GatewaySettings />)

    await screen.findByText('Connections')
    expect(screen.queryByPlaceholderText('Homelab')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Add remote' }))

    const name = screen.getByPlaceholderText('Homelab')

    await waitFor(() => expect(name).toBe(name.ownerDocument.activeElement))
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled()
    expect(screen.getByPlaceholderText('https://gateway.example.com/hermes')).not.toBeNull()
  })

  it('switches directly back to Local from an active saved remote', async () => {
    const remote = config({
      connections: [
        {
          id: 'home-lab',
          name: 'Home Lab',
          remoteAuthMode: 'token',
          remoteOauthConnected: false,
          remoteTokenPreview: 'se…et',
          remoteTokenSet: true,
          remoteUrl: 'https://home.example'
        }
      ],
      mode: 'remote',
      remoteTokenPreview: 'se…et',
      remoteTokenSet: true,
      remoteUrl: 'https://home.example',
      selectedConnectionId: 'home-lab',
      selectedConnectionName: 'Home Lab'
    })

    const local = config({ connections: remote.connections })

    getConnectionConfig.mockResolvedValue(remote)
    selectConnectionConfig.mockResolvedValue(local)
    render(<GatewaySettings embedded />)

    await screen.findByText('Home Lab')
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))

    await waitFor(() => expect(selectConnectionConfig).toHaveBeenCalledWith('local'))
    await waitFor(() =>
      expect(screen.getByText('Local gateway').parentElement?.parentElement?.textContent).toContain('Active')
    )
  })

  it('edits a saved remote without switching away from Local', async () => {
    const local = config({
      connections: [
        {
          id: 'home-lab',
          name: 'Home Lab',
          remoteAuthMode: 'token',
          remoteOauthConnected: false,
          remoteTokenPreview: null,
          remoteTokenSet: false,
          remoteUrl: 'https://home.example'
        }
      ]
    })

    getConnectionConfig.mockResolvedValue(local)
    render(<GatewaySettings />)

    await screen.findByText('Home Lab')
    fireEvent.click(screen.getByRole('button', { name: 'Change Home Lab' }))

    expect(screen.getByDisplayValue('Home Lab')).not.toBeNull()
    expect(selectConnectionConfig).not.toHaveBeenCalled()
    expect(screen.getByText('Local gateway').parentElement?.parentElement?.textContent).toContain('Active')
  })

  it('connects a saved remote immediately from the picker', async () => {
    const connection = {
      id: 'home-lab',
      name: 'Home Lab',
      remoteAuthMode: 'token' as const,
      remoteOauthConnected: false,
      remoteTokenPreview: 'se…et',
      remoteTokenSet: true,
      remoteUrl: 'https://home.example'
    }

    const local = config({ connections: [connection] })

    const remote = config({
      connections: [connection],
      mode: 'remote',
      remoteTokenPreview: connection.remoteTokenPreview,
      remoteTokenSet: true,
      remoteUrl: connection.remoteUrl,
      selectedConnectionId: connection.id,
      selectedConnectionName: connection.name
    })

    getConnectionConfig.mockResolvedValue(local)
    selectConnectionConfig.mockResolvedValue(remote)
    render(<GatewaySettings />)

    await screen.findByText('Home Lab')
    const savedConnection = screen.getByRole('group', { name: 'Home Lab' })

    expect(savedConnection.getAttribute('aria-current')).toBeNull()
    fireEvent.click(within(savedConnection).getByRole('button', { name: 'Connect' }))

    await waitFor(() => expect(selectConnectionConfig).toHaveBeenCalledWith('home-lab'))
    await waitFor(() =>
      expect(screen.getByRole('group', { name: 'Home Lab' }).getAttribute('aria-current')).toBe('true')
    )
  })

  it('opens Cloud management without changing the active Local target', async () => {
    window.hermesDesktop.cloud.status = vi.fn(() => new Promise<DesktopCloudStatus>(() => undefined))
    render(<GatewaySettings />)

    await screen.findByText('Connections')
    fireEvent.click(screen.getByRole('button', { name: 'Choose' }))

    expect(selectConnectionConfig).not.toHaveBeenCalled()
    expect(screen.getByText('Local gateway').parentElement?.parentElement?.textContent).toContain('Active')
    expect(screen.getByRole('button', { name: 'Sign in to Hermes Cloud' })).not.toBeNull()
  })

  it('keeps Local recovery enabled when an environment override is broken', async () => {
    const forced = config({
      envOverride: true,
      mode: 'remote',
      remoteUrl: 'https://broken.example'
    })

    const local = config()

    getConnectionConfig.mockResolvedValue(forced)
    selectConnectionConfig.mockResolvedValue(local)
    render(<GatewaySettings embedded />)

    await screen.findByText('Connections')
    const connect = screen.getByRole('button', { name: 'Connect' })

    expect(connect).not.toHaveProperty('disabled', true)
    fireEvent.click(connect)
    await waitFor(() => expect(selectConnectionConfig).toHaveBeenCalledWith('local'))
  })
})
