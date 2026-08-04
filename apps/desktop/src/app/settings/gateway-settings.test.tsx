import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopCloudAgent } from '@/global'
import type { ProfileInfo } from '@/types/hermes'

const getConnectionConfig = vi.fn()
const saveConnectionConfig = vi.fn()
const profiles = atom<ProfileInfo[]>([])
const starredCloudAgentIds = atom<string[]>([])
const refreshCloudAgentStars = vi.fn()
const setCloudAgentStarred = vi.fn()

vi.mock('@/store/profile', () => ({
  $profiles: profiles,
  refreshActiveProfile: vi.fn()
}))

vi.mock('@/store/gateway-switcher', () => ({
  $starredCloudAgentIds: starredCloudAgentIds,
  refreshCloudAgentStars,
  setCloudAgentStarred
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

const cloudAgent = (id: string, name = id): DesktopCloudAgent => ({
  dashboardGatewayState: 'active',
  dashboardUrl: `https://${id}.example.com`,
  id,
  name,
  status: 'running'
})

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
  starredCloudAgentIds.set([])
  refreshCloudAgentStars.mockReset().mockResolvedValue([])
  setCloudAgentStarred.mockReset().mockResolvedValue([])
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
  })

  it('stars and unstars a discovered Cloud agent without adding a separate saved-connection list', async () => {
    const { GatewaySettings } = await import('./gateway-settings')
    const agent = cloudAgent('prod', 'Production')
    starredCloudAgentIds.set([])
    setCloudAgentStarred.mockResolvedValue(['prod'])
    Object.assign(window.hermesDesktop, {
      cloud: {
        discover: vi.fn().mockResolvedValue({ agents: [agent], org: { id: 'team', slug: 'team' } }),
        login: vi.fn(),
        logout: vi.fn(),
        status: vi.fn().mockResolvedValue({ portalBaseUrl: 'https://portal.example', signedIn: true })
      }
    })

    render(<GatewaySettings />)
    await screen.findByText('Local gateway')
    fireEvent.click(screen.getByRole('button', { name: /Hermes Cloud/ }))
    expect(await screen.findByText('Production')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Star Production' }))

    await waitFor(() => expect(setCloudAgentStarred).toHaveBeenCalledWith('prod', true))
    expect(screen.queryByText('Gateway favorites')).toBeNull()
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
})
