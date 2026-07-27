import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileInfo } from '@/types/hermes'

const getConnectionConfig = vi.fn()
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
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: { getConnectionConfig }
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
})
