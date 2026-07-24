import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const profiles = atom<Array<{ name: string }>>([])

vi.mock('@/store/profile', () => ({
  $profiles: profiles,
  refreshActiveProfile: vi.fn().mockResolvedValue(undefined)
}))

vi.mock('@/store/notifications', () => ({
  notify: vi.fn(),
  notifyError: vi.fn()
}))

const baseConfig = {
  cloudOrg: '',
  envOverride: false,
  mode: 'local' as const,
  remoteAuthMode: 'token' as const,
  remoteOauthConnected: false,
  remoteTokenPreview: null,
  remoteTokenSet: false,
  remoteUrl: ''
}

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true })
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      applyConnectionConfig: vi.fn(),
      getConnectionConfig: vi.fn().mockResolvedValue(baseConfig),
      oauthLoginConnectionConfig: vi.fn().mockResolvedValue({
        baseUrl: 'http://192.0.2.10:9119',
        connected: false,
        ok: true
      }),
      oauthLogoutConnectionConfig: vi.fn(),
      probeConnectionConfig: vi.fn().mockResolvedValue({
        authMode: 'oauth',
        baseUrl: 'http://192.0.2.10:9119',
        error: null,
        providers: [{ displayName: 'Username & Password', name: 'basic', supportsPassword: true }],
        reachable: true,
        version: '0.18.2'
      }),
      saveConnectionConfig: vi.fn().mockImplementation(input =>
        Promise.resolve({
          ...baseConfig,
          mode: input.mode,
          remoteAuthMode: input.remoteAuthMode,
          remoteUrl: input.remoteUrl
        })
      ),
      testConnectionConfig: vi.fn()
    }
  })
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.clearAllMocks()
})

describe('GatewaySettings remote auth discovery', () => {
  it('offers username/password sign-in after a basic-auth gateway probe', async () => {
    const { GatewaySettings } = await import('./gateway-settings')

    render(<GatewaySettings />)
    await screen.findByRole('button', { name: /Remote gateway/ })

    fireEvent.click(screen.getByRole('button', { name: /Remote gateway/ }))
    fireEvent.change(screen.getByPlaceholderText('https://gateway.example.com/hermes'), {
      target: { value: 'http://192.0.2.10:9119' }
    })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(500)
    })

    await waitFor(() =>
      expect(window.hermesDesktop.probeConnectionConfig).toHaveBeenCalledWith('http://192.0.2.10:9119')
    )
    const signIn = await screen.findByRole('button', { name: 'Sign in' })
    expect(screen.getByText(/uses a username and password/i)).toBeTruthy()
    expect(screen.queryByText('Session token')).toBeNull()

    fireEvent.click(signIn)

    await waitFor(() =>
      expect(window.hermesDesktop.saveConnectionConfig).toHaveBeenCalledWith({
        mode: 'remote',
        profile: undefined,
        remoteAuthMode: 'oauth',
        remoteUrl: 'http://192.0.2.10:9119'
      })
    )
    expect(window.hermesDesktop.oauthLoginConnectionConfig).toHaveBeenCalledWith('http://192.0.2.10:9119')
  })
})
