import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionConfig, DesktopConnectionProbeResult } from '@/global'

import { GatewaySettings } from './gateway-settings'

// The dedicated boot-failure reauth button (65712bf78) clears the OAuth
// partition before opening the login window so a stale identity-provider
// cookie can't bounce a fresh sign-in back into the same broken session. The
// routine Settings -> Gateway "Sign in" button is the OTHER door into the
// exact same oauth-login IPC call and was left doing a bare login with no
// logout first — this proves it now clears the partition first too.

const OAUTH_URL = 'https://gateway.example.com'

function baseConfig(patch: Partial<DesktopConnectionConfig> = {}): DesktopConnectionConfig {
  return {
    envOverride: false,
    mode: 'remote',
    profile: null,
    remoteAuthMode: 'oauth',
    remoteOauthConnected: false,
    remoteTokenPreview: null,
    remoteTokenSet: true, // lets authResolved settle without waiting on the debounced probe
    remoteUrl: OAUTH_URL,
    cloudOrg: ''
  }
}

function probeResult(): DesktopConnectionProbeResult {
  return {
    baseUrl: OAUTH_URL,
    reachable: true,
    authMode: 'oauth',
    providers: [{ name: 'nous', displayName: 'Nous', supportsPassword: false }],
    version: null,
    error: null
  }
}

function stubDesktop(config: DesktopConnectionConfig) {
  const calls: string[] = []

  const oauthLogoutConnectionConfig = vi.fn(async () => {
    calls.push('logout')

    return { ok: true, connected: false }
  })

  const oauthLoginConnectionConfig = vi.fn(async () => {
    calls.push('login')

    return { ok: true, baseUrl: OAUTH_URL, connected: true }
  })

  const original = window.hermesDesktop
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value: {
      getConnectionConfig: async () => config,
      saveConnectionConfig: async (payload: unknown) => ({ ...config, ...(payload as object) }),
      probeConnectionConfig: async () => probeResult(),
      oauthLoginConnectionConfig,
      oauthLogoutConnectionConfig
    }
  })

  return {
    calls,
    oauthLoginConnectionConfig,
    oauthLogoutConnectionConfig,
    restore: () => Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: original })
  }
}

afterEach(cleanup)

describe('GatewaySettings sign-in', () => {
  it('clears the OAuth partition before opening the login window', async () => {
    const desktop = stubDesktop(baseConfig())

    try {
      render(<GatewaySettings embedded />)

      // The auth-mode probe is debounced 500ms; the button only renders once
      // it resolves. Its label is "Sign in with <provider>" (the probe's
      // single non-password provider, "Nous") — an exact match, since "Sign
      // in to Hermes Cloud" (the cloud-mode panel, always rendered alongside
      // the mode picker) also contains the substring "sign in".
      const signInButton = await screen.findByRole('button', { name: 'Sign in with Nous' }, { timeout: 3000 })
      fireEvent.click(signInButton)

      await waitFor(() => expect(desktop.oauthLoginConnectionConfig).toHaveBeenCalled(), { timeout: 2000 })

      expect(desktop.oauthLogoutConnectionConfig).toHaveBeenCalledWith(OAUTH_URL)
      expect(desktop.oauthLoginConnectionConfig).toHaveBeenCalledWith(OAUTH_URL)
      // Logout must complete before the login window opens, not just both fire.
      expect(desktop.calls).toEqual(['logout', 'login'])
    } finally {
      desktop.restore()
    }
  })
})
