import { describe, expect, it, vi } from 'vitest'

import { type CloudAgentSessionBridge, reestablishCloudAgentSession } from './cloud-agent-session'

function bridge(overrides: {
  signedIn?: boolean
  login?: { ok: boolean; signedIn: boolean; cancelled?: boolean }
  connected?: boolean
}) {
  const desktop = {
    cloud: {
      status: vi.fn(async () => ({ signedIn: overrides.signedIn ?? false })),
      login: vi.fn(async () => overrides.login ?? { ok: true, signedIn: true }),
      agentSignIn: vi.fn(async () => ({ baseUrl: 'https://agent.example', connected: overrides.connected ?? true }))
    },
    oauthLogoutConnectionConfig: vi.fn(async () => undefined)
  }

  return desktop satisfies CloudAgentSessionBridge
}

describe('reestablishCloudAgentSession', () => {
  it('skips the browser when the portal session is live and exchanges for the agent', async () => {
    const desktop = bridge({ signedIn: true })
    const onBrowserSignIn = vi.fn()

    await expect(reestablishCloudAgentSession(desktop, 'https://agent.example', { onBrowserSignIn })).resolves.toBe(
      'connected'
    )
    expect(desktop.oauthLogoutConnectionConfig).toHaveBeenCalledWith('https://agent.example')
    expect(desktop.cloud.login).not.toHaveBeenCalled()
    expect(onBrowserSignIn).not.toHaveBeenCalled()
    // No agent id from the renderer: the audience is resolved in main.
    expect(desktop.cloud.agentSignIn).toHaveBeenCalledWith('https://agent.example')
  })

  it('reports the pending browser sign-in and ends it even on success', async () => {
    const desktop = bridge({ signedIn: false })
    const onBrowserSignIn = vi.fn()

    await expect(reestablishCloudAgentSession(desktop, 'https://agent.example', { onBrowserSignIn })).resolves.toBe(
      'connected'
    )
    expect(onBrowserSignIn.mock.calls).toEqual([[true], [false]])
  })

  it('returns cancelled (no exchange) when the user backs out in the browser', async () => {
    const desktop = bridge({ login: { ok: false, signedIn: false, cancelled: true } })
    const onBrowserSignIn = vi.fn()

    await expect(reestablishCloudAgentSession(desktop, 'https://agent.example', { onBrowserSignIn })).resolves.toBe(
      'cancelled'
    )
    expect(desktop.cloud.agentSignIn).not.toHaveBeenCalled()
    expect(onBrowserSignIn.mock.calls).toEqual([[true], [false]])
  })

  it('returns portal-incomplete when the browser sign-in did not complete', async () => {
    const desktop = bridge({ login: { ok: false, signedIn: false } })

    await expect(reestablishCloudAgentSession(desktop, 'https://agent.example')).resolves.toBe('portal-incomplete')
    expect(desktop.cloud.agentSignIn).not.toHaveBeenCalled()
  })

  it('clears the pending state when the browser sign-in throws', async () => {
    const desktop = bridge({})
    desktop.cloud.login.mockRejectedValueOnce(new Error('boom'))
    const onBrowserSignIn = vi.fn()

    await expect(reestablishCloudAgentSession(desktop, 'https://agent.example', { onBrowserSignIn })).rejects.toThrow(
      'boom'
    )
    expect(onBrowserSignIn.mock.calls).toEqual([[true], [false]])
  })
})
