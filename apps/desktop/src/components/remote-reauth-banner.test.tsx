import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { HermesConnection } from '@/global'
import { $remoteReauthExhausted, latchRemoteReauthExhausted } from '@/store/remote-reauth'
import { $connection } from '@/store/session'

import { RemoteReauthBanner } from './remote-reauth-banner'

const GATEWAY = 'https://gateway.example.test'
const OTHER = 'https://other.example.test'

const connection = (baseUrl: string) =>
  ({
    authMode: 'oauth',
    baseUrl,
    isFullscreen: false,
    logs: [],
    mode: 'remote',
    nativeOverlayWidth: 0,
    token: '',
    wsUrl: 'wss://gateway.example.test/api/ws'
  }) as unknown as HermesConnection

describe('RemoteReauthBanner', () => {
  const oauthLoginConnectionConfig = vi.fn()

  beforeEach(() => {
    ;(window as unknown as { hermesDesktop?: unknown }).hermesDesktop = {
      oauthLoginConnectionConfig
    }
    $connection.set(connection(GATEWAY))
    $remoteReauthExhausted.set({})
  })

  afterEach(() => {
    cleanup()
    oauthLoginConnectionConfig.mockReset()
    $connection.set(null)
    $remoteReauthExhausted.set({})
    delete (window as unknown as { hermesDesktop?: unknown }).hermesDesktop
  })

  it('offers one manual sign-in for the ACTIVE gateway once automatic sign-in exhausted', () => {
    latchRemoteReauthExhausted(GATEWAY)

    render(<RemoteReauthBanner />)

    expect(screen.getByRole('alert')).toBeTruthy()
    expect(screen.getByText('Reconnection required')).toBeTruthy()
    expect(screen.getByText(/gateway\.example\.test/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Sign in' })).toBeTruthy()
  })

  it('never covers the workspace for another gateway’s failure (background saved row)', () => {
    latchRemoteReauthExhausted(OTHER)

    render(<RemoteReauthBanner />)

    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('renders nothing without an active connection', () => {
    $connection.set(null)

    latchRemoteReauthExhausted(GATEWAY)

    render(<RemoteReauthBanner />)

    expect(screen.queryByRole('alert')).toBeNull()
  })

  it('a successful manual sign-in clears the banner', async () => {
    latchRemoteReauthExhausted(GATEWAY)
    oauthLoginConnectionConfig.mockResolvedValue({ baseUrl: GATEWAY, connected: true, ok: true })

    render(<RemoteReauthBanner />)
    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }))

    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
    expect(oauthLoginConnectionConfig).toHaveBeenCalledWith(GATEWAY)
  })

  it('keeps the banner (and the way forward) when the manual sign-in fails', async () => {
    latchRemoteReauthExhausted(GATEWAY)
    oauthLoginConnectionConfig.mockResolvedValue({
      baseUrl: GATEWAY,
      connected: false,
      error: 'bad credentials',
      ok: false
    })

    render(<RemoteReauthBanner />)
    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }))

    await waitFor(() => expect(screen.getByRole('button', { name: 'Sign in' })).toBeTruthy())
    expect(screen.getByRole('alert')).toBeTruthy()
  })

  it('disables the action while the manual sign-in is in flight', async () => {
    latchRemoteReauthExhausted(GATEWAY)

    let settle!: (value: unknown) => void

    oauthLoginConnectionConfig.mockReturnValue(
      new Promise(resolve => {
        settle = resolve
      })
    )

    render(<RemoteReauthBanner />)
    fireEvent.click(screen.getByRole('button', { name: 'Sign in' }))

    await waitFor(() => expect(screen.getByText('Signing in…')).toBeTruthy())

    const button = screen.getByRole('button', { name: 'Signing in…' }) as HTMLButtonElement

    expect(button.disabled).toBe(true)

    settle({ baseUrl: GATEWAY, connected: true, ok: true })

    await waitFor(() => expect(screen.queryByRole('alert')).toBeNull())
  })
})
