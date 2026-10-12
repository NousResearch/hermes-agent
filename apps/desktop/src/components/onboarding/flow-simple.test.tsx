import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopConnectionKind } from '@/global'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { setInterfaceMode } from '@/store/interface-mode'
import {
  $desktopOnboarding,
  type OnboardingContext,
  type OnboardingFlow,
  recheckExternalSignin
} from '@/store/onboarding'
import { makeOAuthProvider } from '@/test/oauth-provider'

import { FlowPanel } from './flow'

import { DesktopOnboardingOverlay } from '.'

vi.mock('@/store/onboarding', async importOriginal => ({
  ...(await importOriginal<object>()),
  recheckExternalSignin: vi.fn(async () => undefined)
}))

// Never answers, so the overlay is observed on exactly the state each case set up.
const pendingGateway = (() => new Promise(() => {})) as OnboardingContext['requestGateway']

const ctx: OnboardingContext = { requestGateway: pendingGateway, scope: { connectionId: 'fixture', profile: 'work' } }

const anthropic = { ...makeOAuthProvider('anthropic', 'Anthropic Account'), flow: 'external' as const }
const openAuthInTerminal = vi.fn<(providerId: string, opts?: { profile?: string }) => Promise<{ ok: boolean }>>()

function registerConnection(kind: DesktopConnectionKind) {
  $connectionsRegistry.set({
    version: 1,
    primary: 'fixture',
    secureTokenStorage: true,
    connections: [{ id: 'fixture', kind, label: 'Fixture', tokenSet: false, tokenPreview: null }]
  })
}

function renderFlow(flow: OnboardingFlow) {
  return render(<FlowPanel ctx={ctx} flow={flow} leaving={false} onBegin={() => undefined} />)
}

beforeEach(() => {
  setInterfaceMode('simple')
  openAuthInTerminal.mockResolvedValue({ ok: true })
  Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { openAuthInTerminal } })
})

afterEach(() => {
  cleanup()
  openAuthInTerminal.mockReset()
  setInterfaceMode('advanced')
  $connectionsRegistry.set(null)
})

describe('Simple Claude sign-in in the terminal', () => {
  it('opens one terminal per sign-in on a local backend, pinned to the profile', async () => {
    registerConnection('local')
    const flow: OnboardingFlow = { copied: false, provider: anthropic, status: 'external_pending' }

    const { rerender } = renderFlow(flow)
    rerender(<FlowPanel ctx={ctx} flow={flow} leaving={false} onBegin={() => undefined} />)

    await waitFor(() => expect(openAuthInTerminal).toHaveBeenCalledTimes(1))
    expect(openAuthInTerminal).toHaveBeenCalledWith('anthropic', { profile: 'work' })
    expect(screen.queryByText(anthropic.cli_command)).toBeNull()
  })

  it('never opens a local terminal for a remote backend; it shows the command instead', () => {
    registerConnection('remote')
    renderFlow({ copied: false, provider: anthropic, status: 'external_pending' })

    expect(openAuthInTerminal).not.toHaveBeenCalled()
    expect(screen.getByText(anthropic.cli_command)).toBeTruthy()
  })

  it('moves on by itself once the backend sees the Claude sign-in, and keeps failed checks silent', async () => {
    vi.useFakeTimers()
    registerConnection('local')
    let loggedIn = false
    let calls = 0

    const api = vi.fn(async () => {
      calls += 1

      if (calls === 1) {
        throw new Error('backend restarting')
      }

      return { providers: [{ ...anthropic, status: { ...anthropic.status, logged_in: loggedIn } }] }
    })

    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api, openAuthInTerminal } })

    try {
      renderFlow({ copied: false, provider: anthropic, status: 'external_pending' })

      await vi.advanceTimersByTimeAsync(6_000)
      expect(api).toHaveBeenCalledTimes(2)
      expect(recheckExternalSignin).not.toHaveBeenCalled()
      expect(screen.queryByRole('alert')).toBeNull()

      loggedIn = true
      await vi.advanceTimersByTimeAsync(9_000)

      expect(recheckExternalSignin).toHaveBeenCalledOnce()
    } finally {
      vi.useRealTimers()
    }
  })

  it('does not cut the sign-in short when a Claude credential already existed', async () => {
    vi.useFakeTimers()
    vi.mocked(recheckExternalSignin).mockClear()
    registerConnection('local')

    const api = vi.fn(async () => ({ providers: [{ ...anthropic, status: { ...anthropic.status, logged_in: true } }] }))

    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api, openAuthInTerminal } })

    try {
      renderFlow({ copied: false, provider: anthropic, status: 'external_pending' })

      await vi.advanceTimersByTimeAsync(9_000)
      expect(api).toHaveBeenCalledTimes(3)
      expect(recheckExternalSignin).not.toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
    }
  })

  it('falls back to the command when no terminal could be opened', async () => {
    registerConnection('local')
    openAuthInTerminal.mockResolvedValue({ ok: false })
    renderFlow({ copied: false, provider: anthropic, status: 'external_pending' })

    expect(await screen.findByText(anthropic.cli_command)).toBeTruthy()
  })
})

describe('Simple sign-in errors', () => {
  it('names the tile in plain words and keeps the backend wording behind Details', () => {
    renderFlow({
      message: "No Anthropic credentials found. Run 'hermes auth add anthropic' to sign in, or set ANTHROPIC_API_KEY.",
      provider: anthropic,
      status: 'error'
    })

    expect(screen.getByText(/doesn't see your Claude sign-in yet/)).toBeTruthy()
    expect(screen.queryByRole('button', { name: 'Use an API key' })).toBeNull()

    fireEvent.click(screen.getByText('Details'))
    expect(screen.getByText(/ANTHROPIC_API_KEY/)).toBeTruthy()
  })

  it('keeps the backend message as the headline in Advanced', () => {
    setInterfaceMode('advanced')
    renderFlow({ message: 'Sign-in failed upstream.', provider: anthropic, status: 'error' })

    expect(screen.getByText('Sign-in failed upstream.')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Use an API key' })).toBeTruthy()
  })
})

describe('Simple when the sign-in list cannot load', () => {
  it('stays on the account screen with a retry instead of an API key form', () => {
    registerConnection('local')
    $desktopOnboarding.set({
      configured: false,
      firstRunSkipped: false,
      flow: { status: 'idle' },
      freeTierReady: false,
      localEndpoint: false,
      manual: false,
      mode: 'apikey',
      modeChosen: true,
      providers: [],
      reason: null,
      requested: false
    })

    render(<DesktopOnboardingOverlay enabled profile="default" requestGateway={pendingGateway} />)

    expect(screen.getByText(/couldn't load the sign-in options/)).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Try again' })).toBeTruthy()
    expect(screen.queryByRole('textbox')).toBeNull()
  })
})
