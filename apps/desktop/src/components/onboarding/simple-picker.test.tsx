import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import type { DesktopConnectionKind } from '@/global'
import { $connectionsRegistry } from '@/store/connection-registry-state'
import { $interfaceMode, setInterfaceMode } from '@/store/interface-mode'
import { $desktopOnboarding, type DesktopOnboardingState, type OnboardingContext } from '@/store/onboarding'
import { makeOAuthProvider } from '@/test/oauth-provider'

import { SimplePicker } from './simple-picker'

import { DesktopOnboardingOverlay } from '.'

const ctx: OnboardingContext = {
  requestGateway: async () => undefined as never,
  scope: { connectionId: 'fixture', profile: null }
}

// Never answers: the readiness effect stays in flight, so the overlay is
// observed on exactly the state each case set up.
const pendingGateway = (() => new Promise(() => {})) as OnboardingContext['requestGateway']

const anthropic = { ...makeOAuthProvider('anthropic', 'Anthropic Account'), flow: 'external' as const }

function registerConnection(kind: DesktopConnectionKind) {
  $connectionsRegistry.set({
    version: 1,
    primary: 'fixture',
    secureTokenStorage: true,
    connections: [{ id: 'fixture', kind, label: 'Fixture', tokenSet: false, tokenPreview: null }]
  })
}

function firstRun(overrides: Partial<DesktopOnboardingState> = {}): DesktopOnboardingState {
  return {
    configured: false,
    flow: { status: 'idle' },
    mode: 'oauth',
    providers: [makeOAuthProvider('nous', 'Nous Portal'), makeOAuthProvider('openai-codex', 'OpenAI Codex'), anthropic],
    reason: null,
    requested: false,
    firstRunSkipped: false,
    manual: false,
    localEndpoint: false,
    freeTierReady: false,
    modeChosen: false,
    ...overrides
  }
}

afterEach(() => {
  cleanup()
  setInterfaceMode('advanced')
  window.localStorage.clear()
  $connectionsRegistry.set(null)
  $desktopOnboarding.set(firstRun({ providers: null }))
})

describe('SimplePicker', () => {
  it('shows a tile only for account sign-ins the backend offers, and no key form', () => {
    registerConnection('local')
    $desktopOnboarding.set(firstRun())

    render(<SimplePicker ctx={ctx} />)

    expect(screen.getByText('Nous')).toBeTruthy()
    expect(screen.getByText('ChatGPT')).toBeTruthy()
    // This backend has no xai-oauth, so there is no Grok door to a failing flow.
    expect(screen.queryByText('Grok')).toBeNull()
    expect(screen.queryByRole('textbox')).toBeNull()
  })

  it('starts the terminal sign-in from the Claude tile on a local backend', () => {
    registerConnection('local')
    $desktopOnboarding.set(firstRun())

    render(<SimplePicker ctx={ctx} />)
    fireEvent.click(screen.getByText('Claude'))

    expect($desktopOnboarding.get().flow).toMatchObject({ status: 'external_pending', provider: { id: 'anthropic' } })
  })

  it('offers no Claude tile for a remote backend or one without Anthropic', () => {
    registerConnection('remote')
    $desktopOnboarding.set(firstRun())

    const { unmount } = render(<SimplePicker ctx={ctx} />)

    expect(screen.queryByText('Claude')).toBeNull()
    unmount()

    registerConnection('local')
    $desktopOnboarding.set(firstRun({ providers: [makeOAuthProvider('nous', 'Nous Portal')] }))
    render(<SimplePicker ctx={ctx} />)

    expect(screen.queryByText('Claude')).toBeNull()
  })
})

describe('DesktopOnboardingOverlay first-run mode choice', () => {
  it('asks once; Simple shows the tiles and "Switch to Advanced" brings the full picker', () => {
    $desktopOnboarding.set(firstRun())

    render(<DesktopOnboardingOverlay enabled profile="default" requestGateway={pendingGateway} />)

    fireEvent.click(screen.getByRole('button', { name: /^Simple/ }))

    expect($interfaceMode.get()).toBe('simple')
    expect(window.localStorage.getItem('hermes-onboarding-mode-chosen-v1')).toBe('1')
    expect(screen.queryByText('How do you want to use Hermes?')).toBeNull()
    expect(screen.getByText('Needs ChatGPT Plus or Pro')).toBeTruthy()
    expect(screen.getByRole('button', { name: "I'll choose a provider later" })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Switch to Advanced' }))

    expect($interfaceMode.get()).toBe('advanced')
    expect(screen.getByRole('button', { name: "I'll choose a provider later" })).toBeTruthy()
  })

  it.each<[string, Partial<DesktopOnboardingState>]>([
    ['hit a credential wall', { requested: true }],
    ['was configured before or already answered', { modeChosen: true }]
  ])('never asks a user who %s', (_, overrides) => {
    $desktopOnboarding.set(firstRun(overrides))

    render(<DesktopOnboardingOverlay enabled profile="default" requestGateway={pendingGateway} />)

    expect(screen.queryByText('How do you want to use Hermes?')).toBeNull()
    expect(screen.getByText("Let's get you setup with Hermes Agent")).toBeTruthy()
  })
})
