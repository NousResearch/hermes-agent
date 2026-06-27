import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { $desktopOnboarding, type DesktopOnboardingState, type OnboardingContext } from '@/store/onboarding'
import type { OAuthProvider } from '@/types/hermes'

import { Picker } from './desktop-onboarding-overlay'

function provider(id: string, name = id): OAuthProvider {
  return {
    cli_command: `hermes login ${id}`,
    docs_url: `https://example.com/${id}`,
    flow: 'pkce',
    id,
    name,
    status: { logged_in: false }
  }
}

function setProviders(providers: OAuthProvider[]) {
  $desktopOnboarding.set({
    configured: false,
    flow: { status: 'idle' },
    mode: 'oauth',
    providers,
    reason: null,
    needsCredential: false,
    requested: false,
    firstRunSkipped: false,
    manual: false,
    localEndpoint: false,
    managedAvailable: false,
    managedError: null,
    managedSubmitting: false
  } satisfies DesktopOnboardingState)
}

const ctx: OnboardingContext = { requestGateway: async () => undefined as never }

afterEach(() => {
  cleanup()

  try {
    window.localStorage.clear()
  } catch {
    // jsdom localStorage should always be present; ignore if not.
  }

  $desktopOnboarding.set({
    configured: null,
    flow: { status: 'idle' },
    mode: 'oauth',
    providers: null,
    reason: null,
    needsCredential: false,
    requested: false,
    firstRunSkipped: false,
    manual: false,
    localEndpoint: false,
    managedAvailable: false,
    managedError: null,
    managedSubmitting: false
  })
})

describe('onboarding Picker', () => {
  it('features Nous Portal and hides other providers behind a disclosure', () => {
    setProviders([provider('anthropic', 'Anthropic Claude'), provider('nous', 'Nous Portal')])
    render(<Picker ctx={ctx} />)

    expect(screen.getByText('Nous Portal')).toBeTruthy()
    expect(screen.getByText('Recommended')).toBeTruthy()
    expect(screen.queryByText('Anthropic API Key')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Other providers' }))

    expect(screen.getByText('Anthropic API Key')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Collapse' })).toBeTruthy()
  })

  it('shows every provider directly when Nous Portal is absent', () => {
    setProviders([provider('anthropic', 'Anthropic Claude'), provider('openai-codex', 'OpenAI Codex / ChatGPT')])
    render(<Picker ctx={ctx} />)

    expect(screen.getByText('Anthropic API Key')).toBeTruthy()
    expect(screen.getByText('OpenAI OAuth (ChatGPT)')).toBeTruthy()
    expect(screen.queryByText('Other sign-in options')).toBeNull()
    expect(screen.queryByText('Recommended')).toBeNull()
  })

  it('offers "choose later" on first run and persists the skip', () => {
    setProviders([provider('nous', 'Nous Portal')])
    render(<Picker ctx={ctx} />)

    const skip = screen.getByRole('button', { name: "I'll choose a provider later" })

    fireEvent.click(skip)

    expect($desktopOnboarding.get().firstRunSkipped).toBe(true)
    expect(window.localStorage.getItem('hermes-onboarding-skipped-v1')).toBe('1')
  })

  it('hides "choose later" in manual (add-provider) mode', () => {
    setProviders([provider('nous', 'Nous Portal')])
    $desktopOnboarding.set({ ...$desktopOnboarding.get(), manual: true })
    render(<Picker ctx={ctx} />)

    expect(screen.queryByRole('button', { name: "I'll choose a provider later" })).toBeNull()
  })

  it('leads the API-key form with DeepSeek and collapses international providers', () => {
    // No OAuth providers → the API-key form renders directly. The backend model
    // catalog is unavailable in tests, so it falls back to the curated options.
    setProviders([])
    render(<Picker ctx={ctx} />)

    // Domestic providers shown up front (DeepSeek default + the rest).
    expect(screen.getByText('DeepSeek')).toBeTruthy()
    expect(screen.getByText('DashScope (Qwen)')).toBeTruthy()
    // International providers stay hidden until the disclosure is opened.
    expect(screen.queryByText('OpenRouter')).toBeNull()
    expect(screen.queryByText('OpenAI')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'More (needs a VPN)' }))

    expect(screen.getByText('OpenRouter')).toBeTruthy()
    expect(screen.getByText('OpenAI')).toBeTruthy()
  })

  it('shows a clean add-key prompt when a provider is seeded without a key', () => {
    // The DeepSeek-seed happy path: needsCredential → land on the key form with a
    // friendly prompt instead of the raw runtime error.
    $desktopOnboarding.set({ ...$desktopOnboarding.get(), configured: false, mode: 'apikey', needsCredential: true })
    render(<Picker ctx={ctx} />)

    expect(screen.getByText('Your provider is selected — just add its API key to start chatting.')).toBeTruthy()
    expect(screen.getByText('DeepSeek')).toBeTruthy()
  })
})
