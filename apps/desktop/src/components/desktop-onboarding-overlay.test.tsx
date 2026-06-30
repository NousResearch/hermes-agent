import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { $desktopOnboarding, type DesktopOnboardingState, type OnboardingContext } from '@/store/onboarding'
import type { OAuthProvider } from '@/types/hermes'

import { FlowPanel, Picker } from './desktop-onboarding-overlay'

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
    requested: false,
    firstRunSkipped: false,
    manual: false,
    localEndpoint: false
  } satisfies DesktopOnboardingState)
}

const ctx: OnboardingContext = { requestGateway: async () => undefined as never }

function ensureLocalStorage() {
  if (window.localStorage) {
    return
  }

  const store = new Map<string, string>()
  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: {
      clear: () => store.clear(),
      getItem: (key: string) => store.get(key) ?? null,
      removeItem: (key: string) => store.delete(key),
      setItem: (key: string, value: string) => {
        store.set(key, String(value))
      }
    }
  })
}

beforeEach(() => {
  ensureLocalStorage()
})

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
    requested: false,
    firstRunSkipped: false,
    manual: false,
    localEndpoint: false
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
})

describe('onboarding FlowPanel error recovery', () => {
  it('offers direct API-key recovery for missing provider credentials', () => {
    $desktopOnboarding.set({
      configured: false,
      flow: {
        status: 'error',
        message:
          'Connected, but Hermes still cannot resolve a usable provider. No usable credentials found for openrouter.'
      },
      mode: 'oauth',
      providers: [provider('nous', 'Nous Portal')],
      reason: null,
      requested: true,
      firstRunSkipped: false,
      manual: false,
      localEndpoint: false
    })

    render(
      <FlowPanel
        ctx={ctx}
        flow={$desktopOnboarding.get().flow}
        leaving={false}
        onBegin={() => undefined}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'I have an API key' }))

    expect($desktopOnboarding.get().flow.status).toBe('idle')
    expect($desktopOnboarding.get().mode).toBe('apikey')
  })

  it('does not show API-key recovery for non-credential runtime errors', () => {
    render(
      <FlowPanel
        ctx={ctx}
        flow={{ status: 'error', message: 'Selected runtime is not available.' }}
        leaving={false}
        onBegin={() => undefined}
      />
    )

    expect(screen.queryByRole('button', { name: 'I have an API key' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Pick a different provider' })).toBeTruthy()
  })
})
