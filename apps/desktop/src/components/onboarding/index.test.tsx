import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $desktopOnboarding, type DesktopOnboardingState, type OnboardingContext } from '@/store/onboarding'
import { makeOAuthProvider } from '@/test/oauth-provider'
import type { OAuthProvider } from '@/types/hermes'

import { ApiKeyForm, Picker } from '.'

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
    localEndpoint: false,
    freeTierReady: false
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
    requested: false,
    firstRunSkipped: false,
    manual: false,
    localEndpoint: false,
    freeTierReady: false
  })
})

describe('onboarding Picker', () => {
  it('features Nous Portal and hides other providers behind a disclosure', () => {
    setProviders([makeOAuthProvider('anthropic', 'Anthropic Claude'), makeOAuthProvider('nous', 'Nous Portal')])
    render(<Picker ctx={ctx} />)

    expect(screen.getByText('Nous Portal')).toBeTruthy()
    expect(screen.getByText('Recommended')).toBeTruthy()
    // Fireworks stays behind the disclosure with the other alternatives; only
    // Nous Portal is visible before the user expands the list.
    expect(screen.queryByText('Fireworks AI')).toBeNull()
    expect(screen.queryByText('Anthropic Account')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Other providers' }))

    expect(screen.getByText('Fireworks AI')).toBeTruthy()
    expect(screen.getByText('Anthropic Account')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Collapse' })).toBeTruthy()
  })

  it('shows every provider directly when Nous Portal is absent', () => {
    setProviders([
      makeOAuthProvider('anthropic', 'Anthropic Claude'),
      makeOAuthProvider('openai-codex', 'OpenAI Codex / ChatGPT')
    ])
    render(<Picker ctx={ctx} />)

    expect(screen.getByText('Fireworks AI')).toBeTruthy()
    expect(screen.getByText('Anthropic Account')).toBeTruthy()
    expect(screen.getByText('ChatGPT or Codex Subscription')).toBeTruthy()
    expect(screen.queryByText('Other sign-in options')).toBeNull()
    expect(screen.queryByText('Recommended')).toBeNull()
  })

  it('offers "choose later" on first run and persists the skip', () => {
    setProviders([makeOAuthProvider('nous', 'Nous Portal')])
    render(<Picker ctx={ctx} />)

    const skip = screen.getByRole('button', { name: "I'll choose a provider later" })

    fireEvent.click(skip)

    expect($desktopOnboarding.get().firstRunSkipped).toBe(true)
    expect(window.localStorage.getItem('hermes-onboarding-skipped-v1')).toBe('1')
  })

  it('hides "choose later" in manual (add-provider) mode', () => {
    setProviders([makeOAuthProvider('nous', 'Nous Portal')])
    $desktopOnboarding.set({ ...$desktopOnboarding.get(), manual: true })
    render(<Picker ctx={ctx} />)

    expect(screen.queryByRole('button', { name: "I'll choose a provider later" })).toBeNull()
  })
})

describe('ApiKeyForm manual local-model fallback', () => {
  it('reveals the model-name input only after the endpoint enumerates no models, then forwards the name', async () => {
    // First Connect: reachable endpoint, empty /v1/models — the wizard must ask
    // for a manual model name instead of dead-ending. Second Connect: the typed
    // name is forwarded as the 5th onSave argument.
    const onSave = vi
      .fn<
        (
          envKey: string,
          value: string,
          name: string,
          apiKey?: string,
          modelName?: string
        ) => Promise<{ message?: string; needsModelInput?: boolean; ok: boolean }>
      >()
      .mockResolvedValueOnce({
        ok: false,
        needsModelInput: true,
        message: "Connected, but it didn't enumerate any models at /v1/models."
      })
      .mockResolvedValueOnce({ ok: true })

    render(<ApiKeyForm canGoBack={false} initialEnvKey="OPENAI_BASE_URL" onBack={() => undefined} onSave={onSave} />)

    fireEvent.change(screen.getByPlaceholderText('http://127.0.0.1:8000/v1'), {
      target: { value: 'https://api.cohere.ai/compatibility/v1' }
    })

    // Hidden on the happy path — discovery hasn't failed yet.
    expect(screen.queryByPlaceholderText('Model name (e.g. command-a-plus-05-2026)')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))

    await waitFor(() => {
      expect(screen.getByPlaceholderText('Model name (e.g. command-a-plus-05-2026)')).toBeTruthy()
    })

    fireEvent.change(screen.getByPlaceholderText('Model name (e.g. command-a-plus-05-2026)'), {
      target: { value: 'command-a-plus-05-2026' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }))

    await waitFor(() => {
      // apiKey is the (empty) local-key field, forwarded as-is for the local option.
      expect(onSave).toHaveBeenLastCalledWith(
        'OPENAI_BASE_URL',
        'https://api.cohere.ai/compatibility/v1',
        'Local / custom endpoint',
        '',
        'command-a-plus-05-2026'
      )
    })
  })
})
