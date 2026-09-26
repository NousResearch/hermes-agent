import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'
import { requestGatewayForAgent } from '@/store/gateway'
import { $desktopOnboarding, type DesktopOnboardingState, type OnboardingContext } from '@/store/onboarding'
import { makeOAuthProvider } from '@/test/oauth-provider'
import type { OAuthProvider } from '@/types/hermes'

import { DesktopOnboardingOverlay, Picker } from '.'

vi.mock('@/store/gateway', async importActual => ({
  ...(await importActual<Record<string, unknown>>()),
  requestGatewayForAgent: vi.fn(async () => ({ ok: true }))
}))

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

afterEach(() => {
  cleanup()
  vi.mocked(requestGatewayForAgent).mockReset()
  vi.mocked(requestGatewayForAgent).mockResolvedValue({ ok: true })

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
  it('prioritizes own keys and ChatGPT while keeping other providers available', () => {
    setProviders([
      makeOAuthProvider('openai-codex', 'OpenAI Codex'),
      makeOAuthProvider('anthropic', 'Anthropic Claude'),
      makeOAuthProvider('nous', 'Nous Portal')
    ])
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <Picker ctx={ctx} />
      </I18nProvider>
    )
    expect(screen.getByText('OpenRouter')).toBeTruthy()
    expect(screen.getByText('Gemini API')).toBeTruthy()
    expect(screen.getByText('ChatGPT / Codex')).toBeTruthy()
    expect(screen.queryByText('Nous Portal')).toBeNull()
    expect(screen.queryByText('Anthropic API Key')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Other providers' }))
    expect(screen.getByText('Anthropic API Key')).toBeTruthy()
    expect(screen.queryByText('Nous Portal')).toBeNull()
  })

  it('offers "choose later" on first run and persists the skip', () => {
    setProviders([makeOAuthProvider('nous', 'Nous Portal')])
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <Picker ctx={ctx} />
      </I18nProvider>
    )

    const skip = screen.getByRole('button', { name: "I'll choose a provider later" })

    fireEvent.click(skip)

    expect($desktopOnboarding.get().firstRunSkipped).toBe(true)
    expect(window.localStorage.getItem('hermes-onboarding-skipped-v1')).toBe('1')
  })

  it('hides "choose later" in manual (add-provider) mode', () => {
    setProviders([makeOAuthProvider('nous', 'Nous Portal')])
    $desktopOnboarding.set({ ...$desktopOnboarding.get(), manual: true })
    render(
      <I18nProvider configClient={null} initialLocale="en">
        <Picker ctx={ctx} />
      </I18nProvider>
    )

    expect(screen.queryByRole('button', { name: "I'll choose a provider later" })).toBeNull()
  })

  it('routes manual overlay RPC through the captured remote target after active request prop changes', async () => {
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        api: vi.fn(async ({ path }: { path: string }) => {
          if (path.startsWith('/api/model/options')) {
            return {
              model: 'llama-3',
              provider: 'fixture',
              providers: [{ authenticated: true, models: ['llama-3'], name: 'Fixture', slug: 'fixture' }]
            }
          }

          if (path.startsWith('/api/model/recommended-default')) {
            return { model: 'llama-3', provider: 'fixture', free_tier: null }
          }

          return { ok: true }
        })
      }
    })
    $desktopOnboarding.set({
      configured: true,
      flow: {
        copied: false,
        provider: { ...makeOAuthProvider('fixture'), flow: 'external', cli_command: 'hermes login fixture' },
        status: 'external_pending'
      },
      mode: 'oauth',
      providers: [makeOAuthProvider('fixture')],
      reason: null,
      requested: true,
      firstRunSkipped: false,
      manual: true,
      targetProfile: 'research',
      targetScope: { connectionId: 'remote-a', profile: 'research' },
      localEndpoint: false
    })

    const activeBRequest: OnboardingContext['requestGateway'] = vi.fn(async () => ({ ok: true }) as never)

    const queryClient = new QueryClient()

    render(
      <QueryClientProvider client={queryClient}>
        <I18nProvider configClient={null} initialLocale="en">
          <DesktopOnboardingOverlay enabled onCompleted={vi.fn()} profile="other" requestGateway={activeBRequest} />
        </I18nProvider>
      </QueryClientProvider>
    )

    fireEvent.click(await screen.findByRole('button', { name: "I've signed in" }))

    await waitFor(() =>
      expect(requestGatewayForAgent).toHaveBeenCalledWith('remote-a', 'research', 'reload.env', undefined)
    )
    expect(requestGatewayForAgent).toHaveBeenCalledWith('remote-a', 'research', 'setup.status', undefined)
    expect(activeBRequest).not.toHaveBeenCalled()
  })
})
