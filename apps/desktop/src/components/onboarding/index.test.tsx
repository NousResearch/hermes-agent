import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as ModelOptionsModule from '@/lib/model-options'
import { $desktopOnboarding, type DesktopOnboardingState, type OnboardingContext } from '@/store/onboarding'
import { $connection } from '@/store/session'
import { $settingsOwner } from '@/store/settings-scope'
import { makeOAuthProvider } from '@/test/oauth-provider'
import type { OAuthProvider } from '@/types/hermes'

const requestModelOptions = vi.hoisted(() => vi.fn())

vi.mock('@/lib/model-options', async importOriginal => ({
  ...(await importOriginal<typeof ModelOptionsModule>()),
  requestModelOptions
}))

import { FlowPanel } from './flow'

import { DesktopOnboardingOverlay, Picker } from '.'

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

beforeEach(() => {
  requestModelOptions.mockResolvedValue({ model: '', provider: '', providers: [] })
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
    localEndpoint: false,
    freeTierReady: false
  })
  $connection.set(null)
  vi.clearAllMocks()
})

describe('onboarding Picker', () => {
  it('uses the captured owner for catalog REST recovery', async () => {
    const scope = {
      connectionId: null,
      profile: 'remote-profile',
      legacyConnection: {
        baseUrl: 'https://legacy.example',
        headers: { 'Cf-Access-Client-Id': 'private-client' },
        mode: 'remote',
        token: 'private-token'
      }
    } as const

    setProviders([makeOAuthProvider('nous', 'Nous Portal')])
    render(<Picker ctx={{ profile: 'remote-profile', requestGateway: vi.fn(), scope }} />)

    await waitFor(() =>
      expect(requestModelOptions).toHaveBeenCalledWith(
        expect.objectContaining({ explicitOnly: false, profile: 'remote-profile', scope })
      )
    )
  })

  it('features Nous Portal and hides other providers behind a disclosure', () => {
    setProviders([makeOAuthProvider('anthropic', 'Anthropic Claude'), makeOAuthProvider('nous', 'Nous Portal')])
    render(<Picker ctx={ctx} />)

    expect(screen.getByText('Nous Portal')).toBeTruthy()
    expect(screen.getByText('Recommended')).toBeTruthy()
    // Fireworks stays behind the disclosure with the other alternatives; only
    // Nous Portal is visible before the user expands the list.
    expect(screen.queryByText('Fireworks AI')).toBeNull()
    expect(screen.queryByText('Anthropic API Key')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Other providers' }))

    expect(screen.getByText('Fireworks AI')).toBeTruthy()
    expect(screen.getByText('Anthropic API Key')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Collapse' })).toBeTruthy()
  })

  it('shows Fireworks first in the expanded list, ahead of other OAuth providers', () => {
    setProviders([
      makeOAuthProvider('openai-codex', 'OpenAI Codex / ChatGPT'),
      makeOAuthProvider('minimax-oauth', 'MiniMax'),
      makeOAuthProvider('nous', 'Nous Portal')
    ])
    render(<Picker ctx={ctx} />)
    fireEvent.click(screen.getByRole('button', { name: 'Other providers' }))

    const labels = screen
      .getAllByRole('button')
      .map(el => el.textContent ?? '')
      .filter(text => /Nous Portal|Fireworks AI|ChatGPT or Codex|MiniMax|OpenRouter/.test(text))

    const indexOf = (needle: string) => labels.findIndex(text => text.includes(needle))
    expect(indexOf('Nous Portal')).toBeGreaterThanOrEqual(0)
    expect(indexOf('Fireworks AI')).toBeGreaterThan(indexOf('Nous Portal'))
    expect(indexOf('ChatGPT or Codex')).toBeGreaterThan(indexOf('Fireworks AI'))
    expect(indexOf('MiniMax')).toBeGreaterThan(indexOf('ChatGPT or Codex'))
  })

  it('shows every provider directly when Nous Portal is absent', () => {
    setProviders([
      makeOAuthProvider('anthropic', 'Anthropic Claude'),
      makeOAuthProvider('openai-codex', 'OpenAI Codex / ChatGPT')
    ])
    render(<Picker ctx={ctx} />)

    expect(screen.getByText('Fireworks AI')).toBeTruthy()
    expect(screen.getByText('Anthropic API Key')).toBeTruthy()
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

describe('onboarding model confirmation owner routing', () => {
  it('uses an opaque cache key and the captured owner for REST recovery', async () => {
    const scope = {
      connectionId: null,
      profile: 'remote-profile',
      legacyConnection: {
        baseUrl: 'https://legacy.example',
        headers: { 'Cf-Access-Client-Id': 'private-client' },
        mode: 'remote',
        token: 'private-token'
      }
    } as const

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

    render(
      <QueryClientProvider client={client}>
        <FlowPanel
          ctx={{ profile: 'remote-profile', requestGateway: vi.fn(), scope }}
          flow={{
            currentModel: 'fixture/model',
            label: 'Fixture',
            providerSlug: 'fixture',
            saving: false,
            status: 'confirming_model'
          }}
          leaving={false}
          onBegin={vi.fn()}
        />
      </QueryClientProvider>
    )

    await waitFor(() =>
      expect(requestModelOptions).toHaveBeenCalledWith(
        expect.objectContaining({ explicitOnly: false, profile: 'remote-profile', scope })
      )
    )
    const serializedKeys = JSON.stringify(client.getQueryCache().getAll().map(query => query.queryKey))
    expect(serializedKeys).not.toContain('private-token')
    expect(serializedKeys).not.toContain('private-client')
    expect(serializedKeys).toContain('legacy:')
  })
})

describe('DesktopOnboardingOverlay owner routing', () => {
  it('fails closed when a legacy Settings owner changes mid-flow', async () => {
    $connection.set({ mode: 'remote', baseUrl: 'https://legacy-a.example', token: 'token-a' } as never)
    const scope = $settingsOwner.get()
    const provider = { ...makeOAuthProvider('fixture'), flow: 'external' as const }
    const requestGateway = vi.fn(async () => ({ ok: true }))

    expect(scope?.legacyConnection).toBeTruthy()
    $desktopOnboarding.set({
      ...$desktopOnboarding.get(),
      configured: true,
      flow: { status: 'external_pending', provider, copied: false },
      manual: true,
      requested: true,
      targetProfile: 'default',
      targetScope: scope ?? undefined
    })
    render(
      <DesktopOnboardingOverlay
        enabled
        profile="default"
        requestGateway={requestGateway as OnboardingContext['requestGateway']}
      />
    )

    $connection.set({ mode: 'remote', baseUrl: 'https://legacy-b.example', token: 'token-b' } as never)
    fireEvent.click(screen.getByRole('button', { name: "I've signed in" }))

    await waitFor(() => expect($desktopOnboarding.get().flow.status).toBe('error'))
    expect(requestGateway).not.toHaveBeenCalled()
    const flow = $desktopOnboarding.get().flow
    expect(flow.status === 'error' ? flow.message : '').toContain('Settings gateway changed')
  })
})
