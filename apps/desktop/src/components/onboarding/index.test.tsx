import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ModelSettings } from '@/app/settings/model-settings'
import type * as ModelOptionsModule from '@/lib/model-options'
import { closeManualOnboarding } from '@/store/onboarding'
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

    const serializedKeys = JSON.stringify(
      client
        .getQueryCache()
        .getAll()
        .map(query => query.queryKey)
    )

    expect(serializedKeys).not.toContain('private-token')
    expect(serializedKeys).not.toContain('private-client')
    expect(serializedKeys).toContain('legacy:')
  })
})

describe('DesktopOnboardingOverlay owner routing', () => {
  it.each(['openai-codex', 'custom:lab', 'retired-provider'])(
    'keeps the named Settings profile through the real %s onboarding handoff',
    async slug => {
      const scope = { connectionId: 'local', profile: 'research' }
      const provider = makeOAuthProvider('openai-codex', 'Codex')

      const api = vi.fn(async ({ path }: { path: string }) => {
        if (path === '/api/model/info') {return { provider: slug, model: '' }}

        if (path.startsWith('/api/model/options')) {
          return {
            providers:
              slug === 'openai-codex'
                ? [{ slug, name: 'Codex', models: [], authenticated: false, auth_type: 'oauth' }]
                : []
          }
        }

        if (path === '/api/model/auxiliary') {return { main: { provider: slug, model: '' }, tasks: [] }}

        if (path === '/api/model/moa') {return null}

        if (path === '/api/config') {return { config: {} }}

        if (path === '/api/providers/oauth') {return { providers: [provider] }}

        if (path === '/api/providers/oauth/openai-codex/start') {
          return { flow: 'pkce', session_id: 'fixture', auth_url: 'https://example.com/oauth', expires_in: 600 }
        }

        return { ok: true }
      })

      const priorBridge = window.hermesDesktop
      Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { api } })
      const open = vi.spyOn(window, 'open').mockReturnValue(null)
      const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

      try {
        render(
          <MemoryRouter>
            <QueryClientProvider client={client}>
              <ModelSettings scopeProfile={scope} />
              <DesktopOnboardingOverlay enabled profile="default" requestGateway={ctx.requestGateway} />
            </QueryClientProvider>
          </MemoryRouter>
        )
        fireEvent.click(
          await screen.findByRole('button', {
            name: slug === 'openai-codex' ? 'Set up Codex' : 'Set up provider'
          })
        )
        await waitFor(() =>
          expect($desktopOnboarding.get()).toMatchObject({
            manual: true,
            targetProfile: 'research',
            targetScope: scope,
            localEndpoint: slug === 'custom:lab'
          })
        )

        if (slug === 'openai-codex') {
          await waitFor(() =>
            expect(api).toHaveBeenCalledWith(
              expect.objectContaining({
                path: '/api/providers/oauth/openai-codex/start',
                ...scope
              })
            )
          )
          expect(open).toHaveBeenCalled()
        }
      } finally {
        cleanup()
        closeManualOnboarding()
        client.clear()
        open.mockRestore()
        Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: priorBridge })
      }
    }
  )

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

    await waitFor(() =>
      expect($desktopOnboarding.get()).toMatchObject({
        flow: { status: 'idle' },
        manual: false,
        targetProfile: undefined,
        targetScope: undefined
      })
    )
  })
})
