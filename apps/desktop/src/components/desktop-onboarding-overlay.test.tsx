import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $desktopBoot } from '@/store/boot'
import { $desktopOnboarding, type DesktopOnboardingState, type OnboardingContext } from '@/store/onboarding'
import type { OAuthProvider } from '@/types/hermes'

import { DesktopOnboardingOverlay, Picker } from './desktop-onboarding-overlay'

const getGlobalModelOptions = vi.fn()

class ResizeObserverMock {
  observe = vi.fn()
  unobserve = vi.fn()
  disconnect = vi.fn()
}

vi.stubGlobal('ResizeObserver', ResizeObserverMock)

vi.mock('@/hermes', async importOriginal => {
  const actual = (await importOriginal()) as object

  return {
    ...actual,
    getGlobalModelOptions: () => getGlobalModelOptions()
  }
})

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
    manual: false
  } satisfies DesktopOnboardingState)
}

const ctx: OnboardingContext = { requestGateway: async () => undefined as never }

function renderOverlay() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false }
    }
  })

  $desktopBoot.set({
    error: null,
    fakeMode: false,
    message: 'ready',
    phase: 'ready',
    progress: 100,
    running: false,
    timestamp: Date.now(),
    visible: false
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <DesktopOnboardingOverlay enabled requestGateway={ctx.requestGateway} />
    </QueryClientProvider>
  )
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  getGlobalModelOptions.mockReset()
  $desktopOnboarding.set({
    configured: null,
    flow: { status: 'idle' },
    mode: 'oauth',
    providers: null,
    reason: null,
    requested: false,
    manual: false
  })
  $desktopBoot.set({
    error: null,
    fakeMode: false,
    message: 'Starting Hermes Desktop…',
    phase: 'renderer.init',
    progress: 2,
    running: true,
    timestamp: Date.now(),
    visible: true
  })
})

describe('onboarding Picker', () => {
  it('features Nous Portal and hides other providers behind a disclosure', () => {
    setProviders([provider('anthropic', 'Anthropic Claude'), provider('nous', 'Nous Portal')])
    render(<Picker ctx={ctx} />)

    expect(screen.getByText('Nous Portal')).toBeTruthy()
    expect(screen.getByText('Recommended')).toBeTruthy()
    expect(screen.queryByText('Anthropic Claude')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Other providers' }))

    expect(screen.getByText('Anthropic Claude')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Collapse' })).toBeTruthy()
  })

  it('shows every provider directly when Nous Portal is absent', () => {
    setProviders([provider('anthropic', 'Anthropic Claude'), provider('openai-codex', 'OpenAI Codex / ChatGPT')])
    render(<Picker ctx={ctx} />)

    expect(screen.getByText('Anthropic Claude')).toBeTruthy()
    expect(screen.getByText('OpenAI Codex / ChatGPT')).toBeTruthy()
    expect(screen.queryByText('Other sign-in options')).toBeNull()
    expect(screen.queryByText('Recommended')).toBeNull()
  })

  it('collects endpoint, API key, and default model for custom providers', () => {
    setProviders([])
    render(<Picker ctx={ctx} />)

    fireEvent.click(screen.getByRole('button', { name: /Local \/ custom endpoint/i }))

    expect(screen.getByPlaceholderText('http://127.0.0.1:8000/v1')).toBeTruthy()
    expect(screen.getByPlaceholderText('Paste API key (optional for local endpoints)')).toBeTruthy()
    expect(screen.getByPlaceholderText('Provider name')).toBeTruthy()
    expect(screen.getByPlaceholderText('Select or enter a model id')).toBeTruthy()
  })

  it('scopes the confirmation model picker to the newly configured custom provider', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [
        {
          name: 'Pinche',
          slug: 'custom:pinche',
          models: ['codex-auto-review', 'gpt-5.3-codex', 'gpt-5.4', 'gpt-5.4-mini', 'gpt-5.5', 'gpt-image-2']
        },
        { name: 'GitHub Copilot', slug: 'copilot', models: ['codex-auto-review'] }
      ]
    })
    $desktopOnboarding.set({
      configured: false,
      flow: {
        status: 'confirming_model',
        currentModel: 'gpt-5.5',
        label: 'Pinche',
        providerSlug: 'custom:pinche',
        saving: false
      },
      mode: 'oauth',
      providers: [],
      reason: null,
      requested: false,
      manual: false
    } satisfies DesktopOnboardingState)

    renderOverlay()
    fireEvent.click(screen.getByRole('button', { name: 'Change' }))

    await waitFor(() => expect(screen.getByText('gpt-5.5')).toBeTruthy())
    expect(screen.getByText('gpt-5.4')).toBeTruthy()
    expect(screen.getByText('gpt-5.3-codex')).toBeTruthy()
    expect(screen.getByText('Pinche')).toBeTruthy()
    expect(screen.queryByText('GitHub Copilot')).toBeNull()
  })
})
