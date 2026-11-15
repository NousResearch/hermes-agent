import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { $desktopOnboarding, type DesktopOnboardingState, type OnboardingContext } from '@/store/onboarding'
import type { OAuthProvider } from '@/types/hermes'

import { FlowPanel } from './flow'

// Only the catalog fetch is replaced; the model assignment keeps its real path
// down to window.hermesDesktop.api so the test observes the wire body.
vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getGlobalModelOptions: async () => ({
    providers: [
      { free_tier: false, models: ['gpt-5.6-terra'], name: 'OpenAI OAuth (ChatGPT)', slug: 'openai' },
      { models: ['deepseek/deepseek-v4-flash-0731'], name: 'Nous Portal', slug: 'nous' }
    ]
  })
}))

// The real picker is a cmdk/Radix dialog; stand in a button that reports the
// same {provider, model} selection shape the real one emits.
vi.mock('@/components/model-picker', () => ({
  ModelPickerDialog: ({
    onSelect,
    open
  }: {
    onSelect: (selection: { model: string; provider: string }) => void
    open: boolean
  }) =>
    open ? (
      <button onClick={() => onSelect({ model: 'deepseek/deepseek-v4-flash-0731', provider: 'nous' })} type="button">
        pick-nous-model
      </button>
    ) : null
}))

const ctx: OnboardingContext = { requestGateway: async () => undefined as never }

function confirmingModelState(): DesktopOnboardingState {
  return {
    configured: false,
    flow: {
      status: 'confirming_model',
      currentModel: 'gpt-5.6-terra',
      label: 'OpenAI OAuth (ChatGPT)',
      providerSlug: 'openai',
      saving: false
    },
    mode: 'oauth',
    providers: null,
    reason: null,
    requested: false,
    firstRunSkipped: false,
    manual: false,
    localEndpoint: false,
    freeTierReady: false
  }
}

function Harness() {
  const state = $desktopOnboarding.get()

  return (
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <FlowPanel ctx={ctx} flow={state.flow} leaving={false} onBegin={() => undefined} />
    </QueryClientProvider>
  )
}

beforeEach(() => {
  $desktopOnboarding.set(confirmingModelState())
})

afterEach(() => {
  cleanup()
  $desktopOnboarding.set({ ...confirmingModelState(), configured: null, flow: { status: 'idle' } })
})

describe('ConfirmingModelPanel model pick', () => {
  it('persists a cross-provider pick against the picked model provider, not the sign-in provider', async () => {
    const calls: { body?: unknown; path: string }[] = []

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        api: async ({ body, path }: { body?: unknown; path: string }) => {
          calls.push({ body, path })

          if (path === '/api/model/set') {
            return { ok: true, provider: 'nous', model: 'deepseek/deepseek-v4-flash-0731' }
          }

          throw new Error(`unexpected api path: ${path}`)
        }
      }
    })

    render(<Harness />)

    // The Pro badge only renders once the catalog query has resolved — the
    // relabel below reads the picked provider's name from that catalog.
    await screen.findByText('Pro')

    // The user signed in with OpenAI OAuth; the picker offers a deepseek
    // model that only Nous Portal serves.
    fireEvent.click(screen.getByRole('button', { name: 'Change' }))
    fireEvent.click(await screen.findByRole('button', { name: 'pick-nous-model' }))

    await waitFor(() => expect(calls.some(c => c.path === '/api/model/set')).toBe(true))

    expect(calls.find(c => c.path === '/api/model/set')?.body).toMatchObject({
      scope: 'main',
      provider: 'nous',
      model: 'deepseek/deepseek-v4-flash-0731'
    })

    const flow = $desktopOnboarding.get().flow
    expect(flow.status).toBe('confirming_model')

    if (flow.status === 'confirming_model') {
      expect(flow.providerSlug).toBe('nous')
      expect(flow.label).toBe('Nous Portal')
    }
  })
})

describe('ConfirmingModelPanel sign-in result', () => {
  it('names the account and what moved before Begin', async () => {
    $desktopOnboarding.set({
      ...confirmingModelState(),
      flow: {
        currentModel: 'gpt-5.6-terra',
        label: 'OpenAI OAuth (ChatGPT)',
        providerSlug: 'openai',
        saving: false,
        signedIn: { email: 'someone@example.com', handoff: { lines: ['Your apps moved.'], state: 'claimed' } },
        status: 'confirming_model'
      }
    })

    render(<Harness />)

    expect(screen.getByText('someone@example.com')).toBeTruthy()
    expect(screen.getByText('Your apps moved.')).toBeTruthy()
  })
})

describe('external CLI sign-in step', () => {
  const longCommand = '/Users/someone/.nvm/current/bin/claude auth login'

  beforeEach(() => {
    $desktopOnboarding.set({
      ...confirmingModelState(),
      flow: {
        copied: false,
        provider: {
          cli_command: longCommand,
          docs_url: '',
          flow: 'external',
          id: 'claude-subscription-directsdk-experimental',
          name: 'Claude Subscription DirectSDK (Experimental)',
          status: { logged_in: false }
        },
        status: 'external_pending'
      }
    })
  })

  it('keeps the command selectable and the Copy and continue actions rendered', () => {
    render(<Harness />)

    const command = screen.getByText(longCommand, { exact: false }).closest('code')

    const selectable = command?.querySelector('[data-selectable-text="true"]')

    // Only the command is selectable: a copied "$" prompt would not run in a shell.
    expect(selectable?.textContent).toBe(longCommand)
    expect(screen.getByRole('button', { name: 'Copy' })).toBeTruthy()
    expect(screen.getByRole('button', { name: "I've signed in" })).toBeTruthy()
  })
})

describe('awaiting_user device-code input', () => {
  const provider: OAuthProvider = {
    cli_command: '',
    docs_url: '',
    flow: 'pkce',
    id: 'openai',
    name: 'OpenAI OAuth (ChatGPT)',
    status: { logged_in: false }
  }
  const start = { auth_url: 'https://example.com/auth', expires_in: 600, flow: 'pkce' as const, session_id: 's1' }

  function awaitingUserState(): DesktopOnboardingState {
    return { ...confirmingModelState(), flow: { code: '', provider, start, status: 'awaiting_user' } }
  }

  // submitOnboardingCode flips the flow to 'submitting' before its (here
  // rejected) API call; the calls array is the observable for "submitted".
  function trackSubmissions(): { calls: unknown[] } {
    const calls: unknown[] = []

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        api: async (request: unknown) => {
          calls.push(request)

          throw new Error('not expected to resolve in this test')
        }
      }
    })

    return { calls }
  }

  it('submits the code on a bare Enter', async () => {
    const { calls } = trackSubmissions()
    $desktopOnboarding.set(awaitingUserState())

    render(<Harness />)

    const input = screen.getByPlaceholderText('Paste authorization code')
    fireEvent.change(input, { target: { value: 'auth-code-123' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    await waitFor(() => expect(calls.length).toBeGreaterThan(0))
  })

  it('does not submit on an in-flight IME composition Enter', () => {
    const { calls } = trackSubmissions()
    $desktopOnboarding.set(awaitingUserState())

    render(<Harness />)

    const input = screen.getByPlaceholderText('Paste authorization code')
    fireEvent.change(input, { target: { value: '中文' } })
    fireEvent.keyDown(input, { isComposing: true, key: 'Enter' })

    expect(calls).toHaveLength(0)
    expect($desktopOnboarding.get().flow.status).toBe('awaiting_user')
  })

  it('does not submit on the legacy IME commit Enter (keyCode 229)', () => {
    const { calls } = trackSubmissions()
    $desktopOnboarding.set(awaitingUserState())

    render(<Harness />)

    const input = screen.getByPlaceholderText('Paste authorization code')
    fireEvent.change(input, { target: { value: '中文' } })
    fireEvent.keyDown(input, { key: 'Enter', keyCode: 229 })

    expect(calls).toHaveLength(0)
    expect($desktopOnboarding.get().flow.status).toBe('awaiting_user')
  })
})
