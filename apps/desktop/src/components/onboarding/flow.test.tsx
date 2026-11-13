import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'
import { $desktopOnboarding, type DesktopOnboardingState, type OnboardingContext } from '@/store/onboarding'

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

  it('lets the step shrink to the card so a long command cannot push Copy and the footer out', () => {
    render(<Harness />)

    const command = screen.getByText(longCommand, { exact: false }).closest('code')

    if (!command) {
      throw new Error('command code element missing')
    }

    // Grid tracks default to min-width:auto; without these the one-line command sets the step's width.
    const step = command.closest('.grid')

    if (!step) {
      throw new Error('step grid missing')
    }

    expect(step.classList.contains('min-w-0')).toBe(true)
    expect(command.classList.contains('truncate')).toBe(false)
    expect(command.classList.contains('break-all')).toBe(true)
  })

  it('keeps the command selectable and the Copy and continue actions rendered', () => {
    render(<Harness />)

    const command = screen.getByText(longCommand, { exact: false }).closest('code')

    expect(command?.getAttribute('data-selectable-text')).toBe('true')
    expect(screen.getByRole('button', { name: 'Copy' })).toBeTruthy()
    expect(screen.getByRole('button', { name: "I've signed in" })).toBeTruthy()
  })
})
