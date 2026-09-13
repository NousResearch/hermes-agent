import type { ModelOptionsResult } from '@hermes/shared'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ConfirmHost } from '@/components/confirm-host'
import { I18nProvider } from '@/i18n'
import { $confirmRequest } from '@/store/confirm'
import { $customModels } from '@/store/custom-models'
import { $knownModels, $visibleModels, modelVisibilityKey, setVisibleModels } from '@/store/model-visibility'
import { $collapsedProviders } from '@/store/provider-collapse'
import { stubMenuDomApis, stubResizeObserver } from '@/test/jsdom'

import { ModelVisibilityDialog } from './model-visibility-dialog'

vi.mock('@/lib/model-options', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestModelOptions: vi.fn()
}))

import { requestModelOptions } from '@/lib/model-options'

stubResizeObserver()

const OPTIONS: ModelOptionsResult = {
  providers: [
    { authenticated: true, models: ['gpt-5.5', 'gpt-6'], name: 'OpenAI Codex', slug: 'openai-codex' },
    { authenticated: true, models: ['qwen3-coder'], name: 'Qwen', slug: 'qwen' }
  ]
}

function renderDialog() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <I18nProvider>
        <ModelVisibilityDialog onOpenChange={() => {}} onOpenProviders={() => {}} open />
        <ConfirmHost />
      </I18nProvider>
    </QueryClientProvider>
  )
}

const isShown = async (label: RegExp) =>
  (await screen.findByRole('switch', { name: label })).getAttribute('aria-checked') === 'true'

beforeEach(() => {
  window.localStorage.clear()
  $customModels.set([])
  $visibleModels.set(null)
  $knownModels.set(null)
  vi.mocked(requestModelOptions).mockResolvedValue(OPTIONS)
})

afterEach(() => {
  cleanup()
  $confirmRequest.set(null)
  vi.clearAllMocks()
})

describe('Edit Models reset', () => {
  it('restores the default shortlist when a model is stuck hidden by the known snapshot', async () => {
    // A curation saved while the snapshot machinery was live: gpt-6 was listed
    // when the user kept gpt-5.5 and switched it off, so the snapshot counts it
    // as judged and the default rule never re-admits it — the dialog shows it
    // switched off with no per-model way back that survives a catalog change.
    setVisibleModels(new Set([modelVisibilityKey('openai-codex', 'gpt-5.5')]), OPTIONS.providers!)

    renderDialog()

    expect(await isShown(/gpt-6/i)).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: /reset to defaults/i }))
    fireEvent.click(await screen.findByRole('button', { name: /^reset$/i }))

    await waitFor(async () => expect(await isShown(/gpt-6/i)).toBe(true))
    expect(await isShown(/gpt-5\.5/i)).toBe(true)
    expect(window.localStorage.getItem('hermes.desktop.visible-models')).toBeNull()
    expect(window.localStorage.getItem('hermes.desktop.known-models')).toBeNull()
  })

  it('keeps the user’s choices when the reset is cancelled', async () => {
    setVisibleModels(new Set([modelVisibilityKey('openai-codex', 'gpt-5.5')]), OPTIONS.providers!)

    renderDialog()
    await isShown(/gpt-6/i)

    fireEvent.click(screen.getByRole('button', { name: /reset to defaults/i }))
    fireEvent.click(await screen.findByRole('button', { name: /cancel/i }))

    await waitFor(() => expect($confirmRequest.get()).toBeNull())
    expect(await isShown(/gpt-6/i)).toBe(false)
  })

  it('keeps a custom model stored and switched on where the defaults would hide it', async () => {
    // An aggregator row defaults to its featured shortlist; the typed id is
    // appended after the catalog, so the bare default rule would hide it.
    vi.mocked(requestModelOptions).mockResolvedValue({
      providers: [
        {
          authenticated: true,
          featured_models: ['openai/gpt-6'],
          models: ['openai/gpt-5', 'openai/gpt-6', 'anthropic/claude-x'],
          name: 'OpenRouter',
          slug: 'openrouter'
        }
      ]
    })
    $customModels.set([{ model: 'acme/model-x', provider: 'openrouter' }])
    $visibleModels.set(new Set([modelVisibilityKey('openrouter', 'acme/model-x')]))

    renderDialog()

    expect(await isShown(/gpt-6/i)).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: /reset to defaults/i }))
    fireEvent.click(await screen.findByRole('button', { name: /^reset$/i }))

    await waitFor(async () => expect(await isShown(/gpt-6/i)).toBe(true))
    // The custom row's <label> also wraps its remove button, so its switch has
    // no accessible name of its own; find it through the row instead.
    const customRow = screen.getByText('Model X').closest('label')!
    expect(within(customRow).getByRole('switch').getAttribute('aria-checked')).toBe('true')
    expect(screen.getByRole('button', { name: /remove custom model/i })).toBeTruthy()
    expect($customModels.get()).toEqual([{ model: 'acme/model-x', provider: 'openrouter' }])
  })

  it('offers no reset while the list is still the default', async () => {
    renderDialog()

    expect(await isShown(/gpt-6/i)).toBe(true)
    expect(screen.queryByRole('button', { name: /reset to defaults/i })).toBeNull()
  })

  it('does not offer reset before the provider catalog has loaded', () => {
    $visibleModels.set(new Set([modelVisibilityKey('openai-codex', 'gpt-5.5')]))
    vi.mocked(requestModelOptions).mockReturnValue(new Promise(() => {}))

    renderDialog()

    expect(screen.queryByRole('button', { name: /reset to defaults/i })).toBeNull()
  })

stubMenuDomApis()
stubResizeObserver()
afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.clearAllMocks()
})

it('updates an open Models popup from the Nous catalog without changing visibility', async () => {
  vi.useFakeTimers({ toFake: ['setInterval', 'clearInterval'] })
  $collapsedProviders.set([])
  const price = { input: '$0.20', output: '$0.80', cache: '$0.02', free: false }
  const nous = { slug: 'nous', name: 'Nous Portal', models: ['vendor/model'], pricing: { 'vendor/model': price } }

  vi.mocked(requestModelOptions).mockResolvedValue({
    providers: [nous, { ...nous, slug: 'another-provider', name: 'Another provider' }]
  } as Awaited<ReturnType<typeof requestModelOptions>>)
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={client}>
      <ModelVisibilityDialog onOpenChange={vi.fn()} onOpenProviders={vi.fn()} open />
    </QueryClientProvider>
  )
  expect(await screen.findByText('In $0.20')).toBeTruthy()
  expect(screen.getAllByText('USD / 1M tokens')).toHaveLength(1)
  expect(screen.getByText('Cache $0.02')).toBeTruthy()
  const switches = screen.getAllByRole('switch').map(control => control.getAttribute('aria-checked'))

  vi.mocked(requestModelOptions).mockResolvedValue({
    providers: [{ ...nous, pricing: { 'vendor/model': { ...price, input: '$0.10', discount_percent: 50 } } }]
  } as Awaited<ReturnType<typeof requestModelOptions>>)
  // Advance the open editor's normal refresh cadence, without manual refresh.
  await act(() => vi.advanceTimersByTimeAsync(300_000))
  await waitFor(() => expect(screen.getByText('In $0.10')).toBeTruthy())
  expect(screen.getByText('-50%')).toBeTruthy()
  expect(screen.getAllByRole('switch')[0].getAttribute('aria-checked')).toBe(switches[0])
  expect(screen.queryByText('In $0.20')).toBeNull()
})
