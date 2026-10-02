import type { ModelOptionsResult } from '@hermes/shared'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { ConfirmHost } from '@/components/confirm-host'
import { I18nProvider } from '@/i18n'
import { $confirmRequest } from '@/store/confirm'
import { $customModels } from '@/store/custom-models'
import { $knownModels, $visibleModels, modelVisibilityKey, setVisibleModels } from '@/store/model-visibility'
import { stubResizeObserver } from '@/test/jsdom'

import { ModelVisibilityDialog } from './model-visibility-dialog'

vi.mock('@/lib/model-options', async importOriginal => ({
  ...(await importOriginal<Record<string, unknown>>()),
  requestModelOptions: vi.fn()
}))

import { requestModelOptions } from '@/lib/model-options'

stubResizeObserver()

const OPTIONS: ModelOptionsResult = {
  providers: [
    {
      authenticated: true,
      models: ['gpt-5.5', 'gpt-6'],
      name: 'OpenAI Codex',
      slug: 'openai-codex'
    },
    {
      authenticated: true,
      models: ['qwen3-coder'],
      name: 'Qwen',
      slug: 'qwen'
    }
  ]
}

function renderDialog() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })

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

describe('provider focus', () => {
  it('searches other providers and restores the selected provider when cleared', async () => {
    renderDialog()
    await screen.findByRole('switch', { name: /gpt-6/i })
    fireEvent.click(screen.getByRole('button', { name: /^Qwen/ }))
    const search = screen.getByRole('textbox')
    fireEvent.change(search, { target: { value: 'gpt-6' } })
    expect(await screen.findByRole('switch', { name: /gpt-6/i })).toBeTruthy()
    expect(screen.queryByRole('switch', { name: /qwen/i })).toBeNull()
    fireEvent.change(search, { target: { value: '' } })
    expect(await screen.findByRole('switch', { name: /qwen/i })).toBeTruthy()
    expect($visibleModels.get()).toBeNull()
  })

  it('bulk toggles the active provider only and preserves another provider hide-all', async () => {
    setVisibleModels(new Set(['openai-codex::', 'qwen::qwen3-coder']), OPTIONS.providers!)
    renderDialog()
    await screen.findByRole('switch', { name: /gpt-6/i })
    fireEvent.click(screen.getByRole('button', { name: /^Qwen/ }))
    fireEvent.click(screen.getByRole('checkbox', { name: 'Qwen' }))
    expect($visibleModels.get()).toEqual(new Set(['openai-codex::', 'qwen::']))
    fireEvent.click(screen.getByRole('checkbox', { name: 'Qwen' }))
    expect($visibleModels.get()).toEqual(new Set(['openai-codex::', 'qwen::qwen3-coder']))
    expect(JSON.parse(window.localStorage.getItem('hermes.desktop.visible-models')!)).toEqual(
      expect.arrayContaining(['openai-codex::', 'qwen::qwen3-coder'])
    )
  })

  it('keeps custom model creation available when a search has no matches', async () => {
    renderDialog()
    await screen.findByRole('switch', { name: /gpt-6/i })
    fireEvent.change(screen.getByRole('textbox'), {
      target: { value: 'acme/new-model' }
    })
    fireEvent.click(await screen.findByRole('button', { name: /acme\/new-model.*Qwen/i }))
    expect($customModels.get()).toContainEqual({
      provider: 'qwen',
      model: 'acme/new-model'
    })
    expect(screen.getByText('New Model')).toBeTruthy()
    expect(screen.getByRole('button', { name: /^Qwen/ }).getAttribute('aria-pressed')).toBe('true')
  })

  it('shows only the selected provider models without changing visibility', async () => {
    renderDialog()
    await screen.findByRole('switch', { name: /gpt-6/i })
    expect(screen.queryByRole('switch', { name: /qwen/i })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: /^Qwen/ }))
    expect(await screen.findByRole('switch', { name: /qwen/i })).toBeTruthy()
    expect(screen.queryByRole('switch', { name: /gpt-6/i })).toBeNull()
    expect($visibleModels.get()).toBeNull()
  })
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
})
