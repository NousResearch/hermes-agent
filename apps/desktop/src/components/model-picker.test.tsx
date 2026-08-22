import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, describe, expect, it, vi } from 'vitest'

// jsdom has no ResizeObserver; cmdk's CommandList requires one.
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver
}
// jsdom has no scrollIntoView; cmdk scrolls the selected item into view.
if (typeof Element.prototype.scrollIntoView !== 'function') {
  Element.prototype.scrollIntoView = () => {}
}

import { ModelPickerDialog } from './model-picker'

// cmdk pushes the selected item's `value` (`provider.slug:model`) into a
// controlled CommandInput when the selection changes. That leaked value used
// to survive the dialog closing, so the next open filtered the list down to
// just the current model. The picker must own its search term: cleared on
// select, and reset every time the dialog opens.

vi.mock('@/lib/model-options', () => ({
  modelOptionsQueryKey: () => ['test-model-options'],
  requestModelOptions: async () => ({
    model: 'gyz-model',
    provider: 'gyz',
    providers: [
      {
        name: 'Sakiko Dev',
        slug: 'custom:sakiko-dev',
        models: ['tokenrhythm/kimi-k3', 'agy/gemini-3.6-flash-high']
      },
      { name: 'gyz', slug: 'gyz', models: ['gyz-model'] }
    ]
  })
}))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      common: { cancel: 'Cancel' },
      modelPicker: {
        addProvider: 'Add provider',
        current: 'Current:',
        loadFailed: 'Failed to load',
        noAuthenticatedProviders: 'No authenticated providers.',
        noModels: 'No matching models',
        pro: 'PRO',
        search: 'Search models…',
        title: 'Switch model',
        unknown: 'unknown'
      }
    }
  })
}))

function renderPicker(overrides: Record<string, unknown> = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const props = {
    currentModel: 'gyz-model',
    currentProvider: 'gyz',
    onOpenChange: vi.fn(),
    onSelect: vi.fn(),
    open: true,
    ...overrides
  }
  return {
    onOpenChange: props.onOpenChange,
    onSelect: props.onSelect,
    ...render(
      <QueryClientProvider client={client}>
        <ModelPickerDialog {...props} />
      </QueryClientProvider>
    )
  }
}

afterEach(cleanup)

describe('ModelPickerDialog search ownership', () => {
  it('clears the search box when a model is selected', async () => {
    renderPicker()

    const input = await screen.findByPlaceholderText('Search models…')
    await screen.findByText('tokenrhythm/kimi-k3')
    fireEvent.click(screen.getByText('tokenrhythm/kimi-k3'))

    await waitFor(() => expect((input as HTMLInputElement).value).toBe(''))
  })

  it('resets the search box every time the dialog opens', async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const props = {
      currentModel: 'gyz-model',
      currentProvider: 'gyz',
      onOpenChange: vi.fn(),
      onSelect: vi.fn()
    }
    const view = render(
      <QueryClientProvider client={client}>
        <ModelPickerDialog {...props} open={true} />
      </QueryClientProvider>
    )

    const input = await screen.findByPlaceholderText('Search models…')
    await screen.findByText('tokenrhythm/kimi-k3')
    fireEvent.change(input, { target: { value: 'kimi' } })
    expect((input as HTMLInputElement).value).toBe('kimi')

    view.rerender(
      <QueryClientProvider client={client}>
        <ModelPickerDialog {...props} open={false} />
      </QueryClientProvider>
    )
    view.rerender(
      <QueryClientProvider client={client}>
        <ModelPickerDialog {...props} open={true} />
      </QueryClientProvider>
    )

    // The dialog content is a fresh DOM subtree after reopening — re-query.
    const reopened = await screen.findByPlaceholderText('Search models…')
    await waitFor(() => expect((reopened as HTMLInputElement).value).toBe(''))
  })
})
