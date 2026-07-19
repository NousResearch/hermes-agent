import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ModelOptionsResponse } from '@/types/hermes'

import { $customModels, addCustomModel } from '../store/custom-models'
import { $connection } from '../store/session'

import { ModelPickerDialog } from './model-picker'

const requestModelOptions = vi.fn<() => Promise<ModelOptionsResponse>>()

vi.mock('@/lib/model-options', () => ({
  requestModelOptions: () => requestModelOptions()
}))

// jsdom lacks the APIs cmdk relies on: scrollIntoView + ResizeObserver.
function installCmdkDomStubs() {
  ;

(HTMLElement.prototype as unknown as { scrollIntoView: () => void }).scrollIntoView = () => {}

  ;(globalThis as unknown as { ResizeObserver: unknown }).ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

installCmdkDomStubs()

function renderPicker(onSelect = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <ModelPickerDialog
        currentModel=""
        currentProvider=""
        onOpenChange={vi.fn()}
        onSelect={onSelect}
        open
      />
    </QueryClientProvider>
  )

  return { onSelect }
}

beforeEach(() => {
  window.localStorage.clear()
  $connection.set(null)
  $customModels.set({})
  requestModelOptions.mockReset()
})

describe('ModelPickerDialog custom models', () => {
  it('renders a custom model for a provider whose backend catalog is empty (relaxed gate)', async () => {
    // Provider has zero backend models — the old `models.length > 0` gate
    // dropped it entirely, so the custom entry vanished on reopen.
    requestModelOptions.mockResolvedValue({ providers: [{ name: 'xAI', slug: 'xai', models: [] }] })
    addCustomModel('xai', 'grok-custom')

    renderPicker()

    expect(await screen.findByText('grok-custom')).toBeTruthy()
    // Tagged as custom.
    expect(screen.getByText('custom')).toBeTruthy()
  })

  it('selecting a custom model fires onSelect with { provider, model }', async () => {
    requestModelOptions.mockResolvedValue({ providers: [{ name: 'xAI', slug: 'xai', models: [] }] })
    addCustomModel('xai', 'grok-custom')

    const { onSelect } = renderPicker()

    const row = await screen.findByText('grok-custom')
    fireEvent.click(row)

    await waitFor(() =>
      expect(onSelect).toHaveBeenCalledWith({ provider: 'xai', model: 'grok-custom' })
    )
  })

  it('a provider with neither backend nor custom models is not listed', async () => {
    requestModelOptions.mockResolvedValue({ providers: [{ name: 'Empty', slug: 'empty', models: [] }] })

    renderPicker()

    // Wait for load to settle, then confirm the empty provider produced no rows.
    await waitFor(() => expect(requestModelOptions).toHaveBeenCalled())
    expect(screen.queryByText('Empty')).toBeNull()
  })
})
