import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { MoaConfigResponse, ModelOptionProvider } from '@/types/hermes'

const saveMoaModels = vi.fn()
const getMoaModels = vi.fn()

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
  Element.prototype.setPointerCapture = vi.fn()
})

vi.mock('@/hermes', () => ({
  createMoaModelsSaveRequest: (body: MoaConfigResponse) => () => saveMoaModels(body),
  getMoaModels: () => getMoaModels(),
  saveMoaModels: (body: MoaConfigResponse) => saveMoaModels(body),
  setApiRequestProfile: vi.fn()
}))

import { MoaPresetStudio } from './moa-preset-studio'

const providers: ModelOptionProvider[] = [
  { authenticated: true, models: ['hermes-4'], name: 'Nous', slug: 'nous' },
  {
    authenticated: true,
    models: ['anthropic/claude-opus-4.8'],
    name: 'OpenRouter',
    slug: 'openrouter'
  }
]

const configFixture = (name = 'default'): MoaConfigResponse => ({
  active_preset: name,
  aggregator: { model: 'anthropic/claude-opus-4.8', provider: 'openrouter' },
  aggregator_temperature: null,
  default_preset: name,
  enabled: true,
  max_tokens: 4096,
  presets: {
    [name]: {
      aggregator: { model: 'anthropic/claude-opus-4.8', provider: 'openrouter' },
      aggregator_temperature: null,
      enabled: true,
      fanout: 'user_turn',
      max_tokens: 4096,
      reference_max_tokens: 600,
      reference_models: [{ continuity_id: `${name}-advisor`, model: 'hermes-4', provider: 'nous' }],
      reference_temperature: null
    }
  },
  reference_models: [{ continuity_id: `${name}-advisor`, model: 'hermes-4', provider: 'nous' }],
  reference_temperature: null
})

function renderStudio(config = configFixture()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return {
    client,
    ...render(
      <QueryClientProvider client={client}>
        <MoaPresetStudio config={config} onUseMoaPreset={vi.fn(() => true)} providers={providers} />
      </QueryClientProvider>
    )
  }
}

beforeEach(() => {
  getMoaModels.mockReset()
  saveMoaModels.mockReset()
  getMoaModels.mockResolvedValue(configFixture())
  saveMoaModels.mockImplementation((body: MoaConfigResponse) => Promise.resolve(body))
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe('MoaPresetStudio explicit Save transaction', () => {
  it('accumulates edits locally and writes once only after Save changes is clicked', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true })
    renderStudio()

    const enabled = screen.getByRole('switch', { name: 'Preset enabled' })
    const outputLimit = screen.getByRole('spinbutton', { name: 'Aggregator output limit' })

    fireEvent.click(enabled)
    fireEvent.change(outputLimit, { target: { value: '8192' } })

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000)
    })

    expect(saveMoaModels).not.toHaveBeenCalled()
    expect((screen.getByRole('button', { name: 'Use in this chat' }) as HTMLButtonElement).disabled).toBe(true)

    const save = screen.getByRole('button', { name: 'Save changes' }) as HTMLButtonElement
    expect(save.disabled).toBe(false)
    expect(
      screen
        .getByRole('spinbutton', { name: 'Aggregator temperature' })
        .compareDocumentPosition(save) & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy()

    fireEvent.click(save)

    await waitFor(() => expect(saveMoaModels).toHaveBeenCalledOnce())
    const body = saveMoaModels.mock.calls[0][0] as MoaConfigResponse
    expect(body.presets.default).toMatchObject({ enabled: false, max_tokens: 8192 })
  })

  it('keeps a completed new preset unsaved until Save changes is clicked', async () => {
    renderStudio()

    fireEvent.change(screen.getByRole('textbox', { name: 'New preset name' }), {
      target: { value: 'manual preset' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Add preset' }))
    fireEvent.click(screen.getByRole('combobox', { name: 'Reference 1 provider' }))
    fireEvent.click(await screen.findByRole('option', { name: 'Nous' }))
    fireEvent.click(screen.getByRole('combobox', { name: 'Reference 1 model' }))
    fireEvent.click(await screen.findByRole('option', { name: 'hermes-4' }))
    fireEvent.click(screen.getByRole('combobox', { name: 'Aggregator provider' }))
    fireEvent.click(await screen.findByRole('option', { name: 'OpenRouter' }))
    fireEvent.click(screen.getByRole('combobox', { name: 'Aggregator model' }))
    fireEvent.click(await screen.findByRole('option', { name: 'anthropic/claude-opus-4.8' }))

    expect(saveMoaModels).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))

    await waitFor(() => expect(saveMoaModels).toHaveBeenCalledOnce())
    const body = saveMoaModels.mock.calls[0][0] as MoaConfigResponse
    expect(body.presets['manual preset']).toMatchObject({
      aggregator: { model: 'anthropic/claude-opus-4.8', provider: 'openrouter' },
      reference_models: [{ model: 'hermes-4', provider: 'nous' }]
    })
  })

  it('disables Save changes for an incomplete draft and performs no write', () => {
    renderStudio()

    fireEvent.change(screen.getByRole('spinbutton', { name: 'Aggregator output limit' }), {
      target: { value: '' }
    })

    const save = screen.getByRole('button', { name: 'Save changes' }) as HTMLButtonElement
    expect(save.disabled).toBe(true)
    fireEvent.click(save)
    expect(saveMoaModels).not.toHaveBeenCalled()
  })

  it('retains the draft after a failed save and permits a deliberate retry', async () => {
    saveMoaModels
      .mockRejectedValueOnce(new Error('disk is read-only'))
      .mockImplementationOnce((body: MoaConfigResponse) => Promise.resolve(body))
    renderStudio()

    fireEvent.change(screen.getByRole('spinbutton', { name: 'Aggregator output limit' }), {
      target: { value: '8192' }
    })
    fireEvent.click(screen.getByRole('button', { name: 'Save changes' }))

    await waitFor(() => expect(screen.getByRole('alert').textContent).toContain('disk is read-only'))
    expect((screen.getByRole('spinbutton', { name: 'Aggregator output limit' }) as HTMLInputElement).value).toBe('8192')

    const retry = screen.getByRole('button', { name: 'Save changes' }) as HTMLButtonElement
    expect(retry.disabled).toBe(false)
    fireEvent.click(retry)

    await waitFor(() => expect(saveMoaModels).toHaveBeenCalledTimes(2))
  })

  it('discards unsaved edits on teardown and authoritative source replacement without writing', () => {
    const view = renderStudio(configFixture('profile-a'))

    fireEvent.change(screen.getByRole('spinbutton', { name: 'Aggregator output limit' }), {
      target: { value: '8192' }
    })

    view.rerender(
      <QueryClientProvider client={view.client}>
        <MoaPresetStudio config={configFixture('profile-b')} onUseMoaPreset={vi.fn()} providers={providers} />
      </QueryClientProvider>
    )
    view.unmount()

    expect(saveMoaModels).not.toHaveBeenCalled()
  })
})
