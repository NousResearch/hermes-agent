import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

// Radix Select calls scrollIntoView / pointer-capture APIs jsdom lacks.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const getGlobalModelOptions = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: () => getGlobalModelOptions()
}))

const { DelegationModelProviderField, INHERIT_VALUE, MODEL_ONLY_VALUE } = await import(
  './delegation-model-provider-field'
)

beforeEach(() => {
  getGlobalModelOptions.mockResolvedValue({
    providers: [
      { name: 'Anthropic', slug: 'anthropic', models: ['claude-sonnet-4-6', 'claude-opus-4-6'] },
      { name: 'OpenAI', slug: 'openai', models: ['gpt-5.1'] },
      { name: 'Custom Endpoint', slug: 'custom', models: [] }
    ]
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderField(model: string, provider: string, onChange = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={client}>
      <DelegationModelProviderField model={model} onChange={onChange} provider={provider} />
    </QueryClientProvider>
  )

  return onChange
}

function renderFieldWithRerender(model: string, provider: string, onChange = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const view = render(
    <QueryClientProvider client={client}>
      <DelegationModelProviderField model={model} onChange={onChange} provider={provider} />
    </QueryClientProvider>
  )

  return (nextModel: string, nextProvider: string) =>
    view.rerender(
      <QueryClientProvider client={client}>
        <DelegationModelProviderField model={nextModel} onChange={onChange} provider={nextProvider} />
      </QueryClientProvider>
    )
}

describe('DelegationModelProviderField', () => {
  it('renders "Inherit from main agent" by default when both fields are empty', async () => {
    renderField('', '')

    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())

    expect(screen.getByText('Inherit from main agent')).toBeTruthy()
    // No model select trigger should be present in inherit mode
    expect(screen.queryByLabelText('Subagent Model')).toBeNull()
  })

  it('renders provider and model selects when configured', async () => {
    renderField('claude-sonnet-4-6', 'anthropic')

    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())

    expect(screen.getByLabelText('Subagent Provider')).toBeTruthy()
    expect(screen.getByLabelText('Subagent Model')).toBeTruthy()
    expect(screen.getByText('claude-sonnet-4-6')).toBeTruthy()
  })

  it('represents model-only override without provider as Custom model', async () => {
    renderField('my-custom-model', '')

    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())

    expect(screen.getByText('Custom model (use parent credentials)')).toBeTruthy()
    expect(screen.getByLabelText('Subagent Model')).toBeTruthy()
    expect(screen.getByText('my-custom-model')).toBeTruthy()
  })

  it('resyncs when external config changes (e.g. profile switch)', async () => {
    const rerender = renderFieldWithRerender('', '')

    await waitFor(() => expect(screen.getByText('Inherit from main agent')).toBeTruthy())

    rerender('gpt-5.1', 'openai')

    await waitFor(() => {
      expect(screen.getByText('OpenAI')).toBeTruthy()
      expect(screen.getByText('gpt-5.1')).toBeTruthy()
    })
  })

  it('clearing to Inherit emits { model: "", provider: "" } atomically', async () => {
    const onChange = renderField('gpt-5.1', 'openai')

    await waitFor(() => expect(screen.getByText('OpenAI')).toBeTruthy())

    // Radix Select Trigger
    const trigger = screen.getByLabelText('Subagent Provider')
    fireEvent.click(trigger)

    // Pick inherit
    const inheritOption = await screen.findByRole('option', { name: 'Inherit from main agent' })
    fireEvent.click(inheritOption)

    expect(onChange).toHaveBeenCalledWith({ model: '', provider: '' })
  })

  it('switching provider resets model and emits atomically', async () => {
    const onChange = renderField('gpt-5.1', 'openai')

    await waitFor(() => expect(screen.getByText('OpenAI')).toBeTruthy())

    const trigger = screen.getByLabelText('Subagent Provider')
    fireEvent.click(trigger)

    const anthropicOption = await screen.findByRole('option', { name: 'Anthropic' })
    fireEvent.click(anthropicOption)

    // Should switch provider to anthropic and clear model because gpt-5.1 is not an anthropic model
    expect(onChange).toHaveBeenCalledWith({ model: '', provider: 'anthropic' })
  })

  it('switching to Custom model preserves draft model with empty provider', async () => {
    const onChange = renderField('special-model', 'openai')

    await waitFor(() => expect(screen.getByText('OpenAI')).toBeTruthy())

    const trigger = screen.getByLabelText('Subagent Provider')
    fireEvent.click(trigger)

    const customModelOption = await screen.findByRole('option', {
      name: 'Custom model (use parent credentials)'
    })

    fireEvent.click(customModelOption)

    expect(onChange).toHaveBeenCalledWith({ model: 'special-model', provider: '' })
  })
})
