import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { userEvent } from '@testing-library/user-event'
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

async function renderField(
  model: string,
  provider: string,
  onChange = vi.fn(),
  parentModelLabel?: string
) {
  const { DelegationModelProviderField } = await import('./delegation-model-provider-field')

  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={client}>
      <DelegationModelProviderField
        model={model}
        onChange={onChange}
        parentModelLabel={parentModelLabel}
        provider={provider}
      />
    </QueryClientProvider>
  )

  return onChange
}

describe('DelegationModelProviderField', () => {
  it('renders "Inherit from main agent" as the default (both fields empty)', async () => {
    await renderField('', '')

    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())

    // The provider select should show the inherit option.
    expect(screen.getByText('Inherit from main agent')).toBeTruthy()
    // No model dropdown when inheriting.
    expect(screen.queryByRole('combobox', { name: /model/i })).toBeNull()
  })

  it('shows a "currently inheriting" label when parentModelLabel is provided', async () => {
    await renderField('', '', vi.fn(), 'anthropic / claude-sonnet-4-6')

    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())

    expect(screen.getByText(/Currently inheriting/)).toBeTruthy()
    expect(screen.getByText(/claude-sonnet-4-6/)).toBeTruthy()
  })

  it('does not show inheriting label when parentModelLabel is absent', async () => {
    await renderField('', '')

    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())

    expect(screen.queryByText(/Currently inheriting/)).toBeNull()
  })

  it('selecting "Inherit from main agent" writes "" to both keys', async () => {
    const onChange = await renderField('claude-sonnet-4-6', 'anthropic')

    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())

    // Open the provider select and pick inherit.
    const user = userEvent.setup()

    // The provider trigger shows the provider name currently selected.
    const providerTrigger = screen.getByRole('combobox')

    await user.click(providerTrigger)
    await user.click(screen.getByText('Inherit from main agent'))

    expect(onChange).toHaveBeenCalledWith({ model: '', provider: '' })
  })

  it('selecting a provider resets the model to empty', async () => {
    const onChange = await renderField('gpt-5.1', 'openai')

    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())

    const user = userEvent.setup()
    const providerTrigger = screen.getByRole('combobox')

    await user.click(providerTrigger)
    await user.click(screen.getByText('Anthropic'))

    expect(onChange).toHaveBeenCalledWith({ model: '', provider: 'anthropic' })
  })

  it('falls back to free-text model input when provider has no catalog', async () => {
    await renderField('', 'custom')

    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())

    // The model field should be a text input, not a select.
    expect(screen.getByPlaceholderText('Custom model ID…')).toBeTruthy()
  })

  it('selecting a model emits the complete pair', async () => {
    const onChange = await renderField('', 'anthropic')

    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())

    const user = userEvent.setup()

    // There should be two comboboxes: provider + model.
    const comboboxes = screen.getAllByRole('combobox')
    expect(comboboxes).toHaveLength(2)

    // Open the model select (second combobox) and pick a model.
    await user.click(comboboxes[1])
    await user.click(screen.getByText('claude-opus-4-6'))

    expect(onChange).toHaveBeenCalledWith({ model: 'claude-opus-4-6', provider: 'anthropic' })
  })
})
