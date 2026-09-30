import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileScope } from '@/hermes'

const getGlobalModelOptions = vi.fn()
vi.mock('@/hermes', async () => ({
  ...(await vi.importActual('@/api/client')),
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args)
}))
const { DelegationModelProviderField } = await import('./delegation-model-provider-field')

const catalog = {
  providers: [
    { name: 'Anthropic', slug: 'anthropic', models: ['claude-sonnet-4-6'] },
    { name: 'OpenAI', slug: 'openai', models: ['gpt-5.1'] },
    { name: 'Lab', slug: 'lab', aliases: ['custom:lab', 'Lab'], models: ['test-model'] }
  ]
}

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})
beforeEach(() => {
  getGlobalModelOptions.mockReset().mockResolvedValue(catalog)
})
afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function mount(
  model = 'gpt-5.1',
  provider = 'openai',
  scope: ProfileScope = { connectionId: 'local', profile: 'B' },
  baseUrl = ''
) {
  const onChange = vi.fn()
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: 60_000 } } })

  const view = render(
    <QueryClientProvider client={client}>
      <DelegationModelProviderField
        baseUrl={baseUrl}
        model={model}
        onChange={onChange}
        provider={provider}
        scope={scope}
      />
    </QueryClientProvider>
  )

  return { onChange, client, view }
}

async function provider(name: string) {
  fireEvent.click(screen.getByLabelText('Subagent Provider'))
  fireEvent.click(await screen.findByRole('option', { name }))
}

async function model(name: string) {
  fireEvent.click(screen.getByLabelText('Subagent Model'))
  fireEvent.click(await screen.findByRole('option', { name }))
}

describe('delegation route regressions', () => {
  it('requests the exact config owner', async () => {
    const owner = { connectionId: 'remote-B', profile: 'B' }
    mount('gpt-5.1', 'openai', owner)
    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalledWith(undefined, owner))
  })
  it('keeps an unfinished switch local beyond autosave, then applies one complete route', async () => {
    const { onChange } = mount()
    await provider('Anthropic')
    await new Promise(resolve => setTimeout(resolve, 650))
    expect(onChange).not.toHaveBeenCalled()
    expect((screen.getByRole('button', { name: 'Apply' }) as HTMLButtonElement).disabled).toBe(true)
    await model('claude-sonnet-4-6')
    expect(onChange).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    expect(onChange).toHaveBeenCalledExactlyOnceWith({
      model: 'claude-sonnet-4-6',
      provider: 'anthropic',
      resetDirectEndpoint: true
    })
  })
  it('cancels a provider draft', async () => {
    const { onChange } = mount()
    await provider('Anthropic')
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }))
    expect(screen.getByLabelText('Subagent Provider').textContent).toBe('OpenAI')
    expect(screen.getByLabelText('Subagent Model').textContent).toBe('gpt-5.1')
    expect(onChange).not.toHaveBeenCalled()
  })
  it('requires an explicit parent-model choice for provider-only routing', async () => {
    const { onChange } = mount()
    await provider('Anthropic')
    fireEvent.click(screen.getByRole('checkbox', { name: 'Use parent model' }))
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    expect(onChange).toHaveBeenCalledExactlyOnceWith({ model: '', provider: 'anthropic', resetDirectEndpoint: true })
  })
  it('represents direct endpoint precedence before explicitly replacing it', async () => {
    const { onChange } = mount('old-model', 'openai', undefined, 'https://endpoint.invalid/v1')
    expect(screen.getByLabelText('Subagent Provider').textContent).toBe('Direct endpoint override')
    await provider('Inherit from main agent')
    expect(screen.getByText(/Applying this route clears the direct endpoint/)).toBeTruthy()
    expect(onChange).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    expect(onChange).toHaveBeenCalledExactlyOnceWith({ model: '', provider: '', resetDirectEndpoint: true })
  })
  it.each(['lab', 'custom:lab', 'Lab'])('shows alias %s without rewriting it', async alias => {
    const { onChange } = mount('test-model', alias)
    await waitFor(() => expect(screen.getByLabelText('Subagent Provider').textContent).toBe('Lab'))
    expect(onChange).not.toHaveBeenCalled()
  })
  it('allows manual editing on failure and preserves the draft on retry', async () => {
    getGlobalModelOptions.mockRejectedValueOnce(new Error('503'))
    const { onChange } = mount()
    await screen.findByText('Model catalog unavailable. Enter a provider and model manually, or retry.')
    await provider('Custom provider...')
    fireEvent.change(screen.getByLabelText('Custom subagent provider'), { target: { value: 'custom:other' } })
    await model('Custom model…')
    fireEvent.change(screen.getByLabelText('Subagent Model'), { target: { value: 'other-model' } })
    fireEvent.click(screen.getByRole('button', { name: 'Retry' }))
    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalledTimes(2))
    expect((screen.getByLabelText('Custom subagent provider') as HTMLInputElement).value).toBe('custom:other')
    expect((screen.getByLabelText('Subagent Model') as HTMLInputElement).value).toBe('other-model')
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    expect(onChange).toHaveBeenCalledExactlyOnceWith({
      model: 'other-model',
      provider: 'custom:other',
      resetDirectEndpoint: true
    })
  })
  it.each(['bedrock', 'vertex', 'google', 'google-genai'])(
    'does not mislabel native SDK provider %s as a direct endpoint',
    async nativeProvider => {
      mount('native-model', nativeProvider, undefined, 'https://sdk-region.invalid')
      expect(screen.getByLabelText('Subagent Provider').textContent).toBe(nativeProvider)
    }
  )
  it('can edit a model while preserving a catalog display-name alias with spaces', async () => {
    getGlobalModelOptions.mockResolvedValue({
      providers: [{ name: 'Named Provider', slug: 'named', models: ['old', 'new'] }]
    })
    const { onChange } = mount('old', 'Named Provider')
    await model('new')
    fireEvent.click(screen.getByRole('button', { name: 'Apply' }))
    expect(onChange).toHaveBeenCalledExactlyOnceWith({
      model: 'new',
      provider: 'Named Provider',
      resetDirectEndpoint: false
    })
  })
  it('isolates connection catalogs and late responses', async () => {
    let resolveA!: (value: typeof catalog) => void
    getGlobalModelOptions.mockImplementation((_opts, owner) =>
      owner.connectionId === 'A'
        ? new Promise(resolve => {
            resolveA = resolve
          })
        : Promise.resolve({ providers: [{ name: 'Only B', slug: 'only-b', models: ['b'] }] })
    )
    const { client, view } = mount('', '', { connectionId: 'A', profile: 'same' })
    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalledTimes(1))
    view.rerender(
      <QueryClientProvider client={client}>
        <DelegationModelProviderField
          model=""
          onChange={vi.fn()}
          provider=""
          scope={{ connectionId: 'B', profile: 'same' }}
        />
      </QueryClientProvider>
    )
    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalledTimes(2))
    resolveA(catalog)
    await provider('Only B')
    fireEvent.click(screen.getByLabelText('Subagent Provider'))
    expect(screen.queryByRole('option', { name: 'Anthropic' })).toBeNull()
  })
})
