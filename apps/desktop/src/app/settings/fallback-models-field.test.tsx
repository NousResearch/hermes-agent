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
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args),
  profileScopeKey: (scope: { connectionId?: string; profile?: string }) =>
    `${scope.connectionId ?? ''}::${scope.profile ?? 'default'}`
}))

beforeEach(() => {
  getGlobalModelOptions.mockResolvedValue({
    providers: [
      { name: 'GitHub Copilot', slug: 'copilot', models: ['gpt-5-mini', 'gpt-5.4-mini'] },
      { name: 'OpenAI Codex', slug: 'openai-codex', models: ['gpt-5.4-mini'] },
      { name: 'Nous', slug: 'nous', models: ['hermes-4'] }
    ]
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderField(value: unknown, onChange = vi.fn(), scope?: { connectionId: string; profile: string }) {
  const { FallbackModelsField } = await import('./fallback-models-field')
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  render(
    <QueryClientProvider client={client}>
      <FallbackModelsField onChange={onChange} scope={scope} value={value} />
    </QueryClientProvider>
  )

  return onChange
}

async function renderFieldWithRerender(value: unknown, onChange = vi.fn()) {
  const { FallbackModelsField } = await import('./fallback-models-field')
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const view = render(
    <QueryClientProvider client={client}>
      <FallbackModelsField onChange={onChange} value={value} />
    </QueryClientProvider>
  )

  return (next: unknown) =>
    view.rerender(
      <QueryClientProvider client={client}>
        <FallbackModelsField onChange={onChange} value={next} />
      </QueryClientProvider>
    )
}

const CHAIN = [
  { provider: 'copilot', model: 'gpt-5-mini' },
  { provider: 'openai-codex', model: 'gpt-5.4-mini' }
]

describe('FallbackModelsField', () => {
  it('renders each {provider, model} entry as its own row (never "[object Object]")', async () => {
    await renderField(CHAIN)

    // One Remove control per entry proves the object list became rows — the old
    // generic `list` input stringified the array to "[object Object]".
    expect(screen.getAllByLabelText('Remove')).toHaveLength(2)
    expect(screen.getByText('Add fallback')).toBeTruthy()
    expect(screen.queryByText(/\[object Object\]/)).toBeNull()
    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalled())
  })

  it('loads fallback choices from the selected settings owner', async () => {
    const scope = { connectionId: 'gateway-a', profile: 'research' }

    await renderField([], vi.fn(), scope)

    await waitFor(() => expect(getGlobalModelOptions).toHaveBeenCalledWith(undefined, scope))
  })

  it('removing a row emits the remaining entries with their routing keys intact', async () => {
    // #89184: a hand-written local-gateway chain carries base_url/api_key per
    // entry; the editor only owns provider/model and must not strip the rest.
    const routed = { provider: 'custom', model: 'glm-5.08', base_url: 'http://gw:8080/v1', api_key: '${GW_KEY}' }
    const onChange = await renderField([...CHAIN, routed])

    fireEvent.click(screen.getAllByLabelText('Remove')[0])

    expect(onChange.mock.calls.at(-1)?.[0]).toEqual([{ provider: 'openai-codex', model: 'gpt-5.4-mini' }, routed])
  })

  it('adding a blank row does not persist a partial entry', async () => {
    const onChange = await renderField(CHAIN)

    fireEvent.click(screen.getByText('Add fallback'))

    // The new empty row stays in the UI but only complete pairs are emitted.
    expect(onChange.mock.calls.at(-1)?.[0]).toEqual(CHAIN)
    expect(screen.getAllByLabelText('Remove')).toHaveLength(3)
  })

  it('resyncs rows when persisted config changes', async () => {
    const rerender = await renderFieldWithRerender(CHAIN)
    expect(screen.getAllByLabelText('Remove')).toHaveLength(2)

    rerender([{ provider: 'nous', model: 'hermes-4' }])

    await waitFor(() => expect(screen.getAllByLabelText('Remove')).toHaveLength(1))
  })

  it('keeps a draft row visible after autosave re-renders the same persisted chain', async () => {
    const onChange = vi.fn()
    const rerender = await renderFieldWithRerender([], onChange)

    fireEvent.click(screen.getByText('Add fallback'))

    expect(onChange.mock.calls.at(-1)?.[0]).toEqual([])
    expect(screen.getAllByLabelText('Remove')).toHaveLength(1)

    // Parent autosave echo — same complete chain, new array identity.
    rerender([])

    await waitFor(() => expect(screen.getAllByLabelText('Remove')).toHaveLength(1))
  })
})
