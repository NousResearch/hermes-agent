import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const getGlobalModelInfo = vi.fn()
const getHermesConfigRecord = vi.fn()
const saveHermesConfig = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelInfo: () => getGlobalModelInfo(),
  getHermesConfigRecord: () => getHermesConfigRecord(),
  saveHermesConfig: (config: unknown) => saveHermesConfig(config)
}))

import { ContextWindowDialog } from './context-window-dialog'

function renderDialog() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <ContextWindowDialog onOpenChange={() => {}} open />
    </QueryClientProvider>
  )
}

beforeEach(() => {
  getGlobalModelInfo.mockResolvedValue({
    model: 'hermes-4',
    provider: 'nous',
    auto_context_length: 200_000,
    config_context_length: 0,
    effective_context_length: 200_000
  })
  getHermesConfigRecord.mockResolvedValue({ model: 'hermes-4', model_context_length: 0 })
  saveHermesConfig.mockResolvedValue({ ok: true })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('ContextWindowDialog', () => {
  it('shows the auto-detected value when no override is pinned', async () => {
    renderDialog()

    // 0 means auto-detect: the field stays blank while the resolved window is
    // still displayed, so the user can see what they'd be overriding.
    await waitFor(() => expect(screen.getByText(/Auto-detected: 200k/i)).toBeTruthy())

    const field = screen.getByLabelText('Override') as HTMLInputElement
    expect(field.value).toBe('')
  })

  it('seeds the field from an existing override', async () => {
    getGlobalModelInfo.mockResolvedValue({
      model: 'hermes-4',
      provider: 'nous',
      auto_context_length: 200_000,
      config_context_length: 64_000,
      effective_context_length: 64_000
    })

    renderDialog()

    await waitFor(() => {
      expect((screen.getByLabelText('Override') as HTMLInputElement).value).toBe('64000')
    })
  })

  it('displays the provider-enforced effective window, not the raw vendor figure', async () => {
    // models.dev reports 1.05M for this slug; the provider caps it at 272k.
    getGlobalModelInfo.mockResolvedValue({
      model: 'gpt-5.5',
      provider: 'codex',
      auto_context_length: 272_000,
      config_context_length: 0,
      effective_context_length: 272_000
    })

    renderDialog()

    // "In use" and the figure are separate elements (the number is emphasised),
    // so match on the container's combined text.
    await waitFor(() =>
      expect(screen.getByText((_, el) => /In use:\s*272k/i.test(el?.textContent ?? ''), { selector: 'span' })).toBeTruthy()
    )
  })

  it('persists an explicit override through the existing config surface', async () => {
    renderDialog()

    await waitFor(() => expect(screen.getByLabelText('Override')).toBeTruthy())

    fireEvent.change(screen.getByLabelText('Override'), { target: { value: '128000' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(saveHermesConfig).toHaveBeenCalled())

    const saved = saveHermesConfig.mock.calls[0]?.[0] as Record<string, unknown>
    expect(saved.model_context_length).toBe(128_000)
    // The model must ride along: the backend only folds the override back into
    // the model dict when it can see which model the pin belongs to.
    expect(saved.model).toBe('hermes-4')
  })

  it('clears the override back to auto-detect by persisting 0', async () => {
    getGlobalModelInfo.mockResolvedValue({
      model: 'hermes-4',
      provider: 'nous',
      auto_context_length: 200_000,
      config_context_length: 64_000,
      effective_context_length: 64_000
    })

    renderDialog()

    // Wait for the pin to load: the button is disabled until an override
    // actually exists to clear.
    await waitFor(() =>
      expect((screen.getByRole('button', { name: 'Use auto-detect' }) as HTMLButtonElement).disabled).toBe(false)
    )

    fireEvent.click(screen.getByRole('button', { name: 'Use auto-detect' }))

    await waitFor(() => expect(saveHermesConfig).toHaveBeenCalled())
    expect((saveHermesConfig.mock.calls[0]?.[0] as Record<string, unknown>).model_context_length).toBe(0)
  })

  it('keeps what the user typed when the model-info fetch resolves afterwards', async () => {
    // Regression: the field used to be seeded from an effect that ran when the
    // query settled, so a response arriving after the user started typing
    // clobbered their input. The draft now wins over the fetched value.
    let resolveInfo: (value: unknown) => void = () => {}
    getGlobalModelInfo.mockReturnValueOnce(new Promise(resolve => (resolveInfo = resolve)))

    renderDialog()

    fireEvent.change(screen.getByLabelText('Override'), { target: { value: '96000' } })

    resolveInfo({
      model: 'hermes-4',
      provider: 'nous',
      auto_context_length: 200_000,
      config_context_length: 32_000,
      effective_context_length: 32_000
    })

    await waitFor(() => expect(screen.getByText(/Auto-detected: 200k/i)).toBeTruthy())
    expect((screen.getByLabelText('Override') as HTMLInputElement).value).toBe('96000')
  })

  it('refuses to persist a non-numeric override', async () => {
    renderDialog()

    await waitFor(() => expect(screen.getByLabelText('Override')).toBeTruthy())

    fireEvent.change(screen.getByLabelText('Override'), { target: { value: 'lots' } })
    fireEvent.click(screen.getByRole('button', { name: 'Save' }))

    await waitFor(() => expect(screen.getByText(/Enter a whole number of tokens/i)).toBeTruthy())
    expect(saveHermesConfig).not.toHaveBeenCalled()
  })

  it('tells the user the pin is route-scoped and applies next turn', async () => {
    renderDialog()

    // Both are load-bearing: the pin is dropped fail-closed when the route
    // changes, and config is adopted at the start of the next turn rather
    // than rebuilding the live conversation.
    await waitFor(() => expect(screen.getByText(/Switching either one returns to auto-detect/i)).toBeTruthy())
    expect(screen.getByText(/Takes effect on your next message/i)).toBeTruthy()
  })
})
