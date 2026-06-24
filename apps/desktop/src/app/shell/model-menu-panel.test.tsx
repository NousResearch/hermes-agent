import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import {
  $activeSessionId,
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort
} from '@/store/session'

// Radix calls these on open; jsdom doesn't implement them.
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
  // Pre-session (global) selection: the panel marks the row "current" from the
  // sticky composer stores, so reasoning effort resolves to the live value.
  $activeSessionId.set(null)
  $currentModel.set('qwen3.7-max')
  $currentProvider.set('dashscope')
  $currentReasoningEffort.set('high')
  $currentFastMode.set(false)

  getGlobalModelOptions.mockResolvedValue({
    model: 'qwen3.7-max',
    provider: 'dashscope',
    providers: [
      {
        name: 'DashScope',
        slug: 'dashscope',
        models: ['qwen3.7-max'],
        authenticated: true,
        capabilities: { 'qwen3.7-max': { reasoning: true, fast: false } }
      }
    ]
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

async function renderPanel() {
  const { ModelMenuPanel } = await import('./model-menu-panel')
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  return render(
    <QueryClientProvider client={client}>
      <DropdownMenu open>
        <DropdownMenuContent>
          <ModelMenuPanel onSelectModel={vi.fn()} requestGateway={vi.fn() as never} />
        </DropdownMenuContent>
      </DropdownMenu>
    </QueryClientProvider>
  )
}

describe('ModelMenuPanel reasoning-effort badge', () => {
  // Regression for #51833: the reasoning effort must render as its own badge,
  // distinct from the model name, so "High" doesn't read as part of a
  // differently-named model.
  it('renders the reasoning effort as a separate badge, not appended to the name', async () => {
    await renderPanel()

    const name = await screen.findByText('Qwen3.7 Max')
    const badge = await screen.findByText('High')

    // The effort lives in its own element — the name node never absorbs it.
    expect(name).not.toBe(badge)
    expect(name.contains(badge)).toBe(false)
    expect(name.textContent).toBe('Qwen3.7 Max')

    // …and it carries distinct badge styling (bordered chip), not plain inline
    // tertiary text bleeding into the model name.
    expect(badge.className).toContain('border')
    expect(badge.className).toContain('rounded-sm')
  })

  it('drops the effort badge entirely when the model has no reasoning support', async () => {
    getGlobalModelOptions.mockResolvedValue({
      model: 'qwen3.7-max',
      provider: 'dashscope',
      providers: [
        {
          name: 'DashScope',
          slug: 'dashscope',
          models: ['qwen3.7-max'],
          authenticated: true,
          capabilities: { 'qwen3.7-max': { reasoning: false, fast: false } }
        }
      ]
    })

    await renderPanel()

    await screen.findByText('Qwen3.7 Max')
    await waitFor(() => expect(screen.queryByText('High')).toBeNull())
    expect(screen.queryByText('Med')).toBeNull()
  })
})
