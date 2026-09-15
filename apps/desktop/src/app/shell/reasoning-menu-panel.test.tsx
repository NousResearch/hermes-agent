import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { modelOptionsQueryKey } from '@/lib/model-options'
import {
  $activeSessionId,
  $currentFastMode,
  $currentModel,
  $currentProvider,
  $currentReasoningEffort
} from '@/store/session'

import { ReasoningMenuPanel } from './reasoning-menu-panel'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const SESSION_ID = 'runtime-1'

const getGlobalModelOptions = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args),
  setApiRequestProfile: vi.fn()
}))

// A custom endpoint the user added. `capabilities` is what the backend reports
// from models.dev/builtin tables keyed on someone else's provider; here it
// claims this model has no reasoning control, the verdict that used to render
// "No options for this model" and hide the composer's effort pill.
const CUSTOM_PROVIDER = {
  aliases: ['custom:my-endpoint'],
  capabilities: { 'deepseek-v4.1-flash': { fast: false, reasoning: false } },
  is_user_defined: true,
  models: ['deepseek-v4.1-flash'],
  name: 'My Endpoint',
  slug: 'my-endpoint'
}

// A curated row: its catalog genuinely knows the model has no thinking knob, so
// the dead-end is the right answer and must stay.
const CURATED_PROVIDER = {
  capabilities: { 'gpt-4o': { fast: false, reasoning: false } },
  models: ['gpt-4o'],
  name: 'OpenAI',
  slug: 'openai'
}

beforeEach(() => {
  $activeSessionId.set(SESSION_ID)
  $currentFastMode.set(false)
  $currentModel.set('')
  $currentProvider.set('')
  $currentReasoningEffort.set('')
  getGlobalModelOptions.mockResolvedValue({ providers: [CURATED_PROVIDER, CUSTOM_PROVIDER] })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const requestGateway = vi.fn(async (method: string) => {
    if (method === 'model.options') {
      return getGlobalModelOptions()
    }

    return {}
  })

  render(
    <QueryClientProvider client={client}>
      <DropdownMenu open>
        <DropdownMenuContent>
          <ReasoningMenuPanel onSelectModel={vi.fn()} requestGateway={requestGateway as never} />
        </DropdownMenuContent>
      </DropdownMenu>
    </QueryClientProvider>
  )

  // The pre-fetch render looks the same as a resolved one for a model with no
  // verdict, so gate every assertion on the catalog actually landing.
  const catalogLoaded = vi.waitFor(() => {
    expect(client.getQueryData(modelOptionsQueryKey('default', SESSION_ID))).toBeDefined()
  })

  return { catalogLoaded, requestGateway }
}

describe('ReasoningMenuPanel against a custom provider model', () => {
  it('offers the canonical ladder instead of a dead-end when the row says reasoning: false', async () => {
    $currentProvider.set('custom:my-endpoint')
    $currentModel.set('deepseek-v4.1-flash')

    const { catalogLoaded } = renderPanel()
    await catalogLoaded

    // The one shared value set (@hermes/shared), not a second list.
    expect(screen.getByText('Minimal')).toBeTruthy()
    expect(screen.getByText('Medium')).toBeTruthy()
    expect(screen.getByText('High')).toBeTruthy()
    expect(screen.getByText('Ultra')).toBeTruthy()
    expect(screen.queryByText('No options for this model')).toBeNull()
  })

  it('writes the picked level onto the session through config.set', async () => {
    $currentProvider.set('custom:my-endpoint')
    $currentModel.set('deepseek-v4.1-flash')

    const { catalogLoaded, requestGateway } = renderPanel()
    await catalogLoaded

    fireEvent.click(screen.getByText('Max'))

    await vi.waitFor(() => {
      expect(requestGateway).toHaveBeenCalledWith('config.set', {
        key: 'reasoning',
        session_id: SESSION_ID,
        value: 'max'
      })
    })
  })

  it('still dead-ends on a curated row whose catalog really says no reasoning', async () => {
    $currentProvider.set('openai')
    $currentModel.set('gpt-4o')

    const { catalogLoaded } = renderPanel()
    await catalogLoaded

    expect(screen.getByText('No options for this model')).toBeTruthy()
    expect(screen.queryByText('Ultra')).toBeNull()
  })
})
