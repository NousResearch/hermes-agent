import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, findByText, render } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { $activeSessionId, $currentModel, $currentProvider } from '@/store/session'

import { ModelMenuPanel } from './model-menu-panel'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const getGlobalModelOptions = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args)
}))

// The exact payload shape observed live from /api/model/options for a config
// with one bare `model.provider: custom` main model plus two named
// `custom_providers` entries. Reproduces issue #59702: the Desktop picker
// shows only one of these three rows even though the backend/network payload
// is always complete (verified independently 3 ways: REST with refresh, REST
// without refresh, and the live gateway WebSocket JSON-RPC call).
const MAIN_CUSTOM_PROVIDER = {
  api_url: 'http://10.0.0.1:8090/v1',
  is_current: true,
  is_user_defined: true,
  models: ['Qwen3.6-27B-NVFP4-MTP-GGUF.gguf'],
  name: 'Custom endpoint',
  slug: 'custom',
  source: 'model-config',
  total_models: 1
}

const NAMED_CUSTOM_PROVIDER_A = {
  api_url: 'http://10.0.0.2:8080/v1',
  is_current: false,
  is_user_defined: true,
  models: ['gemma-4-12b-omni'],
  name: '3090-gemma4-12b',
  slug: 'custom:3090-gemma4-12b',
  source: 'user-config',
  total_models: 1
}

const NAMED_CUSTOM_PROVIDER_B = {
  api_url: 'http://10.0.0.3:8000/v1',
  is_current: false,
  is_user_defined: true,
  models: ['gemma-4-26b-a4b-nvfp4'],
  name: 'spark-gemma4-26b-a4b',
  slug: 'custom:spark-gemma4-26b-a4b',
  source: 'user-config',
  total_models: 1
}

beforeEach(() => {
  $activeSessionId.set('runtime-1')
  $currentModel.set('Qwen3.6-27B-NVFP4-MTP-GGUF.gguf')
  $currentProvider.set('custom')
  getGlobalModelOptions.mockResolvedValue({
    providers: [MAIN_CUSTOM_PROVIDER, NAMED_CUSTOM_PROVIDER_A, NAMED_CUSTOM_PROVIDER_B]
  })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderPanel(onSelectModel = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <DropdownMenu open>
        <DropdownMenuContent>
          <ModelMenuPanel onSelectModel={onSelectModel} requestGateway={vi.fn() as never} />
        </DropdownMenuContent>
      </DropdownMenu>
    </QueryClientProvider>
  )

  return onSelectModel
}

describe('ModelMenuPanel multiple custom_providers entries (#59702)', () => {
  it('renders all three custom provider groups when three are configured', async () => {
    renderPanel()

    // Wait for the async useQuery to resolve and the panel to re-render.
    await findByText(document.body, 'Custom endpoint')

    expect(document.body.textContent).toContain('Custom endpoint')
    expect(document.body.textContent).toContain('3090-gemma4-12b')
    expect(document.body.textContent).toContain('spark-gemma4-26b-a4b')
  })

  it('renders each provider group model row, not just the first', async () => {
    renderPanel()

    await findByText(document.body, 'Custom endpoint')

    expect(
      document.body.textContent?.includes('Qwen3.6-27B-NVFP4-MTP-GGUF.gguf') ||
        document.body.textContent?.includes('Qwen3.6 27B NVFP4 MTP GGUF.Gguf')
    ).toBe(true)
    expect(document.body.textContent).toContain('Gemma 4 12b Omni')
    // spark's model id renders as its family display name, accept either the
    // raw id or the humanized form. We only care whether the ROW exists at all.
    expect(
      document.body.textContent?.includes('gemma-4-26b-a4b-nvfp4') ||
        document.body.textContent?.includes('Gemma 4 26b A4b Nvfp4')
    ).toBe(true)
  })

  it('renders all three groups via the real gateway.request path (not the REST fallback)', async () => {
    // Prod always goes through gateway.request once connected. model-options.ts
    // only falls back to REST getGlobalModelOptions when `gateway` is undefined.
    // Earlier tests only exercised the REST fallback and passed; this exercises
    // the actual code path a connected Desktop app uses.
    const gatewayRequest = vi.fn().mockResolvedValue({
      providers: [MAIN_CUSTOM_PROVIDER, NAMED_CUSTOM_PROVIDER_A, NAMED_CUSTOM_PROVIDER_B]
    })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

    render(
      <QueryClientProvider client={client}>
        <DropdownMenu open>
          <DropdownMenuContent>
            <ModelMenuPanel
              gateway={{ request: gatewayRequest } as never}
              onSelectModel={vi.fn()}
              requestGateway={vi.fn() as never}
            />
          </DropdownMenuContent>
        </DropdownMenu>
      </QueryClientProvider>
    )

    await findByText(document.body, 'Custom endpoint')

    expect(gatewayRequest).toHaveBeenCalledWith('model.options', expect.objectContaining({ session_id: 'runtime-1' }))
    expect(document.body.textContent).toContain('Custom endpoint')
    expect(document.body.textContent).toContain('3090-gemma4-12b')
    expect(document.body.textContent).toContain('spark-gemma4-26b-a4b')
  })
})
