import type { PluginRestOptions } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { bindApi } from './api'
import { BoardHeaderToggles } from './board'
import { en, KANBAN_LOCALES } from './i18n'
import type { BoardMeta } from './types'

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

const board: BoardMeta = { slug: 'widget', name: 'Widget' }

let client: QueryClient
let disposeApi: () => void
let disposeLocales: () => void
let rest: ReturnType<typeof vi.fn>

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  rest = vi.fn(async (path: string, options?: PluginRestOptions): Promise<unknown> => {
    if (path === '/boards/widget' && options?.method === 'PATCH') {
      const patch = options.body as Record<string, unknown>

      return { board: { ...board, ...patch } }
    }

    throw new Error(`Unexpected REST request: ${path}`)
  })
  disposeApi = bindApi(
    async <T,>(path: string, options?: PluginRestOptions) => (await rest(path, options)) as T,
    { get: (_key, fallback) => fallback, set: vi.fn(), remove: vi.fn() },
    () => vi.fn()
  )
})

afterEach(() => {
  cleanup()
  client.clear()
  disposeApi()
  disposeLocales()
  vi.clearAllMocks()
})

function renderToggles(meta: BoardMeta = board) {
  return render(
    <QueryClientProvider client={client}>
      <BoardHeaderToggles board={meta} />
    </QueryClientProvider>
  )
}

describe('BoardHeaderToggles default rendering', () => {
  it('renders all three switches checked when the board carries no overrides', () => {
    renderToggles()

    const dispatch = screen.getByRole('switch', { name: en.headerDispatch })
    const decompose = screen.getByRole('switch', { name: en.headerDecompose })
    const review = screen.getByRole('switch', { name: en.headerReview })

    expect(dispatch.getAttribute('data-state')).toBe('checked')
    expect(decompose.getAttribute('data-state')).toBe('checked')
    expect(review.getAttribute('data-state')).toBe('checked')
  })

  it('reflects an explicit false override', () => {
    renderToggles({ ...board, dispatch_enabled: false })

    expect(screen.getByRole('switch', { name: en.headerDispatch }).getAttribute('data-state')).toBe('unchecked')
    // Untouched fields still default to checked.
    expect(screen.getByRole('switch', { name: en.headerDecompose }).getAttribute('data-state')).toBe('checked')
  })
})

describe('BoardHeaderToggles click behavior', () => {
  it('sends exactly one field in the PATCH body for the clicked switch', async () => {
    renderToggles()

    fireEvent.click(screen.getByRole('switch', { name: en.headerDecompose }))

    await vi.waitFor(() => {
      const call = rest.mock.calls.find(([path]) => path === '/boards/widget')

      expect(call).toBeTruthy()
    })

    const [, options] = rest.mock.calls.find(([path]) => path === '/boards/widget')!

    expect(options?.method).toBe('PATCH')
    expect(options?.body).toEqual({ auto_decompose_enabled: false })
  })
})
