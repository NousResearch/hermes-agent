import type { PluginRestOptions } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { $boardSlug, bindApi } from './api'
import { BoardSwitcher } from './board-switcher'
import { en, KANBAN_LOCALES } from './i18n'
import type { BoardMeta } from './types'

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

let boards: BoardMeta[]
let client: QueryClient
let disposeApi: () => void
let disposeLocales: () => void
let patched: Array<{ slug: string; patch: Record<string, unknown> }>

const rest = vi.fn(async (path: string, options?: PluginRestOptions): Promise<unknown> => {
  if (path === '/boards' && (!options || options.method === undefined || options.method === 'GET')) {
    return { boards: boards.map(b => ({ ...b })), current: 'proj' }
  }

  const patchMatch = /^\/boards\/([^/]+)$/.exec(path)

  if (patchMatch && options?.method === 'PATCH') {
    const slug = decodeURIComponent(patchMatch[1])
    const patch = options.body as Record<string, unknown>

    patched.push({ slug, patch })
    const idx = boards.findIndex(b => b.slug === slug)

    if (idx >= 0) {
      boards[idx] = { ...boards[idx], ...patch }
    }

    return { board: boards[idx] }
  }

  if (path === '/projects') {
    return { projects: [] }
  }

  throw new Error(`Unexpected REST request: ${path} ${options?.method ?? 'GET'}`)
})

beforeEach(() => {
  patched = []
  boards = [{ slug: 'proj', name: 'Proj', total: 3, is_current: true }]
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  disposeApi = bindApi(
    async <T,>(path: string, options?: PluginRestOptions) => (await rest(path, options)) as T,
    { get: (_key, fallback) => fallback, set: vi.fn(), remove: vi.fn() },
    () => vi.fn()
  )
  $boardSlug.set('')
})

afterEach(() => {
  cleanup()
  client.clear()
  disposeApi()
  disposeLocales()
  vi.clearAllMocks()
})

function openSwitcher() {
  return render(
    <QueryClientProvider client={client}>
      <BoardSwitcher />
    </QueryClientProvider>
  )
}

// Radix's dropdown trigger opens on pointerdown (a synthetic 'click' fireEvent
// alone won't do it) — same technique as project-menu.test.tsx (#67500).
const openTriggerMenu = (trigger: HTMLElement) => {
  fireEvent.pointerDown(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.pointerUp(trigger, { button: 0, pointerType: 'mouse' })
  fireEvent.click(trigger)
}

async function openBoardSettings() {
  openSwitcher()
  openTriggerMenu(await screen.findByRole('button', { name: /Proj/ }))
  const settingsItem = await screen.findByRole('menuitem', { name: en.settingsDots })
  fireEvent.click(settingsItem)
}

describe('per-board dispatch toggles', () => {
  it('defaults every toggle to on when the backend omits the flags (old board.json)', async () => {
    await openBoardSettings()

    const dispatchSwitch = await screen.findByRole('switch', { name: en.boardDispatchEnabled })
    const decomposeSwitch = await screen.findByRole('switch', { name: en.boardAutoDecomposeEnabled })
    const reviewSwitch = await screen.findByRole('switch', { name: en.boardReviewDispatchEnabled })

    expect(dispatchSwitch.getAttribute('aria-checked')).toBe('true')
    expect(decomposeSwitch.getAttribute('aria-checked')).toBe('true')
    expect(reviewSwitch.getAttribute('aria-checked')).toBe('true')
  })

  it('turning a switch off sends exactly that field and nothing else', async () => {
    await openBoardSettings()

    const dispatchSwitch = await screen.findByRole('switch', { name: en.boardDispatchEnabled })
    fireEvent.click(dispatchSwitch)

    await waitFor(() => expect(patched).toContainEqual({ slug: 'proj', patch: { dispatch_enabled: false } }))
    // The other two flags are untouched by this write.
    expect(patched.some(p => 'auto_decompose_enabled' in p.patch || 'review_dispatch_enabled' in p.patch)).toBe(false)
  })

  it('reflects the server value after the toggle round-trips (no stale snapshot)', async () => {
    boards[0].dispatch_enabled = false
    await openBoardSettings()

    const dispatchSwitch = await screen.findByRole('switch', { name: en.boardDispatchEnabled })
    expect(dispatchSwitch.getAttribute('aria-checked')).toBe('false')

    fireEvent.click(dispatchSwitch)
    await waitFor(() => expect(patched).toContainEqual({ slug: 'proj', patch: { dispatch_enabled: true } }))
    await waitFor(async () => {
      const refreshed = await screen.findByRole('switch', { name: en.boardDispatchEnabled })
      expect(refreshed.getAttribute('aria-checked')).toBe('true')
    })
  })
})
