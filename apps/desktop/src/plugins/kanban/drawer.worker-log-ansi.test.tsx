import type { PluginRestOptions } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness drives the host's request scope, as a connection switch does.
// eslint-disable-next-line no-restricted-imports
import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { bindApi } from './api'
import { TaskDrawer } from './drawer'
import { KANBAN_LOCALES } from './i18n'
import type { KanbanTaskDetail } from './types'

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

/** A worker-log tail exactly as a piped stdout lands it: still styled (SGR dim
 *  + SGR reset) and carrying an OSC 8 hyperlink, so a plain log view with no
 *  terminal emulator must not paint the escape sequences. */
const ansiTail = '\x1b[2m  · build started\x1b[0m\nsee \x1b]8;;https://example.invalid/x\x07https://example.invalid/x\x1b\\ for details'
/** The same tail with every escape removed — what the user should actually read. */
const cleanTail = '  · build started\nsee https://example.invalid/x for details'

const legacyDetail: Omit<KanbanTaskDetail, 'attachments'> = {
  task: { id: 't_example', title: 'Example task', body: 'Keep this description readable.', status: 'todo' },
  comments: [],
  events: [],
  links: { parents: [], children: [] },
  runs: []
}

let client: QueryClient
let disposeApi: () => void
let disposeLocales: () => void

const rest = vi.fn(async (path: string, _options?: PluginRestOptions): Promise<unknown> => {
  if (path === '/tasks/t_example') {
    return legacyDetail
  }

  if (path.startsWith('/tasks/t_example/log?')) {
    return { exists: true, content: ansiTail, size_bytes: ansiTail.length, truncated: false }
  }

  if (path === '/profiles') {
    return { profiles: [] }
  }

  if (path === '/orchestration') {
    return { default_assignee: '' }
  }

  throw new Error(`Unexpected REST request: ${path}`)
})

beforeEach(() => {
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  disposeLocales = registerPluginLocales('kanban', KANBAN_LOCALES)
  disposeApi = bindApi(
    async <T,>(path: string, options?: PluginRestOptions) => (await rest(path, options)) as T,
    { get: (_key, fallback) => fallback, set: vi.fn(), remove: vi.fn() },
    () => vi.fn()
  )
})

afterEach(() => {
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.unstubAllGlobals()
  cleanup()
  client.clear()
  disposeApi()
  disposeLocales()
  vi.clearAllMocks()
})

describe('worker log ANSI stripping', () => {
  it('renders the log tail as clean plain text with no escape bytes', async () => {
    render(
      <QueryClientProvider client={client}>
        <TaskDrawer columns={['todo', 'ready', 'done']} id="t_example" onClose={vi.fn()} onOpen={vi.fn()} />
      </QueryClientProvider>
    )

    // TaskDrawer renders its whole UI into a Radix dialog portaled to
    // document.body, so the render container is empty — query the dialog itself.
    const dialog = await screen.findByRole('dialog')

    // The log tab only appears once a non-empty log is known; switch to it.
    const workerLogTab = await screen.findByRole('button', { name: 'Worker log' })
    fireEvent.click(workerLogTab)

    // LogView is THE raw-log viewer: the only selectable-text region that also
    // carries its `overflow-auto` + mono `whitespace-pre-wrap` signature, so the
    // task-title chip and other selectable chips are never mistaken for it.
    const logRegion = await waitFor(() => {
      const candidates = Array.from(
        dialog.querySelectorAll<HTMLElement>('[data-selectable-text="true"]')
      ).filter(el => el.classList.contains('overflow-auto') && el.classList.contains('whitespace-pre-wrap'))
      if (candidates.length !== 1) throw new Error(`expected exactly one LogView region, found ${candidates.length}`)
      return candidates[0]
    })
    await waitFor(() => expect(logRegion.textContent).toBe(cleanTail))

    // The exact escape-free tail, not the ANSI-laced one — the SGR/OSC sequences
    // must have been stripped at the render call site, not passed through.
    expect(logRegion.textContent).not.toContain('\x1b')

    // The whole drawer (the portaled dialog) paints no escape byte anywhere: no
    // lone ESC, no OSC hyperlink marker, no BEL — the tail leaks nowhere.
    expect(dialog.textContent).not.toContain('\x1b')
    expect(dialog.textContent).not.toContain(']8;')
    expect(dialog.textContent).not.toContain('\x07')
    expect(dialog.textContent).toContain(cleanTail)
  })
})
