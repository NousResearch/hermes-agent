import type { PluginRestOptions } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'

import { bindApi, taskKey } from './api'
import { TaskDrawer } from './drawer'
import { en, KANBAN_LOCALES } from './i18n'
import type { KanbanEvent, KanbanRun, KanbanTaskDetail } from './types'

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

const legacyDetail: Omit<KanbanTaskDetail, 'attachments'> = {
  task: { id: 't_example', title: 'Example task', body: 'Keep this description readable.', status: 'todo' },
  comments: [{ id: 1, author: 'test', body: 'Keep this comment readable.', created_at: 0 }],
  events: [],
  links: { parents: [], children: [] },
  runs: []
}

let detail: object
let client: QueryClient
let disposeApi: () => void
let disposeLocales: () => void

const rest = vi.fn(async (path: string, options?: PluginRestOptions): Promise<unknown> => {
  if (path === '/tasks/t_example/attachments' && options?.method === 'POST') {
    detail = { ...legacyDetail, attachments: [{ id: 1, filename: options.upload?.filename }] }

    return { ok: true }
  }

  if (path === '/tasks/t_example') {
    return detail
  }

  if (path.startsWith('/tasks/t_example/log?')) {
    return { exists: false, content: '', size_bytes: 0, truncated: false }
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
  cleanup()
  client.clear()
  disposeApi()
  disposeLocales()
  vi.clearAllMocks()
})

function openDrawer() {
  return render(
    <QueryClientProvider client={client}>
      <TaskDrawer
        columns={['todo', 'ready', 'done']}
        id="t_example"
        lookup={() => undefined}
        onClose={vi.fn()}
        onOpen={vi.fn()}
      />
    </QueryClientProvider>
  )
}

describe('task attachment compatibility', () => {
  it.each([{}, { attachments: null }])(
    'keeps older task details usable without attachment controls (%j)',
    async extra => {
      detail = { ...legacyDetail, ...extra }
      openDrawer()

      expect(await screen.findByRole('heading', { name: legacyDetail.task.title })).toBeTruthy()
      expect(screen.getByText(legacyDetail.task.body!)).toBeTruthy()
      expect(screen.getByText(legacyDetail.comments[0].body)).toBeTruthy()
      expect(screen.queryByRole('button', { name: en.uploadAttachment })).toBeNull()
      expect(screen.queryByText(en.noAttachments)).toBeNull()

      // A later backend response restores the capability without remounting.
      detail = { ...legacyDetail, attachments: [] }
      await act(() => client.invalidateQueries({ queryKey: taskKey('', legacyDetail.task.id) }))
      expect(await screen.findByRole('button', { name: en.uploadAttachment })).toBeTruthy()
      expect(screen.getByText(en.noAttachments)).toBeTruthy()
    }
  )

  it('keeps upload and attachment rendering working for a supported empty list', async () => {
    detail = { ...legacyDetail, attachments: [] }
    const { container } = openDrawer()
    const upload = await screen.findByRole('button', { name: en.uploadAttachment })
    expect(screen.getByText(en.noAttachments)).toBeTruthy()

    const input = container.querySelector<HTMLInputElement>('input[type="file"]')!
    const click = vi.spyOn(input, 'click')
    fireEvent.click(upload)
    expect(click).toHaveBeenCalledOnce()

    const file = new File(['example'], 'example.txt', { type: 'text/plain' })
    const bytes = new ArrayBuffer(7)
    // jsdom's File lacks arrayBuffer; the upload still uses the real REST adapter.
    Object.defineProperty(file, 'arrayBuffer', { value: async () => bytes })
    fireEvent.change(input, { target: { files: [file] } })

    await waitFor(() =>
      expect(rest).toHaveBeenCalledWith('/tasks/t_example/attachments', {
        method: 'POST',
        upload: { filename: file.name, contentType: file.type, bytes }
      })
    )
    expect(await screen.findByText(file.name)).toBeTruthy()
    expect(screen.queryByText(en.noAttachments)).toBeNull()
  })
})

describe('activity folding by attempt', () => {
  const runs: KanbanRun[] = [
    {
      ended_at: 1_699_999_100,
      id: 10,
      outcome: 'crashed',
      profile: 'worker-a',
      started_at: 1_699_999_000,
      status: 'done'
    },
    {
      ended_at: 1_700_000_600,
      id: 11,
      outcome: 'succeeded',
      profile: 'worker-a',
      started_at: 1_700_000_000,
      status: 'done',
      summary: 'first pass summary'
    },
    {
      ended_at: 1_700_001_300,
      error: 'timed out after 300s',
      id: 12,
      outcome: 'timed_out',
      profile: 'worker-b',
      started_at: 1_700_001_000,
      status: 'done'
    }
  ]

  // Events per attempt: two task-scoped rows (no run), then 3 + 2 from the runs.
  const eventsFor = (runId: null | number, count: number, from: number): KanbanEvent[] =>
    Array.from({ length: count }, (_, index) => ({
      created_at: from + index,
      id: (runId ?? 9) * 100 + index,
      kind: 'heartbeat',
      payload: null,
      run_id: runId
    }))

  const events = [
    ...eventsFor(null, 2, 1_699_998_000),
    ...eventsFor(11, 3, 1_700_000_000),
    ...eventsFor(12, 2, 1_700_001_000)
  ]

  /** Fold rows that can actually unfold something (aria-expanded is their mark). */
  const foldRows = (container: HTMLElement) => [
    ...container.querySelectorAll<HTMLElement>('[data-attempt][aria-expanded]')
  ]

  const renderedEvents = (container: HTMLElement) => container.querySelectorAll('[data-event-id]').length

  const hiddenEvents = (container: HTMLElement) =>
    foldRows(container)
      .filter(row => row.getAttribute('aria-expanded') === 'false')
      .reduce((total, row) => total + Number(row.dataset.events), 0)

  const unfoldAll = (container: HTMLElement) => {
    for (const row of foldRows(container).filter(row => row.getAttribute('aria-expanded') === 'false')) {
      fireEvent.click(row)
    }
  }

  it('unfolds the newest attempt only, counting the rest behind fold rows', async () => {
    detail = { ...legacyDetail, events, runs }
    const { container } = openDrawer()

    await screen.findByRole('heading', { name: legacyDetail.task.title })

    expect([...container.querySelectorAll<HTMLElement>('[data-attempt]')].map(row => row.dataset.attempt)).toEqual([
      'task',
      '10',
      '11',
      '12'
    ])
    expect(foldRows(container).map(row => row.getAttribute('aria-expanded'))).toEqual(['false', 'false', 'true'])
    // The run that emitted no events is still listed — it just has no events to unfold.
    expect(container.querySelector('[data-attempt="10"]')?.hasAttribute('aria-expanded')).toBe(false)
    // Nothing but run 12's events is rendered, and its run note stays readable.
    expect(renderedEvents(container)).toBe(2)
    expect(screen.getByText('timed out after 300s')).toBeTruthy()
  })

  it('accounts for every event: folded counts + rendered events = total', async () => {
    detail = { ...legacyDetail, events, runs }
    const { container } = openDrawer()

    await screen.findByRole('heading', { name: legacyDetail.task.title })
    expect(hiddenEvents(container) + renderedEvents(container)).toBe(events.length)

    unfoldAll(container)

    // Every folded attempt unfolded renders exactly the events it accounted for:
    // none dropped, none rendered twice.
    expect(renderedEvents(container)).toBe(events.length)
    expect(foldRows(container).map(row => row.getAttribute('aria-expanded'))).toEqual(['true', 'true', 'true'])
  })

  it('renders fewer nodes folded than unfolded', async () => {
    detail = { ...legacyDetail, events, runs }
    const { container } = openDrawer()

    await screen.findByRole('heading', { name: legacyDetail.task.title })
    const folded = container.querySelectorAll('*').length

    unfoldAll(container)

    expect(folded).toBeLessThan(container.querySelectorAll('*').length)
  })

  it('folds and unfolds from the keyboard on a labelled, focusable row', async () => {
    detail = { ...legacyDetail, events, runs }
    const { container } = openDrawer()

    await screen.findByRole('heading', { name: legacyDetail.task.title })

    const row = container.querySelector<HTMLElement>('[data-attempt="11"]')!
    expect(row.tagName).toBe('BUTTON')
    expect(row.getAttribute('aria-label')).toBe(en.expand('#3'))

    // Enter toggles once: a second activation would unfold and fold it again.
    fireEvent.keyDown(row, { key: 'Enter' })
    expect(row.getAttribute('aria-expanded')).toBe('true')
    expect(row.getAttribute('aria-label')).toBe(en.collapse('#3'))
    expect(renderedEvents(container)).toBe(5)

    fireEvent.click(row)
    expect(row.getAttribute('aria-expanded')).toBe('false')
    expect(renderedEvents(container)).toBe(2)
  })

  it('still renders the fold rows when a card has runs but no events', async () => {
    detail = { ...legacyDetail, runs }
    const { container } = openDrawer()

    await screen.findByRole('heading', { name: legacyDetail.task.title })

    expect(container.querySelectorAll('[data-attempt]').length).toBe(runs.length)
    expect(container.querySelectorAll('[data-attempt][aria-expanded]').length).toBe(0)
    expect(renderedEvents(container)).toBe(0)
  })
})
