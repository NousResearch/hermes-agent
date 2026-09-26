import type { PluginRestOptions } from '@hermes/plugin-sdk'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// Test harness drives the host's request scope, as a connection switch does.
// eslint-disable-next-line no-restricted-imports
import { setApiRequestConnection, setApiRequestProfile } from '@/api/client'
// Test harness supplies the host's locale registration, as plugin loading does.
// eslint-disable-next-line no-restricted-imports
import { registerPluginLocales } from '@/i18n/plugin-i18n'
// Test harness reads the host's toast stack.
// eslint-disable-next-line no-restricted-imports
import { $notifications, clearNotifications } from '@/store/notifications'

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

  if (path.startsWith('/tasks/t_example/comments') && options?.method === 'POST') {
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
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  clearNotifications()
  vi.unstubAllGlobals()
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
  it('downloads the persisted attachment through its original remote owner', async () => {
    const save = vi.fn().mockResolvedValue({ saved: true })
    vi.stubGlobal('hermesDesktop', { saveGatewayFile: save })
    setApiRequestConnection('remote-owner')
    setApiRequestProfile('research')
    detail = {
      ...legacyDetail,
      attachments: [{ id: 42, filename: 'report.md', stored_path: '/persisted/attachments/report.md' }]
    }
    openDrawer()
    const download = await screen.findByRole('button', { name: 'Download report.md' })
    setApiRequestConnection('other-host')
    setApiRequestProfile('other-profile')
    fireEvent.click(download)
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith({
        connectionId: 'remote-owner',
        profile: 'research',
        path: '/persisted/attachments/report.md',
        suggestedName: 'report.md'
      })
    )
    await waitFor(() => expect($notifications.get()[0]).toMatchObject({ kind: 'info', message: 'Saved' }))
  })

  it('keeps a local-backend attachment on this computer after switching to a remote', async () => {
    const save = vi.fn().mockResolvedValue({ saved: true })
    vi.stubGlobal('hermesDesktop', { saveGatewayFile: save })
    setApiRequestConnection('local')
    detail = {
      ...legacyDetail,
      attachments: [{ id: 7, filename: 'notes.txt', stored_path: '/home/me/.hermes/kanban/notes.txt' }]
    }
    openDrawer()
    const download = await screen.findByRole('button', { name: 'Download notes.txt' })
    setApiRequestConnection('remote-host')
    fireEvent.click(download)
    await waitFor(() =>
      expect(save).toHaveBeenCalledWith({
        connectionId: 'local',
        path: '/home/me/.hermes/kanban/notes.txt',
        suggestedName: 'notes.txt'
      })
    )
  })

  it('disables downloads with no persisted path instead of guessing a workspace path', async () => {
    detail = { ...legacyDetail, attachments: [{ id: 1, filename: 'gone.md' }] }
    openDrawer()
    const button = await screen.findByRole('button', { name: 'Download gone.md' })
    expect((button as HTMLButtonElement).disabled).toBe(true)
  })

  it('reports a failed download and allows retry', async () => {
    const save = vi.fn().mockRejectedValue(new Error('File not found'))
    vi.stubGlobal('hermesDesktop', { saveGatewayFile: save })
    detail = { ...legacyDetail, attachments: [{ id: 1, filename: 'gone.md', stored_path: '/persisted/gone.md' }] }
    openDrawer()
    const button = await screen.findByRole('button', { name: 'Download gone.md' })
    fireEvent.click(button)
    await waitFor(() =>
      expect($notifications.get()[0]).toMatchObject({
        kind: 'error',
        message: 'File not found',
        title: 'Download failed'
      })
    )
    await waitFor(() => expect((button as HTMLButtonElement).disabled).toBe(false))
    save.mockResolvedValueOnce({ saved: true })
    fireEvent.click(button)
    await waitFor(() => expect(save).toHaveBeenCalledTimes(2))
  })

  it('disables a pending download and treats save-dialog cancellation quietly', async () => {
    let finish!: (value: { saved: boolean; canceled: boolean }) => void

    const save = vi.fn(
      () =>
        new Promise(resolve => {
          finish = resolve
        })
    )

    vi.stubGlobal('hermesDesktop', { saveGatewayFile: save })
    detail = { ...legacyDetail, attachments: [{ id: 1, filename: 'report.md', stored_path: '/persisted/report.md' }] }
    openDrawer()
    const button = await screen.findByRole('button', { name: 'Download report.md' })
    fireEvent.click(button)
    await waitFor(() => expect((button as HTMLButtonElement).disabled).toBe(true))
    fireEvent.click(button)
    expect(save).toHaveBeenCalledOnce()
    await act(async () => finish({ saved: false, canceled: true }))
    await waitFor(() => expect((button as HTMLButtonElement).disabled).toBe(false))
    expect($notifications.get()).toEqual([])
  })

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
      await act(() => client.invalidateQueries({ queryKey: taskKey('local', '', legacyDetail.task.id) }))
      expect(await screen.findByRole('button', { name: en.uploadAttachment })).toBeTruthy()
      expect(screen.getByText(en.noAttachments)).toBeTruthy()
    }
  )

  it('keeps upload and attachment rendering working for a supported empty list', async () => {
    detail = { ...legacyDetail, attachments: [] }
    openDrawer()
    const upload = await screen.findByRole('button', { name: en.uploadAttachment })
    expect(screen.getByText(en.noAttachments)).toBeTruthy()

    // The modal portals out of the render container; query inside the dialog.
    const input = screen.getByRole('dialog').querySelector<HTMLInputElement>('input[type="file"]')!
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
    // The task modal portals out of the render container — query baseElement.
    const { baseElement: container } = openDrawer()

    await screen.findByRole('heading', { name: legacyDetail.task.title })
    fireEvent.click(await screen.findByRole('button', { name: /^Activity/ }))

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
    // The task modal portals out of the render container — query baseElement.
    const { baseElement: container } = openDrawer()

    await screen.findByRole('heading', { name: legacyDetail.task.title })
    fireEvent.click(await screen.findByRole('button', { name: /^Activity/ }))
    expect(hiddenEvents(container) + renderedEvents(container)).toBe(events.length)

    unfoldAll(container)

    // Every folded attempt unfolded renders exactly the events it accounted for:
    // none dropped, none rendered twice.
    expect(renderedEvents(container)).toBe(events.length)
    expect(foldRows(container).map(row => row.getAttribute('aria-expanded'))).toEqual(['true', 'true', 'true'])
  })

  it('renders fewer nodes folded than unfolded', async () => {
    detail = { ...legacyDetail, events, runs }
    // The task modal portals out of the render container — query baseElement.
    const { baseElement: container } = openDrawer()

    await screen.findByRole('heading', { name: legacyDetail.task.title })
    fireEvent.click(await screen.findByRole('button', { name: /^Activity/ }))
    const folded = container.querySelectorAll('*').length

    unfoldAll(container)

    expect(folded).toBeLessThan(container.querySelectorAll('*').length)
  })

  it('folds and unfolds from the keyboard on a labelled, focusable row', async () => {
    detail = { ...legacyDetail, events, runs }
    // The task modal portals out of the render container — query baseElement.
    const { baseElement: container } = openDrawer()

    await screen.findByRole('heading', { name: legacyDetail.task.title })
    fireEvent.click(await screen.findByRole('button', { name: /^Activity/ }))

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
    // The task modal portals out of the render container — query baseElement.
    const { baseElement: container } = openDrawer()

    await screen.findByRole('heading', { name: legacyDetail.task.title })
    fireEvent.click(await screen.findByRole('button', { name: /^Activity/ }))

    expect(container.querySelectorAll('[data-attempt]').length).toBe(runs.length)
    expect(container.querySelectorAll('[data-attempt][aria-expanded]').length).toBe(0)
    expect(renderedEvents(container)).toBe(0)
  })
})

describe('task modal dialog', () => {
  it('is a modal dialog named by the task title that Esc dismisses', async () => {
    detail = { ...legacyDetail, attachments: [] }
    const onClose = vi.fn()
    render(
      <QueryClientProvider client={client}>
        <TaskDrawer columns={['todo', 'ready', 'done']} id="t_example" onClose={onClose} onOpen={vi.fn()} />
      </QueryClientProvider>
    )

    const dialog = await screen.findByRole('dialog', { name: legacyDetail.task.title })
    fireEvent.keyDown(dialog, { key: 'Escape' })
    expect(onClose).toHaveBeenCalledOnce()
  })

  it('posts a comment from the named icon action in the field', async () => {
    detail = { ...legacyDetail, attachments: [] }
    openDrawer()

    const send = await screen.findByRole('button', { name: en.comment })
    expect((send as HTMLButtonElement).disabled).toBe(true)

    fireEvent.change(screen.getByPlaceholderText(en.addComment), { target: { value: 'looks good' } })
    fireEvent.click(send)

    await waitFor(() =>
      expect(rest).toHaveBeenCalledWith(
        expect.stringMatching(/^\/tasks\/t_example\/comments/),
        expect.objectContaining({ method: 'POST', body: expect.objectContaining({ body: 'looks good' }) })
      )
    )
  })

  it('shows the workspace path as its own value, not prefixed with the raw kind', async () => {
    const path = '/Users/example/.hermes/kanban/workspaces/a_very_long_directory_name_that_must_wrap'
    detail = {
      ...legacyDetail,
      attachments: [],
      task: { ...legacyDetail.task, workspace_kind: 'dir', workspace_path: path }
    }
    openDrawer()

    expect(await screen.findByText(path)).toBeTruthy()
    expect(screen.queryByText(/dir:/)).toBeNull()
  })

  it('renders description and comments as markdown, not raw source', async () => {
    detail = {
      ...legacyDetail,
      attachments: [],
      task: { ...legacyDetail.task, body: '**Goal:** ship it' },
      comments: [{ id: 1, author: 'test', body: 'run `npm test`', created_at: 0 }]
    }
    openDrawer()

    // Formatted runs become their own nodes; the raw markers are gone.
    expect(await screen.findByText('Goal:')).toBeTruthy()
    expect(screen.getByText('npm test')).toBeTruthy()
    expect(screen.queryByText(/\*\*|`/)).toBeNull()
  })

  it('does not repeat the active feed tab as a heading above the tab strip', async () => {
    detail = {
      ...legacyDetail,
      attachments: [],
      events: [{ id: 1, kind: 'created', payload: null, created_at: 0 }]
    }
    openDrawer()

    const commentsLabel = en.comments(legacyDetail.comments.length)
    expect(await screen.findByRole('button', { name: commentsLabel, pressed: true })).toBeTruthy()
    expect(screen.getAllByText(commentsLabel)).toHaveLength(1)
  })
})

describe('dependency chips resolve titles', () => {
  const linkedDetail = {
    ...legacyDetail,
    attachments: [] as [],
    links: { parents: ['t_parent'], children: ['t_child'] },
    link_tasks: [
      { id: 't_parent', title: 'Parent title', status: 'todo' },
      { id: 't_child', title: 'Child title', status: 'running' }
    ]
  }

  it('renders linked task titles, not raw ids, and opens on click', async () => {
    detail = linkedDetail
    const onOpen = vi.fn()
    render(
      <QueryClientProvider client={client}>
        <TaskDrawer columns={['todo', 'ready', 'done']} id="t_example" onClose={vi.fn()} onOpen={onOpen} />
      </QueryClientProvider>
    )

    expect(await screen.findByRole('heading', { name: legacyDetail.task.title })).toBeTruthy()
    expect(screen.getByText('Parent title')).toBeTruthy()
    expect(screen.getByText('Child title')).toBeTruthy()
    expect(screen.queryByText('parent')).toBeNull()

    fireEvent.click(screen.getByText('Parent title'))
    expect(onOpen).toHaveBeenCalledWith('t_parent')
  })

  it('falls back to short ids when the backend omits link_tasks', async () => {
    const { link_tasks: _omit, ...withoutTitles } = linkedDetail
    detail = withoutTitles
    render(
      <QueryClientProvider client={client}>
        <TaskDrawer columns={['todo', 'ready', 'done']} id="t_example" onClose={vi.fn()} onOpen={vi.fn()} />
      </QueryClientProvider>
    )

    expect(await screen.findByText('parent')).toBeTruthy()
    expect(screen.getByText('child')).toBeTruthy()  })
})
