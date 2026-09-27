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
import type { KanbanTaskDetail } from './types'

vi.mock('@/hermes', () => ({ setApiRequestProfile: vi.fn() }))

const META_LINE =
  '> Fleet: revision 3 | point Conductor: tb-cndr | campaign: none | repository: none | canonical status: todo'

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

function openDrawer(fleet = false) {
  return render(
    <QueryClientProvider client={client}>
      <TaskDrawer columns={['todo', 'ready', 'done']} fleet={fleet} id="t_example" onClose={vi.fn()} onOpen={vi.fn()} />
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

describe('task modal dialog', () => {
  it('is a modal dialog named by the task title that Esc dismisses', async () => {
    detail = { ...legacyDetail, attachments: [] }
    const onClose = vi.fn()
    render(
      <QueryClientProvider client={client}>
        <TaskDrawer
          columns={['todo', 'ready', 'done']}
          fleet={false}
          id="t_example"
          onClose={onClose}
          onOpen={vi.fn()}
        />
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
        <TaskDrawer
          columns={['todo', 'ready', 'done']}
          fleet={false}
          id="t_example"
          onClose={vi.fn()}
          onOpen={onOpen}
        />
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
        <TaskDrawer
          columns={['todo', 'ready', 'done']}
          fleet={false}
          id="t_example"
          onClose={vi.fn()}
          onOpen={vi.fn()}
        />
      </QueryClientProvider>
    )

    expect(await screen.findByText('parent')).toBeTruthy()
    expect(screen.getByText('child')).toBeTruthy()
  })
})

describe('fleet detail surface', () => {
  it('reads through the sync decoration and keeps the lifted bookkeeping in its own section', async () => {
    detail = {
      ...legacyDetail,
      attachments: [],
      task: {
        ...legacyDetail.task,
        body: `<!-- fleet-kanban:meta -->\n${META_LINE}\n<!-- /fleet-kanban:meta -->\n\nKeep this description readable.`,
        tenant: 'turnerbook',
        title: '[Sync pending] Example task'
      }
    }
    openDrawer(true)

    expect(await screen.findByRole('heading', { name: 'Example task' })).toBeTruthy()
    expect(screen.getByText('Keep this description readable.')).toBeTruthy()
    expect(screen.getByText(en.sync.pending)).toBeTruthy()
    expect(screen.getByText(en.fleetSync)).toBeTruthy()
    expect(screen.getByText(META_LINE)).toBeTruthy()
    expect(screen.queryByText(/fleet-kanban:meta/)).toBeNull()
  })

  it('shows the same task literally on an ordinary board', async () => {
    detail = {
      ...legacyDetail,
      attachments: [],
      task: {
        ...legacyDetail.task,
        body: `<!-- fleet-kanban:meta -->\n${META_LINE}\n<!-- /fleet-kanban:meta -->\n\nKeep this description readable.`,
        tenant: 'turnerbook',
        title: '[Sync pending] Example task'
      }
    }
    openDrawer(false)

    expect(await screen.findByRole('heading', { name: '[Sync pending] Example task' })).toBeTruthy()
    // Upstream's Markdown renderer hides the leading HTML-comment block, but
    // the ordinary board still keeps its literal title and readable body.
    expect(screen.getByText('Keep this description readable.')).toBeTruthy()
    expect(screen.queryByText(en.sync.pending)).toBeNull()
    expect(screen.queryByText(en.fleetSync)).toBeNull()
  })
})
