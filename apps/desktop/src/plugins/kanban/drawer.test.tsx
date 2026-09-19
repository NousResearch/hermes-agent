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
import type { KanbanTaskDetail } from './types'

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

afterEach(async () => {
  const { setApiRequestConnection, setApiRequestProfile } = await import('@/api/client')
  setApiRequestConnection(null)
  setApiRequestProfile(null)
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
  cleanup()
  client.clear()
  disposeApi()
  disposeLocales()
  vi.clearAllMocks()
})

function openDrawer() {
  return render(
    <QueryClientProvider client={client}>
      <TaskDrawer columns={['todo', 'ready', 'done']} id="t_example" onClose={vi.fn()} onOpen={vi.fn()} />
    </QueryClientProvider>
  )
}

describe('task attachment compatibility', () => {
  it('downloads the persisted attachment through its original remote owner', async () => {
    const { setApiRequestConnection, setApiRequestProfile } = await import('@/api/client')
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
    setApiRequestConnection(null)
    setApiRequestProfile(null)
  })

  it('disables downloads with no persisted path instead of guessing a workspace path', async () => {
    detail = { ...legacyDetail, attachments: [{ id: 1, filename: 'gone.md' }] }
    openDrawer()
    const button = await screen.findByRole('button', { name: 'Download gone.md' })
    expect((button as HTMLButtonElement).disabled).toBe(true)
  })

  it('reports a failed download and allows retry', async () => {
    const { host } = await import('@hermes/plugin-sdk')
    const notify = vi.spyOn(host, 'notify')
    const save = vi.fn().mockRejectedValue(new Error('File not found'))
    vi.stubGlobal('hermesDesktop', { saveGatewayFile: save })
    detail = { ...legacyDetail, attachments: [{ id: 1, filename: 'gone.md', stored_path: '/persisted/gone.md' }] }
    openDrawer()
    const button = await screen.findByRole('button', { name: 'Download gone.md' })
    fireEvent.click(button)
    await waitFor(() => expect(notify).toHaveBeenCalledWith({ kind: 'error', message: 'File not found' }))
    await waitFor(() => expect((button as HTMLButtonElement).disabled).toBe(false))
    save.mockResolvedValueOnce({ saved: true })
    fireEvent.click(button)
    await waitFor(() => expect(save).toHaveBeenCalledTimes(2))
    notify.mockRestore()
  })

  it('disables a pending download and treats save-dialog cancellation quietly', async () => {
    const { host } = await import('@hermes/plugin-sdk')
    const notify = vi.spyOn(host, 'notify')
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
    expect(notify).not.toHaveBeenCalled()
    notify.mockRestore()
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
