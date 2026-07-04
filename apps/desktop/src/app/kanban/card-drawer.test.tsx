// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $kanbanActiveBoard, $kanbanBoard } from '@/store/kanban'
import { $notifications, clearNotifications } from '@/store/notifications'
import type { KanbanAttachment, KanbanTaskDetail } from '@/types/kanban'

import { KanbanCardDrawer } from './card-drawer'

function attachment(id: number, filename: string, extra: Partial<KanbanAttachment> = {}): KanbanAttachment {
  return {
    id,
    task_id: 't_1',
    filename,
    content_type: 'application/pdf',
    size: 2048,
    uploaded_by: 'desktop',
    stored_path: `/data/attachments/t_1/${filename}`,
    created_at: 100,
    ...extra
  }
}

function detailPayload(attachments: KanbanAttachment[]): KanbanTaskDetail {
  return {
    task: {
      id: 't_1',
      title: 'Ship it',
      body: null,
      assignee: 'coder',
      status: 'todo',
      priority: 0,
      created_by: 'test',
      created_at: 1,
      started_at: null,
      completed_at: null,
      workspace_kind: 'scratch',
      workspace_path: null,
      claim_lock: null,
      claim_expires: null,
      tenant: null
    },
    comments: [],
    events: [],
    attachments,
    links: { parents: [], children: [] },
    runs: []
  }
}

describe('KanbanCardDrawer attachments', () => {
  let api: ReturnType<typeof vi.fn>
  let download: ReturnType<typeof vi.fn>
  let upload: ReturnType<typeof vi.fn>
  let selectPaths: ReturnType<typeof vi.fn>
  let revealPath: ReturnType<typeof vi.fn>
  let detail: KanbanTaskDetail

  beforeEach(() => {
    detail = detailPayload([attachment(7, 'spec.pdf'), attachment(8, 'notes.txt', { size: 10, content_type: null })])
    api = vi.fn(async (request: { path: string }) => {
      if (request.path.startsWith('/api/plugins/kanban/tasks/t_1')) {
        return detail
      }

      if (request.path.startsWith('/api/plugins/kanban/assignees')) {
        return { assignees: [] }
      }

      if (request.path.startsWith('/api/plugins/kanban/board')) {
        return { columns: [], tenants: [], assignees: [], latest_event_id: 1, now: 1 }
      }

      throw new Error(`unexpected path ${request.path}`)
    })
    download = vi.fn(async () => ({ path: '/Users/me/Downloads/spec.pdf' }))
    upload = vi.fn(async () => ({ attachment: attachment(9, 'added.png') }))
    selectPaths = vi.fn(async () => ['/tmp/added.png'])
    revealPath = vi.fn(async () => true)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { api, kanbanAttachment: { download, upload }, revealPath, selectPaths }
    })
    $kanbanActiveBoard.set('default')
    $kanbanBoard.set(null)
    clearNotifications()
  })

  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    Reflect.deleteProperty(window, 'hermesDesktop')
    clearNotifications()
  })

  it('renders attachment names and sizes', async () => {
    render(<KanbanCardDrawer onClose={() => undefined} taskId="t_1" />)

    expect(await screen.findByText('spec.pdf')).toBeTruthy()
    expect(screen.getByText('notes.txt')).toBeTruthy()
    expect(screen.getByText('2.0 KB')).toBeTruthy()
    expect(screen.getByText('10 B')).toBeTruthy()
  })

  it('downloads on click and toasts with a reveal action', async () => {
    render(<KanbanCardDrawer onClose={() => undefined} taskId="t_1" />)

    fireEvent.click(await screen.findByRole('button', { name: 'Download spec.pdf' }))

    await waitFor(() => expect(download).toHaveBeenCalledTimes(1))
    expect(download).toHaveBeenCalledWith({
      path: '/api/plugins/kanban/attachments/7?board=default',
      filename: 'spec.pdf'
    })

    await waitFor(() => {
      const toast = $notifications.get()[0]

      expect(toast?.kind).toBe('success')
      expect(toast?.message).toBe('Saved spec.pdf to Downloads')
      expect(toast?.action?.label).toBe('Reveal')
    })

    $notifications.get()[0]?.action?.onClick()
    expect(revealPath).toHaveBeenCalledWith('/Users/me/Downloads/spec.pdf')
  })

  it('surfaces the backend detail when a download fails', async () => {
    download.mockRejectedValueOnce(
      new Error(
        "Error invoking remote method 'hermes:kanban:attachment:download': Error: 404: " +
          '{"detail":"attachment file missing on disk"}'
      )
    )
    render(<KanbanCardDrawer onClose={() => undefined} taskId="t_1" />)

    fireEvent.click(await screen.findByRole('button', { name: 'Download spec.pdf' }))

    await waitFor(() => {
      const toast = $notifications.get()[0]

      expect(toast?.kind).toBe('error')
      expect(toast?.message).toBe('attachment file missing on disk')
    })
  })

  it('uploads picked files and refetches the detail', async () => {
    render(<KanbanCardDrawer onClose={() => undefined} taskId="t_1" />)
    await screen.findByText('spec.pdf')

    const detailFetches = api.mock.calls.filter(([request]) =>
      (request as { path: string }).path.startsWith('/api/plugins/kanban/tasks/t_1')
    ).length

    detail = detailPayload([attachment(7, 'spec.pdf'), attachment(8, 'notes.txt'), attachment(9, 'added.png')])
    fireEvent.click(screen.getByRole('button', { name: /Add attachment/ }))

    await waitFor(() => expect(upload).toHaveBeenCalledTimes(1))
    expect(selectPaths).toHaveBeenCalledWith({ multiple: true, title: 'Add attachment' })
    expect(upload).toHaveBeenCalledWith({
      path: '/api/plugins/kanban/tasks/t_1/attachments?board=default',
      filePath: '/tmp/added.png',
      uploadedBy: 'desktop'
    })

    expect(await screen.findByText('added.png')).toBeTruthy()
    expect(
      api.mock.calls.filter(([request]) =>
        (request as { path: string }).path.startsWith('/api/plugins/kanban/tasks/t_1')
      ).length
    ).toBeGreaterThan(detailFetches)
  })

  it('surfaces the backend detail when an upload is rejected', async () => {
    upload.mockRejectedValueOnce(
      new Error(
        "Error invoking remote method 'hermes:kanban:attachment:upload': Error: 413: " +
          '{"detail":"attachment exceeds 25 MB limit"}'
      )
    )
    render(<KanbanCardDrawer onClose={() => undefined} taskId="t_1" />)
    await screen.findByText('spec.pdf')

    fireEvent.click(screen.getByRole('button', { name: /Add attachment/ }))

    await waitFor(() => {
      const toast = $notifications.get()[0]

      expect(toast?.kind).toBe('error')
      expect(toast?.title).toBe('Attachment upload failed')
      expect(toast?.message).toBe('attachment exceeds 25 MB limit')
    })
  })
})
