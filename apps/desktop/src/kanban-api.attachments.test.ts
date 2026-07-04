// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { downloadKanbanAttachment, uploadKanbanAttachment } from '@/kanban-api'

describe('kanban attachment api wrappers', () => {
  let download: ReturnType<typeof vi.fn>
  let upload: ReturnType<typeof vi.fn>

  beforeEach(() => {
    download = vi.fn(async () => ({ path: '/Users/me/Downloads/spec.pdf' }))
    upload = vi.fn(async () => ({ attachment: null }))
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: { kanbanAttachment: { download, upload } }
    })
  })

  afterEach(() => {
    Reflect.deleteProperty(window, 'hermesDesktop')
  })

  it('download targets the attachment blob route with the board pinned', async () => {
    const result = await downloadKanbanAttachment({ id: 7, filename: 'spec.pdf' }, 'ops board')

    expect(download).toHaveBeenCalledWith({
      path: '/api/plugins/kanban/attachments/7?board=ops%20board',
      filename: 'spec.pdf'
    })
    expect(result.path).toBe('/Users/me/Downloads/spec.pdf')
  })

  it('download omits ?board when no board is given', async () => {
    await downloadKanbanAttachment({ id: 3, filename: 'notes.txt' })

    expect(download).toHaveBeenCalledWith({
      path: '/api/plugins/kanban/attachments/3',
      filename: 'notes.txt'
    })
  })

  it('upload multipart-POSTs the task attachments route with the local path', async () => {
    await uploadKanbanAttachment('t_a/b', '/tmp/report.pdf', 'default')

    expect(upload).toHaveBeenCalledWith({
      path: '/api/plugins/kanban/tasks/t_a%2Fb/attachments?board=default',
      filePath: '/tmp/report.pdf',
      uploadedBy: 'desktop'
    })
  })

  it('rejects with a clear error when the preload bridge predates the channel', async () => {
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: {} })

    expect(() => downloadKanbanAttachment({ id: 1, filename: 'x' })).toThrowError(/updated desktop app/)
    expect(() => uploadKanbanAttachment('t_1', '/tmp/x')).toThrowError(/updated desktop app/)
  })
})
