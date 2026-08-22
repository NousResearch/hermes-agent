import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

const { attachmentDataUrl, messages, notify, previewFile } = vi.hoisted(() => ({
  attachmentDataUrl: vi.fn(),
  messages: { en: undefined as Record<string, unknown> | undefined },
  notify: vi.fn(),
  previewFile: vi.fn()
}))

vi.mock('@hermes/plugin-sdk', () => ({
  Codicon: ({ name }: { name: string }) => <span>{name}</span>,
  host: { notify, previewFile },
  // Resolves against the plugin's real en bundle (wired below, after module
  // load) so the strings under test are the shipped ones.
  usePluginI18n:
    () =>
    (key: string, ...args: unknown[]) => {
      const value = key
        .split('.')
        .reduce<unknown>((node, part) => (node as Record<string, unknown>)?.[part], messages.en)

      return typeof value === 'function' ? value(...args) : String(value ?? key)
    }
}))

vi.mock('./api', () => ({ attachmentDataUrl }))

import { AttachmentList } from './attachment-list'
import { KANBAN_LOCALES } from './i18n'

messages.en = KANBAN_LOCALES.en

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('Kanban attachment list', () => {
  it('opens an attachment in the shared preview pane', async () => {
    previewFile.mockResolvedValue(true)

    render(
      <AttachmentList
        attachments={[
          {
            id: 8,
            filename: 'evidence.png',
            stored_path: '/tmp/evidence.png'
          }
        ]}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Preview evidence.png' }))

    expect(previewFile).toHaveBeenCalledWith('/tmp/evidence.png', 'evidence.png', expect.any(Function))

    await waitFor(() => expect(notify).not.toHaveBeenCalled())
    // The local path resolved, so the remote byte loader is never consulted.
    expect(attachmentDataUrl).not.toHaveBeenCalled()
  })

  it('serves attachment bytes from the backend when the path is not local', async () => {
    // The remote-backend case: `stored_path` names a file on the backend host.
    // Local resolution fails, so the host invokes the fallback loader, which
    // fetches the bytes over the plugin's own REST transport.
    attachmentDataUrl.mockResolvedValue({ contentType: 'image/png', dataUrl: 'data:image/png;base64,AAA' })
    previewFile.mockImplementation(async (_path, _label, fetchBytes) => Boolean(await fetchBytes?.()))

    render(<AttachmentList attachments={[{ id: 8, filename: 'evidence.png', stored_path: '/remote/evidence.png' }]} />)

    fireEvent.click(screen.getByRole('button', { name: 'Preview evidence.png' }))

    await waitFor(() => expect(attachmentDataUrl).toHaveBeenCalledWith(8))
    // It opened from bytes — no error toast.
    await waitFor(() => expect(notify).not.toHaveBeenCalled())
  })

  it('surfaces an error when neither the local path nor the backend bytes resolve', async () => {
    // Backend refused the blob too (missing on disk, over the preview cap, or
    // a type the rail cannot render from bytes).
    attachmentDataUrl.mockRejectedValue(new Error('413 attachment too large to preview'))
    previewFile.mockImplementation(async (_path, _label, fetchBytes) => {
      try {
        return Boolean(await fetchBytes?.())
      } catch {
        return false
      }
    })

    render(<AttachmentList attachments={[{ id: 8, filename: 'evidence.png', stored_path: '/remote/evidence.png' }]} />)

    fireEvent.click(screen.getByRole('button', { name: 'Preview evidence.png' }))

    await waitFor(() =>
      expect(notify).toHaveBeenCalledWith({
        kind: 'error',
        message: 'Cannot preview evidence.png — the file is not reachable from this machine.'
      })
    )
  })

  it('surfaces an error when the path cannot be previewed on this machine', async () => {
    previewFile.mockResolvedValue(false)

    render(<AttachmentList attachments={[{ id: 8, filename: 'evidence.png', stored_path: '/remote/evidence.png' }]} />)

    fireEvent.click(screen.getByRole('button', { name: 'Preview evidence.png' }))

    await waitFor(() =>
      expect(notify).toHaveBeenCalledWith({
        kind: 'error',
        message: 'Cannot preview evidence.png — the file is not reachable from this machine.'
      })
    )
  })

  it('surfaces an error when the preview call rejects', async () => {
    previewFile.mockRejectedValue(new Error('ipc failure'))

    render(<AttachmentList attachments={[{ id: 8, filename: 'evidence.png', stored_path: '/tmp/evidence.png' }]} />)

    fireEvent.click(screen.getByRole('button', { name: 'Preview evidence.png' }))

    await waitFor(() => expect(notify).toHaveBeenCalledWith(expect.objectContaining({ kind: 'error' })))
  })

  it('leaves legacy attachments without a stored path as plain text', () => {
    render(<AttachmentList attachments={[{ id: 9, filename: 'legacy.csv' }]} />)

    expect(screen.queryByRole('button')).toBeNull()
    expect(screen.getByText('legacy.csv')).toBeTruthy()
  })
})
