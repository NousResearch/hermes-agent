import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PreviewAttachment } from '@/components/chat/preview-attachment'

// Behavior contract for the delivered-file card (#97812 follow-up):
// MEDIA-delivered files must offer "Open in system app" (OS file association,
// e.g. WPS for .docx) alongside the preview/download actions, and the button
// must hand the normalized absolute path to the OS as a file:// URL.

describe('PreviewAttachment', () => {
  const previousDesktop = window.hermesDesktop

  afterEach(() => {
    window.hermesDesktop = previousDesktop
    vi.restoreAllMocks()
  })

  function mountDesktopStub() {
    const openExternal = vi.fn(async () => undefined)
    const normalizePreviewTarget = vi.fn(async (_target: string) => ({
      binary: false,
      byteSize: 1024,
      kind: 'file',
      label: 'doc.docx',
      language: 'text',
      mimeType: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      path: 'D:\\work\\doc.docx',
      previewKind: 'binary',
      source: 'file:///D:/work/doc.docx',
      url: 'file:///D:/work/doc.docx'
    }))
    window.hermesDesktop = {
      normalizePreviewTarget,
      openExternal
    } as never

    return { normalizePreviewTarget, openExternal }
  }

  it('renders the open-in-system, download and preview actions', async () => {
    mountDesktopStub()

    await act(async () => {
      render(<PreviewAttachment source="tool-result" target="file:///D:/work/doc.docx" />)
    })

    expect(screen.getByRole('button', { name: 'Open in system app' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Download' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open preview' })).toBeTruthy()
  })

  it('opens the file in the OS via openExternal with a file:// URL', async () => {
    const { normalizePreviewTarget, openExternal } = mountDesktopStub()

    await act(async () => {
      render(<PreviewAttachment source="tool-result" target="file:///D:/work/doc.docx" />)
    })

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Open in system app' }))
    })

    expect(normalizePreviewTarget).toHaveBeenCalledTimes(1)
    expect(normalizePreviewTarget.mock.calls[0][0]).toBe('file:///D:/work/doc.docx')
    expect(openExternal).toHaveBeenCalledTimes(1)
    expect(openExternal).toHaveBeenCalledWith('file:///D:/work/doc.docx')
  })

  it('does not call openExternal for a non-file target', async () => {
    const openExternal = vi.fn(async () => undefined)
    window.hermesDesktop = {
      normalizePreviewTarget: vi.fn(async () => null),
      openExternal
    } as never

    await act(async () => {
      render(<PreviewAttachment source="tool-result" target="https://example.com/report.pdf" />)
    })

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Open in system app' }))
    })

    expect(openExternal).not.toHaveBeenCalled()
  })
})
