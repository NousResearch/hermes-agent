import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $previewTarget } from '@/store/preview'
import { $connection } from '@/store/session'

import { PreviewStatusRow } from './preview-row'

describe('PreviewStatusRow', () => {
  beforeEach(() => {
    $connection.set(null)
    $previewTarget.set(null)
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    $previewTarget.set(null)
    vi.restoreAllMocks()
  })

  it('keeps the preview tooltip label inline inside the portaled decoration', async () => {
    const view = render(
      <PreviewStatusRow
        item={{ cwd: 'C:\\repo', id: 'preview.html', label: 'preview.html', target: 'preview.html' }}
        onDismiss={() => undefined}
      />
    )

    fireEvent.pointerMove(screen.getByText('preview.html'), { pointerType: 'mouse' })
    await screen.findByRole('tooltip')

    const content = document.querySelector<HTMLElement>('[data-slot="tooltip-content"]')
    const label = content?.firstElementChild?.firstElementChild

    expect(content).not.toBeNull()
    expect(view.container.contains(content)).toBe(false)
    expect(label?.classList.contains('inline-flex')).toBe(true)
    expect(label?.classList.contains('flex')).toBe(false)
  })

  it('opens remote file artifacts in the in-app preview instead of the local browser bridge', async () => {
    const remotePath = '/home/agent/report.pdf'
    const openPreviewInBrowser = vi.fn(async () => undefined)

    $connection.set({ mode: 'remote' } as never)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        api: vi.fn(async () => ({ binary: true, byteSize: 42, mimeType: 'application/pdf' })),
        normalizePreviewTarget: vi.fn(async () => ({
          kind: 'file',
          label: 'report.pdf',
          path: remotePath,
          previewKind: 'binary',
          source: remotePath,
          url: 'file:///home/agent/report.pdf'
        })),
        openPreviewInBrowser
      }
    })

    render(
      <PreviewStatusRow
        item={{ cwd: '/home/agent', id: remotePath, label: 'report.pdf', target: remotePath }}
        onDismiss={() => undefined}
      />
    )

    fireEvent.click(screen.getByText('report.pdf'))

    await waitFor(() => {
      expect($previewTarget.get()).toMatchObject({ kind: 'file', path: remotePath })
    })
    expect(openPreviewInBrowser).not.toHaveBeenCalled()
  })

  it('keeps local file artifacts on the browser bridge', async () => {
    const localPath = '/Users/alice/report.pdf'
    const openPreviewInBrowser = vi.fn(async () => undefined)

    $connection.set({ mode: 'local' } as never)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        normalizePreviewTarget: vi.fn(async () => ({
          kind: 'file',
          label: 'report.pdf',
          path: localPath,
          previewKind: 'binary',
          source: localPath,
          url: 'file:///Users/alice/report.pdf'
        })),
        openPreviewInBrowser
      }
    })

    render(
      <PreviewStatusRow
        item={{ cwd: '/Users/alice', id: localPath, label: 'report.pdf', target: localPath }}
        onDismiss={() => undefined}
      />
    )

    fireEvent.click(screen.getByText('report.pdf'))

    await waitFor(() => {
      expect(openPreviewInBrowser).toHaveBeenCalledWith('file:///Users/alice/report.pdf')
    })
    expect($previewTarget.get()).toBeNull()
  })
})
