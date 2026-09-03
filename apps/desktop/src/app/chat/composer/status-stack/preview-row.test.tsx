import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $previewTabs, closeRightRail } from '@/store/preview'
import { $connection } from '@/store/session'

import { PreviewStatusRow } from './preview-row'

describe('PreviewStatusRow', () => {
  beforeEach(() => {
    $connection.set(null)
    closeRightRail()
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    closeRightRail()
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

  it('opens remote non-HTML file artifacts in the in-app preview instead of the local browser bridge', async () => {
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
      expect($previewTabs.get()).toEqual([
        expect.objectContaining({ target: expect.objectContaining({ kind: 'file', path: remotePath }) })
      ])
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
    expect($previewTabs.get()).toEqual([])
  })

  it('keeps remote HTML on the staged browser-open path, not the in-app pane', async () => {
    const remotePath = '/home/agent/index.html'
    const html = '<!doctype html><html><body>hi</body></html>'
    const dataUrl = `data:text/html;base64,${btoa(html)}`
    const openPreviewInBrowser = vi.fn(async () => undefined)
    const saveImageBuffer = vi.fn(async () => '/tmp/staged.html')

    $connection.set({ mode: 'remote' } as never)
    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        api: vi.fn(async () => dataUrl),
        normalizePreviewTarget: vi.fn(async () => ({
          kind: 'file',
          label: 'index.html',
          path: remotePath,
          previewKind: 'html',
          source: remotePath,
          url: 'file:///home/agent/index.html'
        })),
        openPreviewInBrowser,
        saveImageBuffer
      }
    })

    render(
      <PreviewStatusRow
        item={{ cwd: '/home/agent', id: remotePath, label: 'index.html', target: remotePath }}
        onDismiss={() => undefined}
      />
    )

    fireEvent.click(screen.getByText('index.html'))

    await waitFor(() => {
      expect(saveImageBuffer).toHaveBeenCalled()
      expect(openPreviewInBrowser).toHaveBeenCalledWith('file:///tmp/staged.html')
    })
    expect($previewTabs.get()).toEqual([])
  })
})
