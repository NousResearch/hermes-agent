import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { isWebPreviewTarget, PreviewPane } from './preview-pane'

describe('PreviewPane console state', () => {
  beforeEach(() => {
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) =>
      window.setTimeout(() => callback(Date.now()), 0)
    )
    vi.stubGlobal('cancelAnimationFrame', (id: number) => window.clearTimeout(id))
  })

  afterEach(() => {
    cleanup()
    $connection.set(null)
    vi.unstubAllGlobals()
  })

  it('does not watch backend-only remote filesystem previews locally', async () => {
    const watchPreviewFile = vi.fn(async () => ({ id: 'watch-1', path: '/remote/file.txt' }))
    const onPreviewFileChanged = vi.fn(() => vi.fn())
    $connection.set({ mode: 'remote' } as never)
    vi.stubGlobal('window', {
      ...window,
      hermesDesktop: {
        onPreviewFileChanged,
        watchPreviewFile
      }
    })

    await act(async () => {
      render(
        <PreviewPane
          setTitlebarToolGroup={vi.fn()}
          target={{
            kind: 'file',
            label: 'file.txt',
            path: '/remote/file.txt',
            previewKind: 'text',
            source: '/remote/file.txt',
            url: 'file:///remote/file.txt'
          }}
        />
      )
    })

    expect(watchPreviewFile).not.toHaveBeenCalled()
    expect(onPreviewFileChanged).not.toHaveBeenCalled()
  })

  it('routes local HTML through the document preview instead of the live webview', () => {
    expect(
      isWebPreviewTarget({
        kind: 'file',
        label: 'report.html',
        path: '/work/report.html',
        previewKind: 'html',
        renderMode: 'preview',
        source: '/work/report.html',
        url: 'file:///work/report.html'
      })
    ).toBe(false)

    expect(
      isWebPreviewTarget({
        kind: 'url',
        label: 'Development server',
        source: 'http://localhost:5174',
        url: 'http://localhost:5174'
      })
    ).toBe(true)
  })

  it('shows preview, source, diff, and zoom for a local HTML artifact', async () => {
    const originalDesktop = window.hermesDesktop
    const unsubscribe = vi.fn()

    Object.defineProperty(window, 'hermesDesktop', {
      configurable: true,
      value: {
        git: {
          fileDiff: vi.fn(async () => '@@ -1 +1 @@\n-<p>Old</p>\n+<p>Current</p>')
        },
        gitRoot: vi.fn(async () => '/work'),
        onPreviewFileChanged: vi.fn(() => unsubscribe),
        readFileText: vi.fn(async () => ({
          binary: false,
          byteSize: 14,
          contentHash: 'html-v1',
          language: 'html',
          path: '/work/report.html',
          text: '<p>Current</p>'
        })),
        watchPreviewFile: vi.fn(async () => ({ id: 'watch-html', path: '/work/report.html' }))
      }
    })

    const rendered = render(
      <PreviewPane
        target={{
          kind: 'file',
          label: 'report.html',
          path: '/work/report.html',
          previewKind: 'html',
          renderMode: 'preview',
          source: '/work/report.html',
          url: 'file:///work/report.html'
        }}
      />
    )

    expect(await rendered.findByRole('button', { name: 'PREVIEW' })).toBeTruthy()
    expect(rendered.getByRole('button', { name: 'SOURCE' })).toBeTruthy()
    expect(await rendered.findByRole('button', { name: 'DIFF' })).toBeTruthy()
    expect(rendered.getByRole('group', { name: 'HTML preview zoom' })).toBeTruthy()
    expect(rendered.container.querySelector('webview')).toBeNull()

    fireEvent.click(rendered.getByRole('button', { name: 'Increase HTML preview size' }))
    const iframe = await rendered.findByTitle('/work/report.html')

    expect(iframe).toBeInstanceOf(HTMLIFrameElement)
    expect(iframe.parentElement?.parentElement?.classList.contains('relative')).toBe(true)
    fireEvent.click(rendered.getByRole('button', { name: 'Drag HTML preview to pan' }))
    expect(rendered.getByRole('button', { name: 'Drag HTML preview to pan' }).getAttribute('aria-pressed')).toBe('true')

    rendered.unmount()
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: originalDesktop })
  })

  it('does not rebuild the pane titlebar group for streamed console logs', async () => {
    const setTitlebarToolGroup = vi.fn()

    let rendered!: ReturnType<typeof render>
    await act(async () => {
      rendered = render(
        <PreviewPane
          setTitlebarToolGroup={setTitlebarToolGroup}
          target={{
            kind: 'url',
            label: 'Preview',
            source: 'http://localhost:5174',
            url: 'http://localhost:5174'
          }}
        />
      )
    })

    const initialCalls = setTitlebarToolGroup.mock.calls.length
    const webview = rendered.container.querySelector('webview')

    expect(webview).toBeInstanceOf(HTMLElement)

    act(() => {
      webview?.dispatchEvent(
        Object.assign(new Event('console-message'), {
          level: 0,
          message: 'streamed log line',
          sourceId: 'http://localhost:5174/src/main.tsx'
        })
      )
    })

    expect(setTitlebarToolGroup).toHaveBeenCalledTimes(initialCalls)
  })
})
