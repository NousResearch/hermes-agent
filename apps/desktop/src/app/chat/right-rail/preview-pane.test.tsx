import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { PreviewPane } from './preview-pane'
import { forgetPreviewStripTools, previewConsoleState } from './preview-strip-tools'

function stubPdfObjectUrls() {
  const NativeUrl = URL
  let objectUrlIndex = 0
  const createObjectURL = vi.fn((_blob: Blob) => `blob:pdf-preview-${(objectUrlIndex += 1)}`)
  const revokeObjectURL = vi.fn()

  class TestUrl extends NativeUrl {}

  Object.defineProperties(TestUrl, {
    createObjectURL: { configurable: true, value: createObjectURL },
    revokeObjectURL: { configurable: true, value: revokeObjectURL }
  })
  vi.stubGlobal('URL', TestUrl)

  return { createObjectURL, revokeObjectURL }
}

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

  // The console lives in the TAB's store (the toggles sit on the tab, not in the
  // titlebar), so a streamed log has to land in the store keyed by tabId — that
  // is what both the panel in the pane and the button on the tab read.
  it('streams console logs into the tab-keyed console store', async () => {
    const tabId = 'url:http://localhost:5174'

    forgetPreviewStripTools(tabId)

    let rendered!: ReturnType<typeof render>
    await act(async () => {
      rendered = render(
        <PreviewPane
          tabId={tabId}
          target={{
            kind: 'url',
            label: 'Preview',
            source: 'http://localhost:5174',
            url: 'http://localhost:5174'
          }}
        />
      )
    })

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

    expect(previewConsoleState(tabId).$logs.get().at(-1)?.message).toBe('streamed log line')

    forgetPreviewStripTools(tabId)
  })

  it('renders authenticated remote HTML safely and honors source mode', async () => {
    const dataUrl = `data:text/html;base64,${btoa('<h1>remote</h1>')}`

    const target = {
      dataUrl,
      kind: 'file' as const,
      label: 'report.html',
      path: '/srv/report.html',
      previewKind: 'html' as const,
      source: '/srv/report.html',
      url: 'file:///srv/report.html'
    }

    let rendered!: ReturnType<typeof render>
    await act(async () => {
      rendered = render(<PreviewPane target={target} />)
    })

    const iframe = rendered.container.querySelector('iframe')

    expect(rendered.container.querySelector('webview')).toBeNull()
    expect(iframe?.getAttribute('sandbox')).toBe('')
    expect(iframe?.getAttribute('referrerpolicy')).toBe('no-referrer')
    expect(iframe?.getAttribute('srcdoc')).toContain(`default-src 'none'`)
    expect(iframe?.getAttribute('srcdoc')).toContain('<h1>remote</h1>')
    expect(rendered.container.textContent).not.toContain(dataUrl)

    await act(async () => {
      rendered.rerender(
        <PreviewPane target={{ ...target, dataUrl: undefined, renderMode: 'source', transient: true }} />
      )
    })

    expect(rendered.container.querySelector('iframe')).toBeNull()
    const sourceLink = rendered.container.querySelector('a')

    expect(sourceLink?.getAttribute('href')).toBeNull()
    expect(sourceLink?.getAttribute('target')).toBeNull()
    expect(fireEvent.click(sourceLink!)).toBe(false)
  })

  it('renders PDF targets in an embedded viewer', async () => {
    const dataUrl = 'data:application/pdf;base64,JVBERi0xLjQ='
    const readFileDataUrl = vi.fn(async () => dataUrl)
    const { createObjectURL, revokeObjectURL } = stubPdfObjectUrls()
    $connection.set({ mode: 'local' } as never)
    vi.stubGlobal('window', {
      ...window,
      hermesDesktop: {
        readFileDataUrl
      }
    })

    let rendered!: ReturnType<typeof render>
    await act(async () => {
      rendered = render(
        <PreviewPane
          target={{
            kind: 'file',
            label: 'spec.pdf',
            path: '/tmp/spec.pdf',
            previewKind: 'pdf',
            source: '/tmp/spec.pdf',
            url: 'file:///tmp/spec.pdf'
          }}
        />
      )
    })

    await waitFor(
      () => expect(rendered.container.querySelector('iframe')).not.toBeNull(),
      { container: rendered.container }
    )
    expect(rendered.container.querySelector('iframe')?.getAttribute('src')).toBe('blob:pdf-preview-1')
    expect(readFileDataUrl).toHaveBeenCalledWith('/tmp/spec.pdf')
    const blob = createObjectURL.mock.calls[0]?.[0]

    expect(blob).toBeInstanceOf(Blob)
    expect(blob?.type).toBe('application/pdf')
    expect(await blob?.text()).toBe('%PDF-1.4')

    await act(async () => {
      rendered.rerender(
        <PreviewPane
          target={{
            kind: 'file',
            label: 'other.pdf',
            path: '/tmp/other.pdf',
            previewKind: 'pdf',
            source: '/tmp/other.pdf',
            url: 'file:///tmp/other.pdf'
          }}
        />
      )
    })

    await waitFor(() => expect(createObjectURL).toHaveBeenCalledTimes(2), {
      container: rendered.container
    })
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:pdf-preview-1')
    expect(rendered.container.querySelector('iframe')?.getAttribute('src')).toBe('blob:pdf-preview-2')

    rendered.unmount()
    expect(revokeObjectURL).toHaveBeenCalledWith('blob:pdf-preview-2')
  })

  it('accepts case-insensitive metadata and percent-escaped base64', async () => {
    const readFileDataUrl = vi.fn(async () => 'data:APPLICATION/PDF;BASE64,%4AVBERi0xLjQ=')
    const { createObjectURL } = stubPdfObjectUrls()
    $connection.set({ mode: 'local' } as never)
    vi.stubGlobal('window', {
      ...window,
      hermesDesktop: {
        readFileDataUrl
      }
    })

    let rendered!: ReturnType<typeof render>
    await act(async () => {
      rendered = render(
        <PreviewPane
          target={{
            kind: 'file',
            label: 'spec.pdf',
            path: '/tmp/spec.pdf',
            previewKind: 'pdf',
            source: '/tmp/spec.pdf',
            url: 'file:///tmp/spec.pdf'
          }}
        />
      )
    })

    await waitFor(() => expect(createObjectURL).toHaveBeenCalledTimes(1), {
      container: rendered.container
    })
    expect(rendered.container.querySelector('iframe')?.getAttribute('src')).toBe('blob:pdf-preview-1')
  })

  it.each([
    ['a non-PDF MIME type', 'data:text/html;base64,JVBERi0xLjQ=', 'Invalid PDF data URL type'],
    ['bytes without a PDF header', 'data:application/pdf;base64,PGh0bWw+', 'Invalid PDF file header'],
    ['a malformed payload', 'data:application/pdf;base64,%', 'Invalid PDF data URL payload']
  ])('rejects %s before creating an object URL', async (_case, dataUrl, expectedError) => {
    const readFileDataUrl = vi.fn(async () => dataUrl)
    const { createObjectURL } = stubPdfObjectUrls()
    $connection.set({ mode: 'local' } as never)
    vi.stubGlobal('window', {
      ...window,
      hermesDesktop: {
        readFileDataUrl
      }
    })

    let rendered!: ReturnType<typeof render>
    await act(async () => {
      rendered = render(
        <PreviewPane
          target={{
            kind: 'file',
            label: 'spec.pdf',
            path: '/tmp/spec.pdf',
            previewKind: 'pdf',
            source: '/tmp/spec.pdf',
            url: 'file:///tmp/spec.pdf'
          }}
        />
      )
    })

    await waitFor(() => expect(rendered.container.textContent).toContain(expectedError), {
      container: rendered.container
    })
    expect(rendered.container.querySelector('iframe')).toBeNull()
    expect(createObjectURL).not.toHaveBeenCalled()
  })

  it('retries a restored PDF when the filesystem connection becomes remote', async () => {
    const filePath = '/remote/spec.pdf'
    const dataUrl = 'data:application/pdf;base64,JVBERi0xLjQ='
    stubPdfObjectUrls()

    const readFileDataUrl = vi.fn(async () => {
      throw new Error('File preview failed: file does not exist')
    })

    const api = vi.fn(async () => dataUrl)
    $connection.set({ mode: 'local' } as never)
    vi.stubGlobal('window', {
      ...window,
      hermesDesktop: {
        api,
        readFileDataUrl
      }
    })

    let rendered!: ReturnType<typeof render>
    await act(async () => {
      rendered = render(
        <PreviewPane
          target={{
            kind: 'file',
            label: 'spec.pdf',
            path: filePath,
            previewKind: 'pdf',
            source: filePath,
            url: `file://${filePath}`
          }}
        />
      )
    })

    await waitFor(() => expect(readFileDataUrl).toHaveBeenCalledTimes(1), { container: rendered.container })

    await act(async () => {
      $connection.set({ baseUrl: 'http://macmini', mode: 'remote', profile: 'macmini' } as never)
    })

    await waitFor(
      () => expect(rendered.container.querySelector('iframe')).not.toBeNull(),
      { container: rendered.container }
    )
    expect(api).toHaveBeenCalledWith({
      path: `/api/fs/read-data-url?path=${encodeURIComponent(filePath)}`,
      profile: 'macmini'
    })
  })
})
