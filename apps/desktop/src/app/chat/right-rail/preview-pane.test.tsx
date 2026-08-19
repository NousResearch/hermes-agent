import { act, cleanup, fireEvent, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { forgetPreviewConsole, previewConsoleState } from './preview-console-store'
import { PreviewPane } from './preview-pane'

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

    forgetPreviewConsole(tabId)

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

    forgetPreviewConsole(tabId)
  })

  // The bar is chrome for a LIVE page. A file peek, an artifact, and remote
  // HTML in a sandboxed iframe have no webview to navigate.
  it('shows the browser bar only for a live webview preview', async () => {
    let rendered!: ReturnType<typeof render>
    await act(async () => {
      rendered = render(
        <PreviewPane
          target={{ kind: 'url', label: 'Preview', source: 'http://localhost:5174', url: 'http://localhost:5174' }}
        />
      )
    })

    expect(rendered.queryByRole('textbox', { name: 'Address' })).not.toBeNull()

    await act(async () => {
      rendered.rerender(
        <PreviewPane
          target={{
            kind: 'file',
            label: 'notes.txt',
            path: '/tmp/notes.txt',
            previewKind: 'text',
            source: '/tmp/notes.txt',
            url: 'file:///tmp/notes.txt'
          }}
        />
      )
    })

    expect(rendered.queryByRole('textbox', { name: 'Address' })).toBeNull()
  })

  it('drives the webview from the bar and tracks its history', async () => {
    let rendered!: ReturnType<typeof render>
    await act(async () => {
      rendered = render(
        <PreviewPane
          target={{ kind: 'url', label: 'Preview', source: 'http://localhost:5174', url: 'http://localhost:5174' }}
        />
      )
    })

    const webview = rendered.container.querySelector('webview') as HTMLElement & Record<string, unknown>
    const loadURL = vi.fn(async () => undefined)

    Object.assign(webview, {
      canGoBack: () => true,
      canGoForward: () => false,
      goBack: vi.fn(),
      loadURL
    })

    // Back is disabled until the webview reports history, and a navigation is
    // what makes it ask.
    expect((rendered.getByRole('button', { name: 'Back' }) as HTMLButtonElement).disabled).toBe(true)

    act(() => {
      webview.dispatchEvent(Object.assign(new Event('did-navigate'), { url: 'http://localhost:5174/two' }))
    })

    const back = rendered.getByRole('button', { name: 'Back' }) as HTMLButtonElement

    expect(back.disabled).toBe(false)
    fireEvent.click(back)
    expect(webview.goBack).toHaveBeenCalledOnce()

    const address = rendered.getByRole('textbox', { name: 'Address' }) as HTMLInputElement

    expect(address.value).toBe('http://localhost:5174/two')

    fireEvent.focus(address)
    fireEvent.change(address, { target: { value: 'localhost:4000/app' } })
    fireEvent.keyDown(address, { key: 'Enter' })

    // loadURL, not a `src` swap — re-entering the current address must reload.
    // Awaited: navigation first asks main whether the address needs a loopback
    // forward, so the load lands a microtask later.
    await waitFor(() => expect(loadURL).toHaveBeenCalledWith('http://localhost:4000/app'))
    expect(webview.getAttribute('src')).toBe('http://localhost:5174')
  })

  // The webview always runs on THIS machine, so a remote agent's localhost is
  // a different computer's localhost. The failure is honest but baffling
  // without saying so.
  it('explains a failed loopback URL when the agent is on a remote gateway', async () => {
    $connection.set({ mode: 'remote' } as never)

    let rendered!: ReturnType<typeof render>
    await act(async () => {
      rendered = render(
        <PreviewPane
          target={{ kind: 'url', label: 'Preview', source: 'http://localhost:5173', url: 'http://localhost:5173' }}
        />
      )
    })

    const webview = rendered.container.querySelector('webview') as HTMLElement

    await act(async () => {
      webview.dispatchEvent(
        Object.assign(new Event('did-fail-load'), {
          errorCode: -102,
          errorDescription: 'ERR_CONNECTION_REFUSED',
          isMainFrame: true,
          validatedURL: 'http://localhost:5173'
        })
      )
    })

    await waitFor(() => expect(rendered.container.textContent).toContain('machine running your agent'))
  })

  it('stays quiet about loopback when the gateway is local', async () => {
    $connection.set({ mode: 'local' } as never)

    let rendered!: ReturnType<typeof render>
    await act(async () => {
      rendered = render(
        <PreviewPane
          target={{ kind: 'url', label: 'Preview', source: 'http://localhost:5173', url: 'http://localhost:5173' }}
        />
      )
    })

    const webview = rendered.container.querySelector('webview') as HTMLElement

    await act(async () => {
      webview.dispatchEvent(
        Object.assign(new Event('did-fail-load'), {
          errorCode: -102,
          errorDescription: 'ERR_CONNECTION_REFUSED',
          isMainFrame: true,
          validatedURL: 'http://localhost:5173'
        })
      )
    })

    await waitFor(() => expect(rendered.container.textContent).toContain('ERR_CONNECTION_REFUSED'))
    expect(rendered.container.textContent).not.toContain('machine running your agent')
  })

  // A public host fails for ordinary reasons; the remote-vs-local distinction
  // has nothing to do with it.
  it('stays quiet for a non-loopback host on a remote gateway', async () => {
    $connection.set({ mode: 'remote' } as never)

    let rendered!: ReturnType<typeof render>
    await act(async () => {
      rendered = render(
        <PreviewPane
          target={{ kind: 'url', label: 'Preview', source: 'https://example.com', url: 'https://example.com' }}
        />
      )
    })

    const webview = rendered.container.querySelector('webview') as HTMLElement

    await act(async () => {
      webview.dispatchEvent(
        Object.assign(new Event('did-fail-load'), {
          errorCode: -105,
          errorDescription: 'ERR_NAME_NOT_RESOLVED',
          isMainFrame: true,
          validatedURL: 'https://example.com'
        })
      )
    })

    await waitFor(() => expect(rendered.container.textContent).toContain('ERR_NAME_NOT_RESOLVED'))
    expect(rendered.container.textContent).not.toContain('machine running your agent')
  })

  it('surfaces a rejected navigation as a load error', async () => {
    let rendered!: ReturnType<typeof render>
    await act(async () => {
      rendered = render(
        <PreviewPane
          target={{ kind: 'url', label: 'Preview', source: 'http://localhost:5174', url: 'http://localhost:5174' }}
        />
      )
    })

    const webview = rendered.container.querySelector('webview') as HTMLElement & Record<string, unknown>

    Object.assign(webview, { loadURL: vi.fn(async () => Promise.reject(new Error('ERR_CONNECTION_REFUSED'))) })

    const address = rendered.getByRole('textbox', { name: 'Address' }) as HTMLInputElement

    await act(async () => {
      fireEvent.focus(address)
      fireEvent.change(address, { target: { value: 'http://localhost:4000' } })
      fireEvent.keyDown(address, { key: 'Enter' })
    })

    await waitFor(() => expect(rendered.container.textContent).toContain('ERR_CONNECTION_REFUSED'), {
      container: rendered.container
    })
  })

  // `about:blank` in a webview is a white void that reads as broken against
  // the app's chrome — the pane should say it's empty on purpose.
  it('shows the blank-page empty state instead of a white void', async () => {
    let rendered!: ReturnType<typeof render>
    await act(async () => {
      rendered = render(
        <PreviewPane target={{ kind: 'url', label: 'Browser', source: 'about:blank', url: 'about:blank' }} />
      )
    })

    expect(rendered.container.textContent).toContain('Type an address above')

    const webview = rendered.container.querySelector('webview') as HTMLElement

    // Navigating away dismisses it; the bar and the webview stay put.
    act(() => {
      webview.dispatchEvent(Object.assign(new Event('did-navigate'), { url: 'https://example.com' }))
    })

    expect(rendered.container.textContent).not.toContain('Type an address above')
    expect(rendered.queryByRole('textbox', { name: 'Address' })).not.toBeNull()
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

    await waitFor(() => expect(rendered.container.querySelector('iframe')).not.toBeNull(), {
      container: rendered.container
    })
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

    await waitFor(() => expect(rendered.container.querySelector('iframe')).not.toBeNull(), {
      container: rendered.container
    })
    expect(api).toHaveBeenCalledWith({
      path: `/api/fs/read-data-url?path=${encodeURIComponent(filePath)}`,
      profile: 'macmini'
    })
  })
})
