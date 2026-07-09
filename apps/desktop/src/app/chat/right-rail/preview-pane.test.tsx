import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $paneStates } from '@/store/panes'
import { $activeSessionId, $connection, $selectedStoredSessionId } from '@/store/session'

import { PreviewPane } from './preview-pane'

const ORIGINAL_INNER_WIDTH = window.innerWidth

type PreviewTestWebview = HTMLElement & {
  canGoBack?: ReturnType<typeof vi.fn>
  canGoForward?: ReturnType<typeof vi.fn>
  findInPage?: ReturnType<typeof vi.fn>
  goBack?: ReturnType<typeof vi.fn>
  goForward?: ReturnType<typeof vi.fn>
  reload?: ReturnType<typeof vi.fn>
  stopFindInPage?: ReturnType<typeof vi.fn>
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
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: ORIGINAL_INNER_WIDTH })
    $paneStates.set({})
    $activeSessionId.set(null)
    $connection.set(null)
    $selectedStoredSessionId.set(null)
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
  })

  it('does not watch backend-only remote filesystem previews locally', () => {
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

    expect(watchPreviewFile).not.toHaveBeenCalled()
    expect(onPreviewFileChanged).not.toHaveBeenCalled()
  })

  it('falls back to an iframe when Electron webview APIs are unavailable', () => {
    const setTitlebarToolGroup = vi.fn()

    const rendered = render(
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

    const initialCalls = setTitlebarToolGroup.mock.calls.length
    const iframe = rendered.container.querySelector('iframe')
    const latestTools = setTitlebarToolGroup.mock.calls.at(-1)?.[1] ?? []

    expect(iframe).toBeInstanceOf(HTMLIFrameElement)
    expect(iframe?.getAttribute('src')).toBe('http://localhost:5174')
    expect(latestTools.map((tool: { id: string }) => tool.id)).toEqual([
      'preview-annotate',
      'preview-console',
      'preview-devtools',
      'preview-fullscreen'
    ])

    act(() => {
      iframe?.dispatchEvent(new Event('load'))
    })

    expect(setTitlebarToolGroup).toHaveBeenCalledTimes(initialCalls)
  })


  it('navigates the docked preview from the editable address bar', () => {
    const rendered = render(
      <PreviewPane
        setTitlebarToolGroup={vi.fn()}
        target={{
          kind: 'url',
          label: 'Preview',
          source: 'http://localhost:5174',
          url: 'http://localhost:5174'
        }}
      />
    )

    const input = screen.getByLabelText('Preview URL') as HTMLInputElement
    const iframe = rendered.container.querySelector('iframe')

    fireEvent.change(input, { target: { value: 'example.org/docs' } })
    fireEvent.submit(input.closest('form')!)

    expect(iframe?.getAttribute('src')).toBe('https://example.org/docs')
    expect(input.value).toBe('https://example.org/docs')
  })

  it('keeps normal browser chrome in the embedded preview pane', () => {
    render(
      <PreviewPane
        embedded
        setTitlebarToolGroup={vi.fn()}
        target={{
          kind: 'url',
          label: 'Preview',
          source: 'http://localhost:5174',
          url: 'http://localhost:5174'
        }}
      />
    )

    expect(screen.getByLabelText('Preview URL')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Go back in preview' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Go forward in preview' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Reload preview' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open preview in browser' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Find in preview' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Enter preview fullscreen' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Show preview annotations' })).toBeTruthy()
  })

  it('opens preview find with Ctrl+F and reports iframe fallback limitations visibly', () => {
    render(
      <PreviewPane
        embedded
        setTitlebarToolGroup={vi.fn()}
        target={{
          kind: 'url',
          label: 'Preview',
          source: 'http://localhost:5174',
          url: 'http://localhost:5174'
        }}
      />
    )

    fireEvent.keyDown(window, { ctrlKey: true, key: 'f' })

    const findInput = screen.getByLabelText('Find in preview text') as HTMLInputElement

    expect(findInput).toBeTruthy()

    fireEvent.change(findInput, { target: { value: 'DROPShock' } })

    expect(screen.getByText('Find is limited in browser iframe fallback. Open externally or use the page’s own search.')).toBeTruthy()

    fireEvent.keyDown(findInput, { key: 'Escape' })

    expect(screen.queryByLabelText('Find in preview text')).toBeNull()
  })

  it('wires browser find controls to Electron webview find APIs', () => {
    const createdWebviews: PreviewTestWebview[] = []
    const originalCreateElement = document.createElement.bind(document)

    vi.spyOn(document, 'createElement').mockImplementation(((tagName: string, options?: ElementCreationOptions) => {
      if (tagName.toLowerCase() !== 'webview') {
        return originalCreateElement(tagName, options)
      }

      const webview = originalCreateElement('div') as PreviewTestWebview
      webview.findInPage = vi.fn()
      webview.stopFindInPage = vi.fn()
      webview.reload = vi.fn()
      createdWebviews.push(webview)

      return webview
    }) as typeof document.createElement)

    render(
      <PreviewPane
        embedded
        setTitlebarToolGroup={vi.fn()}
        target={{
          kind: 'url',
          label: 'Preview',
          source: 'https://example.com',
          url: 'https://example.com'
        }}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Find in preview' }))

    const findInput = screen.getByLabelText('Find in preview text') as HTMLInputElement

    fireEvent.change(findInput, { target: { value: 'browseros' } })
    expect(createdWebviews[0]?.findInPage).toHaveBeenLastCalledWith('browseros', { forward: true, findNext: false })

    fireEvent.keyDown(findInput, { key: 'Enter' })
    expect(createdWebviews[0]?.findInPage).toHaveBeenLastCalledWith('browseros', { forward: true, findNext: true })

    fireEvent.keyDown(findInput, { key: 'Enter', shiftKey: true })
    expect(createdWebviews[0]?.findInPage).toHaveBeenLastCalledWith('browseros', { forward: false, findNext: true })

    fireEvent.keyDown(findInput, { key: 'Escape' })
    expect(createdWebviews[0]?.stopFindInPage).toHaveBeenCalledWith('clearSelection')
    expect(screen.queryByLabelText('Find in preview text')).toBeNull()
  })

  it('renders website previews directly in the browser tab content by default', () => {
    const rendered = render(
      <PreviewPane
        embedded
        setTitlebarToolGroup={vi.fn()}
        target={{
          kind: 'url',
          label: 'Preview',
          source: 'http://localhost:5174',
          url: 'http://localhost:5174'
        }}
      />
    )

    const viewport = rendered.container.querySelector('[data-preview-viewport]') as HTMLElement | null
    const browserContent = rendered.container.querySelector('[data-preview-browser-content]') as HTMLElement | null
    const iframe = rendered.container.querySelector('iframe')

    expect(viewport).toBeInstanceOf(HTMLElement)
    expect(browserContent).toBeInstanceOf(HTMLElement)
    expect(rendered.container.querySelector('[data-preview-frame]')).toBeNull()
    expect(viewport?.className).toContain('overflow-hidden')
    expect(browserContent?.style.width).toBe('')
    expect(iframe?.parentElement).toBe(browserContent?.querySelector('.absolute.inset-0.flex.bg-transparent'))
  })

  it('offers responsive preview viewport presets without resizing the outer pane', () => {
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1600 })

    const rendered = render(
      <PreviewPane
        embedded
        setTitlebarToolGroup={vi.fn()}
        target={{
          kind: 'url',
          label: 'Preview',
          source: 'http://localhost:5174',
          url: 'http://localhost:5174'
        }}
      />
    )

    const viewport = rendered.container.querySelector('[data-preview-viewport]')

    if (!(viewport instanceof HTMLElement)) {
      throw new Error('missing preview viewport')
    }

    Object.defineProperty(viewport, 'getBoundingClientRect', {
      configurable: true,
      value: () => ({
        bottom: 720,
        height: 720,
        left: 0,
        right: 400,
        top: 0,
        width: 400,
        x: 0,
        y: 0,
        toJSON: () => ({})
      })
    })

    expect(rendered.container.querySelector('[data-preview-frame]')).toBeNull()
    expect(screen.getByRole('button', { name: 'Fit preview to pane' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Set preview viewport to Fold 6:5' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Set preview viewport to iPhone 9:16' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Set preview viewport to Desktop 16:9' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Set preview viewport to Ultrawide 21:9' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Set preview viewport to Desktop 16:9' }))

    let frame = rendered.container.querySelector('[data-preview-frame]') as HTMLElement | null

    expect(frame).toBeInstanceOf(HTMLElement)
    expect($paneStates.get().preview?.widthOverride).toBeUndefined()
    expect(frame?.style.width).toBe('1280px')

    fireEvent.click(screen.getByRole('button', { name: 'Set preview viewport to iPhone 9:16' }))

    frame = rendered.container.querySelector('[data-preview-frame]') as HTMLElement | null

    expect($paneStates.get().preview?.widthOverride).toBeUndefined()
    expect(frame?.style.width).toBe('405px')

    fireEvent.click(screen.getByRole('button', { name: 'Fit preview to pane' }))

    expect(rendered.container.querySelector('[data-preview-frame]')).toBeNull()
  })

  it('clears stale responsive sizing when switching to a local file preview', () => {
    Object.defineProperty(window, 'innerWidth', { configurable: true, value: 1600 })

    const setTitlebarToolGroup = vi.fn()

    const rendered = render(
      <PreviewPane
        embedded
        setTitlebarToolGroup={setTitlebarToolGroup}
        target={{
          kind: 'url',
          label: 'Preview A',
          source: 'http://localhost:5174/a',
          url: 'http://localhost:5174/a'
        }}
      />
    )

    const viewport = rendered.container.querySelector('[data-preview-viewport]')

    if (!(viewport instanceof HTMLElement)) {
      throw new Error('missing preview viewport')
    }

    Object.defineProperty(viewport, 'getBoundingClientRect', {
      configurable: true,
      value: () => ({
        bottom: 720,
        height: 720,
        left: 0,
        right: 400,
        top: 0,
        width: 400,
        x: 0,
        y: 0,
        toJSON: () => ({})
      })
    })

    fireEvent.click(screen.getByRole('button', { name: 'Set preview viewport to Desktop 16:9' }))
    expect((rendered.container.querySelector('[data-preview-frame]') as HTMLElement | null)?.style.width).toBe('1280px')

    rendered.rerender(
      <PreviewPane
        embedded
        setTitlebarToolGroup={setTitlebarToolGroup}
        target={{
          kind: 'file',
          label: 'screenshot.png',
          path: '/tmp/hermes/screenshot.png',
          previewKind: 'image',
          source: '/tmp/hermes/screenshot.png',
          url: 'file:///tmp/hermes/screenshot.png'
        }}
      />
    )

    expect(screen.queryByRole('button', { name: 'Fit preview to pane' })).toBeNull()
    expect(rendered.container.querySelector('[data-preview-frame]')).toBeNull()
    expect((rendered.container.querySelector('[data-preview-browser-content]') as HTMLElement | null)?.style.width).toBe('')
  })

  it('wires preview browser back and forward controls to Electron webview history APIs', () => {
    const createdWebviews: PreviewTestWebview[] = []
    const originalCreateElement = document.createElement.bind(document)

    vi.spyOn(document, 'createElement').mockImplementation(((tagName: string, options?: ElementCreationOptions) => {
      if (tagName.toLowerCase() !== 'webview') {
        return originalCreateElement(tagName, options)
      }

      const webview = originalCreateElement('div') as PreviewTestWebview
      webview.canGoBack = vi.fn(() => true)
      webview.canGoForward = vi.fn(() => true)
      webview.goBack = vi.fn()
      webview.goForward = vi.fn()
      webview.reload = vi.fn()
      createdWebviews.push(webview)

      return webview
    }) as typeof document.createElement)

    render(
      <PreviewPane
        embedded
        setTitlebarToolGroup={vi.fn()}
        target={{
          kind: 'url',
          label: 'Preview',
          source: 'https://example.com/start',
          url: 'https://example.com/start'
        }}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Go back in preview' }))
    fireEvent.click(screen.getByRole('button', { name: 'Go forward in preview' }))

    expect(createdWebviews[0]?.goBack).toHaveBeenCalledTimes(1)
    expect(createdWebviews[0]?.goForward).toHaveBeenCalledTimes(1)
  })

  it('has an explicit fullscreen control that can be exited from the preview flow', async () => {
    let fullscreenElement: Element | null = null

    const requestFullscreen = vi.fn(function request(this: Element) {
      fullscreenElement = this
      document.dispatchEvent(new Event('fullscreenchange'))

      return Promise.resolve()
    })

    const exitFullscreen = vi.fn(() => {
      fullscreenElement = null
      document.dispatchEvent(new Event('fullscreenchange'))

      return Promise.resolve()
    })

    Object.defineProperty(HTMLElement.prototype, 'requestFullscreen', {
      configurable: true,
      value: requestFullscreen
    })
    Object.defineProperty(document, 'exitFullscreen', {
      configurable: true,
      value: exitFullscreen
    })
    Object.defineProperty(document, 'fullscreenElement', {
      configurable: true,
      get: () => fullscreenElement
    })

    render(
      <PreviewPane
        embedded
        setTitlebarToolGroup={vi.fn()}
        target={{
          kind: 'url',
          label: 'Preview',
          source: 'http://localhost:5174',
          url: 'http://localhost:5174'
        }}
      />
    )

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: 'Enter preview fullscreen' }))
    })

    expect(requestFullscreen).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('button', { name: 'Exit preview fullscreen' })).toBeTruthy()

    await act(async () => {
      fireEvent.keyDown(window, { key: 'Escape' })
    })

    expect(exitFullscreen).toHaveBeenCalledTimes(1)
    expect(screen.getByRole('button', { name: 'Enter preview fullscreen' })).toBeTruthy()
  })

  it('provides visible back-out controls for preview overlays and menus', () => {
    const setTitlebarToolGroup = vi.fn()

    const rendered = render(
      <PreviewPane
        embedded
        setTitlebarToolGroup={setTitlebarToolGroup}
        target={{
          kind: 'url',
          label: 'Preview',
          source: 'http://localhost:5174',
          url: 'http://localhost:5174'
        }}
      />
    )

    fireEvent.click(screen.getByRole('button', { name: 'Show preview annotations' }))

    expect(rendered.container.querySelector('[data-preview-annotations]')).toBeTruthy()
    expect(screen.getByLabelText('Preview annotation overlay active')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Back to preview' }))

    expect(rendered.container.querySelector('[data-preview-annotations]')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Show preview debug' }))

    expect(rendered.container.querySelector('[data-preview-debug-overlay]')).toBeTruthy()
    expect(screen.getByLabelText('Preview debug overlay')).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Back to preview' }))

    expect(rendered.container.querySelector('[data-preview-debug-overlay]')).toBeNull()
    expect(screen.queryByText('Preview Console')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Find in preview' }))
    expect(screen.getByLabelText('Find in preview text')).toBeTruthy()

    fireEvent.keyDown(window, { key: 'Escape' })

    expect(screen.queryByLabelText('Find in preview text')).toBeNull()

    const tools = setTitlebarToolGroup.mock.calls.at(-1)?.[1] ?? []
    expect(tools.map((tool: { id: string }) => tool.id)).toContain('preview-annotate')
    expect(tools.map((tool: { id: string }) => tool.id)).toContain('preview-devtools')
  })

  it('scopes Electron preview webview storage to the active chat session', () => {
    const createdWebviews: HTMLElement[] = []
    const originalCreateElement = document.createElement.bind(document)

    vi.spyOn(document, 'createElement').mockImplementation(((tagName: string, options?: ElementCreationOptions) => {
      if (tagName.toLowerCase() !== 'webview') {
        return originalCreateElement(tagName, options)
      }

      const webview = originalCreateElement('div') as unknown as HTMLElement & { reload: () => void }
      webview.reload = vi.fn()
      createdWebviews.push(webview)

      return webview
    }) as typeof document.createElement)

    $activeSessionId.set('chat/with unsafe spaces')

    render(
      <PreviewPane
        setTitlebarToolGroup={vi.fn()}
        target={{
          kind: 'url',
          label: 'Preview',
          source: 'https://example.com',
          url: 'https://example.com'
        }}
      />
    )

    expect(createdWebviews[0]?.getAttribute('partition')).toBe('persist:hermes-preview-chat-with-unsafe-spaces')
  })
})
