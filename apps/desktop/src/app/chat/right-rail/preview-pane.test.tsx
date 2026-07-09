import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $paneStates } from '@/store/panes'
import { $activeSessionId, $connection, $selectedStoredSessionId } from '@/store/session'

import { PreviewPane } from './preview-pane'

const ORIGINAL_INNER_WIDTH = window.innerWidth

type PreviewTestWebview = HTMLElement & {
  findInPage?: ReturnType<typeof vi.fn>
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
      'preview-devtools'
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
    expect(screen.getByRole('button', { name: 'Reload preview' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Open preview in browser' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Find in preview' })).toBeTruthy()
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

    const frame = rendered.container.querySelector('[data-preview-frame]') as HTMLElement | null

    expect(frame).toBeInstanceOf(HTMLElement)
    expect(screen.getByRole('button', { name: 'Fit preview to pane' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Set preview viewport to Fold 6:5' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Set preview viewport to iPhone 9:16' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Set preview viewport to Desktop 16:9' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Set preview viewport to Ultrawide 21:9' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Set preview viewport to Desktop 16:9' }))

    expect($paneStates.get().preview?.widthOverride).toBeUndefined()
    expect(frame?.style.width).toBe('1280px')

    fireEvent.click(screen.getByRole('button', { name: 'Set preview viewport to iPhone 9:16' }))

    expect($paneStates.get().preview?.widthOverride).toBeUndefined()
    expect(frame?.style.width).toBe('405px')

    fireEvent.click(screen.getByRole('button', { name: 'Fit preview to pane' }))

    expect(frame?.style.width).toBe('100%')
  })

  it('makes annotate and debug preview controls visibly toggle in iframe fallback mode', () => {
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

    fireEvent.click(screen.getByRole('button', { name: 'Show preview debug' }))

    expect(rendered.container.querySelector('[data-preview-debug-overlay]')).toBeTruthy()
    expect(screen.getByLabelText('Preview debug overlay')).toBeTruthy()

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
