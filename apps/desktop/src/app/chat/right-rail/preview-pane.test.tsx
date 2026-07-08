import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $paneStates } from '@/store/panes'
import { $activeSessionId, $connection, $selectedStoredSessionId } from '@/store/session'

import { PreviewPane } from './preview-pane'

const ORIGINAL_INNER_WIDTH = window.innerWidth

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
    expect(latestTools.map((tool: { id: string }) => tool.id)).not.toContain('preview-devtools')

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

    const input = screen.getByLabelText('Preview address') as HTMLInputElement
    const iframe = rendered.container.querySelector('iframe')

    fireEvent.change(input, { target: { value: 'example.org/docs' } })
    fireEvent.submit(input.closest('form')!)

    expect(iframe?.getAttribute('src')).toBe('https://example.org/docs')
    expect(input.value).toBe('https://example.org/docs')
  })

  it('offers responsive preview width presets without replacing manual pane resizing', () => {
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

    expect(screen.getByRole('button', { name: 'Set preview to Fold 6:5' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Set preview to iPhone 9:16' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Set preview to Desktop 16:9' })).toBeTruthy()
    expect(screen.getByRole('button', { name: 'Set preview to Ultrawide 21:9' })).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Set preview to Desktop 16:9' }))

    expect($paneStates.get().preview?.widthOverride).toBe(1280)

    fireEvent.click(screen.getByRole('button', { name: 'Set preview to iPhone 9:16' }))

    expect($paneStates.get().preview?.widthOverride).toBe(405)
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
