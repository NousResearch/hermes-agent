import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $connection } from '@/store/session'

import { PreviewPane } from './preview-pane'

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

  it('does not rebuild the pane titlebar group for streamed console logs', () => {
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

  it('uses the webview history and reload controls for URL previews', () => {
    const rendered = render(
      <PreviewPane
        target={{
          kind: 'url',
          label: 'Preview',
          source: 'http://localhost:5174',
          url: 'http://localhost:5174'
        }}
      />
    )

    const webview = rendered.container.querySelector('webview') as
      | (HTMLElement & {
          canGoBack?: () => boolean
          canGoForward?: () => boolean
          goBack?: () => void
          goForward?: () => void
          reload?: () => void
        })
      | null

    const canGoBack = vi.fn(() => true)
    const canGoForward = vi.fn(() => true)
    const goBack = vi.fn()
    const goForward = vi.fn()
    const reload = vi.fn()

    Object.assign(webview ?? {}, { canGoBack, canGoForward, goBack, goForward, reload })

    act(() => {
      webview?.dispatchEvent(new Event('did-navigate'))
    })

    fireEvent.click(rendered.getByRole('button', { name: 'Go back' }))
    fireEvent.click(rendered.getByRole('button', { name: 'Go forward' }))
    fireEvent.click(rendered.getByRole('button', { name: 'Reload preview' }))

    expect(goBack).toHaveBeenCalledOnce()
    expect(goForward).toHaveBeenCalledOnce()
    expect(reload).toHaveBeenCalledOnce()
  })

  it('synchronizes the address and history controls after navigation events', () => {
    const rendered = render(
      <PreviewPane
        target={{
          kind: 'url',
          label: 'Preview',
          source: 'http://localhost:5174',
          url: 'http://localhost:5174'
        }}
      />
    )

    const webview = rendered.container.querySelector('webview') as
      | (HTMLElement & {
          canGoBack?: () => boolean
          canGoForward?: () => boolean
        })
      | null

    let canGoBack = true
    let canGoForward = false

    Object.assign(webview ?? {}, {
      canGoBack: () => canGoBack,
      canGoForward: () => canGoForward
    })

    act(() => {
      webview?.dispatchEvent(Object.assign(new Event('did-navigate'), { url: 'http://127.0.0.1:3104/' }))
    })

    expect((rendered.getByRole('textbox', { name: 'Preview URL' }) as HTMLInputElement).value).toBe(
      'http://127.0.0.1:3104/'
    )
    expect((rendered.getByRole('button', { name: 'Go back' }) as HTMLButtonElement).disabled).toBe(false)
    expect((rendered.getByRole('button', { name: 'Go forward' }) as HTMLButtonElement).disabled).toBe(true)

    canGoBack = false
    canGoForward = true

    act(() => {
      webview?.dispatchEvent(
        Object.assign(new Event('did-navigate-in-page'), { url: 'http://127.0.0.1:3104/settings' })
      )
    })

    expect((rendered.getByRole('textbox', { name: 'Preview URL' }) as HTMLInputElement).value).toBe(
      'http://127.0.0.1:3104/settings'
    )
    expect((rendered.getByRole('button', { name: 'Go back' }) as HTMLButtonElement).disabled).toBe(true)
    expect((rendered.getByRole('button', { name: 'Go forward' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('loads an HTTP URL submitted from the preview address bar', () => {
    const rendered = render(
      <PreviewPane
        target={{
          kind: 'url',
          label: 'Preview',
          source: 'http://localhost:5174',
          url: 'http://localhost:5174'
        }}
      />
    )

    const webview = rendered.container.querySelector('webview') as
      | (HTMLElement & { loadURL?: (url: string) => void })
      | null

    const loadURL = vi.fn()

    Object.assign(webview ?? {}, { loadURL })

    fireEvent.change(rendered.getByRole('textbox', { name: 'Preview URL' }), {
      target: { value: 'http://127.0.0.1:3104/' }
    })
    fireEvent.submit(rendered.getByRole('form', { name: 'Navigate preview' }))

    expect(loadURL).toHaveBeenCalledWith('http://127.0.0.1:3104/')
  })

  it('does not load non-HTTP addresses submitted from the preview address bar', () => {
    const rendered = render(
      <PreviewPane
        target={{
          kind: 'url',
          label: 'Preview',
          source: 'http://localhost:5174',
          url: 'http://localhost:5174'
        }}
      />
    )

    const webview = rendered.container.querySelector('webview') as
      | (HTMLElement & { loadURL?: (url: string) => void })
      | null

    const loadURL = vi.fn()

    Object.assign(webview ?? {}, { loadURL })

    fireEvent.change(rendered.getByRole('textbox', { name: 'Preview URL' }), {
      target: { value: 'file:///private/secret.txt' }
    })
    fireEvent.submit(rendered.getByRole('form', { name: 'Navigate preview' }))

    expect(loadURL).not.toHaveBeenCalled()
  })
})
