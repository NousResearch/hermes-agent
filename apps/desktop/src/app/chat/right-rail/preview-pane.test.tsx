import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { PreviewPane } from './preview-pane'

const WEB_PREVIEW_TARGET = {
  kind: 'url',
  label: 'Preview',
  source: 'http://localhost:5174',
  url: 'http://localhost:5174'
} as const

function renderWebPreviewWithInsertCss() {
  const insertCSS = vi
    .fn()
    .mockResolvedValueOnce('preview-scrollbar-theme-1')
    .mockResolvedValueOnce('preview-scrollbar-theme-2')

  const removeInsertedCSS = vi.fn().mockResolvedValue(undefined)
  const originalCreateElement = document.createElement.bind(document)

  vi.spyOn(document, 'createElement').mockImplementation(((tagName: string, options?: ElementCreationOptions) => {
    const element = originalCreateElement(tagName, options)

    if (tagName.toLowerCase() === 'webview') {
      Object.assign(element, { insertCSS, removeInsertedCSS })
    }

    return element
  }) as typeof document.createElement)

  const rendered = render(<PreviewPane target={WEB_PREVIEW_TARGET} />)
  const webview = rendered.container.querySelector('webview')

  expect(webview).toBeInstanceOf(HTMLElement)

  return { insertCSS, removeInsertedCSS, webview }
}

async function dispatchWebPreviewLoadStop(webview: Element | null) {
  await act(async () => {
    webview?.dispatchEvent(new Event('did-stop-loading'))
    await Promise.resolve()
  })
}

function dispatchWebPreviewLoadStart(webview: Element | null) {
  act(() => {
    webview?.dispatchEvent(new Event('did-start-loading'))
  })
}

async function flushMutationObserver() {
  await act(async () => {
    await Promise.resolve()
  })
}

async function flushPromises() {
  await act(async () => {
    await Promise.resolve()
    await Promise.resolve()
  })
}

function deferred<T>() {
  let reject!: (error: Error) => void
  let resolve!: (value: T) => void

  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })

  return { promise, reject, resolve }
}

describe('PreviewPane', () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    document.documentElement.style.removeProperty('--theme-card-seed')
    document.documentElement.style.removeProperty('--theme-midground')
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

  it('injects themed scrollbar styles into loaded web previews', async () => {
    document.documentElement.style.setProperty('--theme-midground', '#39ff14')

    const { insertCSS, webview } = renderWebPreviewWithInsertCss()

    await dispatchWebPreviewLoadStop(webview)

    expect(insertCSS).toHaveBeenCalledTimes(1)
    expect(insertCSS.mock.calls[0]?.[0]).toContain('hermes-preview-scrollbar-theme')
    expect(insertCSS.mock.calls[0]?.[0]).toContain('::-webkit-scrollbar-thumb')
    expect(insertCSS.mock.calls[0]?.[0]).toContain('#39ff14')
  })

  it('refreshes web preview scrollbar styles when the desktop theme changes', async () => {
    document.documentElement.style.setProperty('--theme-midground', '#39ff14')

    const { insertCSS, webview } = renderWebPreviewWithInsertCss()

    await dispatchWebPreviewLoadStop(webview)

    document.documentElement.style.setProperty('--theme-midground', '#ff00aa')

    await flushMutationObserver()

    expect(insertCSS).toHaveBeenCalledTimes(2)
    expect(insertCSS.mock.calls[1]?.[0]).toContain('#ff00aa')
  })

  it('removes the previous web preview scrollbar style when the desktop theme changes', async () => {
    document.documentElement.style.setProperty('--theme-midground', '#39ff14')

    const { removeInsertedCSS, webview } = renderWebPreviewWithInsertCss()

    await dispatchWebPreviewLoadStop(webview)

    document.documentElement.style.setProperty('--theme-midground', '#ff00aa')

    await flushMutationObserver()

    expect(removeInsertedCSS).toHaveBeenCalledTimes(1)
    expect(removeInsertedCSS).toHaveBeenCalledWith('preview-scrollbar-theme-1')
  })

  it('does not refresh web preview scrollbar styles for unrelated root style changes', async () => {
    document.documentElement.style.setProperty('--theme-midground', '#39ff14')

    const { insertCSS, webview } = renderWebPreviewWithInsertCss()

    await dispatchWebPreviewLoadStop(webview)

    document.documentElement.style.setProperty('--theme-card-seed', '#111111')

    await flushMutationObserver()

    expect(insertCSS).toHaveBeenCalledTimes(1)
  })

  it('ignores web preview scrollbar injection failures after target switch', async () => {
    let rejectInsertCss: ((error: Error) => void) | undefined

    const insertCSS = vi.fn(
      () =>
        new Promise<string>((_resolve, reject) => {
          rejectInsertCss = reject
        })
    )

    const originalCreateElement = document.createElement.bind(document)

    vi.spyOn(document, 'createElement').mockImplementation(((tagName: string, options?: ElementCreationOptions) => {
      const element = originalCreateElement(tagName, options)

      if (tagName.toLowerCase() === 'webview') {
        Object.assign(element, { insertCSS })
      }

      return element
    }) as typeof document.createElement)
    const setTitlebarToolGroup = vi.fn()

    const rendered = render(<PreviewPane setTitlebarToolGroup={setTitlebarToolGroup} target={WEB_PREVIEW_TARGET} />)
    const webview = rendered.container.querySelector('webview')

    expect(webview).toBeInstanceOf(HTMLElement)

    const consoleTool = setTitlebarToolGroup.mock.calls
      .flatMap(([, tools]) => tools)
      .find(tool => tool.id === 'preview-console')

    expect(consoleTool).toBeDefined()

    act(() => {
      consoleTool?.onSelect()
    })

    await dispatchWebPreviewLoadStop(webview)

    rendered.rerender(
      <PreviewPane
        setTitlebarToolGroup={setTitlebarToolGroup}
        target={{
          kind: 'url',
          label: 'Second preview',
          source: 'http://localhost:5175',
          url: 'http://localhost:5175'
        }}
      />
    )

    rejectInsertCss?.(new Error('late insert failure'))

    await flushPromises()

    expect(rendered.queryByText(/Could not theme preview scrollbar/)).toBeNull()
  })

  it('removes stale web preview scrollbar styles when insertions resolve out of order', async () => {
    const firstInsert = deferred<string>()
    const secondInsert = deferred<string>()
    const insertCSS = vi.fn().mockReturnValueOnce(firstInsert.promise).mockReturnValueOnce(secondInsert.promise)
    const removeInsertedCSS = vi.fn().mockResolvedValue(undefined)
    const originalCreateElement = document.createElement.bind(document)

    vi.spyOn(document, 'createElement').mockImplementation(((tagName: string, options?: ElementCreationOptions) => {
      const element = originalCreateElement(tagName, options)

      if (tagName.toLowerCase() === 'webview') {
        Object.assign(element, { insertCSS, removeInsertedCSS })
      }

      return element
    }) as typeof document.createElement)

    document.documentElement.style.setProperty('--theme-midground', '#39ff14')

    const rendered = render(<PreviewPane target={WEB_PREVIEW_TARGET} />)
    const webview = rendered.container.querySelector('webview')

    await dispatchWebPreviewLoadStop(webview)

    document.documentElement.style.setProperty('--theme-midground', '#ff00aa')

    await flushMutationObserver()

    secondInsert.resolve('preview-scrollbar-theme-2')
    await flushPromises()

    firstInsert.resolve('preview-scrollbar-theme-1')
    await flushPromises()

    expect(removeInsertedCSS).toHaveBeenCalledTimes(1)
    expect(removeInsertedCSS).toHaveBeenCalledWith('preview-scrollbar-theme-1')
  })

  it('removes stale web preview scrollbar styles that resolve after navigation starts', async () => {
    const firstInsert = deferred<string>()
    const insertCSS = vi.fn().mockReturnValueOnce(firstInsert.promise)
    const removeInsertedCSS = vi.fn().mockResolvedValue(undefined)
    const originalCreateElement = document.createElement.bind(document)

    vi.spyOn(document, 'createElement').mockImplementation(((tagName: string, options?: ElementCreationOptions) => {
      const element = originalCreateElement(tagName, options)

      if (tagName.toLowerCase() === 'webview') {
        Object.assign(element, { insertCSS, removeInsertedCSS })
      }

      return element
    }) as typeof document.createElement)

    const rendered = render(<PreviewPane target={WEB_PREVIEW_TARGET} />)
    const webview = rendered.container.querySelector('webview')

    await dispatchWebPreviewLoadStop(webview)

    dispatchWebPreviewLoadStart(webview)

    firstInsert.resolve('preview-scrollbar-theme-1')
    await flushPromises()

    expect(removeInsertedCSS).toHaveBeenCalledTimes(1)
    expect(removeInsertedCSS).toHaveBeenCalledWith('preview-scrollbar-theme-1')
  })
})
