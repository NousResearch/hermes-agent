// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const CODE = 'export const answer = 42'

const TOKENS = [
  [
    { content: 'export', htmlStyle: { '--shiki-light': '#6E7781', color: '#6E7781' } },
    { content: ' const answer = 42', htmlStyle: { color: '#ff0000' } }
  ]
]

const { disposeHighlight, startShikiHighlight } = vi.hoisted(() => {
  const disposeHighlight = vi.fn()

  return {
    disposeHighlight,
    startShikiHighlight: vi.fn(() => ({
      dispose: disposeHighlight,
      promise: Promise.resolve(TOKENS)
    }))
  }
})

vi.mock('@/components/chat/shiki-worker-client', () => ({ startShikiHighlight }))

import { SyntaxHighlighter } from '@/components/chat/shiki-highlighter'

let intersectionCallback: IntersectionObserverCallback | null = null

class TestIntersectionObserver {
  constructor(callback: IntersectionObserverCallback) {
    intersectionCallback = callback
  }

  disconnect() {}
  observe() {}
  takeRecords() {
    return []
  }
  unobserve() {}
  root = null
  rootMargin = '300px 0px'
  thresholds = [0]
}

const components = {
  Code: ({ node: _node, ...props }: ComponentProps<'code'> & { node?: unknown }) => <code {...props} />,
  Pre: ({ node: _node, ...props }: ComponentProps<'pre'> & { node?: unknown }) => <pre {...props} />
}

beforeEach(() => {
  disposeHighlight.mockClear()
  startShikiHighlight.mockClear()
  startShikiHighlight.mockImplementation(() => ({
    dispose: disposeHighlight,
    promise: Promise.resolve(TOKENS)
  }))
  intersectionCallback = null
  vi.stubGlobal('IntersectionObserver', TestIntersectionObserver)
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('SyntaxHighlighter viewport and worker lifecycle', () => {
  it('keeps complete off-screen code plain, then highlights it near the viewport', async () => {
    const view = render(<SyntaxHighlighter code={CODE} components={components} language="ts" />)

    expect(view.container.textContent).toContain(CODE)
    expect(startShikiHighlight).not.toHaveBeenCalled()
    expect(intersectionCallback).not.toBeNull()

    act(() => {
      intersectionCallback?.(
        [{ isIntersecting: true } as IntersectionObserverEntry],
        {} as IntersectionObserver
      )
    })

    await waitFor(() => expect(startShikiHighlight).toHaveBeenCalledWith(CODE, 'ts'))
    await waitFor(() => expect(view.container.querySelector('code')?.textContent).toBe(CODE))
    expect(view.container.querySelector('span[style*="#57606a"]')?.textContent).toBe('export')
  })

  it('copies the complete original code after worker highlighting', async () => {
    const writeClipboard = vi.fn(async () => {})
    Object.defineProperty(window, 'hermesDesktop', { configurable: true, value: { writeClipboard } })
    render(<SyntaxHighlighter code={CODE} components={components} language="ts" />)

    act(() => {
      intersectionCallback?.(
        [{ isIntersecting: true } as IntersectionObserverEntry],
        {} as IntersectionObserver
      )
    })
    await waitFor(() => expect(startShikiHighlight).toHaveBeenCalledTimes(1))

    fireEvent.click(screen.getByRole('button', { name: /copy code/i }))

    await waitFor(() => expect(writeClipboard).toHaveBeenCalledWith(CODE))
  })

  it('disposes the worker lease and ignores a late result after unmount', async () => {
    let resolveTokens: (tokens: typeof TOKENS) => void = () => {}
    startShikiHighlight.mockImplementationOnce(() => ({
      dispose: disposeHighlight,
      promise: new Promise(resolve => {
        resolveTokens = resolve
      })
    }))

    const view = render(<SyntaxHighlighter code={CODE} components={components} language="ts" />)
    act(() => {
      intersectionCallback?.(
        [{ isIntersecting: true } as IntersectionObserverEntry],
        {} as IntersectionObserver
      )
    })
    await waitFor(() => expect(startShikiHighlight).toHaveBeenCalledTimes(1))

    view.unmount()
    expect(disposeHighlight).toHaveBeenCalledTimes(1)

    await act(async () => resolveTokens(TOKENS))
  })

  it('keeps plain code usable when Worker support is unavailable', async () => {
    vi.stubGlobal('IntersectionObserver', undefined)
    startShikiHighlight.mockImplementationOnce(() => ({
      dispose: disposeHighlight,
      promise: Promise.reject(new Error('Web Workers are unavailable'))
    }))

    const view = render(<SyntaxHighlighter code={CODE} components={components} language="ts" />)

    await waitFor(() => expect(startShikiHighlight).toHaveBeenCalledTimes(1))
    expect(view.container.querySelector('code')?.textContent).toBe(CODE)
  })
})
