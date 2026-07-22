// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { type ComponentProps, useLayoutEffect, useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ShikiWorkerToken } from './shiki-worker'
import type { ShikiHighlightJob } from './shiki-worker-client'

interface ActualShikiWorkerClient {
  getShikiWorkerClientSnapshotForTests: () => {
    activeGeneration: number | null
    activeLeases: number
    pending: number
  }
  startShikiHighlight: (code: string, language: string) => ShikiHighlightJob
}

const CODE = 'export const answer = 42'

const TOKENS: ShikiWorkerToken[][] = [
  [
    { content: 'export', htmlStyle: { '--shiki-light': '#6E7781', color: '#6E7781' } },
    { content: ' const answer = 42', htmlStyle: { color: '#ff0000' } }
  ]
]

const { disposeHighlight, startShikiHighlight } = vi.hoisted(() => {
  const disposeHighlight = vi.fn()

  return {
    disposeHighlight,
    startShikiHighlight: vi.fn<(code: string, language: string) => ShikiHighlightJob>(
      (_code: string, _language: string) => ({
        dispose: disposeHighlight,
        promise: Promise.resolve(TOKENS)
      })
    )
  }
})

vi.mock('@/components/chat/shiki-worker-client', () => ({
  isShikiWorkerRetryableError: (error: unknown) =>
    error instanceof Error && 'retryable' in error && error.retryable === true,
  startShikiHighlight
}))

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

function CommitProbe({ code, onCommit }: { code: string; onCommit: (visibleCode: null | string) => void }) {
  const ref = useRef<HTMLDivElement | null>(null)

  useLayoutEffect(() => onCommit(ref.current?.querySelector('code')?.textContent ?? null), [code, onCommit])

  return (
    <div ref={ref}>
      <SyntaxHighlighter code={code} components={components} language="ts" />
    </div>
  )
}

beforeEach(() => {
  disposeHighlight.mockClear()
  startShikiHighlight.mockClear()
  startShikiHighlight.mockImplementation((_code: string, _language: string) => ({
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
      intersectionCallback?.([{ isIntersecting: true } as IntersectionObserverEntry], {} as IntersectionObserver)
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
      intersectionCallback?.([{ isIntersecting: true } as IntersectionObserverEntry], {} as IntersectionObserver)
    })
    await waitFor(() => expect(startShikiHighlight).toHaveBeenCalledTimes(1))

    fireEvent.click(screen.getByRole('button', { name: /copy code/i }))

    await waitFor(() => expect(writeClipboard).toHaveBeenCalledWith(CODE))
  })

  it('never commits tokens from the previous code after props change', async () => {
    vi.stubGlobal('IntersectionObserver', undefined)

    const staleTokens: ShikiWorkerToken[][] = [
      [{ content: 'stale highlighted code', htmlStyle: { color: '#ff0000' } }]
    ]

    let resolveStaleTokens: (tokens: ShikiWorkerToken[][]) => void = () => {}
    startShikiHighlight.mockImplementationOnce(() => ({
      dispose: disposeHighlight,
      promise: new Promise(resolve => {
        resolveStaleTokens = resolve
      })
    }))
    startShikiHighlight.mockImplementationOnce(() => ({
      dispose: disposeHighlight,
      promise: new Promise(() => {})
    }))
    const onCommit = vi.fn()
    const view = render(<CommitProbe code={CODE} onCommit={onCommit} />)

    await act(async () => resolveStaleTokens(staleTokens))
    await waitFor(() => expect(view.container.querySelector('code')?.textContent).toBe('stale highlighted code'))

    view.rerender(<CommitProbe code="const replacement = true" onCommit={onCommit} />)

    expect(onCommit).toHaveBeenLastCalledWith('const replacement = true')
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
      intersectionCallback?.([{ isIntersecting: true } as IntersectionObserverEntry], {} as IntersectionObserver)
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

  it('retries all mounted blocks once after a shared worker failure', async () => {
    vi.stubGlobal('IntersectionObserver', undefined)
    const retryable = Object.assign(new Error('shared worker crashed'), { retryable: true })
    const initialRejects: Array<(error: Error) => void> = []
    const retryResolves: Array<(tokens: typeof TOKENS) => void> = []

    startShikiHighlight.mockImplementation(() => {
      const call = startShikiHighlight.mock.calls.length

      return {
        dispose: disposeHighlight,
        promise:
          call <= 2
            ? new Promise((_resolve, reject) => initialRejects.push(reject))
            : new Promise(resolve => retryResolves.push(resolve))
      }
    })

    const view = render(
      <>
        <SyntaxHighlighter code="first block" components={components} language="ts" />
        <SyntaxHighlighter code="second block" components={components} language="ts" />
      </>
    )

    await waitFor(() => expect(startShikiHighlight).toHaveBeenCalledTimes(2))
    await act(async () => initialRejects.forEach(reject => reject(retryable)))
    await waitFor(() => expect(startShikiHighlight).toHaveBeenCalledTimes(4))
    await act(async () => retryResolves.forEach(resolve => resolve(TOKENS)))

    await waitFor(() => expect(view.container.querySelectorAll('span[style*="#57606a"]')).toHaveLength(2))
    expect(startShikiHighlight).toHaveBeenCalledTimes(4)
  })

  it('falls back to complete plain code after the bounded retry also fails', async () => {
    vi.stubGlobal('IntersectionObserver', undefined)
    const retryable = Object.assign(new Error('persistent worker failure'), { retryable: true })

    startShikiHighlight.mockImplementation(() => ({
      dispose: disposeHighlight,
      promise: Promise.reject(retryable)
    }))

    const view = render(<SyntaxHighlighter code={CODE} components={components} language="ts" />)

    await waitFor(() => expect(startShikiHighlight).toHaveBeenCalledTimes(2))
    await act(async () => Promise.resolve())

    expect(startShikiHighlight).toHaveBeenCalledTimes(2)
    expect(view.container.querySelector('code')?.textContent).toBe(CODE)
  })

  it('shares one replacement generation after a transient Worker constructor failure', async () => {
    vi.stubGlobal('IntersectionObserver', undefined)
    const constructorCause = new Error('transient constructor detail')

    class ConstructorRetryWorker {
      static constructorCalls = 0
      static instances: ConstructorRetryWorker[] = []
      messages: Array<{ code: string; id: number; language: string }> = []
      onerror: ((event: ErrorEvent) => void) | null = null
      onmessage: ((event: MessageEvent) => void) | null = null
      terminateCalls = 0

      constructor() {
        ConstructorRetryWorker.constructorCalls += 1

        if (ConstructorRetryWorker.constructorCalls === 1) {
          throw constructorCause
        }

        ConstructorRetryWorker.instances.push(this)
      }

      postMessage(message: { code: string; id: number; language: string }) {
        this.messages.push(message)
      }

      terminate() {
        this.terminateCalls += 1
      }
    }

    vi.stubGlobal('Worker', ConstructorRetryWorker)
    const client = await vi.importActual<ActualShikiWorkerClient>('./shiki-worker-client')
    startShikiHighlight.mockImplementation(client.startShikiHighlight)

    const view = render(
      <>
        <SyntaxHighlighter code="first block" components={components} language="ts" />
        <SyntaxHighlighter code="second block" components={components} language="ts" />
      </>
    )

    await waitFor(() => expect(ConstructorRetryWorker.instances).toHaveLength(1))
    const worker = ConstructorRetryWorker.instances[0]
    await waitFor(() => expect(worker.messages).toHaveLength(2))

    for (const message of worker.messages) {
      worker.onmessage?.({ data: { id: message.id, tokens: TOKENS } } as MessageEvent)
    }

    await waitFor(() => expect(view.container.querySelectorAll('span[style*="#57606a"]')).toHaveLength(2))
    expect(ConstructorRetryWorker.constructorCalls).toBe(2)
    expect(client.getShikiWorkerClientSnapshotForTests()).toMatchObject({ activeLeases: 2, pending: 0 })

    view.unmount()

    expect(worker.terminateCalls).toBe(1)
    expect(client.getShikiWorkerClientSnapshotForTests()).toEqual({
      activeGeneration: null,
      activeLeases: 0,
      pending: 0
    })
  })

  it('falls back without looping when both Worker constructor attempts fail', async () => {
    vi.stubGlobal('IntersectionObserver', undefined)
    const constructorCauses = [new Error('first constructor detail'), new Error('second constructor detail')]

    class AlwaysFailingConstructorWorker {
      static constructorCalls = 0

      constructor() {
        const cause = constructorCauses[AlwaysFailingConstructorWorker.constructorCalls]
        AlwaysFailingConstructorWorker.constructorCalls += 1
        throw cause
      }
    }

    vi.stubGlobal('Worker', AlwaysFailingConstructorWorker)
    const client = await vi.importActual<ActualShikiWorkerClient>('./shiki-worker-client')
    startShikiHighlight.mockImplementation(client.startShikiHighlight)

    const view = render(<SyntaxHighlighter code={CODE} components={components} language="ts" />)

    await waitFor(() => expect(AlwaysFailingConstructorWorker.constructorCalls).toBe(2))
    await act(async () => Promise.resolve())

    expect(AlwaysFailingConstructorWorker.constructorCalls).toBe(2)
    expect(view.container.querySelector('code')?.textContent).toBe(CODE)
    expect(client.getShikiWorkerClientSnapshotForTests()).toEqual({
      activeGeneration: null,
      activeLeases: 0,
      pending: 0
    })
  })
})
