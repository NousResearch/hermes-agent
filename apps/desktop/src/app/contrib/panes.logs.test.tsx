import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type * as HermesApi from '@/hermes'

import { LogsPane } from './panes'

// The pane refetches its tail on an interval; the autoscroll contract (#77794)
// is that refetched data scrolls to the bottom ONLY while the user is already
// at the bottom — a deliberate scroll-up must not be yanked back down by the
// next refetch. jsdom has no layout, so the <pre>'s scroll geometry is stubbed
// per test and `scrollTo` is observed directly; rAF is driven synchronously so
// each effect's frame can be awaited deterministically.
let lines: string[] = []

vi.mock('@/hermes', async importOriginal => ({
  ...(await importOriginal<typeof HermesApi>()),
  getLogs: vi.fn(async () => ({ lines: [...lines] }))
}))

beforeEach(() => {
  lines = ['first line']
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

// jsdom's window has no frame scheduling by default; drive rAF from a
// controllable queue so a test can fire exactly the frames it awaits.
const rafQueue: FrameRequestCallback[] = []

function flushFrames() {
  const queued = rafQueue.splice(0)

  for (const cb of queued) {cb(0)}
}

beforeEach(() => {
  vi.stubGlobal('requestAnimationFrame', (cb: FrameRequestCallback) => {
    rafQueue.push(cb)

    return rafQueue.length
  })
  vi.stubGlobal('cancelAnimationFrame', () => undefined)
})

afterEach(() => {
  vi.unstubAllGlobals()
  rafQueue.length = 0
})

function renderLogsPane() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, refetchInterval: false } }
  })

  render(
    <QueryClientProvider client={queryClient}>
      <LogsPane />
    </QueryClientProvider>
  )

  return queryClient
}

async function refetchedTail(queryClient: QueryClient, pre: HTMLPreElement) {
  lines = ['first line', 'second line']

  await act(async () => {
    await queryClient.invalidateQueries()
  })
  // The pane renders the tail as ONE joined text node inside the <pre>, so
  // assert on its content rather than a per-line element query.
  await waitFor(() => expect(pre.textContent).toContain('second line'))
}

describe('LogsPane', () => {
  it('opts the log tail into text selection', async () => {
    renderLogsPane()

    const pre = (await screen.findByText('first line')).closest('pre')
    expect(pre?.getAttribute('data-selectable-text')).toBe('true')
  })

  it('sticks to the bottom across refetches while the user is at the bottom', async () => {
    const queryClient = renderLogsPane()

    const pre = (await screen.findByText('first line')).closest('pre') as HTMLPreElement
    Object.defineProperty(pre, 'scrollHeight', { configurable: true, value: 1000 })
    Object.defineProperty(pre, 'clientHeight', { configurable: true, value: 800 })
    // Exactly at the bottom: distance from the bottom is 0.
    Object.defineProperty(pre, 'scrollTop', { configurable: true, value: 200 })
    const scrollTo = vi.spyOn(pre, 'scrollTo')

    // Settle the mount frame, then deliver a refetched tail.
    await act(() => void flushFrames())
    scrollTo.mockClear()

    await refetchedTail(queryClient, pre)
    await act(() => void flushFrames())

    // The user was at the bottom, so the fresh tail scrolls into view.
    expect(scrollTo).toHaveBeenCalledWith({ top: 1000 })
  })

  it('does not yank the pane back down when the user has scrolled up', async () => {
    const queryClient = renderLogsPane()

    const pre = (await screen.findByText('first line')).closest('pre') as HTMLPreElement
    Object.defineProperty(pre, 'scrollHeight', { configurable: true, value: 1000 })
    Object.defineProperty(pre, 'clientHeight', { configurable: true, value: 800 })
    const scrollTo = vi.spyOn(pre, 'scrollTo')

    // Settle the mount frame first.
    await act(() => void flushFrames())
    scrollTo.mockClear()

    // The user scrolls up to read an earlier line: distance from the bottom
    // (1000 - 100 - 800 = 100px) is past the 48px stickiness threshold.
    Object.defineProperty(pre, 'scrollTop', { configurable: true, value: 100 })
    fireEvent.scroll(pre)

    await refetchedTail(queryClient, pre)
    await act(() => void flushFrames())

    // The refetched tail rendered, but the pane must not scroll: the user is
    // reading above the bottom.
    expect(scrollTo).not.toHaveBeenCalled()
    expect(pre.textContent).toContain('second line')
  })
})
