import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage } from '@/lib/chat-messages'

const view = { $messages: atom<ChatMessage[]>([]) }
let visible = true

vi.mock('@/app/chat/session-view', () => ({ useSessionView: () => view }))
vi.mock('@/components/pane-shell/pane-visibility', () => ({ usePaneVisible: () => visible }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic: () => {} }))
vi.mock('./use-timeline-history', () => ({
  useTimelineHistory: () => ({ entries: undefined, complete: true, failed: false, loadMore: async () => {} })
}))
vi.mock('@assistant-ui/react', () => ({
  useAui: () => ({ thread: () => ({ getState: () => ({ messages: [] }) }) }),
  useAuiState: (selector: (state: { thread: { messages: ChatMessage[] } }) => unknown) =>
    selector({ thread: { messages: view.$messages.get() } })
}))
// Render the rail's selected identity without involving its separate tick virtualizer.
vi.mock('./timeline-rail', () => ({
  TimelineRail: ({ activeIndex, entries }: { activeIndex: number; entries: { id: string }[] }) => (
    <output data-testid="active">{entries[activeIndex]?.id}</output>
  )
}))

const { ThreadTimeline } = await import('./timeline')

let serial = 0
const frames = new Map<number, FrameRequestCallback>()
const resizes = new Set<{ callback: ResizeObserverCallback; targets: Set<Element> }>()

beforeEach(() => {
  visible = true
  frames.clear()
  resizes.clear()
  view.$messages.set([])
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    frames.set(++serial, callback)

    return serial
  })
  vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id))
  vi.stubGlobal(
    'ResizeObserver',
    class {
      targets = new Set<Element>()
      constructor(public callback: ResizeObserverCallback) {
        resizes.add(this)
      }
      observe(target: Element) {
        this.targets.add(target)
      }
      disconnect() {
        this.targets.clear()
      }
    }
  )
})

afterEach(() => {
  cleanup()
  document.body.innerHTML = ''
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

const rect = (top: number, height = 100): DOMRect =>
  ({ top, bottom: top + height, left: 0, right: 800, width: 800, height, x: 0, y: top, toJSON: () => ({}) }) as DOMRect

async function flushFrame() {
  await act(async () => {
    // Deliver real MutationObserver records before the coalesced animation frame.
    await Promise.resolve()
    const callbacks = [...frames.values()]
    frames.clear()
    callbacks.forEach(callback => callback(performance.now()))
  })
}

function mount(count: number) {
  view.$messages.set(
    Array.from({ length: count }, (_, index) => ({
      id: `u${index}`,
      role: 'user',
      parts: [{ type: 'text', text: `prompt ${index}` }]
    }))
  )
  const root = document.createElement('div')
  root.dataset.sessionAnchor = 'owned'
  const viewport = document.createElement('div')
  viewport.dataset.slot = 'aui_thread-viewport'
  viewport.dataset.following = 'false'
  const content = document.createElement('div')
  content.dataset.slot = 'aui_thread-content'
  viewport.append(content)
  const host = document.createElement('div')
  root.append(viewport, host)
  document.body.append(root)
  const reads = { outer: 0, inner: 0 }
  const tops = Array.from({ length: count }, (_, index) => 100 + index * 123)
  vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue(rect(50, 600))

  const groups = tops.map((_, index) => {
    const group = document.createElement('div')
    group.dataset.slot = 'aui_message-group'
    group.style.contentVisibility = 'auto'
    group.innerHTML = `<div data-slot="aui_turn-pair"><div data-message-id="u${index}"></div></div>`
    content.append(group)
    vi.spyOn(group, 'getBoundingClientRect').mockImplementation(() => {
      reads.outer++

      return rect(50 + tops[index] - viewport.scrollTop)
    })
    vi.spyOn(group.firstElementChild!, 'getBoundingClientRect').mockImplementation(() => {
      reads.inner++

      return rect(50 + tops[index] - viewport.scrollTop)
    })

    return group
  })

  const queries = vi.spyOn(viewport, 'querySelectorAll')
  const ui = render(<ThreadTimeline />, { container: host })

  return { ...ui, viewport, content, groups, reads, tops, queries }
}

describe('timeline position in the real rail component', () => {
  it('keeps exact bidirectional positions with logarithmic outer-box reads and no per-frame DOM sweep', async () => {
    const count = 512
    const ui = mount(count)
    await flushFrame()

    for (const index of [count - 1, 32, 301, 0, 255]) {
      ui.reads.outer = ui.reads.inner = 0
      ui.viewport.scrollTop = ui.tops[index] + 20
      // A burst is one measurement, not one sweep per browser event.
      for (let event = 0; event < 10; event++) {
        fireEvent.scroll(ui.viewport)
      }
      await flushFrame()
      expect(ui.getByTestId('active').textContent).toBe(`u${index}`)
      expect(ui.reads.inner).toBe(0)
      expect(ui.reads.outer).toBeLessThanOrEqual(Math.ceil(Math.log2(count)) + 1)
    }

    const sweeps = () => ui.queries.mock.calls.filter(([selector]) => selector === '[data-message-id]').length
    expect(sweeps()).toBe(1)
    // Markdown growth invalidates geometry, not membership. No stale cached
    // offsets and no full message query after a text-only subtree commit.
    ui.groups[2].append(document.createElement('p'))
    for (let index = 3; index < count; index++) {
      ui.tops[index] += 123
    }
    await flushFrame()
    expect(ui.getByTestId('active').textContent).toBe('u254')
    expect(sweeps()).toBe(1)
  })

  it('refreshes identities after paging and late mounts, measures resize, and releases hidden panes', async () => {
    const ui = mount(16)
    await flushFrame()
    ui.viewport.scrollTop = ui.tops[10] + 20
    fireEvent.scroll(ui.viewport)
    await flushFrame()
    expect(ui.getByTestId('active').textContent).toBe('u10')

    // The outer group survives while the runtime replaces its message identity.
    const message = ui.groups[10].querySelector('[data-message-id]')!
    message.removeAttribute('data-message-id')
    await flushFrame()
    expect(ui.getByTestId('active').textContent).toBe('u9')
    message.setAttribute('data-message-id', 'u10')
    await flushFrame()
    expect(ui.getByTestId('active').textContent).toBe('u10')
    ui.groups[10].remove()
    await flushFrame()
    expect(ui.getByTestId('active').textContent).toBe('u9')
    ui.content.insertBefore(ui.groups[10], ui.groups[11])
    await flushFrame()
    expect(ui.getByTestId('active').textContent).toBe('u10')

    // Late image/tool layout can move the boundary without a scroll event.
    ui.tops[10] += 60
    for (const observer of resizes) {
      if (observer.targets.has(ui.content)) {
        observer.callback([], observer as unknown as ResizeObserver)
      }
    }
    await flushFrame()
    expect(ui.getByTestId('active').textContent).toBe('u9')

    ui.reads.outer = ui.reads.inner = 0
    ui.viewport.dataset.following = 'true'
    fireEvent.scroll(ui.viewport)
    await flushFrame()
    expect(ui.getByTestId('active').textContent).toBe('u15')
    expect(ui.reads.outer + ui.reads.inner).toBe(0)
    visible = false
    ui.rerender(<ThreadTimeline />)
    fireEvent.scroll(ui.viewport)
    ui.groups[0].remove()
    await flushFrame()
    expect(ui.queryByTestId('active')).toBeNull()
    expect(frames.size).toBe(0)
    expect([...resizes].every(observer => observer.targets.size === 0)).toBe(true)
  })
})
