import { cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { setTurnAnchor } from '@/store/thread-scroll'

import { newestPromptTop, useTurnEndAnchor } from './use-turn-end-anchor'

const rect = (top: number, height: number) => ({ bottom: top + height, height, top }) as DOMRect

/**
 * Viewport 600 tall over a 5000px transcript, whose newest prompt sits 1200px
 * above the visible rows — the long turn of #108941. The prompt bubble reports
 * its PINNED (sticky) position while its group reports the layout one.
 */
function transcript() {
  const viewport = window.document.createElement('div')
  const content = window.document.createElement('div')
  viewport.append(content)

  Object.defineProperty(viewport, 'clientHeight', { configurable: true, get: () => 600 })
  Object.defineProperty(viewport, 'scrollHeight', { configurable: true, get: () => 5000 })
  vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue(rect(0, 600))

  const older = window.document.createElement('div')
  older.dataset.slot = 'aui_message-group'
  content.append(older)
  vi.spyOn(older, 'getBoundingClientRect').mockReturnValue(rect(-3000, 2000))

  const turn = window.document.createElement('div')
  turn.dataset.slot = 'aui_message-group'
  const prompt = window.document.createElement('div')
  prompt.dataset.slot = 'aui_user-message-root'
  turn.append(prompt)
  content.append(turn)
  vi.spyOn(turn, 'getBoundingClientRect').mockReturnValue(rect(-1200, 400))
  vi.spyOn(prompt, 'getBoundingClientRect').mockReturnValue(rect(5, 80))

  return { content, stopScroll: vi.fn(), viewport }
}

type Harness = ReturnType<typeof transcript>
type HarnessProps = Parameters<typeof useTurnEndAnchor>[0]

function props(harness: Harness, overrides: Partial<HarnessProps> = {}): HarnessProps {
  return {
    contentRef: { current: harness.content },
    isRunning: true,
    loadSettledRef: { current: true },
    scrollRef: { current: harness.viewport },
    stopScroll: harness.stopScroll,
    ...overrides
  }
}

/** Park the viewport at the bottom of a running turn, the way a follower sees it. */
function mount(harness: Harness, overrides: Partial<HarnessProps> = {}) {
  harness.viewport.scrollTop = 4400

  return renderHook((p: HarnessProps) => useTurnEndAnchor(p), { initialProps: props(harness, overrides) })
}

afterEach(() => {
  cleanup()
  setTurnAnchor('bottom')
  vi.restoreAllMocks()
})

describe('useTurnEndAnchor', () => {
  it('holds the landed-at-the-end viewport under the default anchor', () => {
    const harness = transcript()
    const { rerender } = mount(harness)

    rerender(props(harness, { isRunning: false }))

    expect(harness.viewport.scrollTop).toBe(4400)
    expect(harness.stopScroll).not.toHaveBeenCalled()
  })

  it('settles on the newest prompt when a turn ends, escaping the follow lock first', () => {
    const harness = transcript()

    setTurnAnchor('prompt')
    // A standalone row after the turn (an injected notice) must not become the
    // anchor: the newest group HOLDING a prompt is what the reader asked for.
    const notice = window.document.createElement('div')
    notice.dataset.slot = 'aui_message-group'
    harness.content.append(notice)
    vi.spyOn(notice, 'getBoundingClientRect').mockReturnValue(rect(-700, 100))

    const { rerender } = mount(harness)

    rerender(props(harness, { isRunning: false }))

    expect(harness.viewport.scrollTop).toBe(4400 - 1200)
    expect(harness.stopScroll).toHaveBeenCalledOnce()
  })

  it('keeps the reading position of a reader who scrolled up mid-turn', () => {
    const harness = transcript()

    setTurnAnchor('prompt')
    const { rerender } = mount(harness)
    harness.viewport.scrollTop = 1000

    rerender(props(harness, { isRunning: false }))

    expect(harness.viewport.scrollTop).toBe(1000)
    expect(harness.stopScroll).not.toHaveBeenCalled()
  })

  it('does not move a transcript that is still loading', () => {
    const harness = transcript()

    setTurnAnchor('prompt')
    const { rerender } = mount(harness, { loadSettledRef: { current: false } })

    rerender(props(harness, { isRunning: false, loadSettledRef: { current: false } }))

    expect(harness.viewport.scrollTop).toBe(4400)
  })
})

describe('newestPromptTop', () => {
  it('measures the group that owns the newest prompt, not the bubble', () => {
    const harness = transcript()

    expect(newestPromptTop(harness.content)).toBe(-1200)
  })

  it('has no anchor for a transcript without a prompt', () => {
    expect(newestPromptTop(window.document.createElement('div'))).toBeNull()
  })
})
