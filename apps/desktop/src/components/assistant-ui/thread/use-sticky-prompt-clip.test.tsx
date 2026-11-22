import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { STICKY_PROMPT_CLIP_REMEASURE, useStickyPromptClip } from './use-sticky-prompt-clip'

const rect = (top: number, height: number) => ({ top, bottom: top + height, height }) as DOMRect

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

it('clips only visible covered siblings, follows resize, and releases styles and observers on hide', () => {
  const viewport = window.document.createElement('div')
  const content = window.document.createElement('div')
  viewport.append(content)
  vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue(rect(0, 500))

  const makeGroup = () => {
    const group = window.document.createElement('div')
    group.dataset.slot = 'aui_message-group'
    const editor = window.document.createElement('div')
    editor.style.display = 'contents'
    const prompt = window.document.createElement('div')
    prompt.dataset.slot = 'aui_user-message-root'
    prompt.style.position = 'sticky'
    prompt.style.top = '5px'
    const attachments = window.document.createElement('div')
    const reply = window.document.createElement('div')
    group.append(editor, reply)
    editor.append(prompt, attachments)
    content.append(group)

    return {
      group,
      prompt,
      attachments,
      reply,
      promptRect: vi.spyOn(prompt, 'getBoundingClientRect').mockReturnValue(rect(5, 60)),
      attachmentRect: vi.spyOn(attachments, 'getBoundingClientRect').mockReturnValue(rect(-50, 40)),
      replyRect: vi.spyOn(reply, 'getBoundingClientRect').mockReturnValue(rect(-10, 900))
    }
  }

  const visible = makeGroup()
  const skipped = makeGroup()
  vi.spyOn(visible.group, 'getBoundingClientRect').mockReturnValue(rect(0, 900))
  vi.spyOn(skipped.group, 'getBoundingClientRect').mockReturnValue(rect(900, 900))
  let intersection: IntersectionObserverCallback
  let resize: ResizeObserverCallback
  let frame: FrameRequestCallback | undefined
  const disconnect = vi.fn()
  const observe = vi.fn()
  vi.stubGlobal(
    'IntersectionObserver',
    class {
      constructor(callback: IntersectionObserverCallback) {
        intersection = callback
      }
      observe = observe
      disconnect = disconnect
    }
  )
  vi.stubGlobal(
    'ResizeObserver',
    class {
      constructor(callback: ResizeObserverCallback) {
        resize = callback
      }
      observe() {}
      unobserve() {}
      disconnect = disconnect
    }
  )
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    frame = callback

    return 1
  })
  vi.stubGlobal('cancelAnimationFrame', () => {
    frame = undefined
  })

  const flush = () =>
    act(() => {
      const run = frame
      frame = undefined
      run?.(0)
    })

  const options = {
    scrollRef: { current: viewport },
    contentRef: { current: content },
    paneVisible: true,
    rows: 'initial'
  }

  const { rerender } = renderHook(props => useStickyPromptClip(props), { initialProps: options })
  expect(observe).toHaveBeenCalledWith(skipped.group)
  act(() =>
    intersection(
      [
        {
          target: visible.group,
          isIntersecting: true,
          intersectionRatio: 1,
          boundingClientRect: rect(0, 900),
          intersectionRect: rect(0, 500),
          rootBounds: rect(0, 500),
          time: 0
        }
      ],
      {} as IntersectionObserver
    )
  )
  flush()
  expect(visible.reply.style.getPropertyValue('--sticky-prompt-clip')).toBe('75px')
  expect(visible.attachments.style.getPropertyValue('--sticky-prompt-clip')).toBe('40px')
  expect(visible.prompt.hasAttribute('data-sticky-prompt-clip')).toBe(false)
  expect(skipped.promptRect).not.toHaveBeenCalled()
  expect(skipped.replyRect).not.toHaveBeenCalled()

  // A new message or backfill changes rows while the same prompt is pinned.
  // Its existing mask must survive the commit, before any observer/rAF delivery.
  rerender({ ...options, rows: 'appended' })
  expect(visible.reply.style.getPropertyValue('--sticky-prompt-clip')).toBe('75px')
  expect(disconnect).not.toHaveBeenCalled()

  visible.promptRect.mockReturnValue(rect(5, 120))
  act(() => resize([], {} as ResizeObserver))
  flush()
  expect(visible.reply.style.getPropertyValue('--sticky-prompt-clip')).toBe('135px')

  visible.promptRect.mockReturnValue(rect(80, 120))
  act(() => viewport.dispatchEvent(new Event('scroll')))
  flush()
  expect(visible.reply.hasAttribute('data-sticky-prompt-clip')).toBe(false)
  expect(visible.attachments.hasAttribute('data-sticky-prompt-clip')).toBe(false)

  visible.promptRect.mockReturnValue(rect(5, 120))
  act(() => viewport.dispatchEvent(new Event('scroll')))
  flush()
  expect(visible.reply.hasAttribute('data-sticky-prompt-clip')).toBe(true)
  rerender({ ...options, paneVisible: false })
  expect(visible.reply.style.getPropertyValue('--sticky-prompt-clip')).toBe('')
  expect(disconnect).toHaveBeenCalledTimes(2)
})

const stubObservers = () => {
  const resize: { current: ResizeObserverCallback; observed: Set<Element> } = { current: () => {}, observed: new Set() }
  vi.stubGlobal(
    'IntersectionObserver',
    class {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
  )
  vi.stubGlobal(
    'ResizeObserver',
    class {
      constructor(callback: ResizeObserverCallback) {
        resize.current = callback
      }
      observe(target: Element) {
        resize.observed.add(target)
      }
      unobserve(target: Element) {
        resize.observed.delete(target)
      }
      disconnect() {}
    }
  )
  // Never run rAF callbacks: everything below must happen without one.
  vi.stubGlobal('requestAnimationFrame', () => 1)
  vi.stubGlobal('cancelAnimationFrame', () => undefined)

  return resize
}

const makeRow = (parent: HTMLElement, slot?: string) => {
  const element = window.document.createElement('div')

  if (slot) {
    element.dataset.slot = slot
  }

  parent.append(element)

  return element
}

// A subagent transcript pinned its task prompt while background-process
// notifications (and agent deliveries) streamed in the same turn. Those notes
// share the prompt's data-slot and its `top` inset from styles.css, but they
// are not sticky. Each starts its own message group; one crossing the top edge
// was taken as the "later" prompt and stretched the clip from the pinned card
// down to its own bottom: an opaque band hiding the transcript under the card.
// A viewport resize also left the clip one frame stale (it waited for a rAF).
it('ignores non-sticky prompt-slot rows and re-measures in the resize delivery itself', () => {
  const viewport = window.document.createElement('div')
  const content = makeRow(viewport)
  vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue(rect(0, 500))

  const turn = makeRow(content, 'aui_message-group')
  const prompt = makeRow(turn, 'aui_user-message-root')
  prompt.style.position = 'sticky'
  prompt.style.top = '5px'
  const reply = makeRow(turn)
  const noteGroup = makeRow(content, 'aui_message-group')
  const note = makeRow(noteGroup, 'aui_user-message-root')
  // Same numeric inset as the real prompt; only `position` tells them apart.
  note.style.top = '5px'
  const after = makeRow(noteGroup)

  const promptRect = vi.spyOn(prompt, 'getBoundingClientRect').mockReturnValue(rect(5, 60))
  const replyRect = vi.spyOn(reply, 'getBoundingClientRect').mockReturnValue(rect(-10, 420))
  // The note straddles the top edge and reaches far below the pinned card.
  vi.spyOn(note, 'getBoundingClientRect').mockReturnValue(rect(-20, 200))
  vi.spyOn(after, 'getBoundingClientRect').mockReturnValue(rect(180, 300))
  vi.spyOn(turn, 'getBoundingClientRect').mockReturnValue(rect(-400, 410))
  vi.spyOn(noteGroup, 'getBoundingClientRect').mockReturnValue(rect(-20, 500))
  const resize = stubObservers()

  renderHook(props => useStickyPromptClip(props), {
    initialProps: { scrollRef: { current: viewport }, contentRef: { current: content }, paneVisible: true, rows: 'r' }
  })

  // Exclusion ends at the pinned card (5 + 60 = 65), not at the note's bottom (180).
  expect(reply.style.getPropertyValue('--sticky-prompt-clip')).toBe('75px')
  expect(note.style.getPropertyValue('--sticky-prompt-clip')).toBe('85px')
  expect(after.hasAttribute('data-sticky-prompt-clip')).toBe(false)

  // The card grows (viewport resize / clamp): the ResizeObserver delivery
  // re-measures synchronously, before paint, with no animation frame.
  promptRect.mockReturnValue(rect(5, 90))
  replyRect.mockReturnValue(rect(-5, 420))
  act(() => resize.current([], {} as ResizeObserver))
  expect(reply.style.getPropertyValue('--sticky-prompt-clip')).toBe('100px')
  expect(note.style.getPropertyValue('--sticky-prompt-clip')).toBe('115px')
})

// Growing the viewport clamps scrollTop and can bring an earlier turn to the
// top before IntersectionObserver reports it. The resize delivery must pick it
// up from outer boxes, or that turn paints a frame with no clip at all.
it('refreshes visible turns from outer boxes when the viewport itself resizes', () => {
  const viewport = window.document.createElement('div')
  const content = makeRow(viewport)
  const viewportRect = vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue(rect(0, 100))

  const turn = makeRow(content, 'aui_message-group')
  const prompt = makeRow(turn, 'aui_user-message-root')
  prompt.style.position = 'sticky'
  prompt.style.top = '5px'
  const reply = makeRow(turn)
  const turnRect = vi.spyOn(turn, 'getBoundingClientRect').mockReturnValue(rect(150, 900))
  vi.spyOn(prompt, 'getBoundingClientRect').mockReturnValue(rect(5, 60))
  vi.spyOn(reply, 'getBoundingClientRect').mockReturnValue(rect(-50, 300))
  const resize = stubObservers()

  renderHook(props => useStickyPromptClip(props), {
    initialProps: { scrollRef: { current: viewport }, contentRef: { current: content }, paneVisible: true, rows: 'r' }
  })
  expect(reply.hasAttribute('data-sticky-prompt-clip')).toBe(false)

  expect(resize.observed.has(viewport)).toBe(true)

  viewportRect.mockReturnValue(rect(0, 400))
  turnRect.mockReturnValue(rect(-50, 900))
  act(() => resize.current([{ target: viewport } as unknown as ResizeObserverEntry], {} as ResizeObserver))
  expect(reply.style.getPropertyValue('--sticky-prompt-clip')).toBe('115px')
})

// At the bottom of a fixed-height viewport, collapsing a tall last turn clamps
// scrollTop and brings an earlier turn into view. Only content boxes resize,
// so the clamp (not a viewport entry) must trigger the visibility refresh.
it('refreshes visible turns when a content resize clamps scrollTop', () => {
  const viewport = window.document.createElement('div')
  const content = makeRow(viewport)
  vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue(rect(0, 400))

  const turn = makeRow(content, 'aui_message-group')
  const prompt = makeRow(turn, 'aui_user-message-root')
  prompt.style.position = 'sticky'
  prompt.style.top = '5px'
  const reply = makeRow(turn)
  const last = makeRow(content, 'aui_message-group')
  const turnRect = vi.spyOn(turn, 'getBoundingClientRect').mockReturnValue(rect(-900, 800))
  vi.spyOn(last, 'getBoundingClientRect').mockReturnValue(rect(-100, 2000))
  vi.spyOn(prompt, 'getBoundingClientRect').mockReturnValue(rect(5, 60))
  vi.spyOn(reply, 'getBoundingClientRect').mockReturnValue(rect(-50, 300))
  viewport.scrollTop = 900
  const resize = stubObservers()

  renderHook(props => useStickyPromptClip(props), {
    initialProps: { scrollRef: { current: viewport }, contentRef: { current: content }, paneVisible: true, rows: 'r' }
  })
  expect(reply.hasAttribute('data-sticky-prompt-clip')).toBe(false)

  // The last turn collapses; the browser clamps scrollTop and the earlier turn
  // reaches the top edge. Only the turn and content boxes report a resize.
  viewport.scrollTop = 50
  turnRect.mockReturnValue(rect(-50, 800))
  act(() => resize.current([{ target: content } as unknown as ResizeObserverEntry], {} as ResizeObserver))
  expect(reply.style.getPropertyValue('--sticky-prompt-clip')).toBe('115px')
})

// While a turn streams, list.tsx's bottom-follow ResizeObserver runs after the
// clip measured and re-pins scrollTop. Its synchronous remeasure event must
// move the clip to the final position before paint, without an animation frame.
it('re-measures synchronously when bottom-follow re-pins the transcript', () => {
  const viewport = window.document.createElement('div')
  const content = makeRow(viewport)
  vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue(rect(0, 400))

  const turn = makeRow(content, 'aui_message-group')
  const prompt = makeRow(turn, 'aui_user-message-root')
  prompt.style.position = 'sticky'
  prompt.style.top = '5px'
  const reply = makeRow(turn)
  vi.spyOn(turn, 'getBoundingClientRect').mockReturnValue(rect(-50, 900))
  vi.spyOn(prompt, 'getBoundingClientRect').mockReturnValue(rect(5, 60))
  const replyRect = vi.spyOn(reply, 'getBoundingClientRect').mockReturnValue(rect(-50, 300))
  stubObservers()

  const { rerender } = renderHook(props => useStickyPromptClip(props), {
    initialProps: { scrollRef: { current: viewport }, contentRef: { current: content }, paneVisible: true, rows: 'r' }
  })
  expect(reply.style.getPropertyValue('--sticky-prompt-clip')).toBe('115px')

  // Bottom-follow moved the transcript up by 40px after the clip measured.
  replyRect.mockReturnValue(rect(-90, 300))
  act(() => viewport.dispatchEvent(new Event(STICKY_PROMPT_CLIP_REMEASURE)))
  expect(reply.style.getPropertyValue('--sticky-prompt-clip')).toBe('155px')

  rerender({ scrollRef: { current: viewport }, contentRef: { current: content }, paneVisible: false, rows: 'r' })
  replyRect.mockReturnValue(rect(-120, 300))
  act(() => viewport.dispatchEvent(new Event(STICKY_PROMPT_CLIP_REMEASURE)))
  expect(reply.hasAttribute('data-sticky-prompt-clip')).toBe(false)
})
