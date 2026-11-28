import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { useStickyPromptClip } from './use-sticky-prompt-clip'

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

it('publishes the edge of a prompt at rest, and withdraws it while the prompt is pushed out', () => {
  const viewport = window.document.createElement('div')
  const content = window.document.createElement('div')
  const edge = window.document.createElement('div')
  viewport.append(content)
  vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue(rect(100, 500))

  const group = window.document.createElement('div')
  group.dataset.slot = 'aui_message-group'
  const prompt = window.document.createElement('div')
  prompt.dataset.slot = 'aui_user-message-root'
  prompt.style.top = '48px'
  const reply = window.document.createElement('div')
  group.append(prompt, reply)
  content.append(group)
  vi.spyOn(group, 'getBoundingClientRect').mockReturnValue(rect(0, 900))
  const promptRect = vi.spyOn(prompt, 'getBoundingClientRect').mockReturnValue(rect(148, 44))
  const replyRect = vi.spyOn(reply, 'getBoundingClientRect').mockReturnValue(rect(60, 900))

  let frame: FrameRequestCallback | undefined
  let scrolled: (() => void) | undefined
  vi.spyOn(viewport, 'addEventListener').mockImplementation((type, listener) => {
    if (type === 'scroll') {
      scrolled = listener as () => void
    }
  })

  const Observer = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  }

  vi.stubGlobal('IntersectionObserver', Observer)
  vi.stubGlobal('ResizeObserver', Observer)
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    frame = callback

    return 1
  })
  vi.stubGlobal('cancelAnimationFrame', () => {
    frame = undefined
  })

  const scroll = () =>
    act(() => {
      scrolled?.()
      const run = frame
      frame = undefined
      run?.(0)
    })

  const { unmount } = renderHook(() =>
    useStickyPromptClip({
      contentRef: { current: content },
      edgeRef: { current: edge },
      paneVisible: true,
      rows: 'rows',
      scrollRef: { current: viewport }
    })
  )

  // Pinned 48px below the viewport's top, 44px tall, with the reply running under it.
  expect(edge.hasAttribute('data-sticky-prompt-edge')).toBe(true)
  expect(edge.style.getPropertyValue('--sticky-prompt-edge')).toBe('92px')

  // The next turn pushes it up: no still edge to soften, and the last position is kept to fade from.
  promptRect.mockReturnValue(rect(120, 44))
  scroll()
  expect(edge.hasAttribute('data-sticky-prompt-edge')).toBe(false)
  expect(edge.style.getPropertyValue('--sticky-prompt-edge')).toBe('92px')

  // Back at rest, but its reply starts below it: nothing is covered, so nothing to soften.
  promptRect.mockReturnValue(rect(148, 44))
  replyRect.mockReturnValue(rect(210, 900))
  scroll()
  expect(edge.hasAttribute('data-sticky-prompt-edge')).toBe(false)

  replyRect.mockReturnValue(rect(60, 900))
  scroll()
  expect(edge.hasAttribute('data-sticky-prompt-edge')).toBe(true)

  unmount()
  expect(edge.hasAttribute('data-sticky-prompt-edge')).toBe(false)
  expect(edge.style.getPropertyValue('--sticky-prompt-edge')).toBe('')
})
