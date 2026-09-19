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

it("clips a pinned prompt's own attachments row only past the tuck it sits under", () => {
  const viewport = window.document.createElement('div')
  const content = window.document.createElement('div')
  viewport.append(content)
  vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue(rect(0, 500))

  const makeRow = (tucked: boolean) => {
    const group = window.document.createElement('div')
    group.dataset.slot = 'aui_message-group'
    const prompt = window.document.createElement('div')
    prompt.dataset.slot = 'aui_user-message-root'
    prompt.style.top = '5px'
    const attachments = window.document.createElement('div')
    attachments.dataset.slot = 'aui_user-message-attachments'
    // The row is tucked under the prompt by its own negative top margin, so at
    // rest its box starts inside the prompt's box.
    attachments.style.marginTop = '-12px'
    const reply = window.document.createElement('div')
    group.append(prompt, attachments, reply)
    content.append(group)
    vi.spyOn(group, 'getBoundingClientRect').mockReturnValue(rect(0, 900))

    return {
      attachments,
      // Partially covered (raw distance 15px, well inside the 40px clamp): a
      // tuck deduction leaking onto a non-own row would show as 3px here.
      attachmentRect: vi
        .spyOn(attachments, 'getBoundingClientRect')
        .mockReturnValue(tucked ? rect(53, 40) : rect(50, 40)),
      group,
      prompt,
      promptRect: vi.spyOn(prompt, 'getBoundingClientRect').mockReturnValue(tucked ? rect(5, 60) : rect(-900, 60)),
      reply
    }
  }

  // An older turn whose row has scrolled behind the pinned prompt.
  const older = makeRow(false)
  const pinned = makeRow(true)
  let frame: FrameRequestCallback | undefined
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
      observe() {}
      unobserve() {}
      disconnect() {}
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

  renderHook(() =>
    useStickyPromptClip({
      scrollRef: { current: viewport },
      contentRef: { current: content },
      paneVisible: true,
      rows: 'initial'
    })
  )
  flush()

  // The pinned prompt's row is fully readable at rest: its own tuck is not
  // "covered" content. Siblings — including an older turn's row — still are, and
  // by the FULL distance (a tuck deduction leaking onto them would read 3px).
  expect(pinned.attachments.hasAttribute('data-sticky-prompt-clip')).toBe(false)
  expect(pinned.attachments.style.getPropertyValue('--sticky-prompt-clip')).toBe('')
  expect(older.attachments.style.getPropertyValue('--sticky-prompt-clip')).toBe('15px')

  // Scrolling the transcript on by 30px covers exactly those 30px of the row —
  // not 30px plus the tuck it was already sitting under.
  pinned.attachmentRect.mockReturnValue(rect(23, 40))
  act(() => viewport.dispatchEvent(new Event('scroll')))
  flush()
  expect(pinned.attachments.style.getPropertyValue('--sticky-prompt-clip')).toBe('30px')

  // Scrolling back to rest releases the clip instead of leaving a stale one.
  pinned.attachmentRect.mockReturnValue(rect(53, 40))
  act(() => viewport.dispatchEvent(new Event('scroll')))
  flush()
  expect(pinned.attachments.hasAttribute('data-sticky-prompt-clip')).toBe(false)
  expect(pinned.attachments.style.getPropertyValue('--sticky-prompt-clip')).toBe('')

  // The prompt scrolled out of its sticky range: nothing stays clipped.
  pinned.promptRect.mockReturnValue(rect(80, 60))
  act(() => viewport.dispatchEvent(new Event('scroll')))
  flush()
  expect(pinned.attachments.hasAttribute('data-sticky-prompt-clip')).toBe(false)
  expect(older.attachments.hasAttribute('data-sticky-prompt-clip')).toBe(false)
})
