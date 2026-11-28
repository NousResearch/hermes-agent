import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { $stickyUserMessage } from '@/store/sticky-user-message'

import { useStickyPromptClip } from './use-sticky-prompt-clip'

const rect = (top: number, height: number) => ({ top, bottom: top + height, height }) as DOMRect

afterEach(() => {
  cleanup()
  act(() => $stickyUserMessage.set(true))
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

// One IntersectionObserver and one ResizeObserver per live clip controller.
// Per-instance spies: the off switch disposes the current controller and the
// on switch builds a fresh one, so "everything disconnected" has to be read
// instance by instance.
const stubClipObservers = () => {
  interface Stub {
    disconnect: ReturnType<typeof vi.fn>
    observe: ReturnType<typeof vi.fn>
    unobserve: ReturnType<typeof vi.fn>
  }

  const observers: Stub[] = []

  const makeClass = () =>
    class {
      observe = vi.fn()
      unobserve = vi.fn()
      disconnect = vi.fn()

      constructor(_callback?: unknown) {
        observers.push(this as unknown as Stub)
      }
    }

  vi.stubGlobal('IntersectionObserver', makeClass())
  vi.stubGlobal('ResizeObserver', makeClass())
  let frame: FrameRequestCallback | undefined
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
    frame = callback

    return 1
  })
  vi.stubGlobal('cancelAnimationFrame', () => {
    frame = undefined
  })

  return observers
}

// One pinned prompt with a sibling it covers: geometry identical to the
// scenario below, so the clip arithmetic is the same 75px mask.
const buildPinnedTurn = (viewport: HTMLElement) => {
  const content = window.document.createElement('div')
  viewport.append(content)
  vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue(rect(0, 500))

  const group = window.document.createElement('div')
  group.dataset.slot = 'aui_message-group'
  const prompt = window.document.createElement('div')
  prompt.dataset.slot = 'aui_user-message-root'
  prompt.style.top = '5px'
  const reply = window.document.createElement('div')
  group.append(prompt, reply)
  content.append(group)
  vi.spyOn(group, 'getBoundingClientRect').mockReturnValue(rect(0, 900))
  vi.spyOn(prompt, 'getBoundingClientRect').mockReturnValue(rect(5, 60))
  vi.spyOn(reply, 'getBoundingClientRect').mockReturnValue(rect(-10, 900))

  return { content, reply }
}

const clipOptions = (viewport: HTMLElement, content: HTMLElement, rows: string) => ({
  scrollRef: { current: viewport },
  contentRef: { current: content },
  paneVisible: true,
  rows
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

// The sticky switch (display.sticky_user_message) is the other lever: nothing
// pins while it is off, so the observer must never even exist, and a switch
// made mid-stream must strip any mask the on phase left behind.
it('never observes anything while the switch is off, and starts observing the moment it turns on', () => {
  act(() => $stickyUserMessage.set(false))
  const viewport = window.document.createElement('div')
  const { content, reply } = buildPinnedTurn(viewport)
  const observers = stubClipObservers()

  renderHook(props => useStickyPromptClip(props), { initialProps: clipOptions(viewport, content, 'initial') })

  expect(observers).toHaveLength(0)
  expect(reply.style.getPropertyValue('--sticky-prompt-clip')).toBe('')

  act(() => $stickyUserMessage.set(true))

  // One IntersectionObserver + one ResizeObserver, and the reconcile that
  // builds them masks the covered reply immediately.
  expect(observers).toHaveLength(2)
  expect(reply.style.getPropertyValue('--sticky-prompt-clip')).toBe('75px')
})

it('strips applied clips and disconnects the observers when the switch turns off, and rebuilds both back on', () => {
  const viewport = window.document.createElement('div')
  const { content, reply } = buildPinnedTurn(viewport)
  const observers = stubClipObservers()

  renderHook(props => useStickyPromptClip(props), { initialProps: clipOptions(viewport, content, 'initial') })
  expect(reply.style.getPropertyValue('--sticky-prompt-clip')).toBe('75px')
  expect(reply.hasAttribute('data-sticky-prompt-clip')).toBe(true)

  act(() => $stickyUserMessage.set(false))

  // No surviving mask to crop an unpinned bubble, and no observers left alive.
  expect(reply.style.getPropertyValue('--sticky-prompt-clip')).toBe('')
  expect(reply.hasAttribute('data-sticky-prompt-clip')).toBe(false)
  expect(observers).toHaveLength(2)
  expect(observers.every(observer => observer.disconnect.mock.calls.length === 1)).toBe(true)

  act(() => $stickyUserMessage.set(true))

  // Fresh observers for the rebuilt controller — the disposed pair stays dead.
  expect(observers).toHaveLength(4)
  expect(reply.style.getPropertyValue('--sticky-prompt-clip')).toBe('75px')
  expect(observers.slice(2).every(observer => observer.disconnect.mock.calls.length === 0)).toBe(true)
})
