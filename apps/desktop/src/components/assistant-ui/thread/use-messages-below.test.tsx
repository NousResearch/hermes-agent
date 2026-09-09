import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $threadMessagesBelow, resetThreadScroll } from '@/store/thread-scroll'

import { countMessagesBelow, useMessagesBelow } from './use-messages-below'

function rect(top: number, bottom: number): DOMRect {
  return { top, bottom, height: bottom - top } as DOMRect
}

function transcript() {
  const viewport = window.document.createElement('div')
  const content = window.document.createElement('div')
  viewport.append(content)
  vi.spyOn(viewport, 'getBoundingClientRect').mockReturnValue(rect(0, 600))

  const group = window.document.createElement('div')
  group.dataset.slot = 'aui_message-group'
  content.append(group)
  vi.spyOn(group, 'getBoundingClientRect').mockReturnValue(rect(100, 900))

  const user = window.document.createElement('div')
  user.dataset.slot = 'aui_user-message-root'
  group.append(user)
  vi.spyOn(user, 'getBoundingClientRect').mockReturnValue(rect(100, 180))

  const assistant = window.document.createElement('div')
  assistant.dataset.slot = 'aui_assistant-message-root'
  group.append(assistant)
  const assistantRect = vi.spyOn(assistant, 'getBoundingClientRect').mockReturnValue(rect(200, 900))

  return { viewport, content, assistantRect }
}

afterEach(() => {
  cleanup()
  resetThreadScroll()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('messages below the viewport', () => {
  it('counts the clipped message plus later messages without laying out skipped turns', () => {
    const { viewport, content, assistantRect } = transcript()
    const skipped = window.document.createElement('div')
    skipped.dataset.slot = 'aui_message-group'
    content.append(skipped)
    vi.spyOn(skipped, 'getBoundingClientRect').mockReturnValue(rect(900, 1500))

    const skippedRects = ['aui_user-message-root', 'aui_assistant-message-root'].map(slot => {
      const message = window.document.createElement('div')
      message.dataset.slot = slot
      skipped.append(message)

      return vi.spyOn(message, 'getBoundingClientRect')
    })

    expect(countMessagesBelow(viewport, content)).toBe(3)

    for (const measure of skippedRects) {
      expect(measure).not.toHaveBeenCalled()
    }

    assistantRect.mockReturnValue(rect(200, 600))
    expect(countMessagesBelow(viewport, content)).toBe(2)

    vi.mocked(viewport.getBoundingClientRect).mockReturnValue(rect(0, 1500))
    expect(countMessagesBelow(viewport, content)).toBe(0)
  })

  it('remeasures scroll and resize, ignores hidden panes, and clears at the bottom', () => {
    const { viewport, content, assistantRect } = transcript()
    let frame: FrameRequestCallback | undefined
    let resize: ResizeObserverCallback | undefined
    vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => {
      frame = callback

      return 1
    })
    vi.stubGlobal('cancelAnimationFrame', () => {
      frame = undefined
    })
    vi.stubGlobal(
      'ResizeObserver',
      class {
        constructor(callback: ResizeObserverCallback) {
          resize = callback
        }

        observe() {}
        disconnect() {}
      }
    )

    const flush = () =>
      act(() => {
        const callback = frame
        frame = undefined
        callback?.(0)
      })

    const options = {
      scrollRef: { current: viewport },
      contentRef: { current: content },
      isAtBottom: false,
      paneVisible: true,
      rows: null,
      sessionKey: 'a'
    }

    const { rerender } = renderHook(props => useMessagesBelow(props), { initialProps: options })
    flush()
    expect($threadMessagesBelow.get()).toBe(1)

    assistantRect.mockReturnValue(rect(200, 500))
    viewport.dispatchEvent(new Event('scroll'))
    flush()
    expect($threadMessagesBelow.get()).toBe(0)

    assistantRect.mockReturnValue(rect(200, 900))
    resize?.([], {} as ResizeObserver)
    flush()
    expect($threadMessagesBelow.get()).toBe(1)

    rerender({ ...options, paneVisible: false })
    assistantRect.mockReturnValue(rect(200, 500))
    viewport.dispatchEvent(new Event('scroll'))
    flush()
    expect($threadMessagesBelow.get()).toBe(1)

    rerender({ ...options, isAtBottom: true })
    expect($threadMessagesBelow.get()).toBe(0)
  })
})
