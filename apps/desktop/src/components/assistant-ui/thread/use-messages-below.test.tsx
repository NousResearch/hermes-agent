import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $threadMessagesBelowBySession, resetThreadScroll } from '@/store/thread-scroll'

import { countMessagesBelow, createMessagesBelowReader, useMessagesBelow } from './use-messages-below'

function rect(top: number, bottom: number): DOMRect {
  return { top, bottom, height: bottom - top, width: 800 } as DOMRect
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
  resetThreadScroll('runtime-a')
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

    expect(countMessagesBelow(viewport, content).count).toBe(3)

    for (const measure of skippedRects) {
      expect(measure).not.toHaveBeenCalled()
    }

    assistantRect.mockReturnValue(rect(200, 600))
    expect(countMessagesBelow(viewport, content).count).toBe(2)

    vi.mocked(viewport.getBoundingClientRect).mockReturnValue(rect(0, 1500))
    expect(countMessagesBelow(viewport, content).count).toBe(0)
  })

  it('finds the fold with logarithmic group reads and reuses membership across scroll frames', () => {
    const count = 512
    const viewport = window.document.createElement('div')
    const content = window.document.createElement('div')
    viewport.append(content)
    let scrollTop = 0
    vi.spyOn(viewport, 'getBoundingClientRect').mockImplementation(() => rect(0, 600))
    const reads = { groups: 0, messages: 0 }
    const queries = vi.spyOn(content, 'querySelectorAll')

    for (let index = 0; index < count; index++) {
      const group = window.document.createElement('div')
      group.dataset.slot = 'aui_message-group'
      content.append(group)
      vi.spyOn(group, 'getBoundingClientRect').mockImplementation(() => {
        reads.groups++

        return rect(index * 100 - scrollTop, index * 100 + 100 - scrollTop)
      })

      for (const [slot, start, end] of [
        ['aui_user-message-root', 0, 40],
        ['aui_assistant-message-root', 40, 100]
      ] as const) {
        const message = window.document.createElement('div')
        message.dataset.slot = slot
        group.append(message)
        vi.spyOn(message, 'getBoundingClientRect').mockImplementation(() => {
          reads.messages++

          return rect(index * 100 + start - scrollTop, index * 100 + end - scrollTop)
        })
      }
    }

    const reader = createMessagesBelowReader(viewport, content)

    for (const index of [300, 20, 500, 200]) {
      scrollTop = index * 100 + 50
      reads.groups = reads.messages = 0
      const result = reader.read()
      const straddling = index + 6
      const expected = 1 + (count - straddling - 1) * 2
      expect(result).toEqual({ count: expected, settled: true })
      expect(reads.groups).toBeLessThanOrEqual(Math.ceil(Math.log2(count)) + 2)
      expect(reads.messages).toBe(2)
    }

    expect(queries.mock.calls.filter(([selector]) => selector === '[data-slot="aui_message-group"]').length).toBe(1)
  })

  it('invalidates cached membership only when counted message structure changes', () => {
    const { viewport, content } = transcript()
    const reader = createMessagesBelowReader(viewport, content)
    expect(reader.read().count).toBe(1)

    const paragraph = window.document.createElement('p')
    expect(
      reader.invalidate([{ type: 'childList', addedNodes: [paragraph], removedNodes: [] } as unknown as MutationRecord])
    ).toBe(false)

    const extra = window.document.createElement('div')
    extra.dataset.slot = 'aui_system-message-root'
    expect(
      reader.invalidate([{ type: 'childList', addedNodes: [extra], removedNodes: [] } as unknown as MutationRecord])
    ).toBe(true)
  })

  it('remeasures scroll and resize, ignores hidden panes, and clears at the bottom', () => {
    const { viewport, content, assistantRect } = transcript()
    let frame: FrameRequestCallback | undefined
    let resize: ResizeObserverCallback | undefined
    let mutate: MutationCallback | undefined
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
    vi.stubGlobal(
      'MutationObserver',
      class {
        constructor(callback: MutationCallback) {
          mutate = callback
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
      sessionKey: 'stored-a',
      sessionId: 'runtime-a'
    }

    const { rerender } = renderHook(props => useMessagesBelow(props), { initialProps: options })
    flush()
    expect($threadMessagesBelowBySession.get()['runtime-a'] ?? 0).toBe(1)

    // Non-message subtree mutations do not schedule another geometry read.
    mutate?.([{ type: 'childList', addedNodes: [window.document.createElement('span')], removedNodes: [] } as unknown as MutationRecord], {} as MutationObserver)
    expect(frame).toBeUndefined()

    // Another visible pane reaching bottom must not erase this reader's count.
    const sibling = renderHook(() => useMessagesBelow({ ...options, sessionId: 'runtime-b', isAtBottom: true }))
    expect($threadMessagesBelowBySession.get()['runtime-a']).toBe(1)
    sibling.unmount()

    assistantRect.mockReturnValue(rect(200, 500))
    viewport.dispatchEvent(new Event('scroll'))
    flush()
    expect($threadMessagesBelowBySession.get()['runtime-a'] ?? 0).toBe(0)

    assistantRect.mockReturnValue(rect(200, 900))
    resize?.([], {} as ResizeObserver)
    flush()
    expect($threadMessagesBelowBySession.get()['runtime-a'] ?? 0).toBe(1)

    rerender({ ...options, paneVisible: false })
    assistantRect.mockReturnValue(rect(200, 500))
    viewport.dispatchEvent(new Event('scroll'))
    flush()
    expect($threadMessagesBelowBySession.get()['runtime-a'] ?? 0).toBe(1)

    rerender({ ...options, isAtBottom: true })
    expect($threadMessagesBelowBySession.get()['runtime-a'] ?? 0).toBe(0)
  })

  it('re-measures a frame later when the turn at the fold has no layout boxes yet', () => {
    const { viewport, content, assistantRect } = transcript()
    const straddling = window.document.createElement('div')
    straddling.dataset.slot = 'aui_message-group'
    content.append(straddling)
    vi.spyOn(straddling, 'getBoundingClientRect').mockReturnValue(rect(500, 1500))

    // A content-visibility skipped turn: the group keeps its placeholder box,
    // its messages have none until relevancy updates on the next frame.
    const empty = { top: 0, bottom: 0, height: 0, width: 0 } as DOMRect

    const roots = ['aui_user-message-root', 'aui_assistant-message-root', 'aui_assistant-message-root'].map(slot => {
      const message = window.document.createElement('div')
      message.dataset.slot = slot
      straddling.append(message)

      return vi.spyOn(message, 'getBoundingClientRect').mockReturnValue(empty)
    })

    let frame: FrameRequestCallback | undefined
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
        observe() {}
        disconnect() {}
      }
    )
    vi.stubGlobal(
      'MutationObserver',
      class {
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

    renderHook(() =>
      useMessagesBelow({
        scrollRef: { current: viewport },
        contentRef: { current: content },
        isAtBottom: false,
        paneVisible: true,
        rows: null,
        sessionKey: 'stored-a',
        sessionId: 'runtime-a'
      })
    )

    // First frame: the straddling turn measures empty; nothing is published yet.
    flush()
    expect($threadMessagesBelowBySession.get()['runtime-a']).toBeUndefined()
    expect(frame).toBeDefined()

    // Relevancy updated: two of its three messages sit below the fold.
    roots[0]!.mockReturnValue(rect(500, 580))
    roots[1]!.mockReturnValue(rect(580, 900))
    roots[2]!.mockReturnValue(rect(900, 1500))
    assistantRect.mockReturnValue(rect(200, 500))
    flush()
    expect($threadMessagesBelowBySession.get()['runtime-a']).toBe(2)
    expect(frame).toBeUndefined()
  })
})
