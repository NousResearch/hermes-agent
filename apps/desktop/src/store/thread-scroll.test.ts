import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $threadJumpButtonVisibleBySession,
  $threadScrolledUpBySession,
  isThreadJumpButtonVisible,
  isThreadScrolledUp,
  onScrollToBottomRequest,
  publishThreadAtBottom,
  requestScrollToBottom,
  resetPublishedThreadScroll,
  resetThreadScroll,
  setThreadAtBottom
} from './thread-scroll'

afterEach(() => {
  $threadJumpButtonVisibleBySession.set({})
  $threadScrolledUpBySession.set({})
})

describe('publishThreadAtBottom', () => {
  it('lets the visible pane flash the jump pill when the thread leaves the bottom', () => {
    publishThreadAtBottom(false, { paneVisible: true, sessionId: 'sess-a' })

    expect(isThreadJumpButtonVisible('sess-a')).toBe(true)
    expect(isThreadScrolledUp('sess-a')).toBe(true)
  })

  it('ignores stick-to-bottom misses from a hidden keep-alive pane', () => {
    setThreadAtBottom(true, 'sess-a')

    publishThreadAtBottom(false, { paneVisible: false, sessionId: 'sess-a' })

    expect(isThreadJumpButtonVisible('sess-a')).toBe(false)
    expect(isThreadScrolledUp('sess-a')).toBe(false)
  })

  it("keeps the visible pane's scrolled-up chrome when a hidden pane publishes", () => {
    publishThreadAtBottom(false, { paneVisible: true, sessionId: 'sess-a' })

    publishThreadAtBottom(true, { paneVisible: false, sessionId: 'sess-a' })

    expect(isThreadJumpButtonVisible('sess-a')).toBe(true)
    expect(isThreadScrolledUp('sess-a')).toBe(true)
  })

  it('does not light the jump pill in a sibling split pane (#103586)', () => {
    publishThreadAtBottom(true, { paneVisible: true, sessionId: 'sess-b' })
    publishThreadAtBottom(false, { paneVisible: true, sessionId: 'sess-a' })

    expect(isThreadJumpButtonVisible('sess-a')).toBe(true)
    expect(isThreadJumpButtonVisible('sess-b')).toBe(false)
    expect(isThreadScrolledUp('sess-b')).toBe(false)
  })
})

describe('resetPublishedThreadScroll', () => {
  it('clears the jump pill when the visible pane unmounts', () => {
    setThreadAtBottom(false, 'sess-a')

    resetPublishedThreadScroll({ paneVisible: true, sessionId: 'sess-a' })

    expect(isThreadJumpButtonVisible('sess-a')).toBe(false)
    expect(isThreadScrolledUp('sess-a')).toBe(false)
  })

  it('does not clear the visible pane when a hidden list unmounts', () => {
    setThreadAtBottom(false, 'sess-a')

    resetPublishedThreadScroll({ paneVisible: false, sessionId: 'sess-a' })

    expect(isThreadJumpButtonVisible('sess-a')).toBe(true)
    expect(isThreadScrolledUp('sess-a')).toBe(true)
  })

  it('does not clear a sibling pane when one split pane unmounts', () => {
    setThreadAtBottom(false, 'sess-a')
    setThreadAtBottom(false, 'sess-b')

    resetPublishedThreadScroll({ paneVisible: true, sessionId: 'sess-a' })

    expect(isThreadJumpButtonVisible('sess-a')).toBe(false)
    expect(isThreadJumpButtonVisible('sess-b')).toBe(true)
  })
})

describe('requestScrollToBottom', () => {
  it('routes a scroll request only to its session', () => {
    const sessionA = vi.fn()
    const sessionB = vi.fn()
    const stopA = onScrollToBottomRequest(sessionA, 'session-a')
    const stopB = onScrollToBottomRequest(sessionB, 'session-b')

    requestScrollToBottom('session-b')

    expect(sessionA).not.toHaveBeenCalled()
    expect(sessionB).toHaveBeenCalledOnce()
    stopA()
    stopB()
  })

  it("does not let a late unmount clear a newer session's handler", () => {
    const first = vi.fn()
    const second = vi.fn()
    const stopFirst = onScrollToBottomRequest(first, 'session-a')
    const stopSecond = onScrollToBottomRequest(second, 'session-a')

    stopFirst()
    requestScrollToBottom('session-a')

    expect(first).not.toHaveBeenCalled()
    expect(second).toHaveBeenCalledOnce()
    stopSecond()
  })
})
