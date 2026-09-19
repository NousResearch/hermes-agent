import { afterEach, describe, expect, it, vi } from 'vitest'

import { readKey } from '@/lib/storage'

import {
  $threadJumpButtonVisibleBySession,
  $threadMessagesBelowBySession,
  $threadScrolledUpBySession,
  $turnAnchor,
  onScrollToBottomRequest,
  publishThreadAtBottom,
  publishThreadMessagesBelow,
  requestScrollToBottom,
  resetPublishedThreadScroll,
  resetThreadScroll,
  setThreadAtBottom,
  setTurnAnchor,
  TURN_ANCHOR_STORAGE_KEY,
  turnEndScrollTop
} from './thread-scroll'

afterEach(() => {
  resetThreadScroll('session-a')
  resetThreadScroll('session-b')
  setTurnAnchor('bottom')
})

describe('publishThreadAtBottom', () => {
  it('lets the visible pane flash the jump pill when the thread leaves the bottom', () => {
    publishThreadAtBottom(false, { paneVisible: true, sessionId: 'session-a' })

    expect(Boolean($threadJumpButtonVisibleBySession.get()['session-a'])).toBe(true)
    expect(Boolean($threadScrolledUpBySession.get()['session-a'])).toBe(true)
  })

  it('ignores stick-to-bottom misses from a hidden keep-alive pane', () => {
    setThreadAtBottom(true, 'session-a')

    publishThreadAtBottom(false, { paneVisible: false, sessionId: 'session-a' })

    expect(Boolean($threadJumpButtonVisibleBySession.get()['session-a'])).toBe(false)
    expect(Boolean($threadScrolledUpBySession.get()['session-a'])).toBe(false)
  })

  it("keeps the visible pane's scrolled-up chrome when a hidden pane publishes", () => {
    publishThreadAtBottom(false, { paneVisible: true, sessionId: 'session-a' })

    publishThreadAtBottom(true, { paneVisible: false, sessionId: 'session-a' })

    expect(Boolean($threadJumpButtonVisibleBySession.get()['session-a'])).toBe(true)
    expect(Boolean($threadScrolledUpBySession.get()['session-a'])).toBe(true)
  })
})

describe('resetPublishedThreadScroll', () => {
  it('resets only the unmounting session, including its message count', () => {
    for (const sessionId of ['session-a', 'session-b']) {
      publishThreadAtBottom(false, { paneVisible: true, sessionId })
      publishThreadMessagesBelow(7, { paneVisible: true, sessionId })
    }

    resetPublishedThreadScroll({ paneVisible: true, sessionId: 'session-a' })

    expect($threadJumpButtonVisibleBySession.get()['session-a']).toBeUndefined()
    expect($threadScrolledUpBySession.get()['session-a']).toBeUndefined()
    expect($threadMessagesBelowBySession.get()['session-a']).toBeUndefined()
    expect($threadJumpButtonVisibleBySession.get()['session-b']).toBe(true)
    expect($threadScrolledUpBySession.get()['session-b']).toBe(true)
    expect($threadMessagesBelowBySession.get()['session-b']).toBe(7)
  })

  it('preserves mirror references on no-op ticks, hidden publications, and missing identities', () => {
    publishThreadAtBottom(false, { paneVisible: true, sessionId: 'session-a' })
    publishThreadMessagesBelow(7, { paneVisible: true, sessionId: 'session-a' })
    const flags = $threadScrolledUpBySession.get()
    const jump = $threadJumpButtonVisibleBySession.get()
    const counts = $threadMessagesBelowBySession.get()

    publishThreadAtBottom(false, { paneVisible: true, sessionId: 'session-a' })
    publishThreadMessagesBelow(7, { paneVisible: true, sessionId: 'session-a' })
    publishThreadMessagesBelow(0, { paneVisible: false, sessionId: 'session-a' })
    resetPublishedThreadScroll({ paneVisible: false, sessionId: 'session-a' })
    publishThreadAtBottom(false, { paneVisible: true, sessionId: null })
    publishThreadMessagesBelow(12, { paneVisible: true, sessionId: null })
    resetThreadScroll(null)

    expect($threadScrolledUpBySession.get()).toBe(flags)
    expect($threadJumpButtonVisibleBySession.get()).toBe(jump)
    expect($threadMessagesBelowBySession.get()).toBe(counts)
  })

  it('clears the jump pill when the visible pane unmounts', () => {
    setThreadAtBottom(false, 'session-a')

    resetPublishedThreadScroll({ paneVisible: true, sessionId: 'session-a' })

    expect(Boolean($threadJumpButtonVisibleBySession.get()['session-a'])).toBe(false)
    expect(Boolean($threadScrolledUpBySession.get()['session-a'])).toBe(false)
  })

  it('does not clear the visible pane when a hidden list unmounts', () => {
    setThreadAtBottom(false, 'session-a')

    resetPublishedThreadScroll({ paneVisible: false, sessionId: 'session-a' })

    expect(Boolean($threadJumpButtonVisibleBySession.get()['session-a'])).toBe(true)
    expect(Boolean($threadScrolledUpBySession.get()['session-a'])).toBe(true)
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

// #108941 — where a finished turn leaves the viewport.
describe('turnEndScrollTop', () => {
  // Viewport 600 tall at the top of the screen; the newest prompt sits 1200px
  // above it, in a transcript whose bottom is 4400.
  const parkedAtBottom = {
    anchor: 'prompt' as const,
    atBottom: true,
    maxScrollTop: 4400,
    promptTop: -1200,
    scrollTop: 4400,
    viewportTop: 0
  }

  it('keeps the landed-at-the-end viewport under the default anchor', () => {
    expect(turnEndScrollTop({ ...parkedAtBottom, anchor: 'bottom' })).toBeNull()
  })

  it('settles at the newest prompt, clamped to the scroll range', () => {
    expect(turnEndScrollTop(parkedAtBottom)).toBe(3200)
    // A transcript shorter than the target: land on its bottom, never past it.
    expect(turnEndScrollTop({ ...parkedAtBottom, maxScrollTop: 1000 })).toBe(1000)
    // Prompt already at the top of the view: nothing to do but stay.
    expect(turnEndScrollTop({ ...parkedAtBottom, promptTop: 0 })).toBe(4400)
  })

  it('leaves a reader who scrolled away where they are', () => {
    expect(turnEndScrollTop({ ...parkedAtBottom, atBottom: false, scrollTop: 900 })).toBeNull()
    expect(turnEndScrollTop({ ...parkedAtBottom, promptTop: null })).toBeNull()
  })
})

describe('turn anchor preference', () => {
  it('defaults to the landing every existing install has, storing nothing', () => {
    expect($turnAnchor.get()).toBe('bottom')
    expect(readKey(TURN_ANCHOR_STORAGE_KEY)).toBeNull()
  })

  it('persists the settle-at-my-prompt choice', () => {
    setTurnAnchor('prompt')

    expect(readKey(TURN_ANCHOR_STORAGE_KEY)).toBe('prompt')

    setTurnAnchor('bottom')

    expect(readKey(TURN_ANCHOR_STORAGE_KEY)).toBeNull()
  })
})
