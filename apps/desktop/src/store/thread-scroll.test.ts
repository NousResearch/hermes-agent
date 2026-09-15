import { afterEach, describe, expect, it, vi } from 'vitest'

import { readKey } from '@/lib/storage'

import {
  $threadJumpButtonVisible,
  $threadScrolledUp,
  $turnAnchor,
  onScrollToBottomRequest,
  publishThreadAtBottom,
  requestScrollToBottom,
  resetPublishedThreadScroll,
  resetThreadScroll,
  setThreadAtBottom,
  setTurnAnchor,
  TURN_ANCHOR_STORAGE_KEY,
  turnEndScrollTop
} from './thread-scroll'

afterEach(() => {
  resetThreadScroll()
  setTurnAnchor('bottom')
})

describe('publishThreadAtBottom', () => {
  it('lets the visible pane flash the jump pill when the thread leaves the bottom', () => {
    publishThreadAtBottom(false, { paneVisible: true })

    expect($threadJumpButtonVisible.get()).toBe(true)
    expect($threadScrolledUp.get()).toBe(true)
  })

  it('ignores stick-to-bottom misses from a hidden keep-alive pane', () => {
    setThreadAtBottom(true)

    publishThreadAtBottom(false, { paneVisible: false })

    expect($threadJumpButtonVisible.get()).toBe(false)
    expect($threadScrolledUp.get()).toBe(false)
  })

  it("keeps the visible pane's scrolled-up chrome when a hidden pane publishes", () => {
    publishThreadAtBottom(false, { paneVisible: true })

    publishThreadAtBottom(true, { paneVisible: false })

    expect($threadJumpButtonVisible.get()).toBe(true)
    expect($threadScrolledUp.get()).toBe(true)
  })
})

describe('resetPublishedThreadScroll', () => {
  it('clears the jump pill when the visible pane unmounts', () => {
    setThreadAtBottom(false)

    resetPublishedThreadScroll({ paneVisible: true })

    expect($threadJumpButtonVisible.get()).toBe(false)
    expect($threadScrolledUp.get()).toBe(false)
  })

  it('does not clear the visible pane when a hidden list unmounts', () => {
    setThreadAtBottom(false)

    resetPublishedThreadScroll({ paneVisible: false })

    expect($threadJumpButtonVisible.get()).toBe(true)
    expect($threadScrolledUp.get()).toBe(true)
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
