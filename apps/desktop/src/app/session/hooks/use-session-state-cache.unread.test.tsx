import { act, cleanup, render } from '@testing-library/react'
import type { MutableRefObject } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { AppView } from '@/app/routes'
import { $unreadFinishedSessionIds, setSessionUnread } from '@/store/session'
import { $threadScrolledUp } from '@/store/thread-scroll'

import { useSessionStateCache } from './use-session-state-cache'

type Cache = ReturnType<typeof useSessionStateCache>

const RUNTIME_ID = 'runtime-session'
const STORED_ID = 'stored-session'
const TIP_ONE_ID = 'stored-session-tip-1'
const TIP_TWO_ID = 'stored-session-tip-2'

let focused = true

function Harness({ currentView = 'chat', onReady }: { currentView?: AppView; onReady: (cache: Cache) => void }) {
  const busyRef: MutableRefObject<boolean> = { current: false }

  const cache = useSessionStateCache({
    activeSessionId: RUNTIME_ID,
    busyRef,
    currentView,
    selectedStoredSessionId: STORED_ID,
    setAwaitingResponse: () => undefined,
    setBusy: () => undefined,
    setMessages: () => undefined
  })

  onReady(cache)

  return null
}

describe('useSessionStateCache unread view tracking', () => {
  beforeEach(() => {
    focused = true
    $unreadFinishedSessionIds.set([])
    $threadScrolledUp.set(false)
    vi.spyOn(globalThis.document, 'hasFocus').mockImplementation(() => focused)
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((callback: FrameRequestCallback) => {
      callback(0)

      return null as unknown as number
    })
  })

  afterEach(() => {
    cleanup()
    $unreadFinishedSessionIds.set([])
    $threadScrolledUp.set(false)
    vi.restoreAllMocks()
  })

  it('clears unread when that session is painted in a focused window', () => {
    setSessionUnread(STORED_ID, true)
    let cache!: Cache
    render(<Harness onReady={next => (cache = next)} />)

    act(() => {
      cache.updateSessionState(RUNTIME_ID, state => ({ ...state, busy: false }), STORED_ID)
    })

    expect($unreadFinishedSessionIds.get()).toEqual([])
  })

  it('keeps unread while unfocused, then clears it on window focus', () => {
    focused = false
    setSessionUnread(STORED_ID, true)
    let cache!: Cache
    render(<Harness onReady={next => (cache = next)} />)

    act(() => {
      cache.updateSessionState(RUNTIME_ID, state => ({ ...state, busy: false }), STORED_ID)
    })
    expect($unreadFinishedSessionIds.get()).toEqual([STORED_ID])

    focused = true
    act(() => window.dispatchEvent(new Event('focus')))

    expect($unreadFinishedSessionIds.get()).toEqual([])
  })

  it('keeps unread under another app view, then clears it when the transcript is shown', () => {
    setSessionUnread(STORED_ID, true)
    let cache!: Cache
    const { rerender } = render(<Harness currentView="settings" onReady={next => (cache = next)} />)

    act(() => {
      cache.updateSessionState(RUNTIME_ID, state => ({ ...state, busy: false }), STORED_ID)
    })
    expect($unreadFinishedSessionIds.get()).toEqual([STORED_ID])

    rerender(<Harness currentView="chat" onReady={next => (cache = next)} />)

    expect($unreadFinishedSessionIds.get()).toEqual([])
  })

  it('keeps unread while scrolled up, then clears it on returning to the bottom', () => {
    $threadScrolledUp.set(true)
    setSessionUnread(STORED_ID, true)
    let cache!: Cache
    render(<Harness onReady={next => (cache = next)} />)

    act(() => {
      cache.updateSessionState(RUNTIME_ID, state => ({ ...state, busy: false }), STORED_ID)
    })
    expect($unreadFinishedSessionIds.get()).toEqual([STORED_ID])

    act(() => $threadScrolledUp.set(false))

    expect($unreadFinishedSessionIds.get()).toEqual([])
  })

  it('migrates unread through repeated compression ids and clears the live tip', () => {
    setSessionUnread(STORED_ID, true)
    let cache!: Cache
    const { rerender } = render(<Harness currentView="settings" onReady={next => (cache = next)} />)

    act(() => {
      cache.updateSessionState(RUNTIME_ID, state => state, STORED_ID)
      cache.updateSessionState(RUNTIME_ID, state => state, TIP_ONE_ID)
    })
    expect($unreadFinishedSessionIds.get()).toEqual([TIP_ONE_ID])

    act(() => {
      cache.updateSessionState(RUNTIME_ID, state => state, TIP_TWO_ID)
    })
    expect($unreadFinishedSessionIds.get()).toEqual([TIP_TWO_ID])

    rerender(<Harness currentView="chat" onReady={next => (cache = next)} />)

    expect($unreadFinishedSessionIds.get()).toEqual([])
  })
})
