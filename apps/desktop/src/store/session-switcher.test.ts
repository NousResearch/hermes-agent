import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/types/hermes'

import { $selectedStoredSessionId, $sessions } from './session'
import {
  $switcherIndex,
  $switcherOpen,
  $switcherSessions,
  closeSwitcher,
  commitOnCtrlUp,
  lastVisitedSessionId,
  onSwitcherTabDown,
  onSwitcherTabUp,
  openOrAdvanceSwitcher,
  slotSessionId,
  SWITCHER_REVEAL_MS
} from './session-switcher'

const session = (id: string): SessionInfo => ({ id }) as SessionInfo

const seed = (ids: string[], selected: null | string) => {
  $sessions.set(ids.map(session))
  $selectedStoredSessionId.set(selected)
}

const tabTap = (direction: 1 | -1 = 1) => {
  onSwitcherTabDown()
  const target = openOrAdvanceSwitcher(direction)
  onSwitcherTabUp()

  return target
}

beforeEach(() => {
  vi.useRealTimers()
  closeSwitcher()
  $switcherSessions.set([])
  $switcherIndex.set(0)
})

afterEach(() => {
  seed([], null)
})

describe('openOrAdvanceSwitcher', () => {
  it('does nothing with fewer than two sessions', () => {
    seed(['a'], 'a')
    onSwitcherTabDown()

    expect(openOrAdvanceSwitcher(1)).toBeNull()
  })

  it('jumps immediately on a quick Tab tap without opening the HUD', () => {
    seed(['a', 'b', 'c'], 'a')

    expect(tabTap()).toBe('b')
    expect($switcherOpen.get()).toBe(false)
    expect(commitOnCtrlUp()).toBeNull()
  })

  it('does not open the HUD when Ctrl stays down but Tab was released quickly', () => {
    vi.useFakeTimers()
    seed(['a', 'b', 'c'], 'a')

    tabTap()
    vi.advanceTimersByTime(SWITCHER_REVEAL_MS)

    expect($switcherOpen.get()).toBe(false)
  })

  it('opens the HUD when Tab stays held past the reveal delay', () => {
    vi.useFakeTimers()
    seed(['a', 'b', 'c'], 'a')

    onSwitcherTabDown()
    openOrAdvanceSwitcher(1)
    vi.advanceTimersByTime(SWITCHER_REVEAL_MS)

    expect($switcherOpen.get()).toBe(true)
    onSwitcherTabUp()
  })

  it('opens on a second Tab while Ctrl is still down', () => {
    seed(['a', 'b', 'c'], 'a')

    expect(tabTap()).toBe('b')
    onSwitcherTabDown()
    openOrAdvanceSwitcher(1)
    onSwitcherTabUp()

    expect($switcherOpen.get()).toBe(true)
    expect($switcherIndex.get()).toBe(2)
  })

  it('commits the HUD highlight on Ctrl up', () => {
    seed(['a', 'b', 'c'], 'a')

    expect(tabTap()).toBe('b')
    onSwitcherTabDown()
    openOrAdvanceSwitcher(1)
    onSwitcherTabUp()

    expect(commitOnCtrlUp()).toBe('c')
  })
})

describe('slotSessionId', () => {
  it('reads the armed snapshot while browsing is pending', () => {
    seed(['a', 'b', 'c'], 'a')
    tabTap()
    $sessions.set([session('x')])

    expect(slotSessionId(2)).toBe('b')
  })
})

describe('lastVisitedSessionId', () => {
  it('flips between the two chats you were on, not the sidebar activity order', () => {
    // Sidebar order a, b, c (c least active). Visit a, then c.
    seed(['a', 'b', 'c'], 'a')
    $selectedStoredSessionId.set('c')

    expect(lastVisitedSessionId()).toBe('a')
    $selectedStoredSessionId.set('a')
    expect(lastVisitedSessionId()).toBe('c')
    // A blank draft goes back to the chat just left.
    $selectedStoredSessionId.set(null)
    expect(lastVisitedSessionId()).toBe('a')
  })

  it('skips a chat that left the list but keeps one the list never held', () => {
    seed(['a', 'b', 'c'], 'b')
    $selectedStoredSessionId.set('old-from-search')
    $selectedStoredSessionId.set('c')
    $selectedStoredSessionId.set('a')
    $sessions.set([session('a'), session('b')])

    expect(lastVisitedSessionId()).toBe('old-from-search')
    $selectedStoredSessionId.set('old-from-search')
    expect(lastVisitedSessionId()).toBe('a')
    $selectedStoredSessionId.set('b')
    // c was archived out of the list, so it never comes back as "last".
    expect(lastVisitedSessionId()).toBe('old-from-search')
  })
})
