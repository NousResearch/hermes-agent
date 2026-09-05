import { beforeEach, describe, expect, it, vi } from 'vitest'

const focusOpenSession = vi.fn()
const openSessionTile = vi.fn()
const reuseBlankDraftTile = vi.fn()
const setSessionTileWorkspaceScope = vi.fn()
const openSessionInNewWindow = vi.fn()
const canOpenSessionWindow = vi.fn(() => true)
const workspaceIsPageGet = vi.fn(() => false)

vi.mock('@/store/session-states', () => ({
  focusedSessionNeedsRoute: (focused: 'main' | 'tile' | null, workspaceIsPage: boolean) =>
    !focused || (focused === 'main' && workspaceIsPage),
  focusOpenSession: (...args: unknown[]) => focusOpenSession(...args),
  openSessionTile: (...args: unknown[]) => openSessionTile(...args),
  reuseBlankDraftTile: (...args: unknown[]) => reuseBlankDraftTile(...args),
  setSessionTileWorkspaceScope: (...args: unknown[]) => setSessionTileWorkspaceScope(...args)
}))

vi.mock('@/store/windows', () => ({
  canOpenSessionWindow: () => canOpenSessionWindow(),
  openSessionInNewWindow: (...args: unknown[]) => openSessionInNewWindow(...args)
}))

vi.mock('./routes', () => ({
  $workspaceIsPage: { get: () => workspaceIsPageGet() },
  sessionRoute: (id: string) => `/c/${encodeURIComponent(id)}`
}))

import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

import { mainChatOccupied, openSession, openSessionIntentFromModifiers } from './open-session'

/**
 * The question behind both the sidebar "+" and a palette open: is there a
 * conversation on main that must not be discarded? A create affordance stacks a
 * tab rather than replacing a chat that may still be mid-turn, and an open from
 * nowhere does the same.
 */
describe('mainChatOccupied', () => {
  it('is occupied once a conversation is on screen', () => {
    expect(mainChatOccupied('runtime-a', 'stored-a')).toBe(true)
  })

  it('is occupied by a live runtime whose stored id has not landed yet', () => {
    expect(mainChatOccupied('runtime-a', null)).toBe(true)
  })

  it('is occupied by a selected session still resuming into a runtime', () => {
    expect(mainChatOccupied(null, 'stored-a')).toBe(true)
  })

  it('is free when nothing is open', () => {
    expect(mainChatOccupied(null, null)).toBe(false)
  })
})

describe('openSessionIntentFromModifiers', () => {
  it('defaults to in-place', () => {
    expect(openSessionIntentFromModifiers()).toBe('in-place')
    expect(openSessionIntentFromModifiers(null)).toBe('in-place')
    expect(openSessionIntentFromModifiers({})).toBe('in-place')
  })

  it('returns the caller base for an unmodified select', () => {
    expect(openSessionIntentFromModifiers(undefined, 'stack')).toBe('stack')
    expect(openSessionIntentFromModifiers({}, 'stack')).toBe('stack')
  })

  it('reads ⌘/⌃ as tab and ⇧+mod as window', () => {
    expect(openSessionIntentFromModifiers({ metaKey: true })).toBe('tab')
    expect(openSessionIntentFromModifiers({ ctrlKey: true })).toBe('tab')
    expect(openSessionIntentFromModifiers({ metaKey: true, shiftKey: true })).toBe('window')
    expect(openSessionIntentFromModifiers({ shiftKey: true })).toBe('in-place')
  })

  it('lets modifiers override the base', () => {
    expect(openSessionIntentFromModifiers({ metaKey: true }, 'stack')).toBe('tab')
    expect(openSessionIntentFromModifiers({ metaKey: true, shiftKey: true }, 'stack')).toBe('window')
  })
})

describe('openSession', () => {
  const navigate = vi.fn()

  beforeEach(() => {
    navigate.mockClear()
    focusOpenSession.mockReset()
    openSessionTile.mockReset()
    openSessionInNewWindow.mockReset()
    canOpenSessionWindow.mockReturnValue(true)
    workspaceIsPageGet.mockReturnValue(false)
    reuseBlankDraftTile.mockReset()
    setSessionTileWorkspaceScope.mockReset()
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
  })

  it('in-place routes a TILE hit into main (#92926) — no more dead click', () => {
    focusOpenSession.mockReturnValue('tile')
    openSession('s1', navigate)
    expect(focusOpenSession).toHaveBeenCalledWith('s1', { workspaceMode: 'sessions' })
    expect(navigate).toHaveBeenCalledWith('/c/s1')
    expect(openSessionTile).not.toHaveBeenCalled()
  })

  it('tab intent still fronts an existing tile without navigating', () => {
    focusOpenSession.mockReturnValue('tile')
    openSession('s1', navigate, 'tab')
    expect(navigate).not.toHaveBeenCalled()
    expect(openSessionTile).not.toHaveBeenCalled()
  })

  it('stack spends a blank main even when the session already sits in a tile', () => {
    // stack resolves against main occupancy BEFORE the tile check: with a
    // blank draft on main it becomes in-place, so the #92926 tile-route
    // applies (fronting the tile would leave the blank main showing).
    focusOpenSession.mockReturnValue('tile')
    openSession('s1', navigate, 'stack')
    expect(navigate).toHaveBeenCalledWith('/c/s1')
    expect(openSessionTile).not.toHaveBeenCalled()
  })

  it('in-place focuses main when already selected and not on a page', () => {
    focusOpenSession.mockReturnValue('main')
    openSession('s1', navigate)
    expect(navigate).not.toHaveBeenCalled()
  })

  it('in-place routes when the main session is covered by a page', () => {
    focusOpenSession.mockReturnValue('main')
    workspaceIsPageGet.mockReturnValue(true)
    openSession('s1', navigate)
    expect(navigate).toHaveBeenCalledWith('/c/s1')
  })

  it('in-place routes when the session is not on screen', () => {
    focusOpenSession.mockReturnValue(null)
    openSession('s1', navigate)
    expect(navigate).toHaveBeenCalledWith('/c/s1')
  })

  it('main routes to the workspace even when the session is already open as a tile', () => {
    focusOpenSession.mockReturnValue('tile')
    openSession('s1', navigate, 'main')
    expect(navigate).toHaveBeenCalledWith('/c/s1')
    expect(focusOpenSession).not.toHaveBeenCalled()
    expect(openSessionTile).not.toHaveBeenCalled()
  })

  it('tab focuses an existing open session instead of stacking another', () => {
    focusOpenSession.mockReturnValue('tile')
    openSession('s1', navigate, 'tab')
    expect(focusOpenSession).toHaveBeenCalledWith('s1', { workspaceMode: 'sessions' })
    expect(openSessionTile).not.toHaveBeenCalled()
    expect(navigate).not.toHaveBeenCalled()
  })

  it('tab opens a stacked session tile when not on screen', () => {
    focusOpenSession.mockReturnValue(null)
    openSession('s1', navigate, 'tab')
    expect(openSessionTile).toHaveBeenCalledWith('s1', 'center')
    expect(navigate).not.toHaveBeenCalled()
  })

  it('threads an exact Bot owner into a new session tile', () => {
    const scope = { workspaceMode: 'bots' as const, workspaceOwnerKey: 'connection-a::default' }
    focusOpenSession.mockReturnValue(null)

    openSession('s1', navigate, 'tab', scope)

    expect(setSessionTileWorkspaceScope).toHaveBeenCalledWith('s1', scope)
    expect(focusOpenSession).toHaveBeenCalledWith('s1', scope)
    expect(openSessionTile).toHaveBeenCalledWith('s1', 'center', undefined, undefined, scope)
  })

  it('stack focuses a session that is already on screen', () => {
    $selectedStoredSessionId.set('s0')
    focusOpenSession.mockReturnValue('tile')
    openSession('s1', navigate, 'stack')
    expect(openSessionTile).not.toHaveBeenCalled()
    expect(navigate).not.toHaveBeenCalled()
  })

  it('stack opens a tab rather than taking main from a loaded chat', () => {
    $selectedStoredSessionId.set('s0')
    focusOpenSession.mockReturnValue(null)
    openSession('s1', navigate, 'stack')
    expect(openSessionTile).toHaveBeenCalledWith('s1', 'center')
    expect(navigate).not.toHaveBeenCalled()
  })

  it('stack spends an open blank draft tab before stacking a new one', () => {
    $selectedStoredSessionId.set('s0')
    focusOpenSession.mockReturnValue(null)
    reuseBlankDraftTile.mockReturnValue(true)
    openSession('s1', navigate, 'stack')
    expect(reuseBlankDraftTile).toHaveBeenCalledWith('s1')
    expect(openSessionTile).not.toHaveBeenCalled()
    expect(navigate).not.toHaveBeenCalled()
  })

  it('stack prefers the session already on screen over a blank draft tab', () => {
    $selectedStoredSessionId.set('s0')
    focusOpenSession.mockReturnValue('tile')
    reuseBlankDraftTile.mockReturnValue(true)
    openSession('s1', navigate, 'stack')
    expect(reuseBlankDraftTile).not.toHaveBeenCalled()
    expect(openSessionTile).not.toHaveBeenCalled()
  })

  it('stack opens a tab while main is mid-turn on an unsaved session', () => {
    $activeSessionId.set('runtime-a')
    focusOpenSession.mockReturnValue(null)
    openSession('s1', navigate, 'stack')
    expect(openSessionTile).toHaveBeenCalledWith('s1', 'center')
  })

  it('stack loads into main when it holds only a blank draft', () => {
    focusOpenSession.mockReturnValue(null)
    openSession('s1', navigate, 'stack')
    expect(navigate).toHaveBeenCalledWith('/c/s1')
    expect(openSessionTile).not.toHaveBeenCalled()
  })

  it('window pops out when the bridge supports it', () => {
    openSession('s1', navigate, 'window')
    expect(openSessionInNewWindow).toHaveBeenCalledWith('s1')
    expect(openSessionTile).not.toHaveBeenCalled()
  })

  it('window falls back to a tab when pop-out is unavailable', () => {
    canOpenSessionWindow.mockReturnValue(false)
    focusOpenSession.mockReturnValue(null)
    openSession('s1', navigate, 'window')
    expect(openSessionInNewWindow).not.toHaveBeenCalled()
    expect(openSessionTile).toHaveBeenCalledWith('s1', 'center')
  })

  it('no-ops on an empty id', () => {
    openSession('', navigate)
    expect(navigate).not.toHaveBeenCalled()
    expect(focusOpenSession).not.toHaveBeenCalled()
  })
})
