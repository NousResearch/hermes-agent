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

const sidePaneGet = vi.fn(() => false)
const chatAnchorGet = vi.fn<() => null | string>(() => null)

vi.mock('@/components/pane-shell/tree/store', () => ({
  focusedChatZoneIsSidePane: () => sidePaneGet(),
  lastChatZoneSessionAnchor: () => chatAnchorGet()
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
    sidePaneGet.mockReturnValue(false)
    chatAnchorGet.mockReturnValue(null)
    $activeSessionId.set(null)
    $selectedStoredSessionId.set(null)
  })

  it('in-place focuses an existing tile and does not navigate', () => {
    focusOpenSession.mockReturnValue('tile')
    openSession('s1', navigate)
    expect(focusOpenSession).toHaveBeenCalledWith('s1', { workspaceMode: 'sessions' })
    expect(navigate).not.toHaveBeenCalled()
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

  it('focused opens a tab BESIDE the zone the user last worked in', () => {
    sidePaneGet.mockReturnValue(true)
    chatAnchorGet.mockReturnValue('session-tile:worked-in')
    focusOpenSession.mockReturnValue(null)

    openSession('s1', navigate, 'focused')

    // Explicit anchor: openSessionTile's own fallback asks the hovered → focused
    // → workspace ladder, which a real press on a session row sends to main.
    expect(openSessionTile).toHaveBeenCalledWith('s1', 'center', 'session-tile:worked-in')
    expect(navigate).not.toHaveBeenCalled()
  })

  it('focused leaves the anchor to the ladder when no chat zone was touched', () => {
    sidePaneGet.mockReturnValue(true)
    chatAnchorGet.mockReturnValue(null)
    focusOpenSession.mockReturnValue(null)

    openSession('s1', navigate, 'focused')

    expect(openSessionTile).toHaveBeenCalledWith('s1', 'center', undefined)
  })

  it('focused degrades to the classic in-place resume when main is the target', () => {
    sidePaneGet.mockReturnValue(false)
    focusOpenSession.mockReturnValue(null)

    openSession('s1', navigate, 'focused')

    expect(navigate).toHaveBeenCalledWith('/c/s1')
    expect(openSessionTile).not.toHaveBeenCalled()
  })

  it('focused pulls the chat back when the row is main\u2019s own session and the workspace shows a page', () => {
    sidePaneGet.mockReturnValue(true) // a side chat zone is where the user worked…
    focusOpenSession.mockReturnValue('main') // …but the clicked row IS main's session
    workspaceIsPageGet.mockReturnValue(true) // and main is showing a page
    $selectedStoredSessionId.set('s1')

    openSession('s1', navigate, 'focused')

    // 'tab' would front the workspace tab and return, leaving the page up — the
    // dead click this case exists to avoid.
    expect(navigate).toHaveBeenCalledWith('/c/s1')
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
    expect(openSessionTile).toHaveBeenCalledWith('s1', 'center', undefined)
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
    expect(openSessionTile).toHaveBeenCalledWith('s1', 'center', undefined)
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
    expect(openSessionTile).toHaveBeenCalledWith('s1', 'center', undefined)
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
    expect(openSessionTile).toHaveBeenCalledWith('s1', 'center', undefined)
  })

  it('no-ops on an empty id', () => {
    openSession('', navigate)
    expect(navigate).not.toHaveBeenCalled()
    expect(focusOpenSession).not.toHaveBeenCalled()
  })
})
