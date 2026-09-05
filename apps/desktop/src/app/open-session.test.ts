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

import {
  $activeSessionId,
  $currentCwd,
  $selectedStoredSessionId,
  $sessions,
  $workspaceCwdOwner
} from '@/store/session'

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
    // Reset the workspace atoms between cases. `setOwner(null)` (the no-call
    // path) and a missing release leave whatever the previous test wrote, so
    // without this each new case would observe a stale cwd.
    $currentCwd.set('')
    $workspaceCwdOwner.set(null)
    $sessions.set([])
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

  // The Files pane subscribes to $currentCwd. Until this fix, a sidebar click
  // on an already-open session fronted the tab but never wrote the target's
  // cwd, so the pane stayed on whichever folder the previously-loaded chat
  // owned until the next backend session.info heartbeat landed. These tests
  // pin the new behavior: a focused hit MUST mirror the sidebar row's cwd
  // projection so the panel tracks a click instantly.
  it('mirrors the target session cwd when focused on an existing tile', () => {
    $sessions.set([{ id: 's1', cwd: '/path/B' } as never])
    focusOpenSession.mockReturnValue('tile')
    $currentCwd.set('/path/A')
    $workspaceCwdOwner.set('s0')
    openSession('s1', navigate)
    expect($currentCwd.get()).toBe('/path/B')
    expect($workspaceCwdOwner.get()).toBe('s1')
  })

  it('mirrors the target session cwd when focused on main', () => {
    $sessions.set([{ id: 's1', cwd: '/path/B' } as never])
    focusOpenSession.mockReturnValue('main')
    $currentCwd.set('/path/A')
    $workspaceCwdOwner.set('s0')
    openSession('s1', navigate)
    expect($currentCwd.get()).toBe('/path/B')
    expect($workspaceCwdOwner.get()).toBe('s1')
  })

  it('releases ownership for a detached session stored row', () => {
    // cwd absent: the path on screen is provably still the previous
    // conversation's, so release rather than write '' and collapse the
    // Files/Review panes on every click (#71254).
    $sessions.set([{ id: 's1', cwd: null } as never])
    focusOpenSession.mockReturnValue('tile')
    $currentCwd.set('/path/A')
    $workspaceCwdOwner.set('s0')
    openSession('s1', navigate)
    expect($currentCwd.get()).toBe('/path/A')
    expect($workspaceCwdOwner.get()).not.toBe('s0')
    expect($workspaceCwdOwner.get()).not.toBe('s1')
  })

  it('does not touch cwd when the session is not on screen', () => {
    // A null focus falls through to resumeSession which has its own path —
    // openSession itself must not double-write the cwd here.
    $sessions.set([{ id: 's1', cwd: '/path/B' } as never])
    focusOpenSession.mockReturnValue(null)
    $currentCwd.set('/path/A')
    $workspaceCwdOwner.set('s0')
    openSession('s1', navigate)
    // Forced by an old owner to make sure openSession didn't release it as
    // a side effect; the atom retains whatever it held.
    expect($currentCwd.get()).toBe('/path/A')
  })
})
