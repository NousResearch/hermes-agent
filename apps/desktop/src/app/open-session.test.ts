import { beforeEach, describe, expect, it, vi } from 'vitest'

const focusOpenSession = vi.fn()
const openSessionTile = vi.fn()
const openSessionInNewWindow = vi.fn()
const canOpenSessionWindow = vi.fn(() => true)
const workspaceIsPageGet = vi.fn(() => false)

vi.mock('@/store/session-states', () => ({
  focusedSessionNeedsRoute: (focused: 'main' | 'tile' | null, workspaceIsPage: boolean) =>
    !focused || (focused === 'main' && workspaceIsPage),
  focusOpenSession: (...args: unknown[]) => focusOpenSession(...args),
  openSessionTile: (...args: unknown[]) => openSessionTile(...args)
}))

vi.mock('@/store/windows', () => ({
  canOpenSessionWindow: () => canOpenSessionWindow(),
  openSessionInNewWindow: (...args: unknown[]) => openSessionInNewWindow(...args)
}))

vi.mock('./routes', () => ({
  $workspaceIsPage: { get: () => workspaceIsPageGet() },
  sessionRoute: (id: string) => `/c/${encodeURIComponent(id)}`
}))

import { openSession, openSessionIntentFromModifiers } from './open-session'

describe('openSessionIntentFromModifiers', () => {
  it('defaults to in-place', () => {
    expect(openSessionIntentFromModifiers()).toBe('in-place')
    expect(openSessionIntentFromModifiers(null)).toBe('in-place')
    expect(openSessionIntentFromModifiers({})).toBe('in-place')
  })

  it('reads ⌘/⌃ as tab and ⇧+mod as window', () => {
    expect(openSessionIntentFromModifiers({ metaKey: true })).toBe('tab')
    expect(openSessionIntentFromModifiers({ ctrlKey: true })).toBe('tab')
    expect(openSessionIntentFromModifiers({ metaKey: true, shiftKey: true })).toBe('window')
    expect(openSessionIntentFromModifiers({ shiftKey: true })).toBe('in-place')
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
  })

  it('in-place focuses an existing tile and does not navigate', () => {
    focusOpenSession.mockReturnValue('tile')
    openSession('s1', navigate)
    expect(focusOpenSession).toHaveBeenCalledWith('s1')
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

  it('tab focuses an existing open session instead of stacking another', () => {
    focusOpenSession.mockReturnValue('tile')
    openSession('s1', navigate, 'tab')
    expect(focusOpenSession).toHaveBeenCalledWith('s1')
    expect(openSessionTile).not.toHaveBeenCalled()
    expect(navigate).not.toHaveBeenCalled()
  })

  it('tab opens a stacked session tile when not on screen', () => {
    focusOpenSession.mockReturnValue(null)
    openSession('s1', navigate, 'tab')
    expect(openSessionTile).toHaveBeenCalledWith('s1', 'center')
    expect(navigate).not.toHaveBeenCalled()
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
