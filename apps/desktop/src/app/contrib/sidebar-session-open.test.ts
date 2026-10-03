import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { $sidebarSessionsOpenInNewTab } from '@/store/sidebar-open-preference'

const mocks = vi.hoisted(() => ({
  canOpenSessionWindow: vi.fn(() => true),
  forgetSessionOwnerHintsForSession: vi.fn(),
  openSession: vi.fn(),
  prepareSessionOwnerRetarget: vi.fn(),
  requestSessionResume: vi.fn(),
  sessionOwnerRouteFromRow: vi.fn()
}))

vi.mock('@/store/session', () => ({
  forgetSessionOwnerHintsForSession: (...args: unknown[]) => mocks.forgetSessionOwnerHintsForSession(...args),
  requestSessionResume: (...args: unknown[]) => mocks.requestSessionResume(...args),
  sessionOwnerRouteFromRow: (...args: unknown[]) => mocks.sessionOwnerRouteFromRow(...args)
}))

vi.mock('../open-session', () => ({
  openSession: (...args: unknown[]) => mocks.openSession(...args)
}))

vi.mock('@/store/session-states', () => ({
  prepareSessionOwnerRetarget: (...args: unknown[]) => mocks.prepareSessionOwnerRetarget(...args)
}))

vi.mock('@/store/windows', () => ({
  canOpenSessionWindow: () => mocks.canOpenSessionWindow()
}))

import { openSidebarSession } from './sidebar-session-open'

const navigate = vi.fn()

const session = (profile: string, connectionId: string): SessionInfo =>
  ({ connection_id: connectionId, id: 'shared-id', profile }) as SessionInfo

describe('openSidebarSession', () => {
  beforeEach(() => {
    mocks.canOpenSessionWindow.mockReset()
    mocks.canOpenSessionWindow.mockReturnValue(true)
    mocks.forgetSessionOwnerHintsForSession.mockReset()
    mocks.openSession.mockReset()
    mocks.prepareSessionOwnerRetarget.mockReset()
    mocks.requestSessionResume.mockReset()
    mocks.sessionOwnerRouteFromRow.mockReset()
    navigate.mockReset()
    $sidebarSessionsOpenInNewTab.set(true)
  })

  it('opens duplicate ids in tabs through the clicked exact owner without queueing a main resume', () => {
    const first = session('profile-a', 'connection-a')
    const second = session('profile-b', 'connection-b')
    mocks.sessionOwnerRouteFromRow
      .mockReturnValueOnce({ connectionId: 'connection-a', profile: 'profile-a', targetProfile: 'profile-a' })
      .mockReturnValueOnce({ connectionId: 'connection-b', profile: 'profile-b', targetProfile: 'profile-b' })

    openSidebarSession('shared-id', first, navigate)
    openSidebarSession('shared-id', second, navigate)

    expect(mocks.prepareSessionOwnerRetarget).toHaveBeenNthCalledWith(
      1,
      'shared-id',
      { connectionId: 'connection-a', profile: 'profile-a', targetProfile: 'profile-a' },
      false
    )
    expect(mocks.prepareSessionOwnerRetarget).toHaveBeenNthCalledWith(
      2,
      'shared-id',
      { connectionId: 'connection-b', profile: 'profile-b', targetProfile: 'profile-b' },
      false
    )
    expect(mocks.requestSessionResume).not.toHaveBeenCalled()
    expect(mocks.openSession).toHaveBeenNthCalledWith(1, 'shared-id', navigate, 'tab', {
      ownerRoute: { connectionId: 'connection-a', profile: 'profile-a', targetProfile: 'profile-a' },
      workspaceMode: 'sessions'
    })
    expect(mocks.openSession).toHaveBeenNthCalledWith(2, 'shared-id', navigate, 'tab', {
      ownerRoute: { connectionId: 'connection-b', profile: 'profile-b', targetProfile: 'profile-b' },
      workspaceMode: 'sessions'
    })
    expect(mocks.forgetSessionOwnerHintsForSession).not.toHaveBeenCalled()
  })

  it('queues an exact resume when the persisted preference selects main', () => {
    $sidebarSessionsOpenInNewTab.set(false)
    mocks.sessionOwnerRouteFromRow.mockReturnValue({
      connectionId: 'connection-a',
      profile: 'profile-a',
      targetProfile: 'profile-a'
    })

    openSidebarSession('shared-id', session('profile-a', 'connection-a'), navigate)

    expect(mocks.prepareSessionOwnerRetarget).toHaveBeenCalledWith(
      'shared-id',
      { connectionId: 'connection-a', profile: 'profile-a', targetProfile: 'profile-a' },
      true
    )
    expect(mocks.requestSessionResume).toHaveBeenCalledWith('shared-id', {
      connectionId: 'connection-a',
      profile: 'profile-a',
      targetProfile: 'profile-a'
    })
    expect(mocks.openSession).toHaveBeenCalledWith('shared-id', navigate, 'main', {
      ownerRoute: { connectionId: 'connection-a', profile: 'profile-a', targetProfile: 'profile-a' },
      workspaceMode: 'sessions'
    })
  })

  it('opens an exact-owner window without mutating or queueing the current main surface', () => {
    mocks.sessionOwnerRouteFromRow.mockReturnValue({ connectionId: 'connection-b', profile: 'profile-b' })

    openSidebarSession('shared-id', session('profile-b', 'connection-b'), navigate, 'window')

    expect(mocks.prepareSessionOwnerRetarget).not.toHaveBeenCalled()
    expect(mocks.requestSessionResume).not.toHaveBeenCalled()
    expect(mocks.openSession).toHaveBeenCalledWith('shared-id', navigate, 'window', {
      ownerRoute: { connectionId: 'connection-b', profile: 'profile-b' },
      workspaceMode: 'sessions'
    })
  })

  it('fences the current same-id owner when an unavailable window falls back to a tab', () => {
    mocks.canOpenSessionWindow.mockReturnValue(false)
    mocks.sessionOwnerRouteFromRow.mockReturnValue({ connectionId: 'connection-b', profile: 'profile-b' })

    openSidebarSession('shared-id', session('profile-b', 'connection-b'), navigate, 'window')

    expect(mocks.prepareSessionOwnerRetarget).toHaveBeenCalledWith(
      'shared-id',
      { connectionId: 'connection-b', profile: 'profile-b' },
      false
    )
    expect(mocks.requestSessionResume).not.toHaveBeenCalled()
    expect(mocks.openSession).toHaveBeenCalledWith('shared-id', navigate, 'tab', {
      ownerRoute: { connectionId: 'connection-b', profile: 'profile-b' },
      workspaceMode: 'sessions'
    })
  })

  it('keeps untagged main opens ambient after clearing stale explicit hints', () => {
    const untagged = session('default', '')
    $sidebarSessionsOpenInNewTab.set(false)
    mocks.sessionOwnerRouteFromRow.mockReturnValue(undefined)

    openSidebarSession('shared-id', untagged, navigate)

    expect(mocks.prepareSessionOwnerRetarget).toHaveBeenCalledWith('shared-id', undefined, true)
    expect(mocks.forgetSessionOwnerHintsForSession).toHaveBeenCalledWith('shared-id')
    expect(mocks.requestSessionResume).toHaveBeenCalledWith('shared-id')
    expect(mocks.openSession).toHaveBeenCalledWith('shared-id', navigate, 'main', { workspaceMode: 'sessions' })
  })

  it('keeps a profile-only row ambient with default tab placement', () => {
    mocks.sessionOwnerRouteFromRow.mockReturnValue(undefined)

    openSidebarSession('shared-id', session('remote-profile', ''), navigate)

    expect(mocks.prepareSessionOwnerRetarget).toHaveBeenCalledWith('shared-id', undefined, false)
    expect(mocks.forgetSessionOwnerHintsForSession).toHaveBeenCalledWith('shared-id')
    expect(mocks.requestSessionResume).not.toHaveBeenCalled()
    expect(mocks.openSession).toHaveBeenCalledWith('shared-id', navigate, 'tab', { workspaceMode: 'sessions' })
  })

  it('keeps an ownerless server-search row ambient', () => {
    mocks.sessionOwnerRouteFromRow.mockReturnValue(undefined)

    openSidebarSession('server-only-id', { id: 'server-only-id' } as SessionInfo, navigate)

    expect(mocks.prepareSessionOwnerRetarget).toHaveBeenCalledWith('server-only-id', undefined, false)
    expect(mocks.forgetSessionOwnerHintsForSession).toHaveBeenCalledWith('server-only-id')
    expect(mocks.openSession).toHaveBeenCalledWith('server-only-id', navigate, 'tab', {
      workspaceMode: 'sessions'
    })
  })

  it('preserves explicit window intent without inventing an owner for a generic row', () => {
    mocks.sessionOwnerRouteFromRow.mockReturnValue(undefined)

    openSidebarSession('server-only-id', { id: 'server-only-id' } as SessionInfo, navigate, 'window')

    expect(mocks.prepareSessionOwnerRetarget).not.toHaveBeenCalled()
    expect(mocks.forgetSessionOwnerHintsForSession).toHaveBeenCalledWith('server-only-id')
    expect(mocks.requestSessionResume).not.toHaveBeenCalled()
    expect(mocks.openSession).toHaveBeenCalledWith('server-only-id', navigate, 'window', {
      workspaceMode: 'sessions'
    })
  })
})
