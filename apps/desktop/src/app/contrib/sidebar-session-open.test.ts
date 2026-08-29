import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SessionInfo } from '@/hermes'
import { $sidebarSessionsOpenInNewTab } from '@/store/sidebar-open-preference'

const mocks = vi.hoisted(() => ({
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

import { openSidebarSession } from './sidebar-session-open'

const navigate = vi.fn()

const session = (profile: string, connectionId: string): SessionInfo =>
  ({ connection_id: connectionId, id: 'shared-id', profile }) as SessionInfo

describe('openSidebarSession', () => {
  beforeEach(() => {
    mocks.forgetSessionOwnerHintsForSession.mockReset()
    mocks.openSession.mockReset()
    mocks.prepareSessionOwnerRetarget.mockReset()
    mocks.requestSessionResume.mockReset()
    mocks.sessionOwnerRouteFromRow.mockReset()
    navigate.mockReset()
    $sidebarSessionsOpenInNewTab.set(true)
  })

  it('uses tab intent by default while routing identical stored ids through the clicked row owner', () => {
    const first = session('profile-a', 'connection-a')
    const second = session('profile-b', 'connection-b')
    mocks.sessionOwnerRouteFromRow
      .mockReturnValueOnce({ connectionId: 'connection-a', profile: 'profile-a', targetProfile: 'profile-a' })
      .mockReturnValueOnce({ connectionId: 'connection-b', profile: 'profile-b', targetProfile: 'profile-b' })

    openSidebarSession('shared-id', first, navigate)
    openSidebarSession('shared-id', second, navigate)

    expect(mocks.sessionOwnerRouteFromRow).toHaveBeenNthCalledWith(1, first)
    expect(mocks.sessionOwnerRouteFromRow).toHaveBeenNthCalledWith(2, second)
    expect(mocks.requestSessionResume).toHaveBeenNthCalledWith(1, 'shared-id', {
      connectionId: 'connection-a',
      profile: 'profile-a',
      targetProfile: 'profile-a'
    })
    expect(mocks.requestSessionResume).toHaveBeenNthCalledWith(2, 'shared-id', {
      connectionId: 'connection-b',
      profile: 'profile-b',
      targetProfile: 'profile-b'
    })
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
    expect(mocks.openSession).toHaveBeenNthCalledWith(1, 'shared-id', navigate, 'tab', {
      ownerRoute: { connectionId: 'connection-a', profile: 'profile-a', targetProfile: 'profile-a' },
      workspaceMode: 'sessions'
    })
    expect(mocks.openSession).toHaveBeenNthCalledWith(2, 'shared-id', navigate, 'tab', {
      ownerRoute: { connectionId: 'connection-b', profile: 'profile-b', targetProfile: 'profile-b' },
      workspaceMode: 'sessions'
    })
  })

  it('uses explicit main intent when the persisted preference selects the main tab', () => {
    $sidebarSessionsOpenInNewTab.set(false)
    mocks.sessionOwnerRouteFromRow.mockReturnValue({
      connectionId: 'connection-a',
      profile: 'profile-a',
      targetProfile: 'profile-a'
    })

    openSidebarSession('shared-id', session('profile-a', 'connection-a'), navigate)

    expect(mocks.requestSessionResume).toHaveBeenCalledOnce()
    expect(mocks.prepareSessionOwnerRetarget).toHaveBeenCalledWith(
      'shared-id',
      { connectionId: 'connection-a', profile: 'profile-a', targetProfile: 'profile-a' },
      true
    )
    expect(mocks.openSession).toHaveBeenCalledWith('shared-id', navigate, 'main', {
      ownerRoute: { connectionId: 'connection-a', profile: 'profile-a', targetProfile: 'profile-a' },
      workspaceMode: 'sessions'
    })
  })

  it('keeps untagged rows on the ambient backend after clearing stale explicit hints', () => {
    const untagged = session('default', '')
    mocks.sessionOwnerRouteFromRow.mockReturnValue(undefined)

    openSidebarSession('shared-id', untagged, navigate)

    expect(mocks.sessionOwnerRouteFromRow).toHaveBeenCalledWith(untagged)
    expect(mocks.prepareSessionOwnerRetarget).not.toHaveBeenCalled()
    expect(mocks.forgetSessionOwnerHintsForSession).toHaveBeenCalledWith('shared-id')
    expect(mocks.requestSessionResume).toHaveBeenCalledWith('shared-id')
    expect(mocks.openSession).toHaveBeenCalledWith('shared-id', navigate, 'tab', { workspaceMode: 'sessions' })
  })
})
