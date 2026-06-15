import { describe, expect, it } from 'vitest'

import type { SessionDragPayload } from '@/app/chat/composer/inline-refs'

import { resolveSidebarSessionReleaseDrop } from './session-release-drop'

const RECENT_ROW: SessionDragPayload = {
  archived: false,
  id: 'recent-live',
  pinId: 'recent-root',
  pinned: false,
  profile: 'default',
  title: 'Recent row'
}

const PINNED_ROW: SessionDragPayload = {
  archived: false,
  id: 'pinned-live',
  pinId: 'pinned-root',
  pinned: true,
  profile: 'default',
  title: 'Pinned row'
}

const ARCHIVED_ROW: SessionDragPayload = {
  archived: true,
  id: 'archived-live',
  pinId: 'archived-root',
  pinned: false,
  profile: 'default',
  title: 'Archived row'
}

const anchorPinIdForSessionId = (sessionId: string) =>
  sessionId === 'pinned-live' ? 'pinned-root' : sessionId === 'other-pinned-live' ? 'other-pinned-root' : sessionId

describe('resolveSidebarSessionReleaseDrop', () => {
  it('pins a Sessions row at the released Pinned anchor index', () => {
    expect(
      resolveSidebarSessionReleaseDrop({
        anchor: { before: true, sessionId: 'pinned-live' },
        anchorPinIdForSessionId,
        payload: RECENT_ROW,
        pinnedSessionIds: ['other-pinned-root', 'pinned-root'],
        sectionKey: 'pinned',
        sessionOrderIds: ['recent-live', 'second-live'],
        showAllProfiles: false
      })
    ).toEqual({ index: 1, pinId: 'recent-root', type: 'pin' })
  })

  it('unpins a Pinned row and inserts it into Sessions at the released row slot', () => {
    expect(
      resolveSidebarSessionReleaseDrop({
        anchor: { before: false, sessionId: 'second-live' },
        anchorPinIdForSessionId,
        payload: PINNED_ROW,
        pinnedSessionIds: ['pinned-root'],
        sectionKey: 'sessions',
        sessionOrderIds: ['first-live', 'second-live'],
        showAllProfiles: false
      })
    ).toEqual({
      nextOrder: ['first-live', 'second-live', 'pinned-live'],
      type: 'sessions',
      unpinPinId: 'pinned-root'
    })
  })

  it('unpins a Pinned row at the Sessions section end when there is no row anchor', () => {
    expect(
      resolveSidebarSessionReleaseDrop({
        anchor: null,
        anchorPinIdForSessionId,
        payload: PINNED_ROW,
        pinnedSessionIds: ['pinned-root'],
        sectionKey: 'sessions',
        sessionOrderIds: ['first-live', 'second-live'],
        showAllProfiles: false
      })
    ).toEqual({
      nextOrder: ['first-live', 'second-live', 'pinned-live'],
      type: 'sessions',
      unpinPinId: 'pinned-root'
    })
  })

  it('reorders within Sessions on release using the same stable anchor math', () => {
    expect(
      resolveSidebarSessionReleaseDrop({
        anchor: { before: false, sessionId: 'third-live' },
        anchorPinIdForSessionId,
        payload: { ...RECENT_ROW, id: 'first-live', pinId: 'first-root' },
        pinnedSessionIds: [],
        sectionKey: 'sessions',
        sessionOrderIds: ['first-live', 'second-live', 'third-live'],
        showAllProfiles: false
      })
    ).toEqual({
      nextOrder: ['second-live', 'third-live', 'first-live'],
      type: 'sessions'
    })
  })

  it('restores archived rows released on Sessions and refuses to pin archived rows', () => {
    expect(
      resolveSidebarSessionReleaseDrop({
        anchor: { before: true, sessionId: 'first-live' },
        anchorPinIdForSessionId,
        payload: ARCHIVED_ROW,
        pinnedSessionIds: [],
        sectionKey: 'sessions',
        sessionOrderIds: ['first-live'],
        showAllProfiles: false
      })
    ).toEqual({
      nextOrder: ['archived-live', 'first-live'],
      restoreSessionId: 'archived-live',
      type: 'sessions'
    })

    expect(
      resolveSidebarSessionReleaseDrop({
        anchor: { before: true, sessionId: 'pinned-live' },
        anchorPinIdForSessionId,
        payload: ARCHIVED_ROW,
        pinnedSessionIds: ['pinned-root'],
        sectionKey: 'pinned',
        sessionOrderIds: ['first-live'],
        showAllProfiles: false
      })
    ).toEqual({ type: 'none' })
  })

  it('archives live rows released on Archived', () => {
    expect(
      resolveSidebarSessionReleaseDrop({
        anchor: null,
        anchorPinIdForSessionId,
        payload: RECENT_ROW,
        pinnedSessionIds: [],
        sectionKey: 'archived',
        sessionOrderIds: ['recent-live'],
        showAllProfiles: false
      })
    ).toEqual({ sessionId: 'recent-live', type: 'archive' })
  })
})
