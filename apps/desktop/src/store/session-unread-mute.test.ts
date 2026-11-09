import { backendScopeKey } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createClientSessionState } from '@/lib/chat-runtime'

import { makeSessionInfo } from '../test/session-info'

import { $activeGatewayProfile, $profiles } from './profile'
import { $profileDotStateByScope } from './profile-dot-state'
import {
  $cronSessions,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  $unreadFinishedSessionIds,
  setSessions
} from './session'
import { $sessionDotStateById, $unreadSessionCount, sessionDotStateFor } from './session-dot-state'
import {
  clearAllSessionStates,
  dropTilesForProfile,
  migrateTilesForProfile,
  publishSessionState,
  recordSessionEventScope
} from './session-states'
import {
  $sessionSeenCounts,
  $unreadFinishedMarkers,
  lightDesktopMessagingCompletion,
  markSessionUnreadFinished,
  noteDesktopAdmission
} from './session-unread'
import {
  $sessionUnreadMutes,
  isSessionUnreadMuted,
  setSessionUnreadMuted,
  unreadMuteRowFor
} from './session-unread-mute'
import { $unreadWriteGuard } from './session-unread-remote'

const row = (id = 'helper', profile = 'default', connection_id = 'local') =>
  makeSessionInfo({ id, profile, connection_id })

const dot = (session: ReturnType<typeof row>) => sessionDotStateFor(session, $sessionDotStateById.get())

function reset() {
  clearAllSessionStates()
  $sessions.set([])
  $cronSessions.set([])
  $messagingSessions.set([])
  $selectedStoredSessionId.set(null)
  $unreadFinishedSessionIds.set([])
  $sessionUnreadMutes.set({})
  $sessionSeenCounts.set({})
  $unreadFinishedMarkers.set({})
  $unreadWriteGuard.set(new Map())
  $activeGatewayProfile.set('default')
}

beforeEach(reset)
afterEach(reset)

describe('per-session unread indicator muting', () => {
  it.each(['backend', 'watermark', 'completion'] as const)('masks the %s unread path independently', source => {
    const helper = { ...row(), message_count: 2, unread: source === 'backend' }
    setSessions([helper])

    if (source === 'watermark') {
      setSessions([{ ...helper, message_count: 4 }])
    } else if (source === 'completion') {
      markSessionUnreadFinished(helper.id)
    }

    expect(dot(helper)).toBe('unread')
    setSessionUnreadMuted(helper, true)
    expect(dot(helper)).toBe('idle')
    expect($unreadSessionCount.get()).toBe(0)
    setSessionUnreadMuted(helper, false)
    expect(dot(helper)).toBe('unread')
  })

  it('masks marker, watermark and backend unread without acknowledging them; unmute reveals activity', () => {
    const helper = { ...row(), message_count: 3, unread: true }
    const other = row('other')
    setSessions([helper, other])
    $sessionSeenCounts.set({ default: { helper: 1, other: 0 } })
    markSessionUnreadFinished('helper')
    markSessionUnreadFinished('other')
    expect($unreadSessionCount.get()).toBe(2)

    setSessionUnreadMuted(helper, true)
    expect(dot(helper)).toBe('idle')
    expect(dot(other)).toBe('unread')
    expect($unreadSessionCount.get()).toBe(1)
    expect($unreadFinishedMarkers.get().default).toContain('helper')
    expect($sessionSeenCounts.get().default.helper).toBe(1)
    expect($sessions.get()[0].unread).toBe(true)

    setSessions([{ ...helper, message_count: 9 }, other])
    markSessionUnreadFinished('helper')
    expect(dot(helper)).toBe('idle')
    expect($unreadSessionCount.get()).toBe(1)
    setSessionUnreadMuted(helper, false)
    expect(dot(helper)).toBe('unread')
    expect($unreadSessionCount.get()).toBe(2)
  })

  it('does not let the optimistic backend write guard bypass mute', () => {
    const helper = row()
    setSessions([helper])
    setSessionUnreadMuted(helper, true)
    $unreadWriteGuard.set(new Map([['helper', { at: Date.now(), value: true }]]))
    expect(dot(helper)).toBe('idle')
    setSessionUnreadMuted(helper, false)
    expect(dot(helper)).toBe('unread')
  })

  it('isolates exact ids, profiles and connections, even when listed together', () => {
    const helper = { ...row('shared', 'alpha', 'server-a'), unread: true }
    const otherProfile = { ...row('shared', 'beta', 'server-a'), unread: true }
    const otherConnection = { ...row('shared', 'alpha', 'server-b'), unread: true }
    const similarId = { ...row('shared-extra', 'alpha', 'server-a'), unread: true }
    setSessions([helper, otherProfile, otherConnection, similarId])
    setSessionUnreadMuted(helper, true)
    expect(dot(helper)).toBe('idle')

    for (const other of [otherProfile, otherConnection, similarId]) {
      expect(dot(other)).toBe('unread')
      expect(isSessionUnreadMuted(other)).toBe(false)
    }

    expect($unreadSessionCount.get()).toBe(3)
    expect($profileDotStateByScope.get()[backendScopeKey('server-a', 'alpha')]?.unreadCount).toBe(1)
    expect($profileDotStateByScope.get()[backendScopeKey('server-b', 'alpha')]?.unreadCount).toBe(1)
  })

  it('survives compression and masks aliases held by existing tabs', () => {
    const original = row('root')
    setSessions([original])
    setSessionUnreadMuted(original, true)

    const compressed = {
      ...row('tip-3'),
      _lineage_root_id: 'root',
      _lineage_ids: ['root', 'tip-2', 'tip-3'],
      unread: true
    }

    setSessions([compressed])
    expect(isSessionUnreadMuted(compressed)).toBe(true)
    expect(dot(compressed)).toBe('idle')
    expect($sessionDotStateById.get()['tip-2']).not.toBe('unread')
    setSessionUnreadMuted(compressed, false)
    expect($sessionDotStateById.get()['tip-2']).toBe('unread')
  })

  it('masks an unlisted runtime using its proven owner without losing compression identity', () => {
    const hidden = row('hidden', 'alpha', 'server')
    setSessionUnreadMuted(hidden, true)
    recordSessionEventScope({ session_id: 'hidden-runtime', profile: 'alpha', connectionId: 'server' })
    publishSessionState('hidden-runtime', createClientSessionState('hidden'))
    markSessionUnreadFinished('hidden', 'alpha')
    expect($sessionDotStateById.get().hidden).not.toBe('unread')

    const compressed = { ...hidden, id: 'tip-2', _lineage_root_id: 'hidden', unread: true }
    setSessions([compressed])
    expect($sessionDotStateById.get().hidden).not.toBe('unread')
    expect(dot(compressed)).toBe('idle')
  })

  it('does not hide working or blocking approval/needs-input states', () => {
    const helper = row()
    setSessions([helper])
    setSessionUnreadMuted(helper, true)
    publishSessionState('runtime', { ...createClientSessionState('helper'), busy: true })
    expect(dot(helper)).toBe('working')
    publishSessionState('runtime', { ...createClientSessionState('helper'), busy: true, needsInput: true })
    expect(dot(helper)).toBe('needs-input')
    expect($profileDotStateByScope.get()[backendScopeKey('local', 'default')]?.needsInputCount).toBe(1)
  })

  it('covers cron and desktop-admitted messaging completions', () => {
    const cron = row('cron')
    const messaging = { ...row('messaging'), source: 'telegram' }
    $cronSessions.set([cron])
    $messagingSessions.set([messaging])
    setSessionUnreadMuted(cron, true)
    setSessionUnreadMuted(messaging, true)
    markSessionUnreadFinished('cron')
    noteDesktopAdmission('turn')
    lightDesktopMessagingCompletion('messaging', 'turn', false, 'default')
    expect(dot(cron)).toBe('idle')
    expect(dot(messaging)).toBe('idle')
    expect($unreadSessionCount.get()).toBe(0)
    setSessionUnreadMuted(cron, false)
    setSessionUnreadMuted(messaging, false)
    expect(dot(cron)).toBe('unread')
    expect(dot(messaging)).toBe('unread')
    expect($unreadSessionCount.get()).toBe(1)
  })

  it('suppresses unlisted persisted markers on the profile rail', () => {
    $profiles.set([{ name: 'default' }] as typeof $profiles.value)
    setSessionUnreadMuted(row('hidden'), true)
    $unreadFinishedMarkers.set({ default: ['hidden'] })
    expect($profileDotStateByScope.get()[backendScopeKey('local', 'default')]).toBeUndefined()
    setSessionUnreadMuted(row('hidden'), false)
    expect($profileDotStateByScope.get()[backendScopeKey('local', 'default')]?.unreadCount).toBe(1)
  })

  it('resolves a background-profile menu without using the active profile', () => {
    const helper = row('shared', 'alpha')
    setSessions([row('shared', 'default'), helper])
    expect(unreadMuteRowFor('shared', 'alpha')).toEqual(helper)
  })

  it('migrates on profile rename and drops only the deleted source scope', () => {
    const local = row('helper', 'old')
    const remote = row('helper', 'old', 'remote')
    setSessionUnreadMuted(local, true)
    setSessionUnreadMuted(remote, true)
    migrateTilesForProfile('old', 'new')
    expect(isSessionUnreadMuted(row('helper', 'new'))).toBe(true)
    expect(isSessionUnreadMuted(remote)).toBe(true)
    dropTilesForProfile('old', { connectionId: 'remote', profile: 'old' })
    expect(isSessionUnreadMuted(remote)).toBe(false)
    expect(isSessionUnreadMuted(row('helper', 'new'))).toBe(true)
  })

  it('persists across module reload, including a later unmute', async () => {
    const helper = row('persisted', 'alpha', 'server-a')
    setSessionUnreadMuted(helper, true)
    vi.resetModules()
    const fresh = await import('./session-unread-mute')
    expect(fresh.isSessionUnreadMuted(helper)).toBe(true)
    expect(fresh.isSessionUnreadMuted(row('persisted', 'alpha', 'server-b'))).toBe(false)
    fresh.setSessionUnreadMuted(helper, false)
    vi.resetModules()
    const again = await import('./session-unread-mute')
    expect(again.isSessionUnreadMuted(helper)).toBe(false)
  })
})
