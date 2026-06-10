import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { SessionInfo, SessionPresenceRecord } from '@/types/hermes'

import { $remoteDevices, $remoteSessions, remoteSessionEndpoint } from './remote-sessions'
import { $sessionPresence, $sessions } from './session'

const ENDPOINT = 'ws://192.168.1.20:8664/api/ws'

function presence(over: Partial<SessionPresenceRecord> & { session_id: string }): SessionPresenceRecord {
  return { endpoint: ENDPOINT, host: 'ko-win11', title: 'Remote work', ...over }
}

function localSession(id: string, over: Partial<SessionInfo> = {}): SessionInfo {
  return { id, started_at: 0, message_count: 1, ...over } as SessionInfo
}

describe('$remoteSessions', () => {
  beforeEach(() => {
    $sessions.set([])
    $sessionPresence.set([])
  })

  afterEach(() => {
    $sessions.set([])
    $sessionPresence.set([])
  })

  it('surfaces presence records that have an endpoint and no local twin', () => {
    $sessionPresence.set([presence({ session_id: 'remote-1', updated_at: 5 })])
    const remotes = $remoteSessions.get()
    expect(remotes).toHaveLength(1)
    expect(remotes[0]).toMatchObject({ sessionId: 'remote-1', endpoint: ENDPOINT, host: 'ko-win11' })
  })

  it('excludes records without an endpoint (discovery-only, not attachable)', () => {
    $sessionPresence.set([presence({ session_id: 'r', endpoint: undefined }), presence({ session_id: 'r2', endpoint: '  ' })])
    expect($remoteSessions.get()).toHaveLength(0)
  })

  it('excludes a record whose session is already a local row (by id or key)', () => {
    $sessions.set([localSession('local-1'), localSession('tip-9', { _lineage_root_id: 'root-9' })])
    $sessionPresence.set([
      presence({ session_id: 'local-1' }), // own session, also in local DB
      presence({ session_id: 'rt-9', session_key: 'root-9' }), // key matches a lineage root
      presence({ session_id: 'genuinely-remote' })
    ])
    const ids = $remoteSessions.get().map(r => r.sessionId)
    expect(ids).toEqual(['genuinely-remote'])
  })

  it('dedupes by session id, keeping the freshest record', () => {
    $sessionPresence.set([
      presence({ session_id: 'dup', title: 'old', updated_at: 1 }),
      presence({ session_id: 'dup', title: 'new', updated_at: 9 })
    ])
    const remotes = $remoteSessions.get()
    expect(remotes).toHaveLength(1)
    expect(remotes[0].title).toBe('new')
  })

  it('sorts newest first', () => {
    $sessionPresence.set([
      presence({ session_id: 'a', updated_at: 1 }),
      presence({ session_id: 'b', updated_at: 9 }),
      presence({ session_id: 'c', updated_at: 5 })
    ])
    expect($remoteSessions.get().map(r => r.sessionId)).toEqual(['b', 'c', 'a'])
  })

  it('falls back to sane defaults for missing fields', () => {
    $sessionPresence.set([{ session_id: 'bare', endpoint: ENDPOINT }])
    expect($remoteSessions.get()[0]).toMatchObject({ title: 'Untitled session', host: '', model: '', status: 'idle' })
  })
})

describe('$remoteDevices', () => {
  beforeEach(() => {
    $sessions.set([])
    $sessionPresence.set([])
  })

  afterEach(() => {
    $sessions.set([])
    $sessionPresence.set([])
  })

  it('collapses multiple sessions on the same peer into one device', () => {
    $sessionPresence.set([
      presence({ session_id: 'a', updated_at: 2 }),
      presence({ session_id: 'b', updated_at: 1 }) // same endpoint + host
    ])
    expect($remoteDevices.get()).toEqual([{ endpoint: ENDPOINT, host: 'ko-win11' }])
  })

  it('lists distinct peers sorted by host', () => {
    const OTHER = 'ws://10.0.0.5:8664/api/ws'
    $sessionPresence.set([
      presence({ session_id: 'a', endpoint: ENDPOINT, host: 'ko-win11' }),
      presence({ session_id: 'b', endpoint: OTHER, host: 'aurora-mac' })
    ])
    expect($remoteDevices.get()).toEqual([
      { endpoint: OTHER, host: 'aurora-mac' },
      { endpoint: ENDPOINT, host: 'ko-win11' }
    ])
  })

  it('is empty without reachable peers', () => {
    expect($remoteDevices.get()).toEqual([])
  })
})

describe('remoteSessionEndpoint', () => {
  beforeEach(() => {
    $sessions.set([])
    $sessionPresence.set([])
  })

  it('returns the endpoint for a remote session and null for a local/unknown one', () => {
    $sessionPresence.set([presence({ session_id: 'remote-1' })])
    expect(remoteSessionEndpoint('remote-1')).toBe(ENDPOINT)
    expect(remoteSessionEndpoint('not-a-session')).toBeNull()
  })

  it('returns null once the session also appears locally (attach completed)', () => {
    $sessionPresence.set([presence({ session_id: 'remote-1' })])
    expect(remoteSessionEndpoint('remote-1')).toBe(ENDPOINT)
    // After attaching, the runtime row shows up locally → no longer "remote".
    $sessions.set([localSession('remote-1')])
    expect(remoteSessionEndpoint('remote-1')).toBeNull()
  })
})
