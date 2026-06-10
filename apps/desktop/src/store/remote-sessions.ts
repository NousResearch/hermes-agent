import { computed } from 'nanostores'

import type { SessionInfo, SessionPresenceRecord } from '@/types/hermes'

import { $sessions } from './session'
import { $sessionPresence } from './session'

// ── Remote (cross-device) sessions ──────────────────────────────────────────
// Session presence records (hermes_cli/session_presence.py) are discovered
// across devices when their folder is synced. A record is an ATTACHABLE remote
// session when it advertises a dialable ws `endpoint` (Phase 2a) AND it is not
// already one of this device's own local sessions (which the sidebar lists
// straight from the local state.db). On a single device with no presence sync
// this set is empty, so the sidebar's "Live on other devices" section never
// renders and the common path is untouched.

export interface RemoteSession {
  sessionId: string
  endpoint: string
  title: string
  host: string
  model: string
  status: string
  updatedAt: number
}

// A presence record collides with a local session when either its runtime
// session_id or its stored session_key matches a local row's id (or lineage
// root). Local rows own the canonical view, so we never surface a duplicate.
function localSessionIds(sessions: SessionInfo[]): Set<string> {
  const ids = new Set<string>()
  for (const s of sessions) {
    ids.add(s.id)
    if (s._lineage_root_id) {
      ids.add(s._lineage_root_id)
    }
  }
  return ids
}

function isAttachableRemote(record: SessionPresenceRecord, local: Set<string>): boolean {
  if (!record.endpoint || !record.endpoint.trim()) {
    return false
  }
  if (!record.session_id) {
    return false
  }
  if (local.has(record.session_id)) {
    return false
  }
  return !(record.session_key && local.has(record.session_key))
}

function toRemoteSession(record: SessionPresenceRecord): RemoteSession {
  return {
    sessionId: record.session_id,
    endpoint: record.endpoint!.trim(),
    title: record.title?.trim() || 'Untitled session',
    host: record.host?.trim() || '',
    model: record.model?.trim() || '',
    status: record.status?.trim() || 'idle',
    updatedAt: typeof record.updated_at === 'number' ? record.updated_at : 0
  }
}

// Remote sessions, newest first, deduped by session id (a synced folder can
// briefly hold two records for one session during a write; keep the freshest).
export const $remoteSessions = computed([$sessionPresence, $sessions], (presence, sessions) => {
  const local = localSessionIds(sessions)
  const bySession = new Map<string, RemoteSession>()

  for (const record of presence) {
    if (!isAttachableRemote(record, local)) {
      continue
    }
    const candidate = toRemoteSession(record)
    const existing = bySession.get(candidate.sessionId)
    if (!existing || candidate.updatedAt > existing.updatedAt) {
      bySession.set(candidate.sessionId, candidate)
    }
  }

  return [...bySession.values()].sort((a, b) => b.updatedAt - a.updatedAt)
})

export interface RemoteDevice {
  endpoint: string
  host: string
}

// The distinct peer gateways behind the discovered remote sessions — i.e. the
// devices a NEW session can be created on (channels Phase 3: create-from-
// anywhere). Derived from $remoteSessions, so a device only appears once it has
// advertised at least one session; with no presence sync the list is empty and
// the "new session on another device" affordance never renders.
export const $remoteDevices = computed($remoteSessions, remotes => {
  const byEndpoint = new Map<string, RemoteDevice>()

  for (const remote of remotes) {
    if (!remote.endpoint || byEndpoint.has(remote.endpoint)) {
      continue
    }

    byEndpoint.set(remote.endpoint, { endpoint: remote.endpoint, host: remote.host })
  }

  return [...byEndpoint.values()].sort((a, b) => a.host.localeCompare(b.host))
})

// O(1) endpoint lookup for the resume path: given a session id, return the
// remote gateway endpoint to dial, or null when the session is local.
export function remoteSessionEndpoint(sessionId: string): string | null {
  for (const remote of $remoteSessions.get()) {
    if (remote.sessionId === sessionId) {
      return remote.endpoint
    }
  }
  return null
}
