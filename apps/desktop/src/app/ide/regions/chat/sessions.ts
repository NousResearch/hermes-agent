// The IDE's session slice: list the sessions it owns (source=ide) and create
// new ones. Creation goes through the SAME `session.create` every surface
// uses, so an IDE session gets the full agent (tools, skills, memory); the
// only difference is the wire tag that scopes it to this window.

import { listAllProfileSessions } from '@/api/sessions'
import { $activeGatewayProfile, normalizeProfileKey, resolveNewChatOwnerRoute } from '@/store/profile'
import { $connection, $sessions, setSessionOwnerHint } from '@/store/session'
import { openSessionTile, patchSessionTile } from '@/store/session-states'
import type { SessionCreateResponse, SessionInfo } from '@/types/hermes'

import { $ideWorkspaceRoot } from '../../state'

import { activateIdeChat } from './store'

export const IDE_SESSION_SOURCE = 'ide'

export type IdeRequestGateway = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

/** Recent IDE sessions, newest first — the tab strip's reopen list AND the
 *  source of truth for tab titles (this window's `$sessions` never receives
 *  ide rows from the primary sidebar lists, which exclude them). */
export async function listIdeSessions(): Promise<SessionInfo[]> {
  const profile = normalizeProfileKey($activeGatewayProfile.get()) || 'all'
  const result = await listAllProfileSessions(50, 0, 'exclude', 'recent', profile, { source: IDE_SESSION_SOURCE })

  return result.sessions ?? []
}

/** Upsert listed rows into this window's session store so tab titles resolve. */
export function rememberIdeSessionRows(rows: SessionInfo[]) {
  if (!rows.length) {
    return
  }

  const known = new Map($sessions.get().map(session => [session.id, session]))
  let changed = false

  for (const row of rows) {
    if (!known.has(row.id)) {
      known.set(row.id, row)
      changed = true
    }
  }

  if (changed) {
    $sessions.set([...known.values()])
  }
}

/**
 * Create a new IDE session (source='ide'), open it as a tab, and activate it.
 * Returns the stored session id, or null when the gateway did not return one.
 */
export async function createIdeSession(requestGateway: IdeRequestGateway): Promise<null | string> {
  const cwd = $ideWorkspaceRoot.get()?.trim() ?? ''
  const profile = normalizeProfileKey($activeGatewayProfile.get())

  const params: Record<string, unknown> = {
    cols: 96,
    source: IDE_SESSION_SOURCE,
    ...(cwd ? { cwd } : {}),
    ...(profile ? { profile } : {})
  }

  const created = await requestGateway<SessionCreateResponse>('session.create', params)
  const stored = created?.stored_session_id ?? null

  if (!stored) {
    return null
  }

  // Record the exact owner at creation — the same contract the primary create
  // path follows. The tile's very first `session.resume` resolves ownership
  // from this hint: a brand-new session has no lazy DB row yet, so neither the
  // row tag nor the profile probe can name its backend. The new-chat route is
  // authoritative; the live connection id is the second rung (a legacy
  // profile-only activation yields no route, but the session still belongs to
  // the connected backend it was just created on).
  const route = resolveNewChatOwnerRoute(profile)
  const connectionId = (route?.connectionId ?? $connection.get()?.connectionId ?? '').trim()

  const ownerRoute = connectionId
    ? { connectionId, profile: route?.profile || profile, ...(route?.targetProfile ? { targetProfile: route.targetProfile } : {}) }
    : undefined

  if (ownerRoute) {
    setSessionOwnerHint(stored, ownerRoute)
  }

  openSessionTile(stored)
  patchSessionTile(stored, { ownerRoute })
  activateIdeChat(stored)

  return stored
}
