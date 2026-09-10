import type { SessionInfo } from '@/types/hermes'

const MAX_NOTIFICATION_SESSION_TITLE = 64

export interface NotificationSessionTitleContext {
  connectionId?: null | string
  profile?: null | string
  runtimeSessionId?: null | string
  storedSessionId?: null | string
}

const profileKey = (value: null | string | undefined): string => value?.trim() || 'default'

const compact = (value: null | string | undefined): string => {
  const normalized = value?.replace(/\s+/g, ' ').trim() || ''

  if (normalized.length <= MAX_NOTIFICATION_SESSION_TITLE) {
    return normalized
  }

  return `${normalized.slice(0, MAX_NOTIFICATION_SESSION_TITLE - 1).trimEnd()}…`
}

/** Resolve a chat label only when the event's session identity maps to one
 *  unambiguous visible row. A wrong title is worse than no title when two
 *  connected gateways happen to expose the same session id. */
export function notificationSessionTitle(
  sessions: readonly SessionInfo[],
  context: NotificationSessionTitleContext
): string {
  const ids = new Set([context.runtimeSessionId, context.storedSessionId].filter((id): id is string => Boolean(id)))

  if (ids.size === 0) {
    return ''
  }

  let candidates = sessions.filter(
    session => ids.has(session.id) || Boolean(session._lineage_root_id && ids.has(session._lineage_root_id))
  )

  if (context.connectionId?.trim()) {
    const connectionId = context.connectionId.trim()
    candidates = candidates.filter(session => session.connection_id?.trim() === connectionId)
  } else {
    candidates = candidates.filter(session => !session.connection_id?.trim())
  }

  if (context.profile != null) {
    const profile = profileKey(context.profile)
    candidates = candidates.filter(session => profileKey(session.profile) === profile)
  }

  if (candidates.length !== 1) {
    return ''
  }

  return compact(candidates[0].title) || compact(candidates[0].preview)
}

export function formatSessionNotificationTitle(baseTitle: string, sessionTitle: string): string {
  return sessionTitle ? `${baseTitle} · ${sessionTitle}` : baseTitle
}
