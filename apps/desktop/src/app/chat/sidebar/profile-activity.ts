import { normalizeProfileKey } from '@/store/profile'

export type ProfileActivity = 'idle' | 'needs-input' | 'unread' | 'working'

export interface ProfileActivitySession {
  _lineage_root_id?: null | string
  id: string
  profile?: null | string
}

interface ProfileActivityInput {
  attentionSessionIds: readonly string[]
  sessions: readonly ProfileActivitySession[]
  unreadSessionIds: readonly string[]
  workingSessionIds: readonly string[]
}

const ACTIVITY_PRIORITY: Record<ProfileActivity, number> = {
  idle: 0,
  working: 1,
  unread: 2,
  'needs-input': 3
}

export function profileActivityPriority(activity: ProfileActivity): number {
  return ACTIVITY_PRIORITY[activity]
}

function sessionMatches(ids: ReadonlySet<string>, session: ProfileActivitySession): boolean {
  return ids.has(session.id) || Boolean(session._lineage_root_id && ids.has(session._lineage_root_id))
}

/**
 * Collapse session-level activity into one actionable state per owning profile.
 * A blocking prompt wins over an unseen terminal result, which wins over a
 * running turn. Both live ids and compression lineage roots identify the same row.
 */
export function deriveProfileActivityByProfile({
  attentionSessionIds,
  sessions,
  unreadSessionIds,
  workingSessionIds
}: ProfileActivityInput): Record<string, ProfileActivity> {
  const attention = new Set(attentionSessionIds)
  const unread = new Set(unreadSessionIds)
  const working = new Set(workingSessionIds)
  const byProfile: Record<string, ProfileActivity> = {}

  for (const session of sessions) {
    const activity: ProfileActivity = sessionMatches(attention, session)
      ? 'needs-input'
      : sessionMatches(unread, session)
        ? 'unread'
        : sessionMatches(working, session)
          ? 'working'
          : 'idle'

    if (activity === 'idle') {
      continue
    }

    const profile = normalizeProfileKey(session.profile)
    const previous = byProfile[profile] ?? 'idle'

    if (profileActivityPriority(activity) > profileActivityPriority(previous)) {
      byProfile[profile] = activity
    }
  }

  return byProfile
}
