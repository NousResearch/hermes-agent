import { normalizeProfileKey } from '@/store/profile'
import { sessionScopeKey } from '@/store/session'
import { sessionActivityKey } from '@/store/session-activity'

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

const idMatchesScoped = (ids: Set<string>, profile: null | string | undefined, id: string) => {
  if (ids.has(sessionScopeKey(profile, id))) {
    return true
  }

  const suffix = `\u0000${id}`

  return ![...ids].some(value => value.endsWith(suffix)) && ids.has(id)
}

const sessionMatchesScoped = (ids: Set<string>, session: ProfileActivitySession) =>
  idMatchesScoped(ids, session.profile, session.id) ||
  Boolean(session._lineage_root_id && idMatchesScoped(ids, session.profile, session._lineage_root_id))

const sessionMatchesWorking = (ids: Set<string>, session: ProfileActivitySession) =>
  ids.has(sessionActivityKey(session.profile, session.id)) ||
  Boolean(session._lineage_root_id && ids.has(sessionActivityKey(session.profile, session._lineage_root_id)))

/**
 * Collapse session-level activity into one actionable state per owning profile.
 * A blocking prompt wins. Within one conversation, live work wins over the
 * parent's unseen intermediate completion (background reviewers continue after
 * that completion). Across distinct conversations, an unseen result still wins
 * over a running turn so it is not lost. Both live ids and compression lineage
 * roots identify the same row.
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
    const activity: ProfileActivity = sessionMatchesScoped(attention, session)
      ? 'needs-input'
      : sessionMatchesScoped(unread, session)
        ? 'unread'
        : sessionMatchesWorking(working, session)
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
