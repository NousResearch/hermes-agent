import { computed } from 'nanostores'

import { $sessions, $workingSessionIds, $workingSessionProfiles, UNKNOWN_SESSION_PROFILE_SCOPE } from './session'
import { $subagentsBySession, type SubagentProgress } from './subagents'

const RUNNING = (subagent: SubagentProgress) =>
  subagent.status === 'running' || subagent.status === 'queued' || subagent.handoff === true

const ACTIVITY_SEPARATOR = '\u0000'
const profileKey = (profile: null | string | undefined) => profile?.trim() || 'default'

interface SessionActivityRow {
  _lineage_root_id?: null | string
  id: string
  profile?: null | string
}

export const sessionActivityKey = (profile: null | string | undefined, sessionId: string) =>
  `${profileKey(profile)}${ACTIVITY_SEPARATOR}${sessionId}`

export const sessionIdFromActivityKey = (key: string) => key.slice(key.indexOf(ACTIVITY_SEPARATOR) + 1)

export function deriveSessionActivityKeys(
  foregroundSessionIds: readonly string[],
  subagentsBySession: Record<string, SubagentProgress[]>,
  sessions: readonly SessionActivityRow[] = [],
  foregroundProfiles: Readonly<Record<string, readonly string[]>> = {}
): string[] {
  const active = new Set<string>()

  for (const sessionId of foregroundSessionIds) {
    const scopedProfiles = foregroundProfiles[sessionId]

    const explicitProfiles = scopedProfiles?.filter(profile => profile !== UNKNOWN_SESSION_PROFILE_SCOPE)

    if (scopedProfiles) {
      for (const profile of explicitProfiles ?? []) {
        active.add(sessionActivityKey(profile, sessionId))
      }

      continue
    }

    const matches = sessions.filter(session => session.id === sessionId || session._lineage_root_id === sessionId)

    const matchedProfiles = new Set(matches.map(session => profileKey(session.profile)))

    if (matchedProfiles.size === 1) {
      active.add(sessionActivityKey([...matchedProfiles][0], sessionId))
    }
  }

  for (const [runtimeOwnerId, subagents] of Object.entries(subagentsBySession)) {
    for (const subagent of subagents) {
      if (!RUNNING(subagent)) {
        continue
      }

      const ownerSessionId = subagent.ownerSessionId || runtimeOwnerId
      let profile = subagent.profile

      if (profile === UNKNOWN_SESSION_PROFILE_SCOPE) {
        const matchingProfiles = new Set(
          sessions
            .filter(
              session =>
                session.id === ownerSessionId ||
                session._lineage_root_id === ownerSessionId ||
                session.id === subagent.sessionId
            )
            .map(session => profileKey(session.profile))
        )

        if (matchingProfiles.size !== 1) {
          continue
        }

        profile = [...matchingProfiles][0]
      }

      active.add(sessionActivityKey(profile, ownerSessionId))

      if (subagent.sessionId) {
        active.add(sessionActivityKey(profile, subagent.sessionId))
      }
    }
  }

  for (const session of sessions) {
    if (session._lineage_root_id && active.has(sessionActivityKey(session.profile, session._lineage_root_id))) {
      active.add(sessionActivityKey(session.profile, session.id))
    }
  }

  return [...active]
}

/**
 * Sessions with visible work in flight.
 *
 * Foreground turns live in `$workingSessionIds`. A top-level delegate finishes
 * that parent turn before its independent reviewers finish, so the sidebar and
 * profile rail must additionally project active subagents onto both the parent
 * conversation and each child's own persisted row. This intentionally does not
 * make the composer busy again — its calm background-resume treatment remains
 * authoritative for the foreground chat surface.
 */
export function deriveSessionActivityIds(
  foregroundSessionIds: readonly string[],
  subagentsBySession: Record<string, SubagentProgress[]>,
  sessions: readonly SessionActivityRow[] = []
): string[] {
  return [
    ...new Set(
      deriveSessionActivityKeys(foregroundSessionIds, subagentsBySession, sessions).map(sessionIdFromActivityKey)
    )
  ]
}

export const $sessionActivityKeys = computed(
  [$workingSessionIds, $workingSessionProfiles, $subagentsBySession, $sessions],
  (foregroundSessionIds, foregroundProfiles, subagentsBySession, sessions) =>
    deriveSessionActivityKeys(foregroundSessionIds, subagentsBySession, sessions, foregroundProfiles)
)

export const $sessionActivityIds = computed($sessionActivityKeys, keys => [
  ...new Set(keys.map(sessionIdFromActivityKey))
])
