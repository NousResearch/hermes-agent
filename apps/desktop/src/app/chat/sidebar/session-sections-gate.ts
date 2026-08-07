interface SessionSectionsGateInput {
  showSessionSkeletons: boolean
  localSessionCount: number
  messagingSessionCount: number
  cronSessionCount: number
  projectCount: number
}

/**
 * Whether the sidebar's session sections (recents, per-platform messaging
 * groups, cron) should render at all, vs. the blank "No sessions" state.
 *
 * Regression guard (issue #77816): the gate originally only counted LOCAL
 * (recents) sessions and projects, so a profile whose only conversations
 * came from a messaging platform (or only from cron) -- zero local
 * sessions, zero projects -- saw the entire session area replaced by the
 * blank state even though their messaging/cron sessions existed and were
 * fetched (as $messagingSessions / $cronSessions) and rendered separately
 * inside the very block this gate controls. Messaging and cron sessions
 * must count toward the gate independently of local recents/projects.
 */
export function showSessionSectionsGate({
  showSessionSkeletons,
  localSessionCount,
  messagingSessionCount,
  cronSessionCount,
  projectCount
}: SessionSectionsGateInput): boolean {
  return (
    showSessionSkeletons ||
    localSessionCount > 0 ||
    messagingSessionCount > 0 ||
    cronSessionCount > 0 ||
    projectCount > 0
  )
}
