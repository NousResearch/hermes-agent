// Dev-only: registers the stores worth attributing during a streaming turn.
//
// Deliberately NOT every atom in the app — a churn counter that reports 200
// rows is as useless as none. These are the stores on or adjacent to the
// streaming hot path, plus the sidebar's inputs, so a recording answers one
// question directly: while an agent is typing, what is being published, and
// who re-renders because of it?

import { $cronJobs } from '@/store/cron'
import { $petInfo } from '@/store/pet'
import { $projects, $projectTree } from '@/store/projects'
import {
  $activeSessionId,
  $awaitingResponse,
  $busy,
  $cronSessions,
  $currentCwd,
  $currentUsage,
  $gatewayState,
  $messages,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  $sessionsLoading
} from '@/store/session'
import {
  $attentionSessionIds,
  $focusedRuntimeId,
  $focusedSessionState,
  $focusedStoredSessionId,
  $sessionStates,
  $sessionTiles,
  $stalledSessionIds,
  $workingSessionIds
} from '@/store/session-states'

import { watchAtom } from './atom-churn'

/** Streaming hot path — written per token / per delta flush. */
const HOT = {
  $awaitingResponse,
  $busy,
  // The global mirror the workspace pane paints from.
  $messages,
  // Republished on EVERY delta: the per-session source of truth.
  $sessionStates
}

/** Derived from the hot path. These SHOULD stay quiet during a turn — any
 *  notification here is a candidate for the "wasted" column. */
const DERIVED = {
  $attentionSessionIds,
  $currentUsage,
  $focusedRuntimeId,
  $focusedSessionState,
  $focusedStoredSessionId,
  $stalledSessionIds,
  $workingSessionIds
}

/** Sidebar inputs. Expected to be cold during a turn — if any of these notify
 *  while an agent is typing, that is the bug. */
const SIDEBAR = {
  $activeSessionId,
  $cronJobs,
  $cronSessions,
  $currentCwd,
  $gatewayState,
  $messagingSessions,
  $petInfo,
  $projects,
  $projectTree,
  $selectedStoredSessionId,
  $sessions,
  $sessionsLoading,
  $sessionTiles
}

export function watchSessionAtoms() {
  for (const group of [HOT, DERIVED, SIDEBAR]) {
    for (const [name, store] of Object.entries(group)) {
      watchAtom(name, store)
    }
  }
}
