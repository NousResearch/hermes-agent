import { computed } from 'nanostores'

import type { SessionInfo } from '@/types/hermes'

import { $projectTree } from './projects'
import {
  $cronSessions,
  $currentCwd,
  $messagingSessions,
  $selectedStoredSessionId,
  $sessions,
  $workspaceCwdOwner,
  sessionMatchesStoredId,
  workspaceCwdBelongsToSelectedSession
} from './session'

/**
 * THE conversation → workspace resolution.
 *
 * A conversation's workspace has two possible sources and they must not be
 * confused:
 *
 *  - the LIVE path (`$currentCwd`), which is authoritative only for the
 *    conversation that OWNS it (`$workspaceCwdOwner`);
 *  - the conversation's own stored row, which the sidebar already holds.
 *
 * Surfaces that paint a workspace (the Files pane, the coding rail) used to
 * depend on the first alone. That marker is stamped by whatever happened to run
 * last — a resume's preview paint, a `session.info`, a create — so any
 * conversation whose claim was never stamped read as "no workspace" for the
 * rest of the session, even though a row naming the exact same folder was on
 * screen in its project lane (#108805: identical `cwd` rows, one shows the
 * tree, the rest never do).
 *
 * The second source must therefore be consulted by the resolution itself, and
 * it must be looked for in EVERY list that can hold a row — a conversation
 * drilled into from a project lane lives in the lane's rows, not in the
 * paginated recents slice.
 *
 * What stays sacred: a LIVE path that is NOT owned by the selected conversation
 * is still never published (that is #71254 — the previous conversation's folder
 * must not paint under the newly selected chat). The fallback answers, at most,
 * "the selected conversation's OWN row names this folder"; when no loaded row
 * names one, the resolution returns '' and the panes stay hidden as before.
 */

/**
 * Session rows the sidebar holds for a drilled-in project: its overview preview
 * rows plus every lane row.
 */
export function projectLaneSessionRows(): SessionInfo[] {
  return $projectTree.get().flatMap(project => [
    ...(project.previewSessions ?? []),
    ...project.repos.flatMap(repo => repo.groups.flatMap(group => group.sessions))
  ])
}

/**
 * Every row the renderer currently holds that can describe a conversation:
 * recents first (they can carry a fresher optimistic owner tag), then the cron
 * and messaging slices, then the project lanes. Mirrors the owner-lookup order
 * so both resolution ladders see the same set of rows.
 */
export function candidateSessionRows(): SessionInfo[] {
  const laneRows = projectLaneSessionRows()

  if (!laneRows.length) {
    return $sessions.get()
  }

  return [...$sessions.get(), ...$cronSessions.get(), ...$messagingSessions.get(), ...laneRows]
}

/**
 * The loaded row that describes `storedSessionId`'s own workspace — preferring
 * a row that names a workspace at all, so a stale copy (an optimistic row whose
 * create response omitted `cwd`, a legacy untagged recents copy) cannot answer
 * "no workspace" for a conversation whose lane row knows better.
 *
 * This is a WORKSPACE question, deliberately separate from
 * `cachedSessionRow`'s owner-ROUTING question: that one is fail-closed on
 * purpose (a stale exact route must not be downgraded to another backend's
 * same-named profile) and must not be able to blank a workspace pane.
 */
export function sessionWorkspaceRow(storedSessionId: string): SessionInfo | undefined {
  const candidates = candidateSessionRows().filter(session => sessionMatchesStoredId(session, storedSessionId))

  return candidates.find(session => session.cwd?.trim()) ?? candidates[0]
}

/**
 * The workspace the SELECTED conversation has, or '' when nothing on hand knows
 * one. The single answer every workspace-painting surface reads.
 */
export function selectedSessionWorkspaceCwd(): string {
  const live = $currentCwd.get().trim()

  // The live path wins while the selection owns it: the agent may have relocated
  // itself (`cd` into a new worktree) and that move is the truth, not the row.
  if (workspaceCwdBelongsToSelectedSession()) {
    return live
  }

  const selected = $selectedStoredSessionId.get()

  // A null selection with an unowned path is a released switch: nothing on hand
  // names a workspace for it, so publish the terse empty state rather than the
  // previous conversation's folder.
  if (!selected) {
    return ''
  }

  return sessionWorkspaceRow(selected)?.cwd?.trim() || ''
}

/** Reactive form of {@link selectedSessionWorkspaceCwd} for renderers. */
export const $selectedSessionWorkspaceCwd = computed(
  [$currentCwd, $selectedStoredSessionId, $workspaceCwdOwner, $sessions, $cronSessions, $messagingSessions, $projectTree],
  () => selectedSessionWorkspaceCwd()
)
