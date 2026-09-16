/**
 * Session stamps — one short durable label per session ("Merged", "WIP",
 * "Review", "Handoff", "Hold", or the user's own words), so a long session list
 * stays scannable instead of being a wall of similar titles.
 *
 * Unlike the sidebar's pins (localStorage with a backend mirror, see
 * session-pin-sync), a stamp has NO local copy: `sessions.stamp` on the
 * gateway's state.db is the truth, because it has to agree across two Desktop
 * installs and the CLI. So this module owns the write, patches the cached rows
 * optimistically, and puts the row back when the backend refuses — a stamp that
 * isn't on the server must never sit in the list looking durable.
 */

import { computed, type WritableAtom } from 'nanostores'

import { setSessionStampRemote } from '@/hermes'
import { notifyError } from '@/store/notifications'
import { $archivedSessions } from '@/store/sidebar-archive'
import type { SessionInfo } from '@/types/hermes'

import { $cronSessions, $messagingSessions, $sessions } from './session'

/** The menu's one-tap labels, in the order they are offered. Any other text the
 *  user types is equally valid — these are shortcuts, not an enum. */
export const SESSION_STAMP_PRESETS = ['Merged', 'WIP', 'Review', 'Handoff', 'Hold'] as const

/** Cap on a stamp's length: long enough for "Waiting on CI", short enough to
 *  stay a label beside a title. Mirrors the backend's own limit, which rejects
 *  anything longer, so the UI never sends what the API would 400. */
export const SESSION_STAMP_MAX_LENGTH = 24

/** Trim, collapse whitespace runs, cap the length. `''`, `null` and `undefined`
 *  all mean "no stamp". The ONE normalizer shared by the writer, the chip and
 *  the menu's custom-text input. */
export function normalizeSessionStamp(raw: null | string | undefined): null | string {
  const value = (raw ?? '').trim().replace(/\s+/g, ' ')

  return value ? value.slice(0, SESSION_STAMP_MAX_LENGTH) : null
}

/** Every atom that can hold a session row the user might be looking at. */
const stampAtoms: WritableAtom<SessionInfo[]>[] = [
  $sessions,
  $cronSessions,
  $messagingSessions,
  $archivedSessions
]

function patchRow(rows: SessionInfo[], sessionId: string, stamp: null | string): SessionInfo[] {
  let mutated = false

  const next = rows.map(row => {
    if (row.id !== sessionId || (row.stamp ?? null) === stamp) {
      return row
    }

    mutated = true

    return { ...row, stamp }
  })

  // Preserve reference identity on a no-op: handing React a fresh array with the
  // same rows re-renders the whole expensive tree for nothing.
  return mutated ? next : rows
}

/** Roll one row back to `stamp` in every list, without clobbering the rows a
 *  concurrent poll may have replaced meanwhile. */
function restoreRow(sessionId: string, stamp: null | string): void {
  for (const atom of stampAtoms) {
    atom.set(patchRow(atom.get(), sessionId, stamp))
  }
}

/** The pre-write value of the row, for the rollback path. */
function currentStamp(sessionId: string): null | string {
  for (const atom of stampAtoms) {
    const row = atom.get().find(candidate => candidate.id === sessionId)

    if (row) {
      return row.stamp ?? null
    }
  }

  return null
}

/**
 * Set (or clear) one session's stamp, optimistically. Resolves to whether the
 * backend accepted it — callers rarely care, but the failure path is visible:
 * the row snaps back and a notification says why.
 */
export async function applySessionStamp(
  sessionId: string,
  profile: string | undefined,
  stamp: null | string
): Promise<boolean> {
  const next = normalizeSessionStamp(stamp)
  const previous = currentStamp(sessionId)

  restoreRow(sessionId, next)

  try {
    await setSessionStampRemote(sessionId, next, profile)

    return true
  } catch (error) {
    restoreRow(sessionId, previous)
    notifyError(error, 'Could not set the session stamp')

    return false
  }
}

/**
 * Stored/lineage id -> stamp, over every loaded row.
 *
 * The tab strip subscribes to this through a string selector rather than to the
 * session lists themselves: the map is rebuilt once per list poll (O(rows),
 * once), while a selector keyed on ITS OWN stamp ignores the churn — so a tab
 * repaints when its stamp changes, not on every poll. Same shape as
 * `$sessionDotStateById`.
 */
export const $sessionStamps = computed(
  [$sessions, $cronSessions, $messagingSessions],
  (rows, cronRows, messagingRows) => {
    const stamps = new Map<string, string>()

    for (const session of [...rows, ...cronRows, ...messagingRows]) {
      const stamp = session.stamp

      if (!stamp) {
        continue
      }

      // Compression moves a conversation to a new row id, so a pane or a page
      // holding an older id must still resolve the stamp its session carries.
      stamps.set(session.id, stamp)

      if (session._lineage_root_id) {
        stamps.set(session._lineage_root_id, stamp)
      }

      for (const id of session._lineage_ids ?? []) {
        stamps.set(id, stamp)
      }
    }

    return stamps
  }
)
