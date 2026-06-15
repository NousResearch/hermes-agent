import { deleteSession, setSessionArchived } from '@/hermes'
import { translateNow } from '@/i18n'
import { MESSAGING_SESSION_SOURCE_IDS, normalizeSessionSource } from '@/lib/session-source'
import { clearQueuedPrompts } from '@/store/composer-queue'
import { $pinnedSessionIds } from '@/store/layout'
import { notify, notifyError } from '@/store/notifications'
import { normalizeProfileKey } from '@/store/profile'
import {
  $archivedSessions,
  $cronSessions,
  $messagingSessions,
  $sessions,
  adjustMessagingPlatformTotal,
  adjustSessionProfileTotal,
  sessionPinId,
  setArchivedSessions,
  setArchivedSessionsTotal,
  setCronSessions,
  setMessagingSessions,
  setSessions,
  setSessionsTotal,
  shieldSessionFromMerge
} from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

// A delete that fails with 404 "Session not found" means the row is ALREADY
// gone server-side — deleted on another device, or a remote profile's session
// this device only remembers via a pin. That outcome is success for the user's
// intent, not an error to roll back.
export function isSessionGoneError(err: unknown): boolean {
  const message = err instanceof Error ? err.message : String(err)

  return /session not found/i.test(message)
}

/** Which in-memory list a row renders from. Every count the sidebar shows is
 * scoped to one of these slices, so mutations must adjust the totals of the
 * slice a row actually lives in — decrementing the agent total for a Telegram
 * row is exactly the phantom "Load 1 more" bug this module exists to prevent. */
type SessionSliceName = 'agent' | 'archived' | 'cron' | 'messaging'

interface LocatedSessionRow {
  slice: SessionSliceName
  session: SessionInfo
}

const MESSAGING_SOURCES = new Set<string>(MESSAGING_SESSION_SOURCE_IDS)

function locateStoredSession(sessionId: string): LocatedSessionRow | null {
  const agent = $sessions.get().find(s => s.id === sessionId)

  if (agent) {
    return { session: agent, slice: 'agent' }
  }

  const messaging = $messagingSessions.get().find(s => s.id === sessionId)

  if (messaging) {
    return { session: messaging, slice: 'messaging' }
  }

  const cron = $cronSessions.get().find(s => s.id === sessionId)

  if (cron) {
    return { session: cron, slice: 'cron' }
  }

  const archived = $archivedSessions.get().find(s => s.id === sessionId)

  if (archived) {
    return { session: archived, slice: 'archived' }
  }

  return null
}

/** The slice an unarchived row belongs to, by source. */
function homeSliceFor(session: SessionInfo): SessionSliceName {
  const source = normalizeSessionSource(session.source)

  if (source === 'cron') {
    return 'cron'
  }

  return source && MESSAGING_SOURCES.has(source) ? 'messaging' : 'agent'
}

/** Remove one located row from its slice and keep that slice's totals honest. */
function removeRowFromSlice({ session, slice }: LocatedSessionRow) {
  const drop = (rows: SessionInfo[]) => rows.filter(s => s.id !== session.id)

  switch (slice) {
    case 'agent':
      setSessions(drop)
      setSessionsTotal(prev => Math.max(0, prev - 1))
      adjustSessionProfileTotal(normalizeProfileKey(session.profile), -1)

      break

    case 'archived':
      setArchivedSessions(drop)
      setArchivedSessionsTotal(prev => Math.max(0, prev - 1))

      break

    case 'cron':
      setCronSessions(drop)

      break
    case 'messaging': {
      setMessagingSessions(drop)
      const source = normalizeSessionSource(session.source)

      if (source) {
        adjustMessagingPlatformTotal(source, -1)
      }

      break
    }
  }
}

/** Re-insert a row into a slice (rollback / restore destination), adjusting
 * that slice's totals. No-ops when the row is already present. */
function insertRowIntoSlice(session: SessionInfo, slice: SessionSliceName) {
  const prepend = (rows: SessionInfo[]) => (rows.some(s => s.id === session.id) ? rows : [session, ...rows])

  switch (slice) {
    case 'agent':
      if (!$sessions.get().some(s => s.id === session.id)) {
        setSessionsTotal(prev => prev + 1)
        adjustSessionProfileTotal(normalizeProfileKey(session.profile), 1)
      }

      setSessions(prepend)

      break

    case 'archived':
      if (!$archivedSessions.get().some(s => s.id === session.id)) {
        setArchivedSessionsTotal(prev => prev + 1)
      }

      setArchivedSessions(prepend)

      break

    case 'cron':
      setCronSessions(prepend)

      break
    case 'messaging': {
      const source = normalizeSessionSource(session.source)

      if (source && !$messagingSessions.get().some(s => s.id === session.id)) {
        adjustMessagingPlatformTotal(source, 1)
      }

      setMessagingSessions(prepend)

      break
    }
  }
}

function stripPinsFor(rows: SessionInfo[]): string[] {
  const before = $pinnedSessionIds.get()
  const doomed = new Set<string>()

  for (const session of rows) {
    doomed.add(session.id)
    doomed.add(sessionPinId(session))
  }

  const next = before.filter(id => !doomed.has(id))

  if (next.length !== before.length) {
    $pinnedSessionIds.set(next)
  }

  return before
}

function restorePinsFor(session: SessionInfo, pinnedBefore: string[]) {
  const pinId = sessionPinId(session)
  const lost = pinnedBefore.filter(id => id === session.id || id === pinId)

  if (!lost.length) {
    return
  }

  const current = $pinnedSessionIds.get()
  const missing = lost.filter(id => !current.includes(id))

  if (missing.length) {
    $pinnedSessionIds.set([...current, ...missing])
  }
}

export interface HiddenSessionRow {
  session: SessionInfo
  slice: SessionSliceName
  /** Put the row back where it came from, with that slice's totals re-adjusted. */
  undo: () => void
}

/** Single-row optimistic hide for the legacy per-session actions: removes the
 * row from whichever slice it lives in and keeps that slice's totals honest.
 * Replaces the old "filter $sessions + decrement the agent total" pattern,
 * which skewed counts when the row was actually a messaging/cron row and never
 * touched the per-profile totals the scoped Load-more footer reads. */
export function hideStoredSession(sessionId: string): HiddenSessionRow | null {
  const located = locateStoredSession(sessionId)

  if (!located) {
    return null
  }

  removeRowFromSlice(located)

  return {
    session: located.session,
    slice: located.slice,
    undo: () => insertRowIntoSlice(located.session, located.slice)
  }
}

/** Mirror a just-archived row into the Archived section's slice. */
export function recordSessionArchived(session: SessionInfo) {
  insertRowIntoSlice({ ...session, archived: true }, 'archived')
}

/** Rollback half of {@link recordSessionArchived}. */
export function dropArchivedSessionRow(sessionId: string) {
  const row = $archivedSessions.get().find(s => s.id === sessionId)

  if (row) {
    removeRowFromSlice({ session: row, slice: 'archived' })
  }
}

export interface BulkSessionResult {
  ok: SessionInfo[]
  failed: SessionInfo[]
}

interface BulkOptions {
  /** Runs after the optimistic hide, before the API calls settle — the hook
   * layer uses it to tear down the open chat when it's part of the set. */
  onAfterHide?: (targets: SessionInfo[]) => Promise<void> | void
  /** Suppress this module's toasts (single-row legacy paths own their copy). */
  silent?: boolean
}

function settleOutcome(
  results: PromiseSettledResult<unknown>[],
  targets: LocatedSessionRow[],
  acceptError: (err: unknown) => boolean = () => false
): { ok: LocatedSessionRow[]; failed: Array<{ target: LocatedSessionRow; reason: unknown }> } {
  const ok: LocatedSessionRow[] = []
  const failed: Array<{ reason: unknown; target: LocatedSessionRow }> = []

  results.forEach((result, index) => {
    const target = targets[index]

    if (result.status === 'fulfilled' || acceptError(result.reason)) {
      ok.push(target)
    } else {
      failed.push({ reason: result.reason, target })
    }
  })

  return { failed, ok }
}

/** Soft-archive a set of stored sessions (any slice except Archived itself).
 * Optimistic: rows leave their sections — and join the Archived section —
 * immediately; failures roll back surgically per row. */
export async function archiveStoredSessions(sessionIds: string[], options: BulkOptions = {}): Promise<BulkSessionResult> {
  const targets = [...new Set(sessionIds)]
    .map(locateStoredSession)
    .filter((row): row is LocatedSessionRow => row !== null && row.slice !== 'archived')

  if (!targets.length) {
    return { failed: [], ok: [] }
  }

  const pinnedBefore = stripPinsFor(targets.map(t => t.session))

  for (const target of targets) {
    removeRowFromSlice(target)
    insertRowIntoSlice({ ...target.session, archived: true }, 'archived')
  }

  await options.onAfterHide?.(targets.map(t => t.session))

  const results = await Promise.allSettled(
    targets.map(t => setSessionArchived(t.session.id, true, t.session.profile))
  )

  const { failed, ok } = settleOutcome(results, targets)

  for (const { target } of failed) {
    // Undo: pull the optimistic Archived-section row back out, return the row
    // to its home slice, and re-add any pin the strip removed.
    removeRowFromSlice({ session: { ...target.session, archived: true }, slice: 'archived' })
    insertRowIntoSlice(target.session, target.slice)
    restorePinsFor(target.session, pinnedBefore)
  }

  if (!options.silent) {
    if (ok.length) {
      notify({ durationMs: 2_500, kind: 'success', message: translateNow('sidebar.bulk.archivedToast', ok.length) })
    }

    if (failed.length) {
      notifyError(failed[0].reason, translateNow('sidebar.bulk.archiveFailed', failed.length))
    }
  }

  return { failed: failed.map(f => f.target.session), ok: ok.map(t => t.session) }
}

/** Unarchive a set of Archived-section rows back into the section their source
 * belongs to (recents, a messaging platform, or cron). */
export async function restoreArchivedSessions(sessionIds: string[], options: BulkOptions = {}): Promise<BulkSessionResult> {
  const targets = [...new Set(sessionIds)]
    .map(locateStoredSession)
    .filter((row): row is LocatedSessionRow => row !== null && row.slice === 'archived')

  if (!targets.length) {
    return { failed: [], ok: [] }
  }

  for (const target of targets) {
    removeRowFromSlice(target)
    const restored: SessionInfo = { ...target.session, archived: false }
    insertRowIntoSlice(restored, homeSliceFor(restored))
    // A restored row's recency usually predates the loaded page, so shield it
    // from the next refresh's merge eviction while the totals catch up.
    shieldSessionFromMerge(restored.id)
  }

  const results = await Promise.allSettled(
    targets.map(t => setSessionArchived(t.session.id, false, t.session.profile))
  )

  const { failed, ok } = settleOutcome(results, targets)

  for (const { target } of failed) {
    const restored: SessionInfo = { ...target.session, archived: false }
    removeRowFromSlice({ session: restored, slice: homeSliceFor(restored) })
    insertRowIntoSlice(target.session, 'archived')
  }

  if (!options.silent) {
    if (ok.length) {
      notify({ durationMs: 2_500, kind: 'success', message: translateNow('sidebar.bulk.restoredToast', ok.length) })
    }

    if (failed.length) {
      notifyError(failed[0].reason, translateNow('sidebar.bulk.restoreFailed', failed.length))
    }
  }

  return { failed: failed.map(f => f.target.session), ok: ok.map(t => t.session) }
}

/** Permanently delete a set of stored sessions from any slice (Archived
 * included). 404 "session not found" counts as success — the row is already
 * gone server-side, which satisfies the user's intent. */
export async function deleteStoredSessions(sessionIds: string[], options: BulkOptions = {}): Promise<BulkSessionResult> {
  const targets = [...new Set(sessionIds)]
    .map(locateStoredSession)
    .filter((row): row is LocatedSessionRow => row !== null)

  if (!targets.length) {
    return { failed: [], ok: [] }
  }

  const pinnedBefore = stripPinsFor(targets.map(t => t.session))

  for (const target of targets) {
    removeRowFromSlice(target)
  }

  await options.onAfterHide?.(targets.map(t => t.session))

  const results = await Promise.allSettled(targets.map(t => deleteSession(t.session.id, t.session.profile)))

  const { failed, ok } = settleOutcome(results, targets, isSessionGoneError)

  for (const target of ok) {
    clearQueuedPrompts(target.session.id)
  }

  for (const { target } of failed) {
    insertRowIntoSlice(target.session, target.slice)
    restorePinsFor(target.session, pinnedBefore)
  }

  if (!options.silent) {
    if (ok.length) {
      notify({ durationMs: 2_500, kind: 'success', message: translateNow('sidebar.bulk.deletedToast', ok.length) })
    }

    if (failed.length) {
      notifyError(failed[0].reason, translateNow('sidebar.bulk.deleteFailed', failed.length))
    }
  }

  return { failed: failed.map(f => f.target.session), ok: ok.map(t => t.session) }
}
