import { type SessionInfo, setSessionArchived } from '@/hermes'
import { $pinnedSessionIds } from '@/store/layout'
import { clearNotifications, notify, notifyError } from '@/store/notifications'
import { untombstoneSessions } from '@/store/session-removal'
import { $archivedSessions } from '@/store/sidebar-archive'

import { type ListedSessionSlice, restoreListedSession, sessionMatchesStoredId } from './utils'

interface ArchiveUndoCopy {
  archived: string
  archiveUndo: string
  restored: string
  unarchiveFailed: string
}

export interface ArchivedRow {
  /** Every id the archive tombstoned (stored id, row id, lineage root). */
  ids: (null | string | undefined)[]
  /** The pin id the archive dropped, with its position, or null when it was not pinned. */
  pin: { id: string; index: number } | null
  profile?: string
  row?: SessionInfo
  slice?: ListedSessionSlice
  storedSessionId: string
}

/** Where `storedSessionId`'s pin sat before the archive dropped it, or null. */
export function archivedPin(previousPinned: string[], storedSessionId: string, pinId: string) {
  const index = previousPinned.findIndex(id => id === storedSessionId || id === pinId)

  return index < 0 ? null : { id: previousPinned[index], index }
}

/** Puts an archived sidebar row back where it was: the flag, the row, its pin. */
async function undoArchive(archived: ArchivedRow, copy: ArchiveUndoCopy): Promise<void> {
  clearNotifications()

  try {
    await setSessionArchived(archived.storedSessionId, false, archived.profile)
    untombstoneSessions(archived.ids)

    if (archived.row) {
      restoreListedSession({ ...archived.row, archived: false }, archived.slice)
    }

    const pins = $pinnedSessionIds.get()

    if (archived.pin && !pins.includes(archived.pin.id)) {
      $pinnedSessionIds.set([...pins.slice(0, archived.pin.index), archived.pin.id, ...pins.slice(archived.pin.index)])
    }

    notify({ durationMs: 2_000, kind: 'success', message: copy.restored })
  } catch (err) {
    notifyError(err, copy.unarchiveFailed)
  }
}

/** The "Archived" toast, with an Undo that reverses the archive in place. */
export function notifyArchivedWithUndo(archived: ArchivedRow, copy: ArchiveUndoCopy): void {
  notify({
    action: { label: copy.archiveUndo, onClick: () => void undoArchive(archived, copy) },
    durationMs: 6_000,
    kind: 'success',
    message: copy.archived
  })
}

/**
 * Restore a row from the sidebar's Archived view. Mirrors the Settings → Archived Chats
 * restore (sessions-settings.tsx): flip the persisted flag off, drop the archived-view row,
 * and resurface the session in the sidebar without waiting for a full refresh.
 */
export async function unarchiveListedSession(storedSessionId: string, copy: ArchiveUndoCopy): Promise<void> {
  clearNotifications()

  const archived = $archivedSessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))
  const profile = archived?.profile?.trim() || undefined

  try {
    await setSessionArchived(storedSessionId, false, profile)

    // Drop the archived-view row first so the view reflects the restore even when the
    // session cannot be re-listed below (e.g. a profile the current query does not cover).
    $archivedSessions.set($archivedSessions.get().filter(session => !sessionMatchesStoredId(session, storedSessionId)))

    if (archived) {
      // Lift any optimistic eviction, and re-list through the slice router so a
      // messaging/cron row lands back in its own list, not the sessions one.
      untombstoneSessions([storedSessionId, archived._lineage_root_id])
      restoreListedSession({ ...archived, archived: false })
    }

    notify({ durationMs: 2_000, kind: 'success', message: copy.restored })
  } catch (err) {
    notifyError(err, copy.unarchiveFailed)
  }
}
