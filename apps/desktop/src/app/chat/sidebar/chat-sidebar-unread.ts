import { notifyError } from '@/store/notifications'
import { $sessions } from '@/store/session'
import { markSessionUnread } from '@/store/session-unread-remote'

/**
 * Toggle the persisted read-state watermark for one row. The row's own `unread`
 * prop mirrors what the dot paints; flip it and let the backend become the
 * truth (optimistic update + rollback inside markSessionUnread).
 */
export function toggleSessionUnread(storedId: string, unreadFailedMessage: string): void {
  const row = $sessions.get().find(candidate => candidate.id === storedId)

  if (!row) {
    return
  }

  markSessionUnread(storedId, row.unread !== true).catch(err => notifyError(err, unreadFailedMessage))
}
