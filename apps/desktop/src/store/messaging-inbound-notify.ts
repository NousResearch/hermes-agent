/**
 * Native OS notification for inbound messaging-platform messages (#56187).
 *
 * The turn an inbound Telegram/Discord/WhatsApp message starts runs on the
 * gateway process, not in this renderer — the desktop only learns about it
 * when `sessions.changed` (websocket push) triggers a messaging-list refresh
 * and new rows land in `$messagingSessions`. A row whose `message_count` rose
 * since we last saw it means new persisted messages; the row's preview then
 * names the last thing said. Seed-on-first-sight (same rule as the unread
 * watermarks in `session-unread.ts`) keeps a fresh connect / restart /
 * profile switch from replaying history as a notification storm.
 *
 * Gated twice: `desktop.notify_incoming_messages` (config.yaml, off by
 * default — no unsolicited OS notifications) AND the 'message' per-kind
 * preference in `$nativeNotifyPrefs` (Settings → Notifications). Fires only
 * while the window is unfocused (`shouldFire`), and never for the session
 * currently on screen — its unread affordances already cover it.
 */

import { translateNow } from '@/i18n'

import { $desktopNotifyIncomingMessages } from './desktop-notify-incoming'
import { dispatchNativeNotification } from './native-notifications'
import { $messagingSessions, sessionPinId } from './session'
import { isBrowserWindow, isSecondaryWindow } from './windows'

/** durable lineage id → message_count at last observation. */
const lastCounts = new Map<string, number>()

/** Release observed counts (tests). */
export function resetInboundMessageCountsForTests(): void {
  lastCounts.clear()
}

function onMessagingRowsChange(): void {
  for (const row of $messagingSessions.get()) {
    if (!Number.isFinite(row.message_count)) {
      continue
    }

    const durableId = sessionPinId(row)
    const previous = lastCounts.get(durableId)

    // First sight seeds the baseline; only a rise after that counts as new.
    if (previous === undefined) {
      lastCounts.set(durableId, row.message_count)

      continue
    }

    if (row.message_count > previous) {
      lastCounts.set(durableId, row.message_count)

      // Config gate is checked before dispatch: the per-kind preference can
      // be toggled in Settings while the config key stays the master switch.
      if (!$desktopNotifyIncomingMessages.get()) {
        continue
      }

      dispatchNativeNotification({
        // The preview names the last thing said; the generic body covers a
        // session whose preview is empty. The title shown is always the named
        // one (NAMED_TITLE_KEYS in native-notifications.ts).
        body: row.preview?.trim() || translateNow('notifications.native.messageBody'),
        kind: 'message',
        sessionId: row.id,
        title: translateNow('notifications.native.messageTitle')
      })
    } else if (row.message_count < previous) {
      // Compression rotation or a reset re-projects the row with a lower
      // count: adopt it so the next inbound message diffs from truth.
      lastCounts.set(durableId, row.message_count)
    }
  }
}

// Same wiring rule as session-unread.ts: module-scope listeners, primary
// window only — a secondary window sees a sliver of the list and must not
// fire (or seed) against it. Loaded by contrib/wiring.tsx.
if (!isSecondaryWindow() && !isBrowserWindow()) {
  $messagingSessions.listen(onMessagingRowsChange)
}
