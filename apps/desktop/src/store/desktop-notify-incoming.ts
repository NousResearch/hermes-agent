/**
 * `desktop.notify_incoming_messages` — config gate for the native OS
 * notification fired when an inbound message arrives on a messaging-platform
 * session while the window is unfocused (#56187).
 *
 * Off by default: no unsolicited OS notifications, matching the default in
 * hermes_cli/config_defaults.py. The per-kind toggle in
 * `$nativeNotifyPrefs` (Settings → Notifications) must ALSO be on — both
 * gates are ANDed at dispatch time.
 */

import { atom } from 'nanostores'

export const $desktopNotifyIncomingMessages = atom<boolean>(false)

export function setDesktopNotifyIncomingMessagesFromConfig(value: unknown): void {
  $desktopNotifyIncomingMessages.set(value === true || value === 'true' || value === 1)
}
