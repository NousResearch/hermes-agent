// Post-connect quiet window for native (OS) notifications.
//
// A socket opening replays state that already existed: a session parked on an
// approval re-emits its request so the UI can render the prompt. Those are not
// things that just happened, so launching Hermes — or any reconnect, profile
// switch, or gateway-mode apply — would otherwise fire an OS notification for a
// prompt the user has known about for an hour. The in-app surfaces (sidebar
// row, inline approval bar) still show the prompt immediately; only the OS
// notification is held.
//
// Lives in its own leaf module so `store/gateway` (which marks the baseline)
// and `store/native-notifications` (which reads it) don't import each other.

const SEED_QUIET_MS = 4000

let quietUntil = 0

/** Called on every gateway `open`. Opens the quiet window. */
export function markNativeNotifyBaseline(): void {
  quietUntil = Date.now() + SEED_QUIET_MS
}

/** True while replayed post-connect state should not raise an OS notification. */
export function withinNativeNotifyBaseline(): boolean {
  return Date.now() < quietUntil
}

export function __resetNativeNotifyBaselineForTests(): void {
  quietUntil = 0
}
