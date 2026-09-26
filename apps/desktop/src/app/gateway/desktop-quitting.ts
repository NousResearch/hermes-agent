/** The abort reason local-backend-lifecycle throws once quit teardown has sealed the process. */
const DESKTOP_QUITTING = 'Hermes Desktop is quitting.'

/**
 * True when an IPC rejection is the quit latch, not a failed boot.
 * Painting "Hermes couldn't start" for it invites Repair on a process that
 * can no longer start a backend.
 */
export function isIntentionalDesktopQuitError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? '')

  return message.includes(DESKTOP_QUITTING)
}
