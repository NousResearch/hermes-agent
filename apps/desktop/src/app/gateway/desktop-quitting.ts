/** The abort reason local-backend-lifecycle throws once quit teardown has sealed the process. */
const DESKTOP_QUITTING = 'Hermes Desktop is quitting.'

/**
 * True when an IPC rejection is the quit latch, not a failed boot.
 * Painting "Hermes couldn't start" for it invites Repair on a process that
 * can no longer start a backend.
 */
export function isIntentionalDesktopQuitError(error: unknown): boolean {
  if (typeof error === 'object' && error !== null) {
    if ('intentionalTeardown' in error && (error as { intentionalTeardown?: boolean }).intentionalTeardown) {
      return true
    }

    if ('error' in error && typeof (error as { error?: unknown }).error === 'string') {
      if (((error as { error: string }).error).includes(DESKTOP_QUITTING)) {
        return true
      }
    }
  }

  const message = error instanceof Error ? error.message : String(error ?? '')

  return message.includes(DESKTOP_QUITTING)
}
