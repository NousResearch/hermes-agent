import { redactSecrets } from './ssh-connection'

/**
 * Format a backend stdout/stderr chunk for the local Desktop log ring.
 * Secrets that leak into process output must not persist in hermesLog /
 * desktopLogBuffer (or the on-disk desktop log flush).
 */
export function formatDesktopLogChunk(chunk: unknown): string {
  return redactSecrets(String(chunk || '')).trim()
}
