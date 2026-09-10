/**
 * What a failed update check should say.
 *
 * The Electron bridge/backend already reports WHY a check failed (git's exit
 * code, the remote it contacted, the classified cause). Both update surfaces
 * used to drop that on the floor and print a fixed "we couldn't reach the
 * update server" line instead, which is a false diagnosis for everything that
 * is not actually a network failure — a misconfigured remote, a locked
 * `.git/*.lock`, a disabled credential prompt, a TLS-intercepting proxy. This
 * helper is the single place the detail is surfaced, so a new surface cannot
 * silently reintroduce the bare wording.
 */
export function updateFailureDetail(status: { message?: string } | null | undefined, fallback: string): string {
  const message = typeof status?.message === 'string' ? status.message.trim() : ''

  return message || fallback
}
