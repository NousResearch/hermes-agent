/**
 * The runtime-owner refusal for session writes, recognized as one condition in
 * both of its wire shapes:
 *
 * - The named, retryable 503 (`owner_unavailable: ...`) the session-write
 *   forward answers with when a live gateway owns the profile and cannot be
 *   reached to serve the write (its control plane still starting, draining,
 *   or unreachable).
 * - The bare `Exclusive maintenance refused` 409: the offline store's
 *   reservation refusal, which predates the named shape on older servers and
 *   survived the pre-fix race where discovery's snapshot read `absent` while
 *   the owner already held the lock.
 *
 * Both mean the same thing: a live owner holds the profile, the write is not
 * servable right now, and this is not a content verdict. Sync stores treat
 * it as a deferral (park and retry until the owner serves the write or the
 * deferral bound expires, see session-owner-deferred.ts); user-initiated
 * single actions let it fail visibly.
 */

const OWNER_UNAVAILABLE_MARKER = 'owner_unavailable:'

const EXCLUSIVE_MAINTENANCE_REFUSED = 'Exclusive maintenance refused'

/** True when `error` is the runtime-owner refusal in either wire shape. */
export function isOwnerUnavailableRejection(error: unknown): boolean {
  const message = String((error as { message?: unknown } | null)?.message ?? error ?? '')

  return message.includes(OWNER_UNAVAILABLE_MARKER) || message.includes(EXCLUSIVE_MAINTENANCE_REFUSED)
}
