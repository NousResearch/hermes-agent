// The `hermes:api` owner-unavailable seam, shared by the main-process handler
// and preload so both sides agree on one literal — the direct analogue of
// api-expected-404.ts.
//
// The runtime-owner refusal for session writes (a live gateway owns the
// profile; its control plane is starting or unreachable; the offline store's
// reservation refuses) is a designed, retryable backend verdict: the sync
// stores park the write (session-owner-deferred.ts) and retry until the
// owner serves it. Electron logs "Error occurred in handler for
// 'hermes:api'" with a full stack for every rejected invoke, with no opt-out
// — so the first refusal of every parked write at every boot printed a stack
// into the launching terminal (observed live: four per boot).
//
// The handler therefore RESOLVES with this sentinel for that one expected
// class, and preload rethrows it as an ordinary Error carrying the identical
// message. The renderer sees exactly the rejection it saw before — and
// classifies it via src/api/owner-unavailable.ts — so only Electron's
// logging is bypassed. Any other failure rejects as usual and still logs in
// full.

const HERMES_API_OWNER_UNAVAILABLE = '__hermesOwnerUnavailable__'

// True when `error` is the runtime-owner refusal in either wire shape: the
// named `owner_unavailable:` 503, or the bare `Exclusive maintenance
// refused` 409 that predates it on older servers.
function isOwnerUnavailableHttpError(error: unknown): boolean {
  const message = String((error as { message?: unknown } | null)?.message ?? error ?? '')

  return message.includes('owner_unavailable:') || message.includes('Exclusive maintenance refused')
}

// True when `value` is a handler-resolved owner-unavailable sentinel, not
// real response data. Kept narrow: a plain object whose ONLY key is the
// sentinel and whose value is a string, so a backend payload can't be
// mistaken for one.
function isOwnerUnavailableSentinel(value: unknown): value is Record<string, string> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }

  const keys = Object.keys(value as Record<string, unknown>)

  return (
    keys.length === 1 &&
    keys[0] === HERMES_API_OWNER_UNAVAILABLE &&
    typeof (value as any)[HERMES_API_OWNER_UNAVAILABLE] === 'string'
  )
}

// Restore the caller-visible contract: a sentinel becomes the rejection the
// renderer expects (and classifies); anything else passes through untouched.
function unwrapOwnerUnavailable(value: unknown): unknown {
  if (isOwnerUnavailableSentinel(value)) {
    throw new Error(value[HERMES_API_OWNER_UNAVAILABLE])
  }

  return value
}

export { HERMES_API_OWNER_UNAVAILABLE, isOwnerUnavailableHttpError, isOwnerUnavailableSentinel, unwrapOwnerUnavailable }
