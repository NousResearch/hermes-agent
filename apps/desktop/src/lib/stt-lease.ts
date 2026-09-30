import { setSttLease } from '@/hermes'

// The desktop's voice-input sessions — push-to-talk dictation and the voice
// conversation loop — are the user telling us STT is about to be needed (or no
// longer is). The backend turns that into engine lifecycle: acquiring a lease
// pre-loads the local faster-whisper model (first-use download + load) so the
// transcription request starts hot instead of paying the cold cost inside its
// timeout (issue #105955); releasing drops the refcount but keeps the shared
// model resident for the gateway/CLI surfaces.
//
// This module is the renderer's single choke point for that signal. It dedupes
// (recorder and conversation loop observe the same mic session), serializes
// per lease so a fast on→off→on can't be reordered on the wire, and never
// surfaces failures — warm-up is an optimization; recording must not depend
// on it.

// Per-renderer id so two windows recording at once hold DISTINCT leases —
// window A finishing must not release the engine window B is still using.
const RENDERER_ID = Math.random().toString(36).slice(2, 10)

export const VOICE_INPUT_LEASE = `desktop:voice-input:${RENDERER_ID}`

const sent = new Map<string, boolean>()
const inFlight = new Map<string, Promise<void>>()

/**
 * Bring the backend's view of `lease` in line with `active`. Idempotent: a
 * repeat of the last sent state is a no-op. The initial `false` (nothing was
 * ever acquired) is also skipped — releasing a lease we never held would only
 * churn the backend on app start.
 */
export function syncSttLease(lease: string, active: boolean): Promise<void> {
  const last = sent.get(lease)

  if (last === active || (last === undefined && !active)) {
    return inFlight.get(lease) ?? Promise.resolve()
  }

  sent.set(lease, active)

  const previous = inFlight.get(lease) ?? Promise.resolve()

  const next = previous
    .then(async () => {
      // Latest intent wins: if the lease flipped again while we were queued,
      // the newer call sends its own state and this one has nothing to say.
      if (sent.get(lease) !== active) {
        return
      }

      await setSttLease(lease, active)
    })
    .catch(() => {
      // Backend not up yet / older backend without the endpoint / warm-up
      // failure: forget what we "sent" so the next flip retries honestly.
      if (sent.get(lease) === active) {
        sent.delete(lease)
      }
    })
    .finally(() => {
      if (inFlight.get(lease) === next) {
        inFlight.delete(lease)
      }
    })

  inFlight.set(lease, next)

  return next
}

/** Test seam — forget every sent state. */
export function resetSttLeasesForTests() {
  sent.clear()
  inFlight.clear()
}
