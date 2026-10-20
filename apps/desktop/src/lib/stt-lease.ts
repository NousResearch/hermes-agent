import { type ResolvedOwner, setSttLease } from '@/hermes'

// The desktop's voice-input sessions — push-to-talk dictation and the voice
// conversation loop — are the user telling us STT is about to be needed (or no
// longer is). The backend turns that into engine lifecycle: acquiring a lease
// pre-loads the local faster-whisper model (first-use download + load) so the
// transcription request starts hot instead of paying the cold cost inside its
// timeout (issue #105955); releasing drops the refcount but keeps the shared
// model resident for the gateway/CLI surfaces.
//
// This module is the renderer's single choke point for that signal. It
// serializes per lease so a fast on→off→on can't be reordered on the wire,
// coalesces concurrent identical acquires (recorder and conversation loop
// observe the same mic session), and never surfaces failures — warm-up is an
// optimization; recording must not depend on it.
//
// Three invariants the backend's lease semantics force (they diverge from the
// TTS mirror this was drawn from — the STT engine does NOT pin the model while
// a lease is held):
//
// 1. Registration ≠ readiness. The idle-unload watcher can evict the model
//    between two listening starts (a long reply, a mute), so every acquire is
//    sent: the backend re-warm is a cache hit when the model is resident and a
//    reload when it is not. No client-side timestamp stands in for residency.
// 2. Readiness is carried into transcription. The promise returned for an
//    acquire settles only when that acquire has settled on the wire — the
//    hooks use it as the barrier before submitting audio, so the transcription
//    request's deadline measures decoding, not the loader's cold start.
// 3. A lease belongs to the owner that acquired it. Callers resolve the owner
//    once per voice operation and pass the same value to acquire, transcribe
//    and release; the queue and coalescing identity include it.

// Per-renderer id so two windows recording at once hold DISTINCT leases —
// window A finishing must not release the engine window B is still using.
const RENDERER_ID = Math.random().toString(36).slice(2, 10)

export const VOICE_INPUT_LEASE = `desktop:voice-input:${RENDERER_ID}`

function keyFor(lease: string, owner: ResolvedOwner): string {
  return `${lease}::${owner.connectionId ?? ''}::${owner.profile ?? ''}`
}

/** Latest requested state per (lease, owner): true = acquire, false = release. */
const intent = new Map<string, boolean>()

/** The last queued operation per (lease, owner), until it settles. */
const tails = new Map<string, { active: boolean; promise: Promise<void> }>()

/**
 * Bring the backend's view of `lease` in line with `active`, for `owner`.
 *
 * An acquire coalesces only onto a queued acquire that is still the latest
 * intent — that acquire's settle IS this caller's readiness. Anything queued
 * behind a release is a new acquire. Releasing a lease that was never acquired
 * (or is already released) sends nothing.
 */
export function syncSttLease(lease: string, active: boolean, owner: ResolvedOwner): Promise<void> {
  const key = keyFor(lease, owner)
  const tail = tails.get(key)

  if (!active && intent.get(key) !== true) {
    return tail?.promise ?? Promise.resolve()
  }

  if (active && tail?.active && intent.get(key) === true) {
    return tail.promise
  }

  intent.set(key, active)

  const promise: Promise<void> = (tail?.promise ?? Promise.resolve())
    .then(async () => {
      // Latest intent wins: flipped again while queued → the newer call sends
      // its own state and this one has nothing to say.
      if (intent.get(key) === active) {
        await setSttLease(lease, active, owner)
      }
    })
    .catch(() => {
      // Backend not up yet / older backend without the endpoint: forget what
      // we "sent" so the next flip retries honestly.
      if (intent.get(key) === active) {
        intent.delete(key)
      }
    })
    .finally(() => {
      if (tails.get(key)?.promise === promise) {
        tails.delete(key)
      }
    })

  tails.set(key, { active, promise })

  return promise
}

/** Test seam. */
export function resetSttLeasesForTests() {
  intent.clear()
  tails.clear()
}

// How long one take may sit in "transcribing" waiting on the mic-open
// warm-up. The lease request's own budget is 180 s (a cold first-use model
// download legitimately takes minutes), so an acquire against a backend whose
// model load is wedged under the shared loader lock never answers for the
// full budget — and every later request queues behind it with no log and no
// error, so the take is silently swallowed until the user kills the app
// (#137318). Warm-up is an optimization: past this budget the caller abandons
// the take with a visible "unavailable" notice instead of holding it hostage.
export const STT_WARMUP_BUDGET_MS = 30_000

/**
 * Race one warm-up promise against the take budget. Resolves `'ready'` when
 * the warm-up settles (the normal path — a cache hit is instant) or
 * `'unavailable'` when the budget elapsed first. The warm-up itself is never
 * cancelled: it keeps settling in the background, so the next take on a
 * recovered backend starts hot.
 */
export function awaitSttWarmup(warmup: Promise<void>): Promise<'ready' | 'unavailable'> {
  let timer: ReturnType<typeof setTimeout> | undefined

  const budget = new Promise<'unavailable'>(resolve => {
    timer = setTimeout(() => resolve('unavailable'), STT_WARMUP_BUDGET_MS)
  })

  return Promise.race<'ready' | 'unavailable'>([warmup.then((): 'ready' => 'ready'), budget]).finally(() =>
    clearTimeout(timer)
  )
}
