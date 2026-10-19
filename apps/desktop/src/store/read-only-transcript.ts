/**
 * Read-only stored-transcript resume — the no-owner recovery path (#94724).
 *
 * The fail-closed owner ladder is correct: a session-scoped RPC whose owner
 * nobody can name must NOT ride the ambient socket. But "fail closed" must
 * not mean "locked out of your own history": the transcripts of legacy
 * unowned rows sit fully intact in state.db, reachable over the id-only REST
 * read that needs no live-session routing at all.
 *
 * `resumeWithStoredTranscriptFallback` wraps a live `session.resume` dispatch
 * with exactly that recovery: when — and only when — the resume fails with
 * `SessionOwnerResolutionError` (owner genuinely unresolvable under registry
 * topology), it opens the stored transcript read-only instead of dead-ending.
 * Every other failure keeps its existing semantics. Rows whose owner IS
 * resolvable never take this path, and a later successful live resume (e.g.
 * after the single-match owner backfill stamps the row) clears the flag.
 *
 * A second, unrelated reason a stored transcript is read-only lives here too:
 * a cron run the scheduler no longer owns but never closed (#88443). It keeps
 * its own verdict (see `$cronRunReadOnlyVerdicts` below) so a live resume
 * cannot clear it, and `isStoredTranscriptReadOnly` reports both, so every
 * write path gets one answer.
 */
import { atom } from 'nanostores'

import type { SessionOwnerResolutionError } from './session-owner-resolution'
import { isSessionOwnerResolutionError } from './session-owner-resolution'

/** Stored session ids currently open as read-only stored transcripts. The
 *  composer/submit surfaces consult this to refuse writes into a session
 *  that has no routable live runtime. */
export const $readOnlyStoredTranscripts = atom<ReadonlySet<string>>(new Set())

export function markStoredTranscriptReadOnly(storedSessionId: string): void {
  const id = storedSessionId.trim()

  if (!id || $readOnlyStoredTranscripts.get().has(id)) {
    return
  }

  $readOnlyStoredTranscripts.set(new Set([...$readOnlyStoredTranscripts.get(), id]))
}

export function clearStoredTranscriptReadOnly(storedSessionId: string): void {
  const id = storedSessionId.trim()

  if (!id || !$readOnlyStoredTranscripts.get().has(id)) {
    return
  }

  const next = new Set($readOnlyStoredTranscripts.get())

  next.delete(id)
  $readOnlyStoredTranscripts.set(next)
}

export function isStoredTranscriptReadOnly(storedSessionId: null | string | undefined): boolean {
  if (!storedSessionId) {
    return false
  }

  const id = storedSessionId.trim()

  return $readOnlyStoredTranscripts.get().has(id) || isCronRunReadOnly(id)
}

/**
 * Cron run write gate (#88443), one VERDICT per run session the desktop has
 * evaluated — never a permanent latch.
 *
 * A cron run is an autonomous scheduled execution. A run whose `ended_at` is
 * NULL while the scheduler no longer owns it is a ZOMBIE (watchdog kill,
 * crash, connection drop): resuming it as a desktop chat routes the user's
 * messages into a dead `source='cron'` session. A zombie's verdict is `true`
 * (read-only); a live or closed run's is `false`.
 *
 * The verdict is RE-EVALUATED whenever fresher run data arrives — every Cron
 * surface poll and, authoritatively, right before any send
 * (`refreshCronRunWriteGate`) — so a run that looked idle during a long tool
 * call, then ticked or closed, becomes writable again. It is deliberately
 * separate from the owner-recovery set above: a successful live resume of a
 * zombie must not clear it.
 */
export const $cronRunReadOnlyVerdicts = atom<ReadonlyMap<string, boolean>>(new Map())

export function recordCronRunVerdict(storedSessionId: string, readOnly: boolean): void {
  const id = storedSessionId.trim()
  const current = $cronRunReadOnlyVerdicts.get()

  if (!id || current.get(id) === readOnly) {
    return
  }

  $cronRunReadOnlyVerdicts.set(new Map(current).set(id, readOnly))
}

export function hasCronRunVerdict(storedSessionId: null | string | undefined): boolean {
  return Boolean(storedSessionId && $cronRunReadOnlyVerdicts.get().has(storedSessionId.trim()))
}

export function isCronRunReadOnly(storedSessionId: null | string | undefined): boolean {
  return Boolean(storedSessionId && $cronRunReadOnlyVerdicts.get().get(storedSessionId.trim()) === true)
}

/** Synthetic runtime-id namespace for read-only tiles: a stored transcript
 *  opened without a live runtime still needs a state-cache key, and this
 *  prefix guarantees it can never collide with (or be mistaken for) a real
 *  gateway runtime id. */
export const READ_ONLY_RUNTIME_ID_PREFIX = 'read-only:'

export function readOnlyRuntimeIdFor(storedSessionId: string): string {
  return `${READ_ONLY_RUNTIME_ID_PREFIX}${storedSessionId}`
}

export function isReadOnlyRuntimeId(runtimeId: null | string | undefined): boolean {
  return Boolean(runtimeId?.startsWith(READ_ONLY_RUNTIME_ID_PREFIX))
}

export type StoredTranscriptResumeOutcome<TResumed, TTranscript> =
  | { mode: 'live'; resumed: TResumed }
  | { error: SessionOwnerResolutionError; mode: 'read-only'; transcript: TTranscript }

/**
 * Dispatch a live resume, recovering into a read-only stored-transcript open
 * when the owner is unresolvable. `fetchStoredTranscript` must be an id-only
 * stored read (REST `/api/sessions/:id/messages`) that performs NO gateway
 * routing — that is the whole point of the recovery path.
 *
 * When even the stored read fails, the ORIGINAL owner-resolution error is
 * rethrown: the caller's existing error UX (retry latch, stranded screen)
 * stays authoritative and no misleading transport error replaces the real
 * diagnosis.
 */
export async function resumeWithStoredTranscriptFallback<TResumed, TTranscript>(
  storedSessionId: string,
  resume: () => Promise<TResumed>,
  fetchStoredTranscript: () => Promise<TTranscript>
): Promise<StoredTranscriptResumeOutcome<TResumed, TTranscript>> {
  try {
    const resumed = await resume()

    // A live resume proves the owner is routable again (the backfill stamped
    // the row, or a topology change resolved it) — drop the read-only latch.
    clearStoredTranscriptReadOnly(storedSessionId)

    return { mode: 'live', resumed }
  } catch (error) {
    if (!isSessionOwnerResolutionError(error)) {
      throw error
    }

    let transcript: TTranscript

    try {
      transcript = await fetchStoredTranscript()
    } catch {
      throw error
    }

    markStoredTranscriptReadOnly(storedSessionId)

    return { error, mode: 'read-only', transcript }
  }
}
