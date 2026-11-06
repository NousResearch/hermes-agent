/**
 * The single pending review. Owns the promise the send middleware awaits, the in-flight model call, and
 * the state the dialog renders. Invariants:
 *
 *  - At most one review at a time; a second request is refused, never queued.
 *  - `autoApprove` (Automatic mode): a refinement that comes back with NOTHING
 *    flagged by the preservation and intent checks is approved here, by the
 *    store, and sent; any flag (blocking or review-only) or any failure stops
 *    and waits for the user instead.
 *  - Without `autoApprove` (Approval-First) nothing here resolves to a send:
 *    only `decide` with the user's approve or original does.
 *  - Every review ends in exactly one decision. Approve is accepted only for
 *    the review that is open, only once its refinement is ready, and only
 *    when the preservation check allows it — re-checked HERE, so no UI path
 *    can approve around it.
 *  - Fail closed: no mounted dialog host, a host that never renders the
 *    review, a crash in the dialog, or a plugin unload all resolve to
 *    `cancel` (the caller restores the draft). Nothing here can resolve to a
 *    send without a user action.
 *  - A late result (after Cancel, a retry, or a newer review) is dropped by
 *    review id + its own AbortController, never painted over newer state.
 */

import { atom, type ReadableAtom } from 'nanostores'

import { canApprove, checkPreservation } from './preservation'
import type {
  RefinementAnalysis,
  RefinementDepth,
  RefinementFailure,
  RefineResult,
  ReviewCancelReason,
  ReviewDecision
} from './types'

export interface ReviewState {
  id: number
  /** The composer the drawer opens in (`main` / `tile:<id>`). */
  target: string
  /** Automatic mode: send without a click when nothing is flagged. */
  autoApprove: boolean
  /** Why an automatic send stopped for the user (null = nothing stopped it). */
  attention: 'failed' | 'issues' | null
  original: string
  /** The chat excerpt the refiner saw ('' when none), for the intent check. */
  context: string
  attachmentLabels: readonly string[]
  phase: 'failed' | 'ready' | 'refining'
  refined: null | string
  /** The model returned the draft unchanged. */
  unchanged: boolean
  failure: null | RefinementFailure
  detail: null | string
  /** What the refiner added and why (absent when the model skipped it). */
  analysis: null | RefinementAnalysis
  /** The depth the refiner picked for an Auto depth request (null otherwise). */
  chosenDepth: null | RefinementDepth
}

export interface ReviewRequest {
  target: string
  autoApprove?: boolean
  original: string
  /** The chat excerpt sent with the draft, if any. */
  context?: string
  attachmentLabels: readonly string[]
  /** `attempt` is 1 for the first try, higher for each "Try again". */
  run: (signal: AbortSignal, attempt: number) => Promise<RefineResult>
}

/** How long a mounted host gets to show a new review before we give up and
 *  cancel (a host that is mounted but cannot render must not strand a send). */
export const RENDER_ACK_MS = 2_000

interface Active {
  id: number
  request: ReviewRequest
  controller: AbortController | null
  attempt: number
  resolve: (decision: ReviewDecision) => void
  ackTimer: ReturnType<typeof setTimeout> | null
}

export interface ReviewStore {
  $review: ReadableAtom<null | ReviewState>
  /** A dialog host is mounted; returns its unregister. */
  registerHost: () => () => void
  hasHost: () => boolean
  request: (request: ReviewRequest) => Promise<ReviewDecision>
  /** The host painted review `id`. */
  acknowledge: (id: number) => void
  retry: (id: number) => void
  /** Returns false (and changes nothing) when the decision is not allowed. */
  decide: (id: number, decision: ApproveInput | { kind: 'original' } | { kind: 'cancel' }) => boolean
  cancelActive: (reason: ReviewCancelReason) => void
}

export interface ApproveInput {
  kind: 'approve'
  text: string
  acknowledged: boolean
}

export function createReviewStore(options: { renderAckMs?: number } = {}): ReviewStore {
  const renderAckMs = options.renderAckMs ?? RENDER_ACK_MS
  const $review = atom<null | ReviewState>(null)
  const hosts = new Set<symbol>()
  let seq = 0
  let active: Active | null = null

  const finish = (id: number, decision: ReviewDecision) => {
    if (!active || active.id !== id) {
      return
    }

    const done = active
    active = null
    done.controller?.abort()

    if (done.ackTimer) {
      clearTimeout(done.ackTimer)
    }

    $review.set(null)
    done.resolve(decision)
  }

  const start = (entry: Active) => {
    entry.controller?.abort()
    const controller = new AbortController()
    entry.controller = controller
    const { id } = entry

    const settle = (result: RefineResult) => {
      // Late or superseded: a newer attempt, a retry, or a closed review owns the state now.
      if (active?.id !== id || active.controller !== controller || controller.signal.aborted) {
        return
      }

      const current = $review.get()

      if (!current || current.id !== id) {
        return
      }

      if (!result.ok) {
        $review.set({
          ...current,
          phase: 'failed',
          refined: null,
          unchanged: false,
          analysis: null,
          chosenDepth: null,
          failure: result.failure,
          detail: result.detail ?? null,
          attention: current.autoApprove ? 'failed' : null
        })

        return
      }

      // Automatic mode sends only a refinement nothing is flagged on.
      if (
        current.autoApprove &&
        result.text.trim() &&
        checkPreservation(current.original, result.text, current.context).issues.length === 0
      ) {
        finish(id, { kind: 'approve', text: result.text.trim() })

        return
      }

      $review.set({
        ...current,
        phase: 'ready',
        refined: result.text,
        unchanged: result.unchanged,
        analysis: result.analysis,
        chosenDepth: result.chosenDepth ?? null,
        failure: null,
        detail: null,
        attention: current.autoApprove ? 'issues' : null
      })
    }

    let pending: Promise<RefineResult>

    try {
      entry.attempt += 1
      pending = Promise.resolve(entry.request.run(controller.signal, entry.attempt))
    } catch (error) {
      pending = Promise.reject(error)
    }

    pending.then(settle, (error: unknown) =>
      settle({ ok: false, failure: 'failed', detail: error instanceof Error ? error.message : String(error) })
    )
  }

  return {
    $review,

    registerHost: () => {
      const token = Symbol('prompt-refinement-host')
      hosts.add(token)

      return () => {
        hosts.delete(token)

        if (hosts.size === 0 && active) {
          finish(active.id, { kind: 'cancel', reason: 'no-host' })
        }
      }
    },

    hasHost: () => hosts.size > 0,

    request: request => {
      if (active) {
        return Promise.resolve({ kind: 'cancel', reason: 'busy' })
      }

      if (hosts.size === 0) {
        return Promise.resolve({ kind: 'cancel', reason: 'no-host' })
      }

      const id = ++seq

      return new Promise<ReviewDecision>(resolve => {
        const entry: Active = { id, request, controller: null, attempt: 0, resolve, ackTimer: null }
        active = entry
        entry.ackTimer = setTimeout(() => finish(id, { kind: 'cancel', reason: 'no-host' }), renderAckMs)
        $review.set({
          id,
          target: request.target,
          autoApprove: Boolean(request.autoApprove),
          attention: null,
          analysis: null,
          chosenDepth: null,
          original: request.original,
          context: request.context ?? '',
          attachmentLabels: request.attachmentLabels,
          phase: 'refining',
          refined: null,
          unchanged: false,
          failure: null,
          detail: null
        })
        start(entry)
      })
    },

    acknowledge: id => {
      if (active?.id === id && active.ackTimer) {
        clearTimeout(active.ackTimer)
        active.ackTimer = null
      }
    },

    retry: id => {
      const current = $review.get()

      if (!active || active.id !== id || !current || current.phase === 'refining') {
        return
      }

      $review.set({
        ...current,
        phase: 'refining',
        refined: null,
        unchanged: false,
        analysis: null,
        chosenDepth: null,
        failure: null,
        detail: null,
        attention: null
      })
      start(active)
    },

    decide: (id, decision) => {
      const current = $review.get()

      if (!active || active.id !== id || !current || current.id !== id) {
        return false
      }

      if (decision.kind === 'cancel') {
        finish(id, { kind: 'cancel', reason: 'user' })

        return true
      }

      if (decision.kind === 'original') {
        finish(id, { kind: 'original' })

        return true
      }

      const text = decision.text.trim()

      if (
        current.phase !== 'ready' ||
        !text ||
        !canApprove(checkPreservation(current.original, text, current.context), decision.acknowledged)
      ) {
        return false
      }

      finish(id, { kind: 'approve', text })

      return true
    },

    cancelActive: reason => {
      if (active) {
        finish(active.id, { kind: 'cancel', reason })
      }
    }
  }
}
