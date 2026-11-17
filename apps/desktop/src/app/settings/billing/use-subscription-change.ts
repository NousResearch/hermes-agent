import { useQueryClient } from '@tanstack/react-query'
import { useRef, useState } from 'react'

import type { BillingRefusal } from './api'
import { useBillingApi } from './api'
import type { SubscriptionPreviewResponse } from './types'

export interface DowngradeTarget {
  tierId: string
  tierName: string
}

/**
 * The state machine for one downgrade attempt. Modeled as a discriminated union
 * rather than four independent nullables so impossible combinations (a preview AND a
 * refusal, a "ready" with no quote) simply cannot be represented, and the panel reads
 * exactly one `kind`.
 */
export type DowngradePhase =
  | { kind: 'previewFailed'; refusal: BillingRefusal }
  | { kind: 'previewing' }
  | { kind: 'ready'; preview: SubscriptionPreviewResponse }
  | { kind: 'scheduleFailed'; preview: SubscriptionPreviewResponse; refusal: BillingRefusal }
  | { kind: 'scheduling'; preview: SubscriptionPreviewResponse }

export interface ActiveDowngrade {
  phase: DowngradePhase
  target: DowngradeTarget
}

function invalidateBilling(queryClient: ReturnType<typeof useQueryClient>) {
  return Promise.all([
    queryClient.invalidateQueries({ queryKey: ['billing', 'state'] }),
    queryClient.invalidateQueries({ queryKey: ['billing', 'subscription'] })
  ])
}

/**
 * The in-app downgrade flow: preview (chargeless quote) → confirm → schedule at
 * period end. Typed refusals surface via the shared BillingRefusalInline (which
 * drives the step-up for insufficient_scope, exactly like the auto-reload save);
 * the caller retries by clicking the same button after verifying. On a scheduled
 * success it refetches billing + subscription and calls `onScheduled`.
 *
 * The api comes from `useBillingApi`, which DEV fixtures override with a simulated
 * implementation — so this flow has no fixture/simulation awareness of its own.
 */
export function useDowngradeFlow({ onScheduled }: { onScheduled: () => void }) {
  const api = useBillingApi()
  const queryClient = useQueryClient()
  const [active, setActive] = useState<ActiveDowngrade | null>(null)
  // Monotonic run id discards results from a superseded/cancelled attempt.
  const runIdRef = useRef(0)
  // Synchronous mutex: two clicks in the same tick both see active.busy === null
  // (React hasn't committed the 'schedule' state yet), so guard on a ref too — no
  // double schedule RPC. Cleared on every confirm() exit (below).
  const schedulingRef = useRef(false)

  const runPreview = async (target: DowngradeTarget, runId: number) => {
    const result = await api.previewSubscriptionChange(target.tierId)

    if (runIdRef.current !== runId) {
      return
    }

    setActive({
      phase: result.ok ? { kind: 'ready', preview: result.data } : { kind: 'previewFailed', refusal: result.refusal },
      target
    })
  }

  const begin = (target: DowngradeTarget) => {
    const runId = runIdRef.current + 1

    runIdRef.current = runId
    setActive({ phase: { kind: 'previewing' }, target })
    void runPreview(target, runId)
  }

  const retryPreview = () => {
    if (active) {
      begin(active.target)
    }
  }

  const confirm = async () => {
    // Only a quoted state (ready, or a failed schedule being retried) can commit.
    if (!active || schedulingRef.current || (active.phase.kind !== 'ready' && active.phase.kind !== 'scheduleFailed')) {
      return
    }

    schedulingRef.current = true
    const { target } = active
    const { preview } = active.phase
    const runId = runIdRef.current + 1

    runIdRef.current = runId
    setActive({ phase: { kind: 'scheduling', preview }, target })

    const result = await api.scheduleSubscriptionChange(target.tierId)
    schedulingRef.current = false

    if (runIdRef.current !== runId) {
      return
    }

    if (!result.ok) {
      // A refusal (e.g. insufficient_scope → step-up) leaves the panel open in
      // scheduleFailed, so the same button becomes a manual "Try again" AFTER the
      // user elevates. We deliberately do NOT auto-replay the mutation on step-up
      // success — this matches the auto-reload save's manual-retry pattern.
      setActive({ phase: { kind: 'scheduleFailed', preview, refusal: result.refusal }, target })

      return
    }

    await invalidateBilling(queryClient)
    setActive(null)
    onScheduled()
  }

  const cancel = () => {
    runIdRef.current += 1
    setActive(null)
  }

  // True only while the mutating RPC (schedule) is in flight — used to lock out
  // every other Downgrade tile + Back while one change is committing (the server
  // also 409s overlapping mutations per-org, so this is UI honesty, not the only
  // defense).
  const mutating = active?.phase.kind === 'scheduling'

  return { active, begin, cancel, confirm, mutating, retryPreview }
}

/**
 * The undo for a scheduled downgrade / cancellation: a chargeless
 * `subscription.resume` (no confirm step) that refetches on success. A refusal
 * (e.g. insufficient_scope → step-up) surfaces via `refusal`. The api (real or the
 * DEV-fixture simulation) comes from `useBillingApi`.
 */
export function useResumeFlow() {
  const api = useBillingApi()
  const queryClient = useQueryClient()
  const [busy, setBusy] = useState(false)
  const [refusal, setRefusal] = useState<BillingRefusal | null>(null)
  const runningRef = useRef(false)

  const resume = async () => {
    if (runningRef.current) {
      return
    }

    runningRef.current = true
    setBusy(true)
    setRefusal(null)

    const result = await api.resumeSubscription()

    if (!result.ok) {
      // Refusal → unlock immediately so the user can retry / step up.
      runningRef.current = false
      setBusy(false)
      setRefusal(result.refusal)

      return
    }

    // Success → hold the lock (Undo stays disabled) THROUGH the refetch, so the
    // button never re-enables against the stale, still-pending card.
    await invalidateBilling(queryClient)
    runningRef.current = false
    setBusy(false)
  }

  return { busy, refusal, resume }
}
