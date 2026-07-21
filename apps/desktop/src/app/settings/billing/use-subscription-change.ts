import { useQueryClient } from '@tanstack/react-query'
import { useRef, useState } from 'react'

import type { BillingRefusal } from './api'
import { useBillingApi } from './api'
import type { SubscriptionPreviewResponse } from './types'

// DEV fixtures have no live gateway, so preview / change / resume are simulated
// with canned success after a short delay — the visual loop can click through the
// confirm → schedule → undo path. A non-null value switches the flow to simulation.
export interface SubscriptionSimulation {
  effectiveAt: null | string
}

const SIMULATED_DELAY_MS = 350

const delay = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))

// The canned-success seam is a DEV-fixtures-only affordance; never let a production
// build take the simulated branch even if a `simulate` prop leaks through. Accepts
// the downgrade shape (object | null) and the resume flag (boolean).
const simulationEnabled = (simulate?: boolean | null | SubscriptionSimulation): boolean =>
  import.meta.env.DEV && Boolean(simulate)

export interface DowngradeTarget {
  tierId: string
  tierName: string
}

export interface ActiveDowngrade {
  /** Which RPC is in flight (drives the disabled/"…" states), else null. */
  busy: 'preview' | 'schedule' | null
  /** Which op the current refusal came from, so the retry button re-runs the right one. */
  failedOp: 'preview' | 'schedule' | null
  preview: null | SubscriptionPreviewResponse
  refusal: BillingRefusal | null
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
 */
export function useDowngradeFlow({
  onScheduled,
  simulate
}: {
  onScheduled: () => void
  simulate?: null | SubscriptionSimulation
}) {
  const api = useBillingApi()
  const queryClient = useQueryClient()
  const [active, setActive] = useState<ActiveDowngrade | null>(null)
  // Monotonic run id discards results from a superseded/cancelled attempt.
  const runIdRef = useRef(0)
  const simulated = simulationEnabled(simulate)

  const runPreview = async (target: DowngradeTarget, runId: number) => {
    if (simulated && simulate) {
      await delay(SIMULATED_DELAY_MS)

      if (runIdRef.current !== runId) {
        return
      }

      setActive({
        busy: null,
        failedOp: null,
        preview: {
          effect: 'scheduled',
          effective_at: simulate.effectiveAt,
          ok: true,
          target_tier_name: target.tierName
        },
        refusal: null,
        target
      })

      return
    }

    const result = await api.previewSubscriptionChange(target.tierId)

    if (runIdRef.current !== runId) {
      return
    }

    setActive(
      result.ok
        ? { busy: null, failedOp: null, preview: result.data, refusal: null, target }
        : { busy: null, failedOp: 'preview', preview: null, refusal: result.refusal, target }
    )
  }

  const begin = (target: DowngradeTarget) => {
    const runId = runIdRef.current + 1

    runIdRef.current = runId
    setActive({ busy: 'preview', failedOp: null, preview: null, refusal: null, target })
    void runPreview(target, runId)
  }

  const retryPreview = () => {
    if (active) {
      begin(active.target)
    }
  }

  const confirm = async () => {
    if (!active || active.busy) {
      return
    }

    const { target } = active
    const runId = runIdRef.current + 1

    runIdRef.current = runId
    setActive({ ...active, busy: 'schedule', failedOp: null, refusal: null })

    if (simulated) {
      await delay(SIMULATED_DELAY_MS)

      if (runIdRef.current !== runId) {
        return
      }

      setActive(null)
      onScheduled()

      return
    }

    const result = await api.scheduleSubscriptionChange(target.tierId)

    if (runIdRef.current !== runId) {
      return
    }

    if (!result.ok) {
      // A refusal (e.g. insufficient_scope → step-up) leaves the panel open with
      // failedOp set, so the same button becomes a manual "Try again" AFTER the
      // user elevates. We deliberately do NOT auto-replay the mutation on step-up
      // success — this matches the auto-reload save's manual-retry pattern.
      setActive({ busy: null, failedOp: 'schedule', preview: active.preview, refusal: result.refusal, target })

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
  const mutating = active?.busy === 'schedule'

  return { active, begin, cancel, confirm, mutating, retryPreview }
}

/**
 * The undo for a scheduled downgrade / cancellation: a chargeless
 * `subscription.resume` (no confirm step) that refetches on success. A refusal
 * (e.g. insufficient_scope → step-up) surfaces via `refusal`.
 */
export function useResumeFlow(simulate = false) {
  const api = useBillingApi()
  const queryClient = useQueryClient()
  const [busy, setBusy] = useState(false)
  const [refusal, setRefusal] = useState<BillingRefusal | null>(null)
  const runningRef = useRef(false)
  const simulated = simulationEnabled(simulate)

  const resume = async () => {
    if (runningRef.current) {
      return
    }

    runningRef.current = true
    setBusy(true)
    setRefusal(null)

    if (simulated) {
      await delay(SIMULATED_DELAY_MS)
      runningRef.current = false
      setBusy(false)

      return
    }

    const result = await api.resumeSubscription()

    runningRef.current = false
    setBusy(false)

    if (!result.ok) {
      setRefusal(result.refusal)

      return
    }

    await invalidateBilling(queryClient)
  }

  return { busy, refusal, resume }
}
