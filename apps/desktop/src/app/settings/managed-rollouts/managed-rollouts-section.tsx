import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { getManagedRolloutMessages } from '@/i18n/managed-rollouts'
import type { RolloutDraft as NativeRolloutDraft, RolloutAction, RolloutSnapshot, TargetAttempt } from '@/lib/managed-rollout-contract'
import { makeWaves, MAX_EFFECTIVE_WAVE_SIZE } from '@/lib/managed-rollout-waves'
import {
  $managedRollouts,
  getManagedRolloutSnapshot,
  type ManagedRolloutInventory,
  type ManagedRolloutPreflight,
  type ManagedRolloutResolution,
  type ManagedRolloutsState,
  pollManagedRollouts,
  preflightManagedRollout,
  readManagedRolloutHistory,
  readManagedRolloutInventory,
  resolveManagedRolloutTarget,
  sendManagedRolloutCommand,
  startManagedRollout,
  startManagedRolloutPolling
} from '@/store/managed-rollouts'

import { ActiveRollout } from './active-rollout'
import { AttemptDetail } from './attempt-detail'
import { targetsFromInventory } from './fleet-overview'
import { PreflightReview } from './preflight-review'
import { PreparationPanel } from './preparation-panel'
import { RecoveryPanel } from './recovery-panel'
import { RolloutConfig, type RolloutDraft } from './rollout-config'
import { RolloutControls } from './rollout-controls'
import { RolloutHistory, type RolloutHistoryEntry } from './rollout-history'
import { RolloutSummary } from './rollout-summary'
import type { ManagedRolloutTarget } from './target-row'

function bridgeAvailable(): boolean {
  return typeof window !== 'undefined' && Boolean(window.hermesDesktop?.connections?.managedRollouts)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function snapshotLabel(state: ManagedRolloutsState, noActive: string, activePhase: (phase: string) => string): string {
  if (!state.snapshot) {return noActive}

  return state.active ? activePhase(state.snapshot.phase) : `Last observed: ${activePhase(state.snapshot.phase)}`
}

function currentAttempt(attempt: TargetAttempt) {
  const health = attempt.health

  return {
    installId: attempt.identity.installId,
    phase: attempt.phase,
    receipt: attempt.receipt ? { outcome: attempt.receipt.outcome, correlationId: attempt.receipt.correlationId } : null,
    readiness: health ? {
      ready: false,
      reason: `Last observation: installation ${health.installReady ? 'ready' : 'not ready'}, scope capture ${health.scopeCapture}; ${health.reasons.join(', ') || 'promotion still requires fresh Main verification'}`
    } : null,
    unknown: attempt.phase === 'unverified',
    fenced: attempt.recoveryRequired
  }
}

export function ManagedRolloutsSection({
  history = [],
  onSelect = () => undefined
}: {
  history?: readonly RolloutHistoryEntry[]
  onSelect?: (id: string) => void
}) {
  const { t } = useI18n()
  const messages = getManagedRolloutMessages(t)
  const state = useStore($managedRollouts)
  const available = bridgeAvailable()
  const [inventory, setInventory] = useState<ManagedRolloutInventory | null>(null)
  const [inventoryStatus, setInventoryStatus] = useState<'loading' | 'ready' | 'error'>('loading')
  const [reviewGeneration, setReviewGeneration] = useState(0)
  const [resolution, setResolution] = useState<ManagedRolloutResolution | null>(null)
  const [draft, setDraft] = useState<RolloutDraft | null>(null)
  const [review, setReview] = useState<ManagedRolloutPreflight | null>(null)
  const [retryOf, setRetryOf] = useState<string | null>(null)
  const [retryInstallId, setRetryInstallId] = useState<string | null>(null)
  const [startedRolloutId, setStartedRolloutId] = useState<string | null>(null)
  const [detail, setDetail] = useState<RolloutSnapshot | null>(null)
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(new Set())
  const [nativeHistory, setNativeHistory] = useState<RolloutHistoryEntry[] | null>(null)
  const [pending, setPending] = useState<'prepare' | 'resolve' | 'preflight' | 'start' | 'command' | null>(null)
  const [operatorError, setOperatorError] = useState<string | null>(null)
  const [operatorNotice, setOperatorNotice] = useState<string | null>(null)

  const refreshHistory = useCallback(async () => {
    if (!window.hermesDesktop?.connections?.managedRollouts?.history) {return}
    const rows = await readManagedRolloutHistory()
    setNativeHistory(rows.map(row => ({
      id: row.id,
      phase: row.phase,
      updatedAt: row.updatedAt,
      unresolved: row.unresolvedInstallIds.length,
      archived: row.archived,
      reason: row.unresolvedInstallIds.length ? 'Unresolved installation fence retained' : null
    })))
  }, [])

  const refreshInventory = useCallback(async () => {
    setInventoryStatus('loading')
    setResolution(null)
    setDraft(null)
    setReview(null)
    setReviewGeneration(value => value + 1)

    try {
      const observed = await readManagedRolloutInventory()
      setInventory(observed)
      setInventoryStatus('ready')
    } catch (error) {
      setInventory(null)
      setInventoryStatus('error')
      setOperatorError(errorMessage(error))
    }
  }, [])

  useEffect(() => {
    if (!available) {return}
    const stop = startManagedRolloutPolling()
    void refreshInventory()
    void refreshHistory().catch(error => setOperatorError(errorMessage(error)))

    return stop
  }, [available, refreshHistory, refreshInventory])

  const targets = useMemo(() => inventory ? targetsFromInventory(inventory) : [], [inventory])

  const snapshot = detail && (!state.snapshot || detail.id !== state.snapshot.id || detail.revision >= state.snapshot.revision)
    ? detail : state.snapshot

  const workActive = state.active || Boolean(
    startedRolloutId && snapshot?.id === startedRolloutId &&
    !['completed', 'completed-with-exclusions', 'stopped'].includes(snapshot.phase)
  )

  const activeDraft: RolloutDraft | null = snapshot ? {
    mode: snapshot.promotionPolicy,
    concurrency: snapshot.concurrency,
    canaryInstallId: snapshot.attempts.find(attempt => attempt.wave === 0)?.identity.installId ?? null,
    selectedInstallIds: snapshot.attempts.map(attempt => attempt.identity.installId)
  } : null

  const prepare = async (chosen: ManagedRolloutTarget[]) => {
    setResolution(null)
    setDraft(null)
    setReview(null)
    setOperatorError(null)
    setOperatorNotice(null)
    setPending('prepare')

    try {
      const update = window.hermesDesktop?.connections?.updateManaged

      if (!update) {throw new Error('individual-managed-update-unavailable')}
      const outcomes: string[] = []

      for (const target of chosen) {
        if (!target.connectionId) {throw new Error('selected-connection-unavailable')}
        const result = await update(target.connectionId)
        outcomes.push(messages.labels.preparationOutcome(target.installId, result.outcome, result.correlationId, result.restoreOk))

        if (!result.ok || !result.updateOk || !result.restoreOk) {
          throw new Error(`preparation-refused: ${outcomes.join('; ')}`)
        }
      }

      setOperatorNotice(messages.descriptions.preparationFinished(outcomes.join('; ')))
    } catch (error) {
      setOperatorError(errorMessage(error))
    } finally {
      setPending(null)
      await refreshInventory()
    }
  }

  const resolve = async (chosen: ManagedRolloutTarget[]) => {
    if (!inventory || state.capabilities?.available !== true || pending || workActive) {return}
    setOperatorError(null)
    setOperatorNotice(null)
    setResolution(null)
    setDraft(null)
    setReview(null)
    setPending('resolve')

    try {
      const connectionIds = chosen.map(target => target.connectionId).filter((id): id is string => Boolean(id))

      if (connectionIds.length !== chosen.length) {throw new Error('selected-connection-unavailable')}

      if (retryInstallId && (chosen.length !== 1 || chosen[0].installId !== retryInstallId)) {
        throw new Error('retry-selection-must-match-original-installation')
      }

      const resolved = await resolveManagedRolloutTarget({
        connectionIds,
        inventoryRevision: inventory.inventoryRevision,
        retryOf
      })

      const selectedInstallIds = resolved.plan.waves.flat()

      if (!selectedInstallIds.length) {throw new Error('resolved-plan-has-no-eligible-wave')}
      setResolution(resolved)
      setDraft({
        mode: 'manual',
        concurrency: 1,
        canaryInstallId: selectedInstallIds[0],
        selectedInstallIds
      })
      setOperatorNotice(`Main resolved pinned target ${resolved.plan.target.sha}. Configure and request preflight.`)
    } catch (error) {
      setOperatorError(errorMessage(error))
    } finally {
      setPending(null)
    }
  }

  const requestPreflight = async (next: RolloutDraft) => {
    if (!resolution || !inventory || pending) {return}
    setOperatorError(null)
    setReview(null)
    setPending('preflight')

    try {
      if (!next.canaryInstallId) {throw new Error('canary-required')}
      const waves = makeWaves(next.selectedInstallIds, [next.canaryInstallId], Math.min(MAX_EFFECTIVE_WAVE_SIZE, Math.max(1, next.selectedInstallIds.length - 1)))

      const nativeDraft: NativeRolloutDraft = {
        inventoryRevision: inventory.inventoryRevision,
        targetResolutionId: resolution.resolutionId,
        waves,
        concurrency: next.concurrency,
        promotionPolicy: next.mode,
        retryOf
      }

      const result = await preflightManagedRollout(nativeDraft)
      setReview(result)

      if (!result.ok) {setOperatorNotice('Preflight refused; inspect every blocker and changed row before renewing review.')}
    } catch (error) {
      setOperatorError(errorMessage(error))
    } finally {
      setPending(null)
    }
  }

  const begin = async (_draft: RolloutDraft, token: string) => {
    if (!review?.ok || review.token !== token || !review.requestId || pending) {return}
    setPending('start')
    setOperatorError(null)

    try {
      const accepted = await startManagedRollout({ token, requestId: review.requestId })
      setReview(null)
      setResolution(null)
      setDraft(null)
      setRetryOf(null)
      setRetryInstallId(null)
      setStartedRolloutId(accepted.id)
      setOperatorNotice(`Start accepted for rollout ${accepted.id}; waiting for its authoritative snapshot.`)

      try {
        setDetail(await getManagedRolloutSnapshot(accepted.id))
        await pollManagedRollouts()
        await refreshHistory()
      } catch (error) {
        setOperatorError(`Start was accepted, but snapshot refresh failed: ${errorMessage(error)}`)
      }
    } catch (error) {
      setOperatorError(errorMessage(error))
      throw error
    } finally {
      setPending(null)
    }
  }

  const command = async (action: RolloutAction, installId: string | null = null, reason: string | null = null) => {
    if (!snapshot || pending) {return false}
    setPending('command')
    setOperatorError(null)

    try {
      const ack = await sendManagedRolloutCommand({
        id: snapshot.id,
        expectedRevision: snapshot.revision,
        requestId: globalThis.crypto.randomUUID(),
        action,
        installId,
        reason,
        promotionPolicy: null
      })

      if (!ack.ok) {
        setOperatorError(ack.message ?? ack.code ?? 'managed-rollout-command-refused')

        return false
      }

      try {
        const latest = await getManagedRolloutSnapshot(snapshot.id)

        if (latest) {setDetail(latest)}
        await pollManagedRollouts()
        await refreshHistory()
      } catch (error) {
        setOperatorError(`Command was accepted, but snapshot refresh failed: ${errorMessage(error)}`)
      }

      return true
    } catch (error) {
      setOperatorError(errorMessage(error))

      return false
    } finally {
      setPending(null)
    }
  }

  const selectHistory = async (id: string) => {
    onSelect(id)

    try {
      const selected = await getManagedRolloutSnapshot(id)

      if (!selected) {throw new Error('managed-rollout-detail-unavailable')}
      setDetail(selected)
    } catch (error) {
      setOperatorError(errorMessage(error))
    }
  }

  if (!available) {return null}
  const unsupported = state.status === 'unsupported'
  const historyEntries = nativeHistory ?? history

  return <section aria-label={messages.title} className="grid min-w-0 gap-4">
    <div className="grid gap-1">
      <h2 className="text-sm font-medium">{messages.title}</h2>
      <p aria-live="polite" className="text-xs text-(--ui-text-tertiary)" role="status">
        {unsupported ? messages.warnings.unavailable : snapshotLabel(state, messages.noActive, messages.status.activePhase)}
      </p>
      {state.capabilities?.available === false ? <p className="text-xs text-(--ui-text-primary)">{messages.descriptions.rolloutCapacityUnavailable(state.capabilities.maxInstallations, state.capabilities.reason)}</p> : null}
      {unsupported && state.error ? <p aria-label={state.error} className="text-xs text-amber-600" role="alert">{state.error}</p> : null}
      {state.status === 'error' && state.error ? <p className="text-xs text-amber-600" role="alert">Snapshot unavailable: {state.error}</p> : null}
      {operatorError ? <p className="text-xs text-amber-600" role="alert">{operatorError}</p> : null}
      {operatorNotice ? <p aria-live="polite" className="text-xs">{operatorNotice}</p> : null}
    </div>

    <section aria-label={messages.sections.inventory} className="grid min-w-0 gap-2">
      <p className="text-xs text-(--ui-text-tertiary)">{messages.descriptions.inventoryObserved}</p>
      {inventoryStatus === 'loading' ? <p role="status">{messages.descriptions.inventoryLoading}</p> : null}
      {inventoryStatus === 'error' ? <Button onClick={() => { void refreshInventory() }} type="button" variant="outline">{messages.actions.retryInventory}</Button> : null}
      {inventory && inventoryStatus === 'ready' ? <>
        <p className="text-xs">{messages.labels.inventoryRevision(inventory.inventoryRevision, inventory.capturedMono)}</p>
        <PreparationPanel
          disabled={Boolean(pending) || workActive}
          onPrepare={prepare}
          onReview={resolve}
          reviewDisabled={state.capabilities?.available !== true}
          reviewGeneration={`${inventory.inventoryRevision}:${reviewGeneration}`}
          targets={targets}
        />
      </> : null}
    </section>

    {resolution && draft ? <RolloutConfig
      draft={draft}
      onChange={next => { setDraft(next); setReview(null) }}
      onContinue={next => { void requestPreflight(next) }}
      pending={pending === 'preflight'}
    /> : null}

    {review && draft ? <PreflightReview
      blockers={review.blockers}
      changes={review.changes}
      compatible={review.ok && state.capabilities?.available === true}
      draft={draft}
      expiresAt={review.expiresAt}
      key={review.planDigest ?? `refused:${reviewGeneration}`}
      onRenew={() => { void requestPreflight(draft) }}
      onStart={begin}
      pending={pending === 'start' || pending === 'preflight'}
      plan={review.canonicalPlan}
      planner={() => review.canonicalPlan.waves}
      reviewedToken={review.token}
    /> : null}

    {snapshot && activeDraft ? <>
      <ActiveRollout draft={activeDraft} state={{
        phase: snapshot.phase,
        completed: snapshot.attempts.filter(attempt => attempt.phase === 'updated' || attempt.phase === 'already-current').length,
        total: snapshot.attempts.length,
        receipt: null,
        readiness: null,
        canaryGate: snapshot.canaryApproved ? 'approved' : snapshot.phase === 'attention-required' ? 'failed' : 'pending',
        restartRequired: snapshot.continuationRequired
      }} />
      <p className="text-xs text-(--ui-text-tertiary)">Per-installation receipts and readiness below are the last observed evidence, not a live fleet health claim.</p>
      {snapshot.attempts.map(attempt => {
        const displayed = currentAttempt(attempt)

        return <div className="grid gap-2" key={displayed.installId}>
          <AttemptDetail attempt={{ ...displayed, expanded: expanded.has(displayed.installId) }} onToggle={id => setExpanded(current => {
            const next = new Set(current)

            if (next.has(id)) {next.delete(id)}
            else {next.add(id)}

            return next
          })} />
          {displayed.unknown || displayed.fenced ? <RecoveryPanel
            onExclude={reason => { void command('exclude', displayed.installId, reason) }}
            onRecheck={() => { void command('reprobe', displayed.installId) }}
            onRecover={() => { void command('recover', displayed.installId) }}
            onRetry={() => {
              setRetryOf(snapshot.id)
              setRetryInstallId(displayed.installId)
              setResolution(null)
              setDraft(null)
              setReview(null)
              setOperatorNotice('Retry needs a fresh inventory, target resolution, and review. Select the installation above after refreshing inventory.')
              void refreshInventory()
            }}
            onStop={() => { void command('stop') }}
            target={{ installId: displayed.installId, phase: displayed.phase, unknown: displayed.unknown, fenced: displayed.fenced, reason: attempt.reasons.join(', ') || null }}
          /> : null}
        </div>
      })}
      {state.active && snapshot.id === state.snapshot?.id ? <RolloutControls
        onCommand={action => command(action)}
        onVerify={() => { void command('promote') }}
        phase={snapshot.phase}
      /> : null}
      <RolloutSummary summary={{
        phase: snapshot.phase,
        excluded: snapshot.attempts.filter(attempt => attempt.phase === 'skipped').length,
        unresolved: snapshot.attempts.filter(attempt => attempt.recoveryRequired).length,
        archived: snapshot.archivedAt !== null,
        reason: snapshot.continuationRequired ? 'Fresh human continuation is required' : null
      }} />
    </> : null}

    <RolloutHistory entries={historyEntries} onSelect={id => { void selectHistory(id) }} />
  </section>
}
