import type { RolloutDraft } from './rollout-config'

export interface ActiveRolloutState {
  phase: string
  completed: number
  total: number
  receipt: { outcome: string; correlationId: string } | null
  readiness: { ready: boolean; reason: string | null } | null
  canaryGate: 'pending' | 'approved' | 'failed'
  restartRequired: boolean
}

export function ActiveRollout({ state, draft }: { state: ActiveRolloutState; draft: RolloutDraft }) {
  return <section aria-label="Active managed rollout" className="grid gap-2">
    <p className="text-sm">Phase: <strong>{state.phase}</strong></p>
    <p className="text-xs text-(--ui-text-tertiary)">Progress: {state.completed} of {state.total} targets completed.</p>
    <p className="text-xs text-(--ui-text-tertiary)">Mode: {draft.mode}; canary gate: {state.canaryGate}.</p>
    {state.receipt ? <p className="text-xs">Receipt: {state.receipt.outcome} ({state.receipt.correlationId})</p> : <p className="text-xs text-(--ui-text-tertiary)">Receipt: pending</p>}
    {state.readiness ? <p className="text-xs">Readiness: {state.readiness.ready ? 'ready' : state.readiness.reason ?? 'not ready'}</p> : <p className="text-xs text-(--ui-text-tertiary)">Readiness: pending evidence</p>}
    {state.restartRequired ? <p className="text-xs text-amber-600">Restart requires a fresh review before promotion.</p> : null}
  </section>
}
