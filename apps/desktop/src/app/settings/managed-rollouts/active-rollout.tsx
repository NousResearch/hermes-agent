import { useI18n } from '@/i18n'
import { getManagedRolloutMessages } from '@/i18n/managed-rollouts'

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
  const { t } = useI18n()
  const messages = getManagedRolloutMessages(t)

  return <section aria-label={messages.sections.active} className="grid min-w-0 gap-2">
    <p className="text-sm">{messages.status.activePhase(state.phase)}</p>
    <p className="text-xs text-(--ui-text-tertiary)">{messages.labels.progress(state.completed, state.total)}</p>
    <p className="text-xs text-(--ui-text-tertiary)">{messages.policy.mode(draft.mode)}; {messages.policy.canaryGate(state.canaryGate)}</p>
    {state.receipt ? <p className="text-xs">{messages.labels.receipt(state.receipt.outcome, state.receipt.correlationId)}</p> : <p className="text-xs text-(--ui-text-tertiary)">{messages.labels.receiptPending}</p>}
    {state.readiness ? <p className="text-xs">{messages.labels.readiness(state.readiness.ready ? messages.status.ready : state.readiness.reason ?? messages.status.pending)}</p> : <p className="text-xs text-(--ui-text-tertiary)">{messages.labels.readiness(messages.status.pending)}</p>}
    {state.restartRequired ? <p className="text-xs text-amber-600">{messages.warnings.restartReviewRequired}</p> : null}
  </section>
}
