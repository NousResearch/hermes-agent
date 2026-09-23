import { useI18n } from '@/i18n'
import { getManagedRolloutMessages } from '@/i18n/managed-rollouts'

export interface RolloutSummaryData { phase: string; excluded: number; unresolved: number; archived: boolean; reason: string | null }

export function RolloutSummary({ summary }: { summary: RolloutSummaryData }) {
  const { t } = useI18n()
  const messages = getManagedRolloutMessages(t)

  return <section aria-label={messages.sections.summary} className="grid gap-1 text-sm"><p>{messages.summary.outcome(summary.phase)}</p><p>{messages.summary.excludedTargets(summary.excluded)}</p><p>{messages.summary.unresolvedFences(summary.unresolved)}</p><p>{messages.summary.archive(summary.archived)}</p>{summary.reason ? <p className="text-xs text-(--ui-text-tertiary)">{messages.summary.reason(summary.reason)}</p> : null}</section>
}
