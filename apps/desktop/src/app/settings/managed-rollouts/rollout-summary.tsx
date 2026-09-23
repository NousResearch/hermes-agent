export interface RolloutSummaryData { phase: string; excluded: number; unresolved: number; archived: boolean; reason: string | null }

export function RolloutSummary({ summary }: { summary: RolloutSummaryData }) {
  return <section aria-label="Managed rollout summary" className="grid gap-1 text-sm"><p>Outcome: {summary.phase}</p><p>Excluded targets: {summary.excluded}</p><p>Unresolved fences: {summary.unresolved}</p><p>Archive: {summary.archived ? 'archived' : 'active'}</p>{summary.reason ? <p className="text-xs text-(--ui-text-tertiary)">Reason: {summary.reason}</p> : null}</section>
}
