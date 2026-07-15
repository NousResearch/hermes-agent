import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import type { StudyIntervention, StudyOverviewResponse } from '@/types/hermes'

interface StudyInboxProps {
  onDecide: (proposalId: string, action: 'accept' | 'reject') => Promise<void>
  onStartAgent?: (prompt: string) => void | Promise<void>
  overview: StudyOverviewResponse
}

function interventionPrompt(item: StudyIntervention): string {
  return `Use StudyOS to run ${item.kind} for “${item.capability}”. Target ${item.evidence_dimension}, preserve evaluator and assistance provenance, then record the attempt.`
}

export function StudyInbox({ onDecide, onStartAgent, overview }: StudyInboxProps) {
  const { t } = useI18n()
  const [busyId, setBusyId] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  const decide = async (proposalId: string, action: 'accept' | 'reject') => {
    setBusyId(proposalId)
    setError(null)

    try {
      await onDecide(proposalId, action)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setBusyId(null)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-2xl font-semibold">{t.study.suggestionsTitle}</h3>
        <p className="mt-1 text-sm text-muted-foreground">{t.study.suggestionsDescription}</p>
      </div>
      {error && <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-3 text-sm">{error}</div>}

      <section>
        <h4 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          {t.study.pendingChanges} · {overview.pending_plan_proposals.length}
        </h4>
        {overview.pending_plan_proposals.length === 0 ? (
          <div className="mt-3 rounded-2xl border border-dashed p-5 text-sm text-muted-foreground">
            {t.study.emptyInbox}
          </div>
        ) : (
          <div className="mt-3 space-y-3">
            {overview.pending_plan_proposals.map(proposal => (
              <article className="rounded-3xl border bg-card/70 p-5" key={proposal.proposal_id}>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <h5 className="font-semibold">{proposal.title}</h5>
                    <p className="mt-1 max-w-3xl text-sm text-muted-foreground">{proposal.rationale}</p>
                  </div>
                  <span className="rounded-full bg-amber-500/10 px-3 py-1 text-xs font-medium text-amber-700 dark:text-amber-300">
                    {proposal.items.length} {t.study.actions}
                  </span>
                </div>
                <div className="mt-4 space-y-2">
                  {proposal.items.map(item => (
                    <div className="rounded-xl border bg-background/60 p-3 text-sm" key={item.intervention_id}>
                      <div className="font-medium">{item.capability}</div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        {item.kind} · {item.evidence_dimension} · {item.recommended_activity.duration_minutes}m
                      </div>
                    </div>
                  ))}
                </div>
                <p className="mt-4 text-xs text-muted-foreground">{t.study.proposalNotice}</p>
                <div className="mt-4 flex gap-2">
                  <Button
                    disabled={busyId === proposal.proposal_id}
                    onClick={() => void decide(proposal.proposal_id, 'accept')}
                    size="sm"
                  >
                    {t.study.accept}
                  </Button>
                  <Button
                    disabled={busyId === proposal.proposal_id}
                    onClick={() => void decide(proposal.proposal_id, 'reject')}
                    size="sm"
                    variant="secondary"
                  >
                    {t.study.reject}
                  </Button>
                </div>
              </article>
            ))}
          </div>
        )}
      </section>

      <section>
        <h4 className="text-sm font-semibold uppercase tracking-wide text-muted-foreground">
          {t.study.recommendations} · {overview.intervention_queue.items.length}
        </h4>
        <div className="mt-3 space-y-3">
          {overview.intervention_queue.items.map(item => (
            <article className="rounded-2xl border bg-card/60 p-4" key={item.intervention_id}>
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <div className="font-medium">{item.capability}</div>
                  <p className="mt-1 text-sm text-muted-foreground">{item.reasons[0]}</p>
                </div>
                <span className="rounded-full bg-primary/10 px-3 py-1 text-xs font-medium text-primary">
                  {item.priority_band} · {item.priority_score}
                </span>
              </div>
              {onStartAgent && (
                <Button
                  className="mt-3"
                  onClick={() => void onStartAgent(interventionPrompt(item))}
                  size="sm"
                  variant="secondary"
                >
                  {t.study.startWithAgent}
                </Button>
              )}
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}
