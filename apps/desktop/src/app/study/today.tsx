import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import type { StudyIntervention, StudyOverviewResponse } from '@/types/hermes'

interface StudyTodayProps {
  onOpenReview: () => void
  onStartAgent?: (prompt: string) => void | Promise<void>
  overview: StudyOverviewResponse
}

function nextActionPrompt(action: StudyIntervention): string {
  return [
    'Use StudyOS to guide the following evidence-backed learning activity.',
    `Objective: ${action.capability}`,
    `Evidence target: ${action.evidence_dimension}`,
    `Activity: ${action.kind}`,
    `Keep assistance at ${action.recommended_activity.assistance_level} and record evaluator provenance.`
  ].join('\n')
}

export function StudyToday({ onOpenReview, onStartAgent, overview }: StudyTodayProps) {
  const { t } = useI18n()
  const dimensions = Object.values(overview.evidence.dimensions)
  const observed = dimensions.filter(item => item.status === 'observed').length
  const independentlyVerified = dimensions.filter(item => item.verification_status === 'independent').length
  const nextAction = overview.intervention_queue.items[0]

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">{overview.today}</div>
          <h3 className="mt-1 text-2xl font-semibold">{t.study.todayTitle}</h3>
        </div>
        <div className="text-sm text-muted-foreground">{overview.project.title}</div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Metric label={t.study.scheduledToday} value={overview.today_events.length} />
        <Metric label={t.study.dueNow} value={overview.due_reviews.count} />
        <Metric label={t.study.completedToday} value={overview.completed_today} />
        <Metric label={t.study.activityToday} value={overview.activity_today} />
      </div>

      <div className="grid gap-5 xl:grid-cols-[1.25fr_0.75fr]">
        <section className="rounded-3xl border bg-card/70 p-5">
          <div className="flex items-center justify-between gap-3">
            <h4 className="text-lg font-semibold">{t.study.scheduledToday}</h4>
            <span className="text-xs text-muted-foreground">{overview.project.timezone}</span>
          </div>
          {overview.today_events.length === 0 ? (
            <p className="mt-4 text-sm text-muted-foreground">{t.study.noEventsToday}</p>
          ) : (
            <div className="mt-4 space-y-3">
              {overview.today_events.map(event => (
                <article className="rounded-2xl border bg-background/70 p-4" key={`${event.schedule_id}:${event.id}`}>
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="font-medium">{event.title}</div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        {event.start.slice(11, 16)}–{event.end.slice(11, 16)} · {event.schedule_title}
                      </div>
                    </div>
                    <span className="rounded-full bg-primary/10 px-2.5 py-1 text-xs text-primary">
                      {event.duration_minutes}m
                    </span>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>

        <section className="rounded-3xl border bg-card/70 p-5">
          <h4 className="text-lg font-semibold">{t.study.evidenceStatus}</h4>
          <div className="mt-4 grid grid-cols-2 gap-3">
            <Metric label={t.study.evidenceCoverage} value={`${observed}/${dimensions.length}`} />
            <Metric label={t.study.verifiedDimensions} value={`${independentlyVerified}/${dimensions.length}`} />
            <Metric label={t.study.attempts} value={overview.evidence.attempt_count} />
            <Metric label={t.study.independentEvidence} value={overview.evidence.independently_verified_count} />
          </div>
          <div className="mt-3 text-xs text-muted-foreground">
            {t.study.latestEvidence}:{' '}
            {overview.evidence.latest_evidence_at?.slice(0, 16).replace('T', ' ') ?? t.study.noEvidence}
          </div>
        </section>
      </div>

      <div className="grid gap-5 xl:grid-cols-2">
        <section className="rounded-3xl border bg-card/70 p-5">
          <div className="flex items-center justify-between gap-3">
            <h4 className="text-lg font-semibold">{t.study.dueReviews}</h4>
            <Button onClick={onOpenReview} size="sm" variant="secondary">
              {t.study.openReview}
            </Button>
          </div>
          {overview.due_reviews.items.length === 0 ? (
            <p className="mt-4 text-sm text-muted-foreground">{t.study.noDueReviewsDesc}</p>
          ) : (
            <div className="mt-4 space-y-2">
              {overview.due_reviews.items.slice(0, 4).map(item => (
                <div className="rounded-xl border bg-background/60 px-3 py-2" key={item.path}>
                  <div className="truncate text-sm font-medium">{item.title}</div>
                  <div className="mt-0.5 text-xs text-muted-foreground">
                    {item.subject ?? item.path} · L{item.review_level}
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        <section className="rounded-3xl border border-primary/30 bg-primary/5 p-5">
          <h4 className="text-lg font-semibold">{t.study.nextAction}</h4>
          {nextAction ? (
            <>
              <div className="mt-3 text-base font-medium">{nextAction.capability}</div>
              <p className="mt-2 text-sm text-muted-foreground">{nextAction.reasons[0]}</p>
              <div className="mt-3 text-xs text-muted-foreground">
                {nextAction.evidence_dimension} · {nextAction.recommended_activity.duration_minutes}m ·{' '}
                {nextAction.priority_band}
              </div>
              {onStartAgent && (
                <Button className="mt-4" onClick={() => void onStartAgent(nextActionPrompt(nextAction))} size="sm">
                  {t.study.startWithAgent}
                </Button>
              )}
            </>
          ) : (
            <p className="mt-3 text-sm text-muted-foreground">{t.study.noSuggestions}</p>
          )}
        </section>
      </div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="rounded-2xl border bg-background/60 p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 text-xl font-semibold">{value}</div>
    </div>
  )
}
