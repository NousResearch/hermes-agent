import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useRef, useState } from 'react'

import { CompactMarkdown } from '@/components/chat/compact-markdown'
import { PageLoader } from '@/components/page-loader'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { getStudyReviewDetail, getStudyReviewStats, submitStudyReviewAttempt } from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $reviewCompletedToday, $reviewDueItems, $reviewStats } from '@/store/study-review'
import type {
  StudyReviewDetail,
  StudyReviewItem,
  StudyReviewResult,
  StudyReviewSubmissionResponse
} from '@/types/hermes'

interface ReviewRunnerProps {
  allItems: StudyReviewItem[]
  initialItem: StudyReviewItem
  items: StudyReviewItem[]
  onClose: () => void
  onSessionComplete: (completed: number) => void
  projectId: null | string
}

function formatElapsed(seconds: number): string {
  const minutes = Math.floor(seconds / 60)

  return `${String(minutes).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`
}

export function ReviewRunner({
  allItems,
  initialItem,
  items,
  onClose,
  onSessionComplete,
  projectId
}: ReviewRunnerProps) {
  const { t } = useI18n()
  const completed = useStore($reviewCompletedToday)
  const [activeItem, setActiveItem] = useState(initialItem)
  const [detail, setDetail] = useState<null | StudyReviewDetail>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<null | string>(null)
  const [response, setResponse] = useState('')
  const [confidence, setConfidence] = useState<null | number>(null)
  const [answerRevealed, setAnswerRevealed] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const [lastResult, setLastResult] = useState<null | StudyReviewSubmissionResponse>(null)
  const startedAtRef = useRef<null | number>(null)
  const sessionIdRef = useRef(`desktop-review-${Date.now()}`)

  const openItem = useCallback(async (item: StudyReviewItem) => {
    setLoading(true)
    setError(null)

    try {
      const nextDetail = await getStudyReviewDetail(item.path)
      setActiveItem(item)
      setDetail(nextDetail)
      setResponse('')
      setConfidence(null)
      setAnswerRevealed(false)
      setElapsedSeconds(0)
      startedAtRef.current = Date.now()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void openItem(initialItem)
  }, [initialItem, openItem])

  useEffect(() => {
    if (startedAtRef.current === null) {
      return
    }

    const update = () => setElapsedSeconds(Math.max(0, Math.floor((Date.now() - startedAtRef.current!) / 1000)))
    update()
    const timer = window.setInterval(update, 1000)

    return () => window.clearInterval(timer)
  }, [activeItem])

  const submitResult = useCallback(
    async (result: StudyReviewResult) => {
      if (!projectId || !response.trim() || confidence === null) {
        return
      }

      setSubmitting(true)
      setError(null)

      try {
        const submitted = await submitStudyReviewAttempt({
          project_id: projectId,
          note: activeItem.path,
          response: response.trim(),
          result,
          duration_seconds: Math.max(1, elapsedSeconds),
          self_confidence: confidence,
          transfer_level: 'execution',
          diagnoses: [],
          session_id: sessionIdRef.current
        })

        const remaining = items.filter(item => item.path !== activeItem.path)
        const remainingAll = allItems.filter(item => item.path !== activeItem.path)
        const nextCompleted = completed + submitted.completed_today_increment
        $reviewDueItems.set(remainingAll)
        $reviewCompletedToday.set(nextCompleted)
        void getStudyReviewStats()
          .then(nextStats => $reviewStats.set(nextStats))
          .catch(() => undefined)
        setLastResult(submitted)

        if (remaining[0]) {
          await openItem(remaining[0])
          setLastResult(submitted)
        } else {
          startedAtRef.current = null
          onSessionComplete(nextCompleted)
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      } finally {
        setSubmitting(false)
      }
    },
    [
      activeItem.path,
      allItems,
      completed,
      confidence,
      elapsedSeconds,
      items,
      onSessionComplete,
      openItem,
      projectId,
      response
    ]
  )

  if (loading) {
    return (
      <div>
        <PageLoader label={t.study.loadingReview} />
      </div>
    )
  }

  if (!detail) {
    return (
      <div className="rounded-2xl border border-destructive/40 bg-destructive/10 p-5 text-sm">
        <div>{error || t.study.loadingReview}</div>
        <Button className="mt-4" onClick={onClose} size="sm" variant="secondary">
          {t.study.backToList}
        </Button>
      </div>
    )
  }

  return (
    <div className="flex h-full min-h-[32rem] min-w-0 flex-col gap-4 xl:grid xl:grid-cols-[minmax(13rem,0.55fr)_minmax(0,1.7fr)]">
      <aside className="max-h-52 space-y-2 overflow-y-auto rounded-2xl border bg-card/40 p-3 xl:max-h-none">
        <div className="mb-3 flex items-center justify-between gap-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            {items.length} {t.study.remaining}
          </span>
          <button className="text-xs text-muted-foreground hover:text-foreground" onClick={onClose} type="button">
            {t.study.backToList}
          </button>
        </div>
        {items.map(item => (
          <button
            className={cn(
              'w-full min-w-0 rounded-xl border p-3 text-left text-sm transition-colors hover:bg-accent/60',
              item.path === activeItem.path && 'border-primary/50 bg-primary/10'
            )}
            key={item.path}
            onClick={() => void openItem(item)}
            type="button"
          >
            <div className="line-clamp-2 break-words font-semibold leading-snug">{item.title}</div>
            <div className="mt-1 text-xs text-muted-foreground">Lv.{item.review_level}</div>
          </button>
        ))}
      </aside>

      <section className="flex min-h-0 min-w-0 flex-col overflow-hidden rounded-3xl border bg-card/60 p-6 shadow-sm">
        {!projectId && (
          <div className="mb-4 shrink-0 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-sm">
            {t.study.selectProjectForReview}
          </div>
        )}
        {error && (
          <div className="mb-4 shrink-0 rounded-xl border border-destructive/40 bg-destructive/10 p-3 text-sm">
            {error}
          </div>
        )}
        {lastResult && (
          <div className="mb-4 shrink-0 rounded-xl border border-primary/30 bg-primary/10 p-3 text-sm">
            {t.study.reviewRecorded} · {t.study.nextReview}: {lastResult.review.next_review_at}
          </div>
        )}
        <div className="flex shrink-0 flex-wrap items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-primary">
              {activeItem.subject || t.study.reviewTitle}
            </div>
            <h3 className="mt-2 break-words text-xl font-semibold leading-tight">{activeItem.title}</h3>
          </div>
          <div className="shrink-0 rounded-full border bg-background/70 px-3 py-1 font-mono text-sm">
            {formatElapsed(elapsedSeconds)}
          </div>
        </div>

        <div className="mt-5 max-h-[min(42vh,32rem)] min-h-40 shrink overflow-y-auto rounded-2xl border bg-background/70 p-5">
          <CompactMarkdown text={detail.prompt_markdown} />
        </div>
        <div className="mt-5 shrink-0">
          <label className="text-sm font-semibold" htmlFor="study-review-response">
            {t.study.yourAnswer}
          </label>
          <Textarea
            className="mt-2 min-h-36 resize-y"
            disabled={answerRevealed || submitting}
            id="study-review-response"
            onChange={event => setResponse(event.target.value)}
            placeholder={t.study.answerPlaceholder}
            value={response}
          />
        </div>
        <div className="mt-4 shrink-0">
          <div className="text-sm font-semibold">{t.study.confidence}</div>
          <div className="mt-2 flex gap-2">
            {[1, 2, 3, 4, 5].map(value => (
              <button
                aria-label={`${t.study.confidence} ${value}`}
                className={cn(
                  'size-9 rounded-full border text-sm font-medium transition-colors',
                  confidence === value
                    ? 'border-primary bg-primary text-primary-foreground'
                    : 'bg-background hover:bg-accent'
                )}
                disabled={answerRevealed || submitting}
                key={value}
                onClick={() => setConfidence(value)}
                type="button"
              >
                {value}
              </button>
            ))}
          </div>
        </div>

        {!answerRevealed ? (
          <Button
            className="mt-5"
            disabled={!projectId || !response.trim() || confidence === null}
            onClick={() => setAnswerRevealed(true)}
          >
            {t.study.revealAnswer}
          </Button>
        ) : (
          <div className="mt-5 space-y-4">
            <div className="max-h-[min(32vh,24rem)] overflow-y-auto rounded-2xl border border-amber-500/30 bg-amber-500/10 p-5">
              <div className="mb-3 text-sm font-semibold">{t.study.referenceAnswer}</div>
              {detail.answer_markdown ? (
                <CompactMarkdown text={detail.answer_markdown} />
              ) : (
                <p className="text-sm text-muted-foreground">{t.study.noReferenceAnswer}</p>
              )}
            </div>
            <div>
              <div className="text-sm font-semibold">{t.study.selfGrade}</div>
              <div className="mt-2 flex flex-wrap gap-2">
                <Button disabled={submitting} onClick={() => void submitResult('incorrect')} variant="destructive">
                  {t.study.incorrect}
                </Button>
                <Button disabled={submitting} onClick={() => void submitResult('partial')} variant="secondary">
                  {t.study.partial}
                </Button>
                <Button disabled={submitting} onClick={() => void submitResult('correct')}>
                  {t.study.correct}
                </Button>
              </div>
            </div>
          </div>
        )}
      </section>
    </div>
  )
}
