import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { $annotationContext, $annotations } from '@/store/annotations'
import {
  $planReviews,
  approvePlanReview,
  openPlanReview,
  planReviewsForCurrentScope,
  requestPlanChanges
} from '@/store/plan-review'
import { openReview } from '@/store/review'
import { $activeSessionId, $connection, $currentCwd, $selectedStoredSessionId } from '@/store/session'

function useCurrentPlanReview(includeApproved = true) {
  useStore($connection)
  useStore($currentCwd)
  const reviews = planReviewsForCurrentScope(useStore($planReviews))
  const sessionId = useStore($activeSessionId)
  const storedSessionId = useStore($selectedStoredSessionId)

  return [...reviews]
    .reverse()
    .find(
      item =>
        (item.sessionId === sessionId || item.sessionId === storedSessionId) &&
        (includeApproved || item.status !== 'approved')
    )
}

export function PlanReviews() {
  const { t } = useI18n()
  const copy = t.desktop.annotations
  const annotations = useStore($annotations)
  const annotationContext = useStore($annotationContext)
  // Match both the live gateway id and the durable history id. A brand-new
  // session can produce a plan before it has been flushed to history.
  const review = useCurrentPlanReview()
  const version = review?.versions.at(-1)

  if (!review || !version) {
    return null
  }

  return (
    <section className="grid gap-2 border-b border-(--ui-stroke-tertiary) px-2.5 py-2">
      <div className="min-w-0">
        <div className="text-[0.68rem] font-medium text-(--ui-text-secondary)">
          {review.status === 'approved' ? copy.planApproved : copy.planAvailable}
        </div>
        <div className="truncate font-mono text-[0.62rem] text-(--ui-text-quaternary)" title={version.path}>
          {version.path}
        </div>
      </div>
      <div className="flex flex-wrap gap-1.5">
        <Button onClick={() => void openPlanReview(review.id)} size="xs" variant="secondary">
          {copy.planReview}
        </Button>
        {review.status === 'reviewing' &&
          annotationContext.kind === 'plan' &&
          annotationContext.contentHash === version.contentHash &&
          annotations.length > 0 && (
            <Button onClick={() => requestPlanChanges(review.id)} size="xs" variant="outline">
              {copy.planRequestChanges}
            </Button>
          )}
        {review.status === 'reviewing' && (
          <Button onClick={() => approvePlanReview(review.id)} size="xs">
            {copy.planApprove}
          </Button>
        )}
      </div>
    </section>
  )
}

/** Small, reactive statusbar affordance so a detected plan is discoverable
 * even when the Review pane is closed. The pane remains the review surface. */
export function PlanReviewStatus() {
  const { t } = useI18n()
  const review = useCurrentPlanReview(false)

  if (!review) {
    return null
  }

  return (
    <button
      className="inline-flex h-full items-center gap-1 px-1.5 text-[0.6875rem] text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground"
      onClick={() => {
        openReview()
        void openPlanReview(review.id)
      }}
      title={t.desktop.annotations.planAvailable}
      type="button"
    >
      <Codicon name="comment-discussion" size="0.75rem" />
      <span>{t.desktop.annotations.planReview}</span>
    </button>
  )
}
