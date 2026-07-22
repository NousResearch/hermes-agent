import { Button } from '@/components/ui/button'
import { ExternalLink } from '@/lib/icons'

import { BillingRefusalInline } from './inline-feedback'
import { openExternal } from './open-external'
import { TierArt } from './tier-art'
import type { BillingPlanCardView } from './use-billing-state'
import { useResumeFlow } from './use-subscription-change'

export function CurrentPlanCard({ onViewPlans, plan }: { onViewPlans: () => void; plan: BillingPlanCardView }) {
  const resumeFlow = useResumeFlow()

  return (
    <div className="@container">
      <div className="grid gap-3 py-3 @2xl:grid-cols-[minmax(0,1fr)_minmax(15rem,22rem)] @2xl:items-center">
        <div className="flex min-w-0 items-center gap-3">
          <TierArt name={plan.tierName} />
          <div className="min-w-0">
            <div className="flex min-w-0 flex-wrap items-baseline gap-x-2">
              <span className="truncate text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
                {plan.tierName}
              </span>
              {plan.price && (
                <span className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                  {plan.price}/mo
                </span>
              )}
            </div>
            <div className="mt-1 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              {plan.caption}
            </div>
          </div>
        </div>
        <div className="flex min-w-0 flex-wrap items-center justify-start gap-2 @2xl:justify-end">
          {plan.action && (
            <Button onClick={onViewPlans} size="sm" type="button" variant="outline">
              {plan.action.label}
            </Button>
          )}
          {/* Scheduled downgrade → chargeless undo (subscription.resume), no confirm. */}
          {plan.pending && (
            <Button disabled={resumeFlow.busy} onClick={() => void resumeFlow.resume()} size="sm" type="button">
              {resumeFlow.busy ? 'Undoing…' : 'Undo'}
            </Button>
          )}
          {plan.link && (
            <Button onClick={() => plan.link && openExternal(plan.link.url)} size="sm" type="button" variant="outline">
              {plan.link.label}
              <ExternalLink className="size-3.5" />
            </Button>
          )}
        </div>
      </div>
      <BillingRefusalInline refusal={resumeFlow.refusal} />
    </div>
  )
}
