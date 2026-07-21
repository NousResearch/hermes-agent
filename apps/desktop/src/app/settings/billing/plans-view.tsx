import { Button } from '@/components/ui/button'
import { openExternalLink } from '@/lib/external-link'
import { ChevronLeft, ExternalLink } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { Pill } from '../primitives'

import { TierArt } from './tier-art'
import type { BillingPlanTierView } from './use-billing-state'

function PlanCard({ tier }: { tier: BillingPlanTierView }) {
  const isCurrent = tier.state === 'current'

  return (
    <div
      className={cn(
        'flex min-w-0 flex-col gap-3 rounded-lg border p-4',
        isCurrent ? 'border-(--ui-green)/60 bg-(--ui-green)/5' : 'border-border/70 bg-muted/20'
      )}
    >
      <div className="flex min-w-0 items-center gap-3">
        <TierArt name={tier.name} />
        <div className="min-w-0">
          <div className="truncate text-[length:var(--conversation-text-font-size)] font-medium text-foreground">
            {tier.name}
          </div>
          <div className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {tier.priceDisplay}/mo
          </div>
        </div>
      </div>

      {tier.creditsDisplay && (
        <div className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          {tier.creditsDisplay}
        </div>
      )}

      <div className="mt-auto min-w-0 pt-1">
        {isCurrent && <Pill tone="primary">Current plan</Pill>}

        {tier.state === 'upgrade' && tier.action && (
          <Button onClick={() => openExternalLink(tier.action?.url ?? '')} size="sm" type="button" variant="outline">
            {tier.action.label}
            <ExternalLink className="size-3.5" />
          </Button>
        )}

        {tier.state === 'downgrade' && (
          <div className="flex min-w-0 flex-col gap-1.5">
            <Button disabled size="sm" type="button" variant="outline">
              Downgrade
            </Button>
            {tier.disabledCaption && (
              <span className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                {tier.disabledCaption}
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export function BillingPlansView({ onBack, tiers }: { onBack: () => void; tiers: BillingPlanTierView[] }) {
  return (
    <div className="@container">
      <div className="mb-2.5 flex items-center gap-2 pt-2 text-[length:var(--conversation-text-font-size)] font-medium">
        <Button
          aria-label="Back to billing"
          className="size-7 p-0 text-(--ui-text-tertiary)"
          onClick={onBack}
          size="sm"
          type="button"
          variant="ghost"
        >
          <ChevronLeft className="size-4" />
        </Button>
        <span>Plans</span>
      </div>

      {tiers.length > 0 ? (
        <div className="grid gap-3 @lg:grid-cols-2 @3xl:grid-cols-3">
          {tiers.map(tier => (
            <PlanCard key={tier.tierId} tier={tier} />
          ))}
        </div>
      ) : (
        <div className="rounded-lg border border-border/70 bg-muted/20 p-4 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          No plans are available to change to right now.
        </div>
      )}
    </div>
  )
}
