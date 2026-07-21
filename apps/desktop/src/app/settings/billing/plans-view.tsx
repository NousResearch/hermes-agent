import { useEffect, useRef } from 'react'

import { Button } from '@/components/ui/button'
import { openExternalLink } from '@/lib/external-link'
import { ChevronLeft, ExternalLink } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { Pill } from '../primitives'

import { BillingRefusalInline } from './inline-feedback'
import { TierArt } from './tier-art'
import { type BillingPlanTierView, formatBillingDate, formatMonthlyCreditsDelta } from './use-billing-state'
import { type DowngradePhase, useDowngradeFlow } from './use-subscription-change'

type DowngradeFlow = ReturnType<typeof useDowngradeFlow>

// The human sentence for the panel body, derived purely from the phase. `null` while
// a refusal is the only thing to show (BillingRefusalInline renders that separately).
function previewMessage(phase: DowngradePhase, fallbackTierName: string): null | string {
  if (phase.kind === 'previewing') {
    return 'Checking this change…'
  }

  if (phase.kind === 'previewFailed') {
    return null
  }

  const { preview } = phase
  const targetName = preview.target_tier_name ?? fallbackTierName
  const creditsDelta = formatMonthlyCreditsDelta(preview.monthly_credits_delta)

  switch (preview.effect) {
    case 'blocked':
      return preview.reason ?? 'That change cannot be made here.'

    case 'no_op':
      return `You are already on ${targetName} — nothing to change.`

    case 'scheduled':
      return (
        `Change to ${targetName} — takes effect ${formatBillingDate(preview.effective_at)}. No charge now; ` +
        `you keep your current plan until then.${creditsDelta ? ` Monthly credits change: ${creditsDelta}.` : ''}`
      )

    default:
      return 'This change cannot be scheduled here.'
  }
}

// The in-card preview → confirm panel for a downgrade (mirrors the TUI confirm flow).
function DowngradeConfirm({ flow, tier }: { flow: DowngradeFlow; tier: BillingPlanTierView }) {
  const active = flow.active
  const panelRef = useRef<HTMLDivElement>(null)
  const open = active?.target.tierId === tier.tierId

  // Move focus into the panel on open so keyboard users land on the confirm flow;
  // role="status"/aria-live announces the async preview text as it arrives.
  useEffect(() => {
    if (open) {
      panelRef.current?.focus()
    }
  }, [open])

  if (!active || active.target.tierId !== tier.tierId) {
    return null
  }

  const { phase } = active
  const captionCn = 'text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)'
  const refusal = phase.kind === 'previewFailed' || phase.kind === 'scheduleFailed' ? phase.refusal : null
  const busy = phase.kind === 'previewing' || phase.kind === 'scheduling'
  const message = previewMessage(phase, tier.name)

  const canConfirm =
    (phase.kind === 'ready' && phase.preview.effect === 'scheduled') ||
    phase.kind === 'scheduling' ||
    phase.kind === 'scheduleFailed'

  return (
    <div
      aria-live="polite"
      className="flex min-w-0 flex-col gap-2 rounded-md border border-border/70 bg-background/60 p-3 outline-none"
      ref={panelRef}
      role="status"
      tabIndex={-1}
    >
      {message && <div className={captionCn}>{message}</div>}

      <BillingRefusalInline refusal={refusal} />

      <div className="flex min-w-0 flex-wrap items-center gap-2">
        {phase.kind === 'previewFailed' ? (
          <Button disabled={busy} onClick={flow.retryPreview} size="sm" type="button">
            Try again
          </Button>
        ) : canConfirm ? (
          <Button disabled={busy} onClick={() => void flow.confirm()} size="sm" type="button">
            {phase.kind === 'scheduling' ? 'Scheduling…' : phase.kind === 'scheduleFailed' ? 'Try again' : 'Confirm downgrade'}
          </Button>
        ) : null}
        <Button disabled={busy} onClick={flow.cancel} size="sm" type="button" variant="outline">
          Cancel
        </Button>
      </div>
    </div>
  )
}

function PlanCard({ flow, tier }: { flow: DowngradeFlow; tier: BillingPlanTierView }) {
  const isCurrent = tier.state === 'current'
  const confirming = flow.active?.target.tierId === tier.tierId
  const cardRef = useRef<HTMLDivElement>(null)
  const wasConfirming = useRef(false)

  // When the confirm panel closes (cancel / scheduled), return focus to this tile
  // so keyboard focus is never left detached on the removed panel.
  useEffect(() => {
    if (wasConfirming.current && !confirming) {
      cardRef.current?.focus()
    }

    wasConfirming.current = confirming
  }, [confirming])

  return (
    <div
      className={cn(
        'flex min-w-0 flex-col gap-3 rounded-lg border p-4 outline-none',
        isCurrent ? 'border-(--ui-green)/60 bg-(--ui-green)/5' : 'border-border/70 bg-muted/20'
      )}
      ref={cardRef}
      tabIndex={-1}
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

        {tier.state === 'scheduled' && <Pill>Scheduled</Pill>}

        {tier.state === 'upgrade' && (
          <Button onClick={() => tier.action && openExternalLink(tier.action.url)} size="sm" type="button" variant="outline">
            {tier.action.label}
            <ExternalLink className="size-3.5" />
          </Button>
        )}

        {tier.state === 'downgrade' &&
          (confirming ? (
            <DowngradeConfirm flow={flow} tier={tier} />
          ) : (
            // Disabled while another tile's change is committing — no concurrent mutation.
            <Button
              disabled={flow.mutating}
              onClick={() => flow.begin({ tierId: tier.tierId, tierName: tier.name })}
              size="sm"
              type="button"
              variant="outline"
            >
              Downgrade
            </Button>
          ))}
      </div>
    </div>
  )
}

export function BillingPlansView({ onBack, tiers }: { onBack: () => void; tiers: BillingPlanTierView[] }) {
  // A scheduled downgrade lands the user back on the overview, where the plan card
  // now shows the pending state with its undo.
  const flow = useDowngradeFlow({ onScheduled: onBack })

  return (
    <div className="@container">
      <div className="mb-2.5 flex items-center gap-2 pt-2 text-[length:var(--conversation-text-font-size)] font-medium">
        <Button
          aria-label="Back to billing"
          className="size-7 p-0 text-(--ui-text-tertiary)"
          disabled={flow.mutating}
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
            <PlanCard flow={flow} key={tier.tierId} tier={tier} />
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
