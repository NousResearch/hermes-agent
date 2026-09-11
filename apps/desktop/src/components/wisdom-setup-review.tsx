import { Button } from '@/components/ui/button'
import type { WisdomConsentAction, WisdomConsentInteraction } from '@/hermes'
import type { WisdomTranslations } from '@/i18n/types'

export function legacySetupReview(
  interaction: WisdomConsentInteraction,
  copy: WisdomTranslations
): WisdomConsentInteraction['setup_review'] {
  if (interaction.operation !== 'setup') {
    return undefined
  }

  const { step, setup_instruction: instruction, setup_explanation: explanation } = interaction.facts
  const reviewable = !!instruction && !!step && (step.phase === 'prerequisite' || !!step.command)
  const pending = interaction.state === 'pending' && interaction.expires_at * 1000 > Date.now()

  const labels = {
    defer: copy.notNow,
    inspect: copy.reviewFirst,
    confirm: step?.phase === 'prerequisite' ? copy.confirmSetupPrerequisite : copy.runSetupStep
  }

  // Older APIs accept these three actions only. Never infer recovery support or approve an incomplete review.
  return {
    summary: copy.reviewExact,
    detail: reviewable
      ? [explanation, instruction, pending ? copy.setupStepApprovalNotice : ''].filter(Boolean).join('\n\n')
      : copy.unavailable,
    command: step?.command || '',
    command_label: copy.setupCommand,
    actions: interaction.actions
      .filter(action => action === 'inspect' || (pending && reviewable && !interaction.deferred))
      .map(action => ({ action, label: labels[action], primary: action === 'confirm' }))
  }
}

interface WisdomSetupReviewProps {
  review: NonNullable<WisdomConsentInteraction['setup_review']>
  active: boolean
  busy: boolean
  onAction: (action: WisdomConsentAction) => void
}

export function WisdomSetupReview({ review, active, busy, onAction }: WisdomSetupReviewProps) {
  return (
    <section aria-label={review.summary} className="mt-3 min-w-0">
      <h4 className="text-sm font-medium" role="status">
        {review.summary}
      </h4>
      <p className="mt-2 whitespace-pre-wrap break-words text-sm text-(--ui-text-secondary)">{review.detail}</p>
      {review.command && (
        <div className="mt-3 min-w-0">
          <p className="text-xs font-medium">{review.command_label}</p>
          <pre className="mt-1 overflow-auto whitespace-pre-wrap break-all border border-(--ui-stroke-tertiary) p-2 text-xs">
            <code>{review.command}</code>
          </pre>
        </div>
      )}
      {active && (
        <div className="mt-3 flex flex-wrap gap-2 [&>button]:min-w-0 [&>button]:whitespace-normal [&>button]:break-words">
          {review.actions.map(action => (
            <Button
              disabled={busy}
              key={action.action}
              onClick={() => onAction(action.action)}
              size="sm"
              variant={action.primary ? 'default' : 'outline'}
            >
              {action.label}
            </Button>
          ))}
        </div>
      )}
    </section>
  )
}
