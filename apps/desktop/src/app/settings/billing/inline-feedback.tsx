import { Button } from '@/components/ui/button'
import { openExternalLink } from '@/lib/external-link'
import { ExternalLink } from '@/lib/icons'

import type { BillingRefusal } from './api'
import { resolveRefusal } from './errors'
import { useStepUpFlow } from './use-step-up'

export function StepUpInlineAction({ flow }: { flow: ReturnType<typeof useStepUpFlow> }) {
  if (flow.verification) {
    return (
      <span className="inline-flex min-w-0 flex-wrap items-center gap-2">
        <span className="font-mono text-[0.72rem] font-semibold text-foreground">{flow.verification.code}</span>
        <Button onClick={flow.openVerification} size="sm" type="button" variant="outline">
          Open verification page
          <ExternalLink className="size-3.5" />
        </Button>
      </span>
    )
  }

  if (flow.message) {
    return (
      <span className="inline-flex min-w-0 flex-wrap items-center gap-2">
        <span>
          {flow.message.title}: {flow.message.text}
        </span>
        <Button onClick={flow.dismiss} size="sm" type="button" variant="outline">
          Dismiss
        </Button>
      </span>
    )
  }

  if (flow.phase === 'waiting') {
    return <span>Waiting for verification link…</span>
  }

  return (
    <Button onClick={() => void flow.start()} size="sm" type="button" variant="outline">
      Verify to continue
    </Button>
  )
}

export function BillingRefusalInline({ refusal }: { refusal: BillingRefusal | null }) {
  const stepUp = useStepUpFlow()

  if (!refusal) {
    return null
  }

  const resolved = resolveRefusal(refusal)
  const portalUrl = resolved.action.type === 'portal' ? resolved.action.url : undefined

  return (
    <div className="mt-2 flex min-w-0 flex-wrap items-center gap-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
      <span>
        <span className="font-medium text-foreground">{resolved.title}:</span> {resolved.message}
      </span>
      {resolved.action.type === 'step_up' && <StepUpInlineAction flow={stepUp} />}
      {portalUrl && (
        <Button onClick={() => openExternalLink(portalUrl)} size="sm" type="button" variant="outline">
          Open portal
          <ExternalLink className="size-3.5" />
        </Button>
      )}
    </div>
  )
}
