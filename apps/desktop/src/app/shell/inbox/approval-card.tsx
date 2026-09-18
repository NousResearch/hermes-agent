import { useCallback, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'
import { type InboxRequestApproval, respondToApproval, type InboxRequest } from '@/store/inbox'
import { $gateway } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'

interface ApprovalCardProps {
  approval: InboxRequestApproval
  liveSessionId: string
  onResolved?: () => void
}

const ALLOWED_CHOICES = new Set(['once', 'session', 'always', 'deny'])

function choiceLabel(choice: string, approval: InboxRequestApproval): string {
  switch (choice) {
    case 'once': return 'Approve once'

    case 'session': return approval.allow_session === false ? 'Approve once' : 'Approve for session'

    case 'always': return approval.allow_permanent === false ? 'Approve once' : 'Always allow'

    case 'deny': return 'Deny'

    default: return choice
  }
}

function availableChoices(approval: InboxRequestApproval): string[] {
  return approval.choices.filter(choice => {
    if (!ALLOWED_CHOICES.has(choice)) {return false}

    if (choice === 'session' && approval.allow_session === false) {return false}

    if (choice === 'always' && approval.allow_permanent === false) {return false}

    return true
  })
}

export function ApprovalCard({ approval, liveSessionId, onResolved }: ApprovalCardProps) {
  const [submitting, setSubmitting] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [resolved, setResolved] = useState(false)
  const pinnedProfile = useRef($activeGatewayProfile.get() ?? '')
  const pinnedGateway = useRef($gateway.get())
  const submittingRef = useRef(false)

  const choices = availableChoices(approval)
  const hasRequestId = Boolean(approval.request_id)

  const respond = useCallback(async (choice: string) => {
    if (submittingRef.current || resolved || !hasRequestId) {return}
    submittingRef.current = true
    setSubmitting(choice)
    setError(null)

    try {
      const currentGateway = $gateway.get()
      const currentProfile = $activeGatewayProfile.get() ?? ''

      if (currentGateway !== pinnedGateway.current || currentProfile !== pinnedProfile.current) {
        setError('Profile changed — re-open to act')
        setSubmitting(null)
        return
      }

      const boundRequest: InboxRequest = (method, params) => currentGateway!.request(method, params ?? {})

      const result = await respondToApproval({
        choice,
        liveSessionId,
        profile: currentProfile,
        requestId: approval.request_id || undefined,
        request: boundRequest
      })

      const verifyGateway = $gateway.get()
      const verifyProfile = $activeGatewayProfile.get() ?? ''

      if (verifyGateway !== pinnedGateway.current || verifyProfile !== pinnedProfile.current) {
        setError('Profile changed — re-open to act')
        setSubmitting(null)
        return
      }

      if (result.resolved > 0) {
        setResolved(true)
        onResolved?.()
      } else {
        setError('Request may have been resolved already')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to respond')
    } finally {
      submittingRef.current = false
      setSubmitting(null)
    }
  }, [approval.request_id, hasRequestId, liveSessionId, onResolved, resolved])

  if (resolved) {
    return (
      <div className="rounded-md border border-(--ui-stroke-tertiary) bg-foreground/5 px-3 py-2 text-xs text-muted-foreground/70">
        Resolved
      </div>
    )
  }

  return (
    <div className="rounded-md border border-(--ui-stroke-tertiary) bg-foreground/5">
      <div className="px-3 py-2">
        <div className="flex items-center gap-2 text-xs text-(--ui-text-secondary)">
          <Codicon name="terminal" size="0.8rem" />
          <span className="font-medium">{approval.command || 'Pending approval'}</span>
        </div>
        {approval.description && (
          <p className="mt-1 text-[0.68rem] text-muted-foreground/70">{approval.description}</p>
        )}
      </div>
      {error && (
        <div className="px-3 pb-1 text-[0.62rem] text-destructive">{error}</div>
      )}
      <div className="flex items-center gap-1.5 px-3 pb-2 pt-1">
        {choices.length === 0 && (
          <span className="text-[0.62rem] text-muted-foreground/70">No supported actions available</span>
        )}
        {choices.map(choice => (
          <Button
            className={cn(choice === 'once' && 'relative')}
            disabled={submitting !== null || !hasRequestId}
            key={choice}
            onClick={() => void respond(choice)}
            size="xs"
            variant={choice === 'deny' ? 'text' : choice === 'once' ? 'default' : 'text'}
          >
            {submitting === choice ? '…' : choiceLabel(choice, approval)}
          </Button>
        ))}
      </div>
    </div>
  )
}
