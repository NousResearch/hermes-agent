'use client'

import { useStore } from '@nanostores/react'
import { type FC, useCallback, useMemo, useRef, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import { Button } from '@/components/ui/button'
import { CardStack } from '@/components/ui/card-stack'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { releaseApprovalKey } from '@/lib/keybinds/approval-keys'
import { AlertCircle, ChevronDown, Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import {
  type ApprovalRequest,
  clearApprovalRequest,
  replayPendingApproval,
  sessionApprovalRequests
} from '@/store/prompts'
import { requestForOwnedSession } from '@/store/session-states'

// A session has one response surface, independent of mounted tool rows.
// Parallel tools and replay can deliver several independently correlated asks.
type ApprovalChoice = 'once' | 'session' | 'always' | 'deny'

export const PendingApprovalStack: FC = () => {
  const { t } = useI18n()
  const sessionId = useStore(useSessionView().$runtimeId)
  const requests = useStore(useMemo(() => sessionApprovalRequests(sessionId), [sessionId]))

  if (!requests.length) {
    return null
  }

  return (
    <section
      aria-label={t.assistant.approval.jumpToApproval}
      className="pointer-events-none absolute left-1/2 z-30 w-[calc(100%-2rem)] max-w-2xl -translate-x-1/2"
      data-approval-stack=""
      data-slot="tool-approval-stack"
      style={{ bottom: 'calc(var(--composer-measured-height) + 0.875rem)' }}
    >
      <div className="pointer-events-auto">
        <ApprovalCard key={requests[0].requestId ?? 'legacy'} request={requests[0]} count={requests.length} />
      </div>
    </section>
  )
}

interface ApprovalCardProps {
  request: ApprovalRequest
  count: number
}

const ApprovalCard: FC<ApprovalCardProps> = ({ request, count }) => {
  const { t } = useI18n()
  const copy = t.assistant.approval
  const gateway = useStore($gateway)
  const [submitting, setSubmitting] = useState<ApprovalChoice | null>(null)
  const submittingRef = useRef(false)
  // "Always allow" persists the pattern to ~/.hermes/config.yaml permanently, so
  // it goes through a confirm step rather than firing straight from the menu.
  const [confirmAlways, setConfirmAlways] = useState(false)
  const [showCommand, setShowCommand] = useState(false)
  const busy = submitting !== null
  // false when the backend won't honor a permanent allow (tirith warning) → hide "Always allow".
  const allowPermanent = request.allowPermanent !== false
  const choices = request.choices ?? (request.smartDenied ? ['once', 'deny'] : undefined)
  const allowSession = choices ? choices.includes('session') : true
  const allowAlways = choices ? choices.includes('always') : allowPermanent
  const hasMoreOptions = allowSession || allowAlways
  const hasCommand = request.command.trim().length > 0

  const respond = useCallback(
    async (choice: ApprovalChoice) => {
      const pending = sessionApprovalRequests(request.sessionId).get()
      if (submittingRef.current || !pending.some(item => item.requestId === request.requestId)) {
        return
      }

      if (!gateway) {
        notifyError(new Error(copy.gatewayDisconnected), copy.sendFailed)

        return
      }

      submittingRef.current = true
      setSubmitting(choice)

      try {
        // Route through the session's OWNER (tile route → known profile);
        // ambient only when no owner is known. The ambient socket follows
        // foreground focus, and for a cross-profile session it points at a
        // backend that never held this approval (#91684 client half).
        await requestForOwnedSession<{ resolved?: boolean }>(
          request.sessionId,
          // Bound (not wrapped) so the ambient fallback keeps the exact
          // 2-arg call shape gateway.request callers assert on.
          gateway.request.bind(gateway) as typeof gateway.request,
          'approval.respond',
          {
            all: false,
            choice,
            request_id: request.requestId,
            session_id: request.sessionId ?? undefined
          }
        )
        triggerHaptic(choice === 'deny' ? 'cancel' : 'submit')
        clearApprovalRequest(request.sessionId, request.requestId)
        void replayPendingApproval(gateway, request.sessionId).catch(() => undefined)
      } catch (error) {
        releaseApprovalKey()
        notifyError(error, copy.sendFailed)
        submittingRef.current = false
        setSubmitting(null)
      }
    },
    [copy.gatewayDisconnected, copy.sendFailed, gateway, request.requestId, request.sessionId]
  )

  return (
    <article data-slot="tool-approval-card" data-request-id={request.requestId} className="min-w-0">
      <CardStack
        count={count}
        direction="up"
        backClassName="rounded-3xl border border-(--stroke-nous) bg-(--ui-widget-surface-background) shadow-nous"
      >
        <div className={cn(WIDGET_SHELL_CLASS, 'grid gap-2 shadow-nous')}>
          <div className="flex min-w-0 items-center gap-2 text-xs text-(--ui-text-secondary)">
            <AlertCircle className="size-3.5 shrink-0 text-primary" />
            <span className="min-w-0 flex-1 truncate">{request.description || copy.jumpToApproval}</span>
            {count > 1 && <span className="shrink-0 tabular-nums text-(--ui-text-tertiary)">1 / {count}</span>}
          </div>
          {hasCommand && (
            <pre
              className={cn(
                'font-mono text-xs leading-relaxed text-foreground',
                showCommand ? 'max-h-40 overflow-auto whitespace-pre-wrap break-words' : 'truncate'
              )}
            >
              {request.command.trim()}
            </pre>
          )}
        </div>
      </CardStack>
      <div className="flex items-center gap-2 px-1 py-1.5" data-slot="tool-approval-actions">
        <Button data-approval-run="" disabled={busy} onClick={() => void respond('once')} size="xs">
          {submitting === 'once' ? <Loader2 className="animate-spin" /> : copy.run}
          <span className="opacity-60">↵</span>
        </Button>
        {hasMoreOptions && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button aria-label={copy.moreOptions} disabled={busy} size="icon-xs" variant="ghost">
                <ChevronDown />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" className="min-w-44">
              {allowSession && (
                <DropdownMenuItem onSelect={() => void respond('session')}>{copy.allowSession}</DropdownMenuItem>
              )}
              {allowAlways && (
                <DropdownMenuItem onSelect={() => setTimeout(() => setConfirmAlways(true), 0)}>
                  {copy.alwaysAllowMenu}
                </DropdownMenuItem>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        )}
        <Button data-approval-deny="" disabled={busy} onClick={() => void respond('deny')} size="xs" variant="ghost">
          {submitting === 'deny' ? <Loader2 className="animate-spin" /> : copy.reject}
          <span className="opacity-55">Esc</span>
        </Button>
        {hasCommand && (
          <Button aria-expanded={showCommand} onClick={() => setShowCommand(value => !value)} size="xs" variant="ghost">
            {copy.command}
            <ChevronDown className={cn('transition-transform', showCommand && 'rotate-180')} />
          </Button>
        )}
      </div>

      <Dialog onOpenChange={setConfirmAlways} open={confirmAlways}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{copy.alwaysTitle}</DialogTitle>
            <DialogDescription>{copy.alwaysDescription(request.description)}</DialogDescription>
          </DialogHeader>

          {request.command.trim() && (
            <pre className="max-h-32 overflow-auto whitespace-pre-wrap break-words rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background) px-2.5 py-1.5 font-mono text-xs leading-snug text-foreground">
              {request.command.trim()}
            </pre>
          )}

          <DialogFooter>
            <Button onClick={() => setConfirmAlways(false)} size="sm" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button
              onClick={() => {
                setConfirmAlways(false)
                void respond('always')
              }}
              size="sm"
              variant="destructive"
            >
              {copy.alwaysAllow}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </article>
  )
}
