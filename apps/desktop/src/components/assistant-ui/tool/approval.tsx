'use client'

import { useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import {
  createContext,
  type FC,
  type FormEvent,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { SCAFFOLD_LABEL_CLASS, ScaffoldRow } from '@/components/chat/scaffold-row'
import { Button } from '@/components/ui/button'
import { CardStack, type CardStackAction } from '@/components/ui/card-stack'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { ChevronDown, Loader2 } from '@/lib/icons'
import { releaseApprovalKey } from '@/lib/keybinds/approval-keys'
import { cn } from '@/lib/utils'
import { $gateway } from '@/store/gateway'
import { reconnectAction } from '@/store/gateway-reconnect'
import { notifyError } from '@/store/notifications'
import {
  accessTarget,
  answerApproval,
  type ApprovalRequest,
  clearApprovalRequest,
  isCommandApproval,
  networkTarget,
  networkTargetName,
  replayPendingApproval,
  sessionApprovalRequests,
  sessionApprovalStackSize
} from '@/store/prompts'
import { $showToolActivity } from '@/store/tool-activity'
import { setToolDisclosureOpen } from '@/store/tool-view'

import { isApprovalActivity } from './approval-activity'
import { toolEntryDisclosureId } from './fallback-model/targets'
import { isToolCallPart, summarizeToolRun } from './run-summary'

type ApprovalChoice = 'once' | 'session' | 'always' | 'deny'
// A deny reason is one short line for the agent.
const DENY_REASON_MAX = 500
export const ApprovalPlacementContext = createContext<'inline' | 'floating'>('inline')

// One transcript-owned host for the session. Execution rows never mount,
// register or re-home this queue, including the first delayed tool.start.
export const PendingApprovalStack: FC = () => {
  const { t } = useI18n()
  const placement = useContext(ApprovalPlacementContext)
  const sessionId = useStore(useSessionView().$runtimeId)
  const requests = useStore(useMemo(() => sessionApprovalRequests(sessionId), [sessionId]))
  const total = useStore(useMemo(() => sessionApprovalStackSize(sessionId), [sessionId]))
  const reduced = useReducedMotion()

  return (
    <motion.section
      animate={{ paddingBlock: requests.length ? 8 : 0 }}
      aria-label={t.assistant.approval.jumpToApproval}
      className={cn(
        'min-w-0',
        placement === 'floating' ? 'sticky bottom-4 z-10 mt-auto w-full max-w-xl self-center' : 'w-full max-w-xl'
      )}
      data-approval-placement={placement}
      data-approval-stack=""
      data-session-id={sessionId ?? undefined}
      data-slot="tool-approval-stack"
      initial={false}
      transition={reduced || requests.length ? { duration: 0 } : { duration: 0.22, ease: 'easeInOut' }}
    >
      {/* The run summary explains a command approval; a folder or site card is not about the turn's tools. */}
      <ApprovalActivity floating={placement === 'floating'} visible={requests.some(isCommandApproval)} />
      <ApprovalQueue floating={placement === 'floating'} requests={requests} total={total} />
    </motion.section>
  )
}

function ApprovalActivity({ floating, visible }: { floating: boolean; visible: boolean }) {
  const { t } = useI18n()
  const reduced = useReducedMotion()
  const showToolActivity = useStore($showToolActivity)

  const summary = useAuiState(state => {
    if (!visible) {
      return ''
    }

    const start = state.thread.messages.findLastIndex(message => message.role === 'user')

    const tools = state.thread.messages
      .slice(start + 1)
      .filter(message => message.role === 'assistant')
      .flatMap(message => message.content.filter(isToolCallPart).filter(isApprovalActivity))

    return tools.length
      ? summarizeToolRun(
          tools,
          tools.some(tool => tool.result === undefined)
        )
      : ''
  })

  const disclosureIds = useAuiState(state => {
    if (!visible) {
      return ''
    }

    const start = state.thread.messages.findLastIndex(message => message.role === 'user')

    return state.thread.messages
      .slice(start + 1)
      .filter(message => message.role === 'assistant')
      .flatMap(message =>
        message.content
          .filter(isToolCallPart)
          .filter(isApprovalActivity)
          .map(tool => toolEntryDisclosureId(message.id, tool))
      )
      .join('\n')
  })

  // The pending approval stays. The run summary beside it is tool feed and
  // follows display.tool_progress alongside the other process rows.
  if (!showToolActivity) {
    return null
  }

  return (
    <AnimatePresence initial={false}>
      {visible && (
        <motion.div
          animate={{ height: 'auto', opacity: 1 }}
          className="overflow-hidden"
          exit={{ height: 0, opacity: 0 }}
          initial={{ height: 0, opacity: 0 }}
          key="activity"
          transition={{ duration: reduced ? 0 : 0.22, ease: 'easeInOut' }}
        >
          <div
            className={cn('mb-1 min-w-0', floating && 'rounded bg-(--ui-chat-surface-background)')}
            data-approval-activity=""
            data-glass-opaque={floating ? '' : undefined}
            data-tool-summary=""
          >
            <ScaffoldRow
              onToggle={
                disclosureIds
                  ? () => {
                      for (const id of disclosureIds.split('\n')) {
                        setToolDisclosureOpen(id, true)
                      }
                    }
                  : undefined
              }
              open={false}
            >
              <span className={cn(SCAFFOLD_LABEL_CLASS, 'truncate')}>
                {summary || t.assistant.approval.jumpToApproval}
              </span>
            </ScaffoldRow>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}

async function sendApproval(request: ApprovalRequest, choice: ApprovalChoice, reason?: string) {
  const gateway = $gateway.get()

  if (!gateway) {
    throw new Error('Gateway disconnected')
  }

  if (
    !sessionApprovalRequests(request.sessionId)
      .get()
      .some(item => item.requestId === request.requestId)
  ) {
    return
  }

  await answerApproval(gateway, request, choice, false, reason)
  triggerHaptic(choice === 'deny' ? 'cancel' : 'submit')
  clearApprovalRequest(request.sessionId, request.requestId)
  void replayPendingApproval(gateway, request.sessionId).catch(() => undefined)
}

export function ApprovalQueue({
  requests,
  total,
  floating = false
}: {
  requests: ApprovalRequest[]
  total: number
  floating?: boolean
}) {
  const { t } = useI18n()

  return (
    <CardStack
      getKey={request => request.requestId ?? 'legacy'}
      items={requests}
      onSwipe={(request, _side, action) => {
        void action
          .depart(() => sendApproval(request, 'deny'))
          .catch(error => {
            releaseApprovalKey()
            notifyError(error, t.assistant.approval.sendFailed)
          })
      }}
      surfaceClassName={cn(
        'rounded-xl border bg-(--ui-chat-surface-background)',
        floating ? 'border-(--stroke-nous) shadow-nous' : 'border-(--ui-stroke-secondary)'
      )}
      swipeDirections={['left']}
    >
      {(request, action) => (
        <ApprovalCard
          position={total - requests.length + requests.indexOf(request) + 1}
          request={request}
          stack={action}
          total={total}
        />
      )}
    </CardStack>
  )
}

interface ApprovalCardProps {
  request: ApprovalRequest
  total: number
  position: number
  stack: CardStackAction
}

// A folder or a site is drawn with its spaces as they are: ordinary text joins a run of spaces
// into one, and two names that differ only in spacing would read the same.
const NAME_STYLE = { whiteSpace: 'pre-wrap' } as const

/** The folder or site a folder / site card names, plus its risk line. */
function ApprovalTargetLines({ request }: { request: ApprovalRequest }) {
  const { t } = useI18n()
  const copy = t.assistant.approval
  const access = accessTarget(request)
  const network = networkTarget(request)

  return (
    <div className="grid gap-0.5 px-2.5 py-2 text-xs leading-relaxed" data-slot="tool-approval-target">
      {access ? (
        <p className="m-0 min-w-0 [overflow-wrap:anywhere]">
          <span className="font-mono text-(--ui-text-primary)" style={NAME_STYLE}>
            {access.path}
          </span>
          {access.access && (
            <span className="text-(--ui-text-secondary)">
              {' · '}
              {access.access === 'read_write' ? copy.access.readWrite : copy.access.readOnly}
            </span>
          )}
        </p>
      ) : network ? (
        <p className="m-0 min-w-0 [overflow-wrap:anywhere]">
          <span className="font-mono text-(--ui-text-primary)" style={NAME_STYLE}>
            {networkTargetName(network)}
          </span>
        </p>
      ) : (
        request.description.trim() && <p className="m-0 min-w-0 break-words">{request.description.trim()}</p>
      )}
      {request.risk && (
        <p className="m-0 min-w-0 break-words text-(--ui-text-secondary)" data-slot="tool-approval-risk">
          {request.risk}
        </p>
      )}
    </div>
  )
}

const ApprovalCard: FC<ApprovalCardProps> = ({ request, total, position, stack }) => {
  if (!isCommandApproval(request)) {
    return <AccessApprovalCard position={position} request={request} stack={stack} total={total} />
  }

  return <CommandApprovalCard position={position} request={request} stack={stack} total={total} />
}

/** Focus, submit and gateway handling shared by every approval card. */
function useApprovalResponder(request: ApprovalRequest, stack: CardStackAction) {
  const { t } = useI18n()
  const copy = t.assistant.approval
  const gateway = useStore($gateway)
  const [submitting, setSubmitting] = useState<ApprovalChoice | null>(null)
  const submittingRef = useRef(false)
  // Answering with the pointer moves focus onto the card, and the card then
  // unmounts, which parks focus on <body> — where type-to-focus routes the next
  // keystrokes into the chat composer. For a computer_use flow the agent may
  // have just aimed its input at another pane (terminal, preview), so the
  // approval hands focus back to whatever held it before the press (#113839).
  const focusOrigin = useRef<HTMLElement | null>(null)
  const cardRef = useRef<HTMLElement | null>(null)

  const rememberFocusOrigin = useCallback(() => {
    const active = document.activeElement

    if (active instanceof HTMLElement && active !== document.body && !cardRef.current?.contains(active)) {
      focusOrigin.current = active
    }
  }, [])

  const restoreFocusOrigin = useCallback((): boolean => {
    const origin = focusOrigin.current
    const active = document.activeElement

    focusOrigin.current = null

    // Only when the answer itself is what stranded focus — never steal from a
    // surface the user moved to while the reply was in flight.
    if (origin?.isConnected && (!active || active === document.body || cardRef.current?.contains(active))) {
      origin.focus({ preventScroll: true })

      return true
    }

    return false
  }, [])

  const respond = useCallback(
    async (choice: ApprovalChoice, reason?: string) => {
      const pending = sessionApprovalRequests(request.sessionId).get()

      if (submittingRef.current || !pending.some(item => item.requestId === request.requestId)) {
        return
      }

      if (!gateway) {
        notifyError(new Error(copy.gatewayDisconnected), copy.sendFailed, { action: reconnectAction() })

        return
      }

      submittingRef.current = true
      setSubmitting(choice)

      try {
        await stack.depart(() => sendApproval(request, choice, reason))
        restoreFocusOrigin()
      } catch (error) {
        releaseApprovalKey()
        notifyError(error, copy.sendFailed)
        submittingRef.current = false
        setSubmitting(null)
      }
    },
    [copy.gatewayDisconnected, copy.sendFailed, gateway, request, restoreFocusOrigin, stack]
  )

  return { cardRef, rememberFocusOrigin, respond, restoreFocusOrigin, submitting }
}

/** A sandbox asking for a folder (`access`) or a site (`network`). Allow lasts
 *  until the sandbox is switched off, Always allow saves the rule, and Deny can
 *  carry a reason for the agent. */
const AccessApprovalCard: FC<ApprovalCardProps> = ({ request, total, position, stack }) => {
  const { t } = useI18n()
  const copy = t.assistant.approval
  const kindCopy = request.kind === 'network' ? copy.network : copy.access
  const { cardRef, rememberFocusOrigin, respond, restoreFocusOrigin, submitting } = useApprovalResponder(request, stack)
  const [confirmAlways, setConfirmAlways] = useState(false)
  const [denying, setDenying] = useState(false)
  const denyButtonRef = useRef<HTMLButtonElement | null>(null)
  const present = stack.active
  const busy = submitting !== null || !present || stack.busy
  const allowAlways = request.choices ? request.choices.includes('always') : request.allowPermanent !== false

  // The reason field closes with focus in it: hand focus back to where the user
  // was before Deny, else to Deny itself.
  const cancelDeny = () => {
    setDenying(false)
    setTimeout(() => {
      if (!restoreFocusOrigin()) {
        denyButtonRef.current?.focus({ preventScroll: true })
      }
    }, 0)
  }

  return (
    <article
      aria-hidden={!present || undefined}
      className="min-w-0"
      data-approval-kind={request.kind}
      data-request-id={request.requestId}
      data-slot="tool-approval-card"
      inert={!present}
      onPointerDownCapture={rememberFocusOrigin}
      ref={cardRef}
    >
      <div className="flex items-center gap-2 px-2.5 pt-2 text-xs text-(--ui-text-secondary)">
        <Codicon name={request.kind === 'network' ? 'globe' : 'folder'} size="0.875rem" />
        <span>{kindCopy.title}</span>
        {total > 1 && (
          <span className="ml-auto text-[0.6875rem] tabular-nums text-(--ui-text-tertiary)">
            {position} / {total}
          </span>
        )}
      </div>
      <ApprovalTargetLines request={request} />
      {denying ? (
        <DenyReasonForm
          busy={busy}
          onCancel={cancelDeny}
          onSubmit={reason => void respond('deny', reason)}
          submitting={submitting === 'deny'}
        />
      ) : (
        <div className="flex items-center justify-end gap-1.5 px-2 pb-2 pt-1" data-slot="tool-approval-actions">
          <Button
            data-approval-deny=""
            disabled={busy}
            onClick={() => {
              // Escape reaches this button from the keyboard too; keep where the
              // user was so the reason field can hand focus back.
              rememberFocusOrigin()
              setDenying(true)
            }}
            ref={denyButtonRef}
            size="sm"
            variant="text"
          >
            {copy.deny}
          </Button>
          {allowAlways && (
            <Button disabled={busy} onClick={() => setConfirmAlways(true)} size="sm" variant="secondary">
              {copy.alwaysAllowMenu}
            </Button>
          )}
          {/* No Enter shortcut here: this card can outlive its turn, and an Enter meant for something else must
              not allow a folder or a site. Enter on the focused button still presses it. */}
          <Tip label={copy.allowHint}>
            <Button disabled={busy} onClick={() => void respond('once')} size="sm">
              {submitting === 'once' ? <Loader2 className="size-3 animate-spin" /> : copy.allow}
            </Button>
          </Tip>
        </div>
      )}

      <Dialog onOpenChange={setConfirmAlways} open={confirmAlways}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{kindCopy.alwaysTitle}</DialogTitle>
            <DialogDescription>{kindCopy.alwaysDescription}</DialogDescription>
          </DialogHeader>

          <div className="rounded-md border border-(--ui-stroke-tertiary)">
            <ApprovalTargetLines request={request} />
          </div>

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

/** Deny's optional reason. Enter (or Deny) sends it, Escape (or Cancel) goes
 *  back to the card's answers. While it is open, the stack's Enter / Escape
 *  keys act on these two buttons. */
function DenyReasonForm({
  busy,
  onCancel,
  onSubmit,
  submitting
}: {
  busy: boolean
  onCancel: () => void
  onSubmit: (reason: string) => void
  submitting: boolean
}) {
  const { t } = useI18n()
  const copy = t.assistant.approval
  const [reason, setReason] = useState('')
  const inputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    inputRef.current?.focus({ preventScroll: true })
  }, [])

  const submit = (event: FormEvent) => {
    event.preventDefault()

    if (!busy) {
      onSubmit(reason.trim())
    }
  }

  return (
    <form className="flex items-center gap-1.5 px-2 pb-2 pt-1" data-slot="tool-approval-actions" onSubmit={submit}>
      <Input
        aria-label={copy.denyReason}
        className="min-w-0 flex-1"
        disabled={busy}
        maxLength={DENY_REASON_MAX}
        onChange={event => setReason(event.target.value)}
        onKeyDown={event => {
          if (event.key === 'Escape' && !event.nativeEvent.isComposing) {
            event.preventDefault()
            event.stopPropagation()
            onCancel()
          }
        }}
        placeholder={copy.denyReasonPlaceholder}
        ref={inputRef}
        size="sm"
        value={reason}
      />
      <Button data-approval-deny="" disabled={busy} onClick={onCancel} size="sm" type="button" variant="text">
        {t.common.cancel}
      </Button>
      <Button data-approval-run="" disabled={busy} size="sm" type="submit">
        {submitting ? <Loader2 className="size-3 animate-spin" /> : copy.deny}
        <span className="opacity-60">↵</span>
      </Button>
    </form>
  )
}

const CommandApprovalCard: FC<ApprovalCardProps> = ({ request, total, position, stack }) => {
  const { t } = useI18n()
  const copy = t.assistant.approval
  const { cardRef, rememberFocusOrigin, respond, submitting } = useApprovalResponder(request, stack)
  // "Always allow" persists the pattern to ~/.hermes/config.yaml permanently, so
  // it goes through a confirm step rather than firing straight from the menu.
  const [confirmAlways, setConfirmAlways] = useState(false)

  const present = stack.active
  const busy = submitting !== null || !present || stack.busy

  // false when the backend won't honor a permanent allow → hide "Always allow".
  const allowPermanent = request.allowPermanent !== false
  const choices = request.choices ?? (request.smartDenied ? ['once', 'deny'] : undefined)
  const allowSession = choices ? choices.includes('session') : true
  const allowAlways = choices ? choices.includes('always') : allowPermanent
  const hasMoreOptions = allowSession || allowAlways
  const hasCommand = request.command.trim().length > 0
  // A plugin `approve` rule escalates through the same gate with a synthetic
  // display target (`<tool> (plugin approval rule)`) while the real command /
  // change lives in `description` — the user approves a label they cannot act
  // on otherwise. Show the description whenever the command is missing or one
  // of these synthetic labels; a real command keeps rendering as the command.
  const SYNTHETIC_COMMAND_RE = /^<[^>]+> \(/

  const showsDescription = !hasCommand || SYNTHETIC_COMMAND_RE.test(request.command.trim())

  const details = showsDescription ? request.description.trim() : request.command.trim()

  return (
    <article
      aria-hidden={!present || undefined}
      className="min-w-0"
      data-request-id={request.requestId}
      data-slot="tool-approval-card"
      inert={!present}
      onPointerDownCapture={rememberFocusOrigin}
      ref={cardRef}
    >
      <div className="flex items-center gap-2 px-2.5 pt-2 text-xs text-(--ui-text-secondary)">
        <Codicon name="terminal" size="0.875rem" />
        <span>{showsDescription ? copy.commandDetails : copy.command}</span>
        {total > 1 && (
          <span className="ml-auto text-[0.6875rem] tabular-nums text-(--ui-text-tertiary)">
            {position} / {total}
          </span>
        )}
      </div>
      {details.length > 0 && (
        <pre className="m-0 max-h-40 overflow-auto whitespace-pre-wrap break-words px-2.5 py-2 font-mono text-xs leading-relaxed text-(--ui-text-primary)">
          {details}
        </pre>
      )}
      <div className="flex items-center justify-end gap-1.5 px-2 pb-2 pt-1" data-slot="tool-approval-actions">
        <Button data-approval-deny="" disabled={busy} onClick={() => void respond('deny')} size="sm" variant="text">
          {submitting === 'deny' ? <Loader2 className="size-3 animate-spin" /> : copy.reject}
        </Button>
        {hasMoreOptions && (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button aria-label={copy.moreOptions} disabled={busy} size="sm" variant="secondary">
                {copy.alwaysAllowMenu}
                <ChevronDown className="size-3" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="min-w-44">
              {allowSession && (
                <DropdownMenuItem onSelect={() => void respond('session')}>{copy.allowSession}</DropdownMenuItem>
              )}
              {allowAlways && (
                <DropdownMenuItem
                  onSelect={() => {
                    // Defer one tick so the menu fully unmounts before the dialog
                    // mounts — otherwise Radix's focus-return races the dialog and
                    // dismisses it via onInteractOutside.
                    setTimeout(() => setConfirmAlways(true), 0)
                  }}
                >
                  {copy.alwaysAllowMenu}
                </DropdownMenuItem>
              )}
              <DropdownMenuItem onSelect={() => void respond('deny')} variant="destructive">
                {copy.reject}
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        )}
        <Button data-approval-run="" disabled={busy} onClick={() => void respond('once')} size="sm">
          {submitting === 'once' ? <Loader2 className="size-3 animate-spin" /> : copy.run}
          <span className="opacity-60">↵</span>
        </Button>
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
