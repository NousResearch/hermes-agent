'use client'

import { useStore } from '@nanostores/react'
import { type FormEvent, useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { AlertTriangle, ChevronDown, KeyRound, Loader2, Lock } from '@/lib/icons'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import {
  $approvalRequest,
  type ApprovalRequest,
  clearApprovalRequest,
  $secretRequest,
  $sudoRequest,
  clearSecretRequest,
  clearSudoRequest
} from '@/store/prompts'

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'

// Renders the modal mid-turn prompts the gateway raises and waits on: sudo
// password and skill secret capture. (Dangerous-command / execute_code approval
// is rendered INLINE on the pending tool row instead — see
// components/assistant-ui/tool-approval.tsx — so it reads like an inline "Run"
// affordance rather than a blocking modal.) Each Python-side caller blocks the
// agent thread until the matching `*.respond` RPC lands; without a renderer the
// agent stalls until its timeout and the tool is BLOCKED (the bug this fixes —
// desktop handled clarify.request but not these). Any close path (Esc, backdrop
// click) funnels through Radix's single `onOpenChange(false)` and maps to a
// refusal, so silence is never mistaken for consent, matching the TUI. We
// deliberately do NOT add onEscapeKeyDown / onInteractOutside handlers — they'd
// fire a second `*.respond` alongside onOpenChange (double-send) or block the
// backdrop-dismiss path.

// ---------------------------------------------------------------------------
// Global approval overlay
// ---------------------------------------------------------------------------
// The inline ApprovalBar (tool-approval.tsx) renders inside the tool-group
// body div.  When the group is collapsed (hidden={isGroup && !open}), the bar
// is visually hidden and the user cannot approve or deny — the agent stalls.
// This overlay renders the same approval affordance at the global level
// (fixed bottom) so it is always visible regardless of the tool group's
// collapsed state.  It shares the same $approvalRequest atom; when the inline
// bar resolves the request, this overlay automatically disappears too.

type ApprovalChoice = 'once' | 'session' | 'always' | 'deny'

function ApprovalOverlay() {
  const request = useStore($approvalRequest)
  const gateway = useStore($gateway)
  const [submitting, setSubmitting] = useState<ApprovalChoice | null>(null)
  const [confirmAlways, setConfirmAlways] = useState(false)
  const busy = submitting !== null

  const respond = useCallback(
    async (choice: ApprovalChoice) => {
      if (busy || !$approvalRequest.get()) {
        return
      }

      if (!gateway) {
        notifyError(new Error('Hermes gateway is not connected'), 'Could not send approval response')
        return
      }

      setSubmitting(choice)

      try {
        await gateway.request<{ resolved?: boolean }>('approval.respond', {
          choice,
          session_id: request?.sessionId ?? undefined
        })
        triggerHaptic(choice === 'deny' ? 'cancel' : 'submit')
        clearApprovalRequest()
      } catch (error) {
        notifyError(error, 'Could not send approval response')
        setSubmitting(null)
      }
    },
    [busy, gateway, request?.sessionId]
  )

  // Keyboard shortcuts: ⌘/Ctrl+Enter → Run, Esc → Deny.
  // These fire only when no inline ApprovalBar is mounted (i.e. the tool
  // group is collapsed).  When the group is expanded, the inline bar owns
  // the keyboard and this overlay defers — both share the same atom so the
  // result is identical either way.
  useEffect(() => {
    if (!request || confirmAlways) {
      return
    }

    const onKeyDown = (event: KeyboardEvent) => {
      // If the inline bar is mounted (group expanded), it handles shortcuts.
      const inlineBar = document.querySelector('[data-slot="tool-approval-inline"]')
      if (inlineBar) {
        return
      }

      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault()
        void respond('once')
      } else if (event.key === 'Escape') {
        event.preventDefault()
        void respond('deny')
      }
    }

    window.addEventListener('keydown', onKeyDown, true)
    return () => window.removeEventListener('keydown', onKeyDown, true)
  }, [request, confirmAlways, respond])

  if (!request) {
    return null
  }

  return (
    <div
      className="fixed bottom-4 left-1/2 z-50 -translate-x-1/2 animate-in fade-in slide-in-from-bottom-2"
      data-slot="tool-approval-overlay"
    >
      <div className="flex items-center gap-2 rounded-xl border border-primary/25 bg-background/95 px-3 py-2 shadow-lg backdrop-blur-sm">
        <AlertTriangle className="size-4 shrink-0 text-primary" />
        <span className="max-w-60 truncate text-xs text-(--ui-text-secondary)">
          {request.description || request.command || 'Approval required'}
        </span>
        <span className="mx-1 h-4 w-px bg-border" />
        <Button
          className="h-7 gap-1 rounded-md px-2 text-xs font-medium"
          disabled={busy}
          onClick={() => void respond('once')}
          size="xs"
        >
          {submitting === 'once' ? <Loader2 className="size-3 animate-spin" /> : 'Run'}
          {submitting !== 'once' && <span className="text-[0.625rem] opacity-55">⌘⏎</span>}
        </Button>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              aria-label="More approval options"
              className="h-7 w-7 rounded-md px-0"
              disabled={busy}
              size="xs"
              variant="ghost"
            >
              <ChevronDown className="size-3" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="min-w-40">
            <DropdownMenuItem onSelect={() => void respond('session')}>Allow this session</DropdownMenuItem>
            <DropdownMenuItem
              onSelect={() => {
                setTimeout(() => setConfirmAlways(true), 0)
              }}
            >
              Always allow…
            </DropdownMenuItem>
            <DropdownMenuItem onSelect={() => void respond('deny')} variant="destructive">
              Reject
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
        <Button
          className="h-7 gap-1 rounded-md px-1.5 text-xs text-(--ui-text-tertiary)"
          disabled={busy}
          onClick={() => void respond('deny')}
          size="xs"
          variant="ghost"
        >
          {submitting === 'deny' ? <Loader2 className="size-3 animate-spin" /> : 'Reject'}
          {submitting !== 'deny' && <span className="text-[0.625rem] opacity-55">Esc</span>}
        </Button>

        <Dialog onOpenChange={setConfirmAlways} open={confirmAlways}>
          <DialogContent className="max-w-md">
            <DialogHeader>
              <DialogTitle>Always allow this command?</DialogTitle>
              <DialogDescription>
                This adds the &ldquo;{request.description}&rdquo; pattern to your permanent allowlist (
                <code className="font-mono text-xs">~/.hermes/config.yaml</code>). Hermes won&apos;t ask again for
                commands like this — in this session or any future one.
              </DialogDescription>
            </DialogHeader>
            {request.command.trim() && (
              <pre className="max-h-32 overflow-auto whitespace-pre-wrap break-words rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background) px-2.5 py-1.5 font-mono text-xs leading-snug text-foreground">
                {request.command.trim()}
              </pre>
            )}
            <DialogFooter>
              <Button onClick={() => setConfirmAlways(false)} size="sm" variant="ghost">
                Cancel
              </Button>
              <Button
                onClick={() => {
                  setConfirmAlways(false)
                  void respond('always')
                }}
                size="sm"
                variant="destructive"
              >
                Always allow
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  )
}

function SudoDialog() {
  const { t } = useI18n()
  const copy = t.prompts
  const request = useStore($sudoRequest)
  const gateway = useStore($gateway)
  const [password, setPassword] = useState('')
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    setPassword('')
    setSubmitting(false)
  }, [request?.requestId])

  const send = useCallback(
    async (value: string) => {
      if (!request) {
        return
      }

      if (!gateway) {
        notifyError(new Error(copy.gatewayDisconnected), copy.sudoSendFailed)

        return
      }

      setSubmitting(true)

      try {
        await gateway.request<{ status?: string }>('sudo.respond', {
          password: value,
          request_id: request.requestId
        })
        triggerHaptic('submit')
        clearSudoRequest(request.sessionId, request.requestId)
      } catch (error) {
        notifyError(error, copy.sudoSendFailed)
        setSubmitting(false)
      }
    },
    [copy.gatewayDisconnected, copy.sudoSendFailed, gateway, request]
  )

  // Cancel → empty password. The backend treats an empty sudo response as a
  // failed sudo (no command runs), so closing the dialog is a safe refusal.
  const onOpenChange = useCallback(
    (open: boolean) => {
      if (!open && !submitting && request) {
        void send('')
      }
    },
    [request, send, submitting]
  )

  const onSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      void send(password)
    },
    [password, send]
  )

  if (!request) {
    return null
  }

  return (
    <Dialog onOpenChange={onOpenChange} open>
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle icon={Lock}>{copy.sudoTitle}</DialogTitle>
          <DialogDescription>{copy.sudoDesc}</DialogDescription>
        </DialogHeader>

        <form className="grid gap-3" onSubmit={onSubmit}>
          <Input
            autoFocus
            disabled={submitting}
            onChange={event => setPassword(event.target.value)}
            placeholder={copy.sudoPlaceholder}
            type="password"
            value={password}
          />
          <DialogFooter>
            <Button disabled={submitting} onClick={() => void send('')} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button disabled={submitting} type="submit">
              {submitting ? <Loader2 className="size-3.5 animate-spin" /> : t.common.send}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function SecretDialog() {
  const { t } = useI18n()
  const copy = t.prompts
  const request = useStore($secretRequest)
  const gateway = useStore($gateway)
  const [value, setValue] = useState('')
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    setValue('')
    setSubmitting(false)
  }, [request?.requestId])

  const send = useCallback(
    async (secret: string) => {
      if (!request) {
        return
      }

      if (!gateway) {
        notifyError(new Error(copy.gatewayDisconnected), copy.secretSendFailed)

        return
      }

      setSubmitting(true)

      try {
        await gateway.request<{ status?: string }>('secret.respond', {
          request_id: request.requestId,
          value: secret
        })
        triggerHaptic('submit')
        clearSecretRequest(request.sessionId, request.requestId)
      } catch (error) {
        notifyError(error, copy.secretSendFailed)
        setSubmitting(false)
      }
    },
    [copy.gatewayDisconnected, copy.secretSendFailed, gateway, request]
  )

  const onOpenChange = useCallback(
    (open: boolean) => {
      if (!open && !submitting && request) {
        void send('')
      }
    },
    [request, send, submitting]
  )

  const onSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      void send(value)
    },
    [send, value]
  )

  if (!request) {
    return null
  }

  return (
    <Dialog onOpenChange={onOpenChange} open>
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle icon={KeyRound}>{request.envVar || copy.secretTitle}</DialogTitle>
          <DialogDescription>{request.prompt || copy.secretDesc}</DialogDescription>
        </DialogHeader>

        <form className="grid gap-3" onSubmit={onSubmit}>
          <Input
            autoFocus
            disabled={submitting}
            onChange={event => setValue(event.target.value)}
            placeholder={request.envVar || copy.secretPlaceholder}
            type="password"
            value={value}
          />
          <DialogFooter>
            <Button disabled={submitting} onClick={() => void send('')} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button disabled={submitting || !value} type="submit">
              {submitting ? <Loader2 className="size-3.5 animate-spin" /> : t.common.send}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

export function PromptOverlays() {
  return (
    <>
      <ApprovalOverlay />
      <SudoDialog />
      <SecretDialog />
    </>
  )
}
