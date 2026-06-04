'use client'

import { useStore } from '@nanostores/react'
import { type FormEvent, useCallback, useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { triggerHaptic } from '@/lib/haptics'
import { AlertTriangle, KeyRound, Loader2, Lock } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import {
  $approvalRequest,
  $secretRequest,
  $sudoRequest,
  clearApprovalRequest,
  clearSecretRequest,
  clearSudoRequest
} from '@/store/prompts'

// Renders the three blocking mid-turn prompts the gateway raises and waits on:
// dangerous-command/execute_code approval, sudo password, and skill secret
// capture. Each Python-side caller blocks the agent thread until the matching
// `*.respond` RPC lands; without these overlays the agent stalls until its
// timeout and the tool is BLOCKED (the bug this fixes — desktop handled
// clarify.request but not these). Any close path (Esc, backdrop click) funnels
// through Radix's single `onOpenChange(false)` and maps to a refusal, so silence
// is never mistaken for consent, matching the TUI. We deliberately do NOT add
// onEscapeKeyDown / onInteractOutside handlers — they'd fire a second `*.respond`
// alongside onOpenChange (double-send) or block the backdrop-dismiss path.

const APPROVAL_OPTS = [
  { choice: 'once', label: 'Allow once', variant: 'default' as const },
  { choice: 'session', label: 'Allow this session', variant: 'secondary' as const },
  { choice: 'always', label: 'Always allow', variant: 'secondary' as const },
  { choice: 'deny', label: 'Deny', variant: 'ghost' as const }
]

const APPROVAL_CMD_PREVIEW_LINES = 12

function ApprovalDialog() {
  const request = useStore($approvalRequest)
  const gateway = useStore($gateway)
  const [submitting, setSubmitting] = useState(false)

  const respond = useCallback(
    async (choice: string) => {
      if (!request) {
        return
      }

      if (!gateway) {
        notifyError(new Error('Hermes gateway is not connected'), 'Could not send approval response')

        return
      }

      setSubmitting(true)

      try {
        await gateway.request<{ resolved?: boolean }>('approval.respond', {
          choice,
          session_id: request.sessionId ?? undefined
        })
        triggerHaptic('submit')
        clearApprovalRequest()
      } catch (error) {
        notifyError(error, 'Could not send approval response')
        setSubmitting(false)
      }
    },
    [gateway, request]
  )

  // Esc / backdrop close → deny (consent contract: silence is not consent).
  const onOpenChange = useCallback(
    (open: boolean) => {
      if (!open && !submitting && request) {
        void respond('deny')
      }
    },
    [request, respond, submitting]
  )

  if (!request) {
    return null
  }

  const rawLines = request.command.split('\n')
  const shown = rawLines.slice(0, APPROVAL_CMD_PREVIEW_LINES)
  const overflow = rawLines.length - shown.length

  return (
    <Dialog onOpenChange={onOpenChange} open>
      <DialogContent
        className="max-w-xl border-[color-mix(in_srgb,var(--dt-warning,#d97706)_45%,var(--ui-stroke-secondary))]"
        showCloseButton={false}
      >
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <AlertTriangle className="size-4 text-[var(--dt-warning,#d97706)]" />
            Approval required
          </DialogTitle>
          <DialogDescription>{request.description}</DialogDescription>
        </DialogHeader>

        <pre
          className={cn(
            'max-h-56 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-(--ui-stroke-secondary)',
            'bg-(--ui-chat-surface-background) px-3 py-2 font-mono text-[0.8125rem] leading-snug text-foreground'
          )}
        >
          {shown.join('\n') || ' '}
          {overflow > 0 ? `\n… +${overflow} more line${overflow === 1 ? '' : 's'}` : ''}
        </pre>

        <DialogFooter className="sm:flex-col sm:items-stretch sm:gap-2">
          {APPROVAL_OPTS.map(opt => (
            <Button
              disabled={submitting}
              key={opt.choice}
              onClick={() => void respond(opt.choice)}
              variant={opt.variant}
            >
              {submitting ? <Loader2 className="size-3.5 animate-spin" /> : opt.label}
            </Button>
          ))}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function SudoDialog() {
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
        notifyError(new Error('Hermes gateway is not connected'), 'Could not send sudo password')

        return
      }

      setSubmitting(true)

      try {
        await gateway.request<{ status?: string }>('sudo.respond', {
          password: value,
          request_id: request.requestId
        })
        triggerHaptic('submit')
        clearSudoRequest(request.requestId)
      } catch (error) {
        notifyError(error, 'Could not send sudo password')
        setSubmitting(false)
      }
    },
    [gateway, request]
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
          <DialogTitle className="flex items-center gap-2">
            <Lock className="size-4 text-primary" />
            Administrator password
          </DialogTitle>
          <DialogDescription>
            Hermes needs your sudo password to run a privileged command. It is sent only to your local agent.
          </DialogDescription>
        </DialogHeader>

        <form className="grid gap-3" onSubmit={onSubmit}>
          <Input
            autoFocus
            disabled={submitting}
            onChange={event => setPassword(event.target.value)}
            placeholder="sudo password"
            type="password"
            value={password}
          />
          <DialogFooter>
            <Button disabled={submitting} onClick={() => void send('')} type="button" variant="ghost">
              Cancel
            </Button>
            <Button disabled={submitting} type="submit">
              {submitting ? <Loader2 className="size-3.5 animate-spin" /> : 'Send'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function SecretDialog() {
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
        notifyError(new Error('Hermes gateway is not connected'), 'Could not send secret')

        return
      }

      setSubmitting(true)

      try {
        await gateway.request<{ status?: string }>('secret.respond', {
          request_id: request.requestId,
          value: secret
        })
        triggerHaptic('submit')
        clearSecretRequest(request.requestId)
      } catch (error) {
        notifyError(error, 'Could not send secret')
        setSubmitting(false)
      }
    },
    [gateway, request]
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
          <DialogTitle className="flex items-center gap-2">
            <KeyRound className="size-4 text-primary" />
            {request.envVar || 'Secret required'}
          </DialogTitle>
          <DialogDescription>{request.prompt || 'Hermes needs a credential to continue.'}</DialogDescription>
        </DialogHeader>

        <form className="grid gap-3" onSubmit={onSubmit}>
          <Input
            autoFocus
            disabled={submitting}
            onChange={event => setValue(event.target.value)}
            placeholder={request.envVar || 'secret value'}
            type="password"
            value={value}
          />
          <DialogFooter>
            <Button disabled={submitting} onClick={() => void send('')} type="button" variant="ghost">
              Cancel
            </Button>
            <Button disabled={submitting || !value} type="submit">
              {submitting ? <Loader2 className="size-3.5 animate-spin" /> : 'Send'}
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
      <ApprovalDialog />
      <SudoDialog />
      <SecretDialog />
    </>
  )
}
