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
import { triggerHaptic } from '@/lib/haptics'
import { KeyRound, Loader2, Lock } from '@/lib/icons'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import {
  $approvalInlineVisible,
  $approvalRequest,
  $secretRequest,
  $sudoRequest,
  clearApprovalRequest,
  clearSecretRequest,
  clearSudoRequest
} from '@/store/prompts'

// Renders modal mid-turn prompts the gateway raises and waits on. Sudo password
// and skill secret capture are always modal. Dangerous-command / execute_code
// approval is normally rendered INLINE on the pending tool row (see
// components/assistant-ui/tool-approval.tsx), but this overlay provides a
// fallback when no inline approval control is mounted. Each Python-side caller
// blocks until the matching `*.respond` RPC lands; without a visible renderer
// the agent stalls until its timeout and the tool is BLOCKED. Any close path
// (Esc, backdrop click) funnels through Radix's single `onOpenChange(false)` and
// maps to a refusal, so silence is never mistaken for consent, matching the TUI.
// We deliberately do NOT add onEscapeKeyDown / onInteractOutside handlers —
// they'd fire a second `*.respond` alongside onOpenChange (double-send) or block
// the backdrop-dismiss path.

type ApprovalChoice = 'once' | 'session' | 'deny'

function ApprovalFallbackDialog() {
  const request = useStore($approvalRequest)
  const inlineVisible = useStore($approvalInlineVisible)
  const gateway = useStore($gateway)
  const [submitting, setSubmitting] = useState<ApprovalChoice | null>(null)

  useEffect(() => {
    setSubmitting(null)
  }, [request?.sessionId, request?.command])

  const send = useCallback(
    async (choice: ApprovalChoice) => {
      if (!request) {
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
          session_id: request.sessionId ?? undefined
        })
        triggerHaptic(choice === 'deny' ? 'cancel' : 'submit')
        clearApprovalRequest(request.sessionId)
      } catch (error) {
        notifyError(error, 'Could not send approval response')
        setSubmitting(null)
      }
    },
    [gateway, request]
  )

  const onOpenChange = useCallback(
    (open: boolean) => {
      if (!open && !submitting && request) {
        void send('deny')
      }
    },
    [request, send, submitting]
  )

  if (!request || inlineVisible) {
    return null
  }

  return (
    <Dialog onOpenChange={onOpenChange} open>
      <DialogContent className="max-w-xl" showCloseButton={false}>
        <DialogHeader>
          <DialogTitle>Command approval required</DialogTitle>
          <DialogDescription>
            Hermes is waiting for permission to run this local command. Review it before continuing.
          </DialogDescription>
        </DialogHeader>

        {request.command.trim() && (
          <pre className="max-h-52 overflow-auto whitespace-pre-wrap break-words rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-chat-surface-background) px-2.5 py-2 font-mono text-xs leading-snug text-foreground">
            {request.command.trim()}
          </pre>
        )}

        {request.description && (
          <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {request.description}
          </p>
        )}

        <DialogFooter>
          <Button disabled={submitting !== null} onClick={() => void send('deny')} variant="ghost">
            {submitting === 'deny' ? <Loader2 className="size-3.5 animate-spin" /> : 'Reject'}
          </Button>
          <Button disabled={submitting !== null} onClick={() => void send('session')} variant="outline">
            {submitting === 'session' ? <Loader2 className="size-3.5 animate-spin" /> : 'Allow this session'}
          </Button>
          <Button disabled={submitting !== null} onClick={() => void send('once')}>
            {submitting === 'once' ? <Loader2 className="size-3.5 animate-spin" /> : 'Run once'}
          </Button>
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
        clearSudoRequest(request.sessionId, request.requestId)
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
        clearSecretRequest(request.sessionId, request.requestId)
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
      <ApprovalFallbackDialog />
      <SudoDialog />
      <SecretDialog />
    </>
  )
}
