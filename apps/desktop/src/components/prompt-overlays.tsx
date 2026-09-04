'use client'

import { useStore } from '@nanostores/react'
import { type FormEvent, useCallback, useEffect, useMemo, useState } from 'react'

import { PendingApprovalFallback } from '@/components/assistant-ui/tool/approval'
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
import { isMissingPendingPromptRequest } from '@/lib/gateway-rpc'
import { triggerHaptic } from '@/lib/haptics'
import { Loader2, Lock } from '@/lib/icons'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $activeProfile } from '@/store/profile'
import { clearSecretRequest, clearSudoRequest, sessionSecretRequest, sessionSudoRequest } from '@/store/prompts'

// Renders the modal mid-turn prompts the gateway raises and waits on: sudo
// password and skill secret capture. Dangerous-command / execute_code approval
// prefers the pending tool row, but also has a chat-level fallback when no row
// is mounted (remote gateway sessions can raise the request before the matching
// tool call is visible). Each Python-side caller blocks the agent thread until
// the matching `*.respond` RPC lands; without a renderer the agent stalls until
// its timeout and the tool is BLOCKED. Any close path (Esc, backdrop
// click) funnels through Radix's single `onOpenChange(false)` and maps to a
// refusal, so silence is never mistaken for consent, matching the TUI. We
// deliberately do NOT add onEscapeKeyDown / onInteractOutside handlers — they'd
// fire a second `*.respond` alongside onOpenChange (double-send) or block the
// backdrop-dismiss path.

function SudoDialog({ sessionId }: { sessionId: string | null }) {
  const { t } = useI18n()
  const copy = t.prompts
  const $request = useMemo(() => sessionSudoRequest(sessionId), [sessionId])
  const request = useStore($request)
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
        if (isMissingPendingPromptRequest(error, 'password')) {
          clearSudoRequest(request.sessionId, request.requestId)

          return
        }

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

const activeCredentialCaptures = new Map<string, Promise<void>>()

function SecretDialog({ sessionId }: { sessionId: string | null }) {
  const { locale, t } = useI18n()
  const copy = t.prompts
  const $request = useMemo(() => sessionSecretRequest(sessionId), [sessionId])
  const request = useStore($request)
  const gateway = useStore($gateway)
  const profile = useStore($activeProfile)

  useEffect(() => {
    if (!request || activeCredentialCaptures.has(request.requestId)) {
      return
    }

    const capture = window.hermesDesktop.secureCredential?.capture

    if (!gateway || !capture) {
      notifyError(new Error(copy.gatewayDisconnected), copy.secretSendFailed)

      return
    }

    const task = (async () => {
      try {
        const result = await capture({
          envVar: request.envVar,
          locale,
          profile,
          prompt: request.prompt,
          requestId: request.requestId
        })

        // The gateway receives only a storage receipt. The credential value was
        // already written by Electron main and never entered this renderer.
        await gateway.request<{ status?: string }>('secret.respond', {
          request_id: request.requestId,
          value: result.status === 'saved' ? { stored: true } : ''
        })

        if (result.status === 'saved') {
          triggerHaptic('submit')
        }

        clearSecretRequest(request.sessionId, request.requestId)
      } catch (error) {
        if (isMissingPendingPromptRequest(error, 'value')) {
          clearSecretRequest(request.sessionId, request.requestId)

          return
        }

        notifyError(error, copy.secretSendFailed)
      } finally {
        activeCredentialCaptures.delete(request.requestId)
      }
    })()

    activeCredentialCaptures.set(request.requestId, task)
  }, [copy.gatewayDisconnected, copy.secretSendFailed, gateway, locale, profile, request])

  // The actual input lives in the dedicated native window. Rendering no form
  // here ensures password managers, DOM inspection, and chat capture cannot
  // observe a credential field in the agent-facing renderer.
  return null
}

/** Mid-turn prompt surfaces for ONE session. Mounted by both the primary chat
 *  and each tile with its own session id, so a background/tiled session's
 *  blocking prompt renders instead of silently stalling. */
export function PromptOverlays({ sessionId }: { sessionId: string | null }) {
  return (
    <>
      <PendingApprovalFallback />
      <SudoDialog sessionId={sessionId} />
      <SecretDialog sessionId={sessionId} />
    </>
  )
}
