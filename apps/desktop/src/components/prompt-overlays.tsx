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
import { Field } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { isMissingPendingPromptRequest } from '@/lib/gateway-rpc'
import { triggerHaptic } from '@/lib/haptics'
import { KeyRound, Loader2, Lock, ShieldLock } from '@/lib/icons'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import {
  clearSecretRequest,
  clearSudoRequest,
  clearVaultSaveLoginRequest,
  clearVaultUnlockRequest,
  sessionSecretRequest,
  sessionSudoRequest,
  sessionVaultSaveLoginRequest,
  sessionVaultUnlockRequest
} from '@/store/prompts'
import { ambientRequestFor } from '@/store/session-gone-latch'
import { requestForOwnedSession } from '@/store/session-states'

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

function SecretDialog({ sessionId }: { sessionId: string | null }) {
  const { t } = useI18n()
  const copy = t.prompts
  const $request = useMemo(() => sessionSecretRequest(sessionId), [sessionId])
  const request = useStore($request)
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
        if (isMissingPendingPromptRequest(error, 'value')) {
          clearSecretRequest(request.sessionId, request.requestId)

          return
        }

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

/** Masked master-password card for an external password manager (1Password / Bitwarden).
 *  Mirrors SecretDialog's contract: closing without submitting answers "" (keep locked); a late
 *  answer after expiry is tolerated by the backend. The value only lives in this component. */
function VaultUnlockDialog({ sessionId }: { sessionId: string | null }) {
  const { t } = useI18n()
  const copy = t.prompts
  const $request = useMemo(() => sessionVaultUnlockRequest(sessionId), [sessionId])
  const request = useStore($request)
  const gateway = useStore($gateway)
  const [value, setValue] = useState('')
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    setValue('')
    setSubmitting(false)
  }, [request?.requestId])

  const send = useCallback(
    async (password: string) => {
      if (!request) {
        return
      }

      if (!gateway) {
        notifyError(new Error(copy.gatewayDisconnected), copy.vaultUnlockSendFailed)

        return
      }

      setSubmitting(true)

      try {
        // A master password must reach the backend that raised the prompt, not whatever
        // gateway is foreground right now (background profile tiles have their own socket).
        await requestForOwnedSession<{ status?: string }>(
          request.sessionId,
          ambientRequestFor(gateway),
          'vault.unlock.respond',
          { request_id: request.requestId, password }
        )
        triggerHaptic('submit')
        clearVaultUnlockRequest(request.sessionId, request.requestId)
      } catch (error) {
        if (isMissingPendingPromptRequest(error, 'password')) {
          clearVaultUnlockRequest(request.sessionId, request.requestId)

          return
        }

        notifyError(error, copy.vaultUnlockSendFailed)
        setSubmitting(false)
      } finally {
        setValue('')
      }
    },
    [copy.gatewayDisconnected, copy.vaultUnlockSendFailed, gateway, request]
  )

  if (!request) {
    return null
  }

  return (
    <Dialog onOpenChange={open => !open && !submitting && void send('')} open>
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle icon={ShieldLock}>{copy.vaultUnlockTitle(request.displayName)}</DialogTitle>
          <DialogDescription>{copy.vaultUnlockDesc(request.displayName)}</DialogDescription>
        </DialogHeader>

        <form
          className="grid gap-3"
          onSubmit={event => {
            event.preventDefault()
            void send(value)
          }}
        >
          <Input
            autoComplete="current-password"
            autoFocus
            disabled={submitting}
            onChange={event => setValue(event.target.value)}
            placeholder={copy.vaultUnlockPlaceholder}
            type="password"
            value={value}
          />
          <DialogFooter>
            <Button disabled={submitting} onClick={() => void send('')} type="button" variant="ghost">
              {copy.vaultUnlockKeepLocked}
            </Button>
            <Button disabled={submitting || !value} type="submit">
              {submitting ? <Loader2 className="size-3.5 animate-spin" /> : copy.vaultUnlockConfirm}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

/** "Save this login" card: the agent is on a sign-in page for `site` with nothing in the vault.
 *  Identifier is plain, password masked; the pair goes to `vault.save_login.respond` as JSON and
 *  the backend stores it encrypted and fills the page. Closing answers "" (don't save). */
function VaultSaveLoginDialog({ sessionId }: { sessionId: string | null }) {
  const { t } = useI18n()
  const copy = t.prompts
  const $request = useMemo(() => sessionVaultSaveLoginRequest(sessionId), [sessionId])
  const request = useStore($request)
  const gateway = useStore($gateway)
  const [identifier, setIdentifier] = useState('')
  const [password, setPassword] = useState('')
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    setIdentifier('')
    setPassword('')
    setSubmitting(false)
  }, [request?.requestId])

  const send = useCallback(
    async (login: string) => {
      if (!request) {
        return
      }

      if (!gateway) {
        notifyError(new Error(copy.gatewayDisconnected), copy.vaultSaveSendFailed)

        return
      }

      setSubmitting(true)

      try {
        await requestForOwnedSession<{ status?: string }>(
          request.sessionId,
          ambientRequestFor(gateway),
          'vault.save_login.respond',
          { login, request_id: request.requestId }
        )
        triggerHaptic('submit')
        clearVaultSaveLoginRequest(request.sessionId, request.requestId)
      } catch (error) {
        if (isMissingPendingPromptRequest(error, 'login')) {
          clearVaultSaveLoginRequest(request.sessionId, request.requestId)

          return
        }

        notifyError(error, copy.vaultSaveSendFailed)
        setSubmitting(false)
      } finally {
        setIdentifier('')
        setPassword('')
      }
    },
    [copy.gatewayDisconnected, copy.vaultSaveSendFailed, gateway, request]
  )

  if (!request) {
    return null
  }

  const canSave = Boolean(identifier.trim()) && Boolean(password)

  return (
    <Dialog onOpenChange={open => !open && !submitting && void send('')} open>
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle icon={ShieldLock}>{copy.vaultSaveTitle(request.site)}</DialogTitle>
          <DialogDescription>{copy.vaultSaveDesc(request.origin)}</DialogDescription>
        </DialogHeader>

        <form
          className="grid gap-3"
          onSubmit={event => {
            event.preventDefault()

            if (canSave) {
              void send(JSON.stringify({ identifier: identifier.trim(), password }))
            }
          }}
        >
          <Field htmlFor="vault-save-identifier" label={copy.vaultSaveIdentifierLabel}>
            <Input
              autoComplete="username"
              autoFocus
              disabled={submitting}
              id="vault-save-identifier"
              onChange={event => setIdentifier(event.target.value)}
              placeholder={copy.vaultSaveIdentifierPlaceholder}
              value={identifier}
            />
          </Field>
          <Field htmlFor="vault-save-password" label={copy.vaultSavePasswordPlaceholder}>
            <Input
              autoComplete="current-password"
              disabled={submitting}
              id="vault-save-password"
              onChange={event => setPassword(event.target.value)}
              type="password"
              value={password}
            />
          </Field>
          <p className="text-xs text-muted-foreground">{copy.vaultSaveFootnote}</p>
          <DialogFooter>
            <Button disabled={submitting} onClick={() => void send('')} type="button" variant="ghost">
              {copy.vaultSaveDecline}
            </Button>
            <Button disabled={submitting || !canSave} type="submit">
              {submitting ? <Loader2 className="size-3.5 animate-spin" /> : copy.vaultSaveConfirm}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
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
      <VaultUnlockDialog sessionId={sessionId} />
      <VaultSaveLoginDialog sessionId={sessionId} />
    </>
  )
}
