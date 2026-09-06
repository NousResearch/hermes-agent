import { useStore } from '@nanostores/react'
import { useQueryClient } from '@tanstack/react-query'
import type * as React from 'react'
import { type ReactNode, useEffect, useId } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import type { ModelSelection } from '@/app/shell/model-menu-panel'
import { DeviceCode } from '@/components/onboarding/flow'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  preventCloseButtonAutoFocus
} from '@/components/ui/dialog'
import { getGlobalModelOptions } from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { CheckCircle2, Loader2 } from '@/lib/icons'
import { FREE_TIER_MODEL, NOUS_PROVIDER_ID, refreshFreeTierStatus } from '@/store/free-tier'
import {
  $freeTierSignIn,
  beginFreeTierSignIn,
  claimFreeTierSignIn,
  closeFreeTierSignIn,
  copyFreeTierCode,
  copyFreeTierUrl,
  freeTierSignInClaim,
  type FreeTierSignInFailure,
  releaseFreeTierSignIn
} from '@/store/free-tier-sign-in'
import { refreshOnboardingProviders } from '@/store/onboarding'
import { $currentModel, setModelPickerOpen } from '@/store/session'

interface FreeTierSignInDialogProps {
  /** The app's model-assignment path. Used only to re-home a live session that
   *  is still pointed at the free-tier model once a real default arrives. */
  onSelectModel?: (selection: ModelSelection) => Promise<boolean> | void
}

/**
 * The one sign-in surface for the free tier. Every entry point (Settings ›
 * Billing, the statusbar chip, the first-launch intro) calls
 * `openFreeTierSignIn()`; this host owns the gateway requester, drives the
 * transfer, and paints one screen per state.
 *
 * Mounted once at the shell. A second mount claims nothing and renders null, so
 * two hosts can never stack two dialogs over one flow.
 */
export function FreeTierSignInDialog({ onSelectModel }: FreeTierSignInDialogProps) {
  const id = useId()
  const claim = useStore(freeTierSignInClaim())
  const state = useStore($freeTierSignIn)
  const { requestGateway } = useGatewayRequest()
  const queryClient = useQueryClient()
  const { t } = useI18n()
  const copy = t.freeTier
  const owned = claim === id

  useEffect(() => {
    claimFreeTierSignIn(id)

    return () => releaseFreeTierSignIn(id)
  }, [id])

  // An entry point can only record the intent — it has no requester of its own.
  // The owner picks that up and starts the real flow.
  useEffect(() => {
    if (owned && state.status === 'requested') {
      void beginFreeTierSignIn(requestGateway)
    }
  }, [owned, requestGateway, state.status])

  if (!owned || state.status === 'closed' || state.status === 'requested') {
    return null
  }

  // Everything that changed when the tokens landed: the account's billing, the
  // model catalog (paid models are reachable now), the free-tier verdict the
  // chrome paints, and the cached OAuth rows. Runs when the user leaves the
  // completed screen, by either door.
  const settle = (model: null | string) => {
    void queryClient.invalidateQueries({ queryKey: ['billing'] })
    void queryClient.invalidateQueries({ queryKey: ['model-options'] })
    void getGlobalModelOptions({ refresh: true }).catch(() => undefined)
    void refreshFreeTierStatus(requestGateway)
    void refreshOnboardingProviders()

    // Only re-home a session still sitting on the free-tier model: a user who
    // already picked something of their own keeps it.
    if (model && $currentModel.get() === FREE_TIER_MODEL) {
      void onSelectModel?.({ model, provider: NOUS_PROVIDER_ID })
    }
  }

  const finish = (model: null | string, after?: () => void) => {
    settle(model)
    closeFreeTierSignIn()
    after?.()
  }

  const retry = () => void beginFreeTierSignIn(requestGateway)

  return (
    <Dialog onOpenChange={open => !open && closeFreeTierSignIn()} open>
      <DialogContent onOpenAutoFocus={preventCloseButtonAutoFocus}>
        {state.status === 'setting_up' && (
          <Screen heading={copy.signInHeading}>
            <Spinner>{state.minting ? copy.settingUp : copy.waiting}</Spinner>
          </Screen>
        )}

        {state.status === 'code' && (
          <Screen body={copy.codeBody} heading={copy.signInHeading}>
            <DeviceCode code={state.code} copied={state.codeCopied} onCopy={() => void copyFreeTierCode()} />
            <div className="flex min-w-0 items-center justify-between gap-3">
              <a
                className="min-w-0 truncate text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary) underline underline-offset-2 hover:text-foreground"
                href={state.url}
                rel="noreferrer"
                target="_blank"
              >
                {state.url}
              </a>
              <Button onClick={() => void copyFreeTierUrl()} size="xs" type="button" variant="text">
                {state.urlCopied ? t.common.copied : copy.copyLink}
              </Button>
            </div>
            <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              {copy.doNotShare}
            </p>
            <div className="flex items-center justify-between gap-3">
              <Spinner>{copy.waiting}</Spinner>
              <Button onClick={() => closeFreeTierSignIn()} size="sm" type="button" variant="ghost">
                {t.common.cancel}
              </Button>
            </div>
          </Screen>
        )}

        {state.status === 'finishing' && (
          <Screen body={copy.finishingBody} heading={copy.finishingHeading}>
            <Spinner>{copy.waiting}</Spinner>
          </Screen>
        )}

        {state.status === 'completed' && (
          <Screen
            body={copy.connectorsKept}
            heading={state.email ? copy.signedInAs(state.email) : copy.signedIn}
            icon={CheckCircle2}
          >
            {state.model && (
              <div className="flex min-w-0 flex-wrap items-center gap-2">
                <span className="font-mono text-[0.625rem] uppercase tracking-[0.2em] text-muted-foreground">
                  {copy.defaultModel}
                </span>
                <span className="min-w-0 truncate font-mono text-[length:var(--conversation-text-font-size)]">
                  {state.model}
                </span>
                <Button
                  onClick={() => finish(state.model, () => setModelPickerOpen(true))}
                  size="inline"
                  type="button"
                  variant="text"
                >
                  {copy.change}
                </Button>
              </div>
            )}
            <Actions>
              <Button onClick={() => finish(state.model)} type="button">
                {copy.done}
              </Button>
            </Actions>
          </Screen>
        )}

        {state.status === 'already_signed_in' && (
          <Screen body={copy.alreadySignedInBody} heading={copy.alreadySignedInHeading}>
            <Actions>
              <Button onClick={() => closeFreeTierSignIn()} type="button">
                {copy.done}
              </Button>
            </Actions>
          </Screen>
        )}

        {state.status === 'failed' && (
          <Screen body={failureBody(state.kind, state.message, copy)} heading={failureHeading(state.kind, copy)}>
            <Actions>
              <Button onClick={() => closeFreeTierSignIn()} size="sm" type="button" variant="text">
                {copy.notNow}
              </Button>
              <Button onClick={retry} type="button">
                {state.kind === 'superseded' ? copy.startAgain : copy.tryAgain}
              </Button>
            </Actions>
          </Screen>
        )}
      </DialogContent>
    </Dialog>
  )
}

type FreeTierCopy = Translations['freeTier']

function failureHeading(kind: FreeTierSignInFailure, copy: FreeTierCopy): string {
  return kind === 'timed_out' ? copy.timedOutHeading : copy.didNotComplete
}

function failureBody(kind: FreeTierSignInFailure, message: null | string, copy: FreeTierCopy): string {
  switch (kind) {
    case 'rejected':
      return copy.rejectedBody

    case 'retired':
      return copy.retiredBody

    case 'superseded':
      return copy.supersededBody

    case 'timed_out':
      return copy.timedOutBody

    default:
      // The backend's own wording when it sent one — it names the specific
      // refusal (a busy account, a transport failure) better than we can.
      return message ?? copy.errorBody
  }
}

function Screen({
  body,
  children,
  heading,
  icon
}: {
  body?: string
  children: ReactNode
  heading: string
  icon?: React.ComponentType<{ className?: string }>
}) {
  return (
    <>
      <DialogHeader>
        <DialogTitle icon={icon}>{heading}</DialogTitle>
        {body ? <DialogDescription>{body}</DialogDescription> : null}
      </DialogHeader>
      {children}
    </>
  )
}

function Spinner({ children }: { children: ReactNode }) {
  return (
    <span
      className="flex items-center gap-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)"
      role="status"
    >
      <Loader2 className="size-3 animate-spin" />
      {children}
    </span>
  )
}

function Actions({ children }: { children: ReactNode }) {
  return <div className="flex items-center justify-end gap-2">{children}</div>
}
