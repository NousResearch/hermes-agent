import { useStore } from '@nanostores/react'
import { useQueryClient } from '@tanstack/react-query'
import { type ReactNode, useEffect, useId } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { CAPABILITIES_ROUTE } from '@/app/routes'
import type { ModelSelection } from '@/app/shell/model-menu-panel'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  preventCloseButtonAutoFocus
} from '@/components/ui/dialog'
import { getGlobalModelOptions, type ProfileScope } from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { CheckCircle2, Loader2 } from '@/lib/icons'
import {
  $freeTierStatus,
  FREE_TIER_MODEL,
  friendlyWait,
  NOUS_PROVIDER_ID,
  refreshFreeTierStatus
} from '@/store/free-tier'
import {
  $freeTierSignIn,
  $freeTierTurnCompleted,
  beginFreeTierSignIn,
  claimFreeTierSignIn,
  closeFreeTierSignIn,
  freeTierSignInClaim,
  type FreeTierSignInFailure,
  type FreeTierSignInState,
  handoffStillMoving,
  releaseFreeTierSignIn,
  reopenFreeTierSignInPage,
  sameGatewayRoute,
  stopFreeTierOffer,
  syncFreeTierOffer
} from '@/store/free-tier-sign-in'
import { $localModelOfferPending } from '@/store/local-model-offer'
import { refreshOnboardingProviders } from '@/store/onboarding'
import { $onboardingSurfaceClear } from '@/store/onboarding-presence'
import { $currentModel, setModelPickerOpen } from '@/store/session'
import type { AppHandoffStatus } from '@/types/hermes'

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

  // The sign-in offer rides every status read, and retries once the first-run
  // onboarding leaves the screen. A completed free-tier turn re-reads the
  // status, since that is when the backend arms the next offer. Only the
  // owner (main windows) times it.
  useEffect(() => {
    if (!owned) {
      return
    }

    const sync = () => syncFreeTierOffer($freeTierStatus.get(), requestGateway)

    const stops = [
      $freeTierStatus.listen(sync),
      $onboardingSurfaceClear.listen(sync),
      $localModelOfferPending.listen(sync),
      $freeTierTurnCompleted.listen(() => void refreshFreeTierStatus(requestGateway, sameGatewayRoute()))
    ]

    sync()

    return () => {
      stops.forEach(stop => stop())
      stopFreeTierOffer()
    }
  }, [owned, requestGateway])

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
        {state.status === 'offer' && (
          <Screen
            body={state.kind === 'connector' ? t.connectorOffer.body : copy.offer.body}
            heading={state.kind === 'connector' ? t.connectorOffer.heading : copy.offer.heading}
          >
            <Actions>
              <Button onClick={() => closeFreeTierSignIn()} size="sm" type="button" variant="text">
                {copy.offer.notNow}
              </Button>
              <Button onClick={retry} type="button">
                {copy.offer.signIn}
              </Button>
            </Actions>
          </Screen>
        )}

        {state.status === 'setting_up' && (
          <Screen heading={copy.signInHeading}>
            <Spinner>{state.minting ? copy.settingUp : copy.waiting}</Spinner>
          </Screen>
        )}

        {state.status === 'code' && (
          <Screen heading={copy.finishInBrowser}>
            {/* The code is only there to compare with the browser page, so it stays small. */}
            <p className="flex flex-wrap items-baseline gap-x-2 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
              <span className="font-mono text-sm tracking-widest text-foreground">{state.code}</span>
              {copy.doNotShare}
            </p>
            <div className="flex items-center justify-between gap-3">
              <Spinner>{copy.waiting}</Spinner>
              <div className="flex items-center gap-2">
                <Button onClick={() => void reopenFreeTierSignInPage()} size="sm" type="button" variant="text">
                  {copy.openPageAgain}
                </Button>
                <Button onClick={() => closeFreeTierSignIn()} size="sm" type="button" variant="ghost">
                  {t.common.cancel}
                </Button>
              </div>
            </div>
          </Screen>
        )}

        {state.status === 'finishing' && (
          <Screen body={copy.finishingBody} heading={copy.finishingHeading}>
            <Spinner>{copy.waiting}</Spinner>
          </Screen>
        )}

        {state.status === 'completed' && <SignedInScreen copy={copy} finish={finish} state={state} />}

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
          <Screen
            body={failureBody(state.kind, state.message, state.retryAfter, copy)}
            heading={failureHeading(state.kind, copy)}
          >
            <Actions>
              <Button onClick={() => closeFreeTierSignIn()} size="sm" type="button" variant="text">
                {state.kind === 'unavailable' ? copy.done : copy.notNow}
              </Button>
              {/* A terminal refusal (this version, a locked session, a proof-of-work
                  request) has nothing to retry: the backend's sentence names the way out. */}
              {state.kind === 'unavailable' ? null : (
                <Button onClick={retry} type="button">
                  {copy.tryAgain}
                </Button>
              )}
            </Actions>
          </Screen>
        )}
      </DialogContent>
    </Dialog>
  )
}

type FreeTierCopy = Translations['freeTier']

function failureHeading(kind: FreeTierSignInFailure, copy: FreeTierCopy): string {
  switch (kind) {
    case 'busy':
      return copy.busyHeading

    case 'timed_out':
      return copy.timedOutHeading

    default:
      return copy.didNotComplete
  }
}

function failureBody(
  kind: FreeTierSignInFailure,
  message: null | string,
  retryAfter: number,
  copy: FreeTierCopy
): string {
  switch (kind) {
    case 'busy':
      // The backend's sentence already names the wait it was given; ours fills
      // in when an older backend sent none.
      return message ?? copy.busyBody(friendlyWait(retryAfter || 60))

    case 'timed_out':
      return copy.timedOutBody

    case 'unreachable':
      return message ?? copy.unreachableBody

    default:
      // The backend's own wording when it sent one — it names the specific
      // refusal (this version, a locked session) better than we can.
      return message ?? copy.errorBody
  }
}

function Screen({ body, children, heading }: { body?: string; children: ReactNode; heading: string }) {
  return (
    <>
      <DialogHeader>
        <DialogTitle>{heading}</DialogTitle>
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

/** The result screen: who is signed in, set large, then what the free tier's apps did. */
function SignedInScreen({
  copy,
  finish,
  state
}: {
  copy: FreeTierCopy
  finish: (model: null | string, after?: () => void) => void
  state: Extract<FreeTierSignInState, { status: 'completed' }>
}) {
  const { email, handoff, model, scope } = state

  return (
    <>
      <DialogHeader>
        <DialogTitle icon={CheckCircle2}>{email ? copy.signedInAsLabel : copy.signedIn}</DialogTitle>
        {email ? (
          <p className="min-w-0 truncate text-xl font-semibold tracking-tight text-foreground">{email}</p>
        ) : null}
        {handoff?.lines.length ? null : <DialogDescription>{copy.completedBody}</DialogDescription>}
      </DialogHeader>
      {handoff ? <HandoffResult handoff={handoff} /> : null}
      {model && (
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <span className="font-mono text-[0.625rem] uppercase tracking-[0.2em] text-muted-foreground">
            {copy.defaultModel}
          </span>
          <span className="min-w-0 truncate font-mono text-[length:var(--conversation-text-font-size)]">{model}</span>
          <Button
            onClick={() => finish(model, () => setModelPickerOpen(true))}
            size="inline"
            type="button"
            variant="text"
          >
            {copy.change}
          </Button>
        </div>
      )}
      <Actions>
        {connectorsTarget(handoff, scope) ? (
          <Button
            onClick={() => finish(model, () => openConnectors(handoff, scope))}
            size="sm"
            type="button"
            variant="text"
          >
            {handoff?.state === 'claimed' ? copy.reconnect : copy.openConnectors}
          </Button>
        ) : null}
        <Button onClick={() => finish(model)} type="button">
          {copy.done}
        </Button>
      </Actions>
    </>
  )
}

/** What the free tier's apps did, in the backend's own lines. While the move
 *  is still retried the line is a small note that updates itself. */
function HandoffResult({ handoff }: { handoff: AppHandoffStatus }) {
  if (!handoff.lines.length) {
    return null
  }

  if (handoffStillMoving(handoff)) {
    return <Spinner>{handoff.lines.join(' ')}</Spinner>
  }

  return (
    <div className="grid gap-1 text-[length:var(--conversation-text-font-size)] text-foreground">
      {handoff.lines.map(line => (
        <p key={line}>{line}</p>
      ))}
    </div>
  )
}

/** Where the completed screen's connectors button goes: the one app to
 *  reconnect, the whole list for several, or the list to check when the apps
 *  probably moved already. Null when there is nothing to do there. A sign-in
 *  made for another profile names it (and its connection), the same way a
 *  chat's connector links do, so the page acts in that profile only. */
export function connectorsTarget(handoff: AppHandoffStatus | null, scope?: ProfileScope): null | string {
  const dropped = handoff?.state === 'claimed' ? (handoff.dropped_toolkits ?? []) : []
  const checkMoved = handoff?.state === 'gone' && handoff.probably_moved === true

  if (!dropped.length && !checkMoved) {
    return null
  }

  const params = new URLSearchParams({ tab: 'connectors' })

  if (dropped.length === 1) {
    params.set('connector', dropped[0])
  }

  const profile = (typeof scope === 'object' ? scope?.profile : scope)?.trim()
  const connectionId = typeof scope === 'object' ? scope?.connectionId?.trim() : null

  if (profile) {
    params.set('profile', profile)

    if (connectionId) {
      params.set('connection', connectionId)
    }
  }

  return `${CAPABILITIES_ROUTE}?${params.toString()}`
}

function openConnectors(handoff: AppHandoffStatus | null, scope?: ProfileScope) {
  const target = connectorsTarget(handoff, scope)

  if (target) {
    window.location.hash = `#${target}`
  }
}
