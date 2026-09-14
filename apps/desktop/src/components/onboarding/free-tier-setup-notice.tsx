import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { type Translations, useI18n } from '@/i18n'
import {
  $freeTierStatus,
  type FreeTierSetupFailure,
  freeTierSetupFailure,
  friendlyWait,
  provisionFreeTier,
  refreshFreeTierStatus
} from '@/store/free-tier'
import { type OnboardingContext, refreshOnboarding } from '@/store/onboarding'

type SetupFailedCopy = Translations['freeTier']['setupFailed']

/** One sentence per backend code. The backend's own sentence is the fallback for
 *  a code this build does not know, so a newer backend still reads as words. */
export function setupFailureCopy(failure: FreeTierSetupFailure, copy: SetupFailedCopy): string {
  switch (failure.code) {
    case 'anon_gate_closed':
      return copy.gateClosed

    case 'anon_gate_paused':
      return copy.paused

    case 'anon_rate_limited':
      return copy.rateLimited(friendlyWait(failure.retryAfter || 60))

    case 'anon_unreachable':
      return copy.unreachable

    case 'anon_server_error':
      return copy.serverError

    case 'anon_pow_required':
      return copy.powRequired

    case 'anon_account_locked':
      return copy.locked

    default:
      return failure.message || copy.generic
  }
}

/**
 * The first-launch notice for a free tier that could not be set up: the boot
 * bootstrap tried, the account service refused or could not be reached, and
 * the user is looking at the provider picker with no idea why. Says what
 * happened in one sentence, offers the user's own retry when a later attempt
 * can succeed, and points at the Nous row below when signing in can help
 * (never when the same service is the one that is unreachable).
 *
 * Renders nothing unless the backend reported a failure, so an older backend
 * or a healthy boot leaves the picker exactly as it was.
 */
export function FreeTierSetupNotice({ ctx }: { ctx: OnboardingContext }) {
  const { t } = useI18n()
  const status = useStore($freeTierStatus)
  const failure = freeTierSetupFailure(status)
  const [retrying, setRetrying] = useState(false)
  const copy = t.freeTier.setupFailed

  // The verdict is a local, zero-network read; make sure it is fresh for the
  // picker even before the ambient status round has run.
  useEffect(() => {
    void refreshFreeTierStatus(ctx.requestGateway)
  }, [ctx])

  if (!failure) {
    return null
  }

  const retry = async () => {
    if (retrying) {
      return
    }

    setRetrying(true)

    try {
      const next = await provisionFreeTier(ctx.requestGateway)

      if (next?.has_guest) {
        // The identity exists now: re-run the readiness round so the picker
        // gives way to the ready screen.
        await refreshOnboarding(ctx)
      }
    } finally {
      setRetrying(false)
    }
  }

  return (
    <div
      className="grid gap-2 rounded-2xl border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary)/40 px-4 py-3 text-sm"
      data-testid="free-tier-setup-notice"
      role="status"
    >
      <p>{setupFailureCopy(failure, copy)}</p>
      {failure.door === 'sign_in' ? <p className="text-muted-foreground">{copy.signInBelow}</p> : null}
      {failure.retryable ? (
        <div>
          <Button disabled={retrying} onClick={() => void retry()} size="sm" type="button" variant="outline">
            {retrying ? copy.retrying : copy.tryAgain}
          </Button>
        </div>
      ) : null}
    </div>
  )
}
