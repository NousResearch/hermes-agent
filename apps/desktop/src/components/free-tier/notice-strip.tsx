import { useStore } from '@nanostores/react'
import { useEffect, useId } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { StatusRow } from '@/components/chat/status-row'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import {
  ackFreeTierNotice,
  claimFreeTierNotice,
  freeTierNoticeClaim,
  releaseFreeTierNotice
} from '@/store/free-tier'
import { openFreeTierSignIn } from '@/store/free-tier-sign-in'
import { setModelPickerOpen } from '@/store/session'

/**
 * Which mounted composer gets to paint the strip. Several can be on screen at
 * once (split zones, a popout mid-dock); the first to mount claims it and the
 * rest report false, so one pending notice never paints N times. The CALLER
 * asks, so a non-owning stack adds no empty row to its card.
 */
export function useFreeTierNoticeOwner(): boolean {
  const id = useId()
  const claim = useStore(freeTierNoticeClaim())

  useEffect(() => {
    claimFreeTierNotice(id)

    return () => releaseFreeTierNotice(id)
  }, [id])

  // When the owner unmounts it releases the claim; a composer still mounted takes it over,
  // so the notice does not vanish until some later mount.
  useEffect(() => {
    if (claim === null) {
      claimFreeTierNotice(id)
    }
  }, [claim, id])

  return claim === id
}

/**
 * The quiet half of the free-tier introduction: shown when the user already has
 * a provider of their own carrying inference, so the free models are an offer
 * rather than the only road. It lives in the composer status stack — the same
 * lane as the billing wall — and never blocks the composer.
 *
 * Backend-latched: `notice_pending` is the only source of truth, so any of the
 * three actions retires it everywhere at once and no renderer flag can strand a
 * strip the backend considers seen.
 */
export function FreeTierNoticeStrip() {
  const { requestGateway } = useGatewayRequest()
  const { t } = useI18n()
  const copy = t.freeTier

  const consume = (after?: () => void) => {
    void ackFreeTierNotice(requestGateway)
    after?.()
  }

  return (
    <StatusRow
      leading={<Codicon aria-hidden className="text-(--ui-text-tertiary)" name="account" size="0.8rem" />}
      trailing={
        <>
          <Button
            className="text-foreground/90 hover:text-foreground"
            onClick={() => consume(() => setModelPickerOpen(true))}
            size="micro"
            type="button"
            variant="text"
          >
            {copy.openModelPicker}
          </Button>
          <Button
            className="text-foreground/90 hover:text-foreground"
            onClick={() => consume(() => openFreeTierSignIn())}
            size="micro"
            type="button"
            variant="text"
          >
            {copy.signIn}
          </Button>
          <Button
            className="text-muted-foreground/75 hover:text-foreground/90"
            onClick={() => consume()}
            size="micro"
            type="button"
            variant="text"
          >
            {copy.dismiss}
          </Button>
        </>
      }
      trailingVisible
    >
      <span className="min-w-0 truncate text-[0.73rem] leading-4 text-foreground/92">
        <span className="font-medium">{copy.stripTitle}</span>
        <span className="text-muted-foreground/80"> {copy.stripBody}</span>
      </span>
    </StatusRow>
  )
}
