import { useStore } from '@nanostores/react'
import { useEffect, useId } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  preventCloseButtonAutoFocus
} from '@/components/ui/dialog'
import { type I18nContextValue, useI18n } from '@/i18n'
import { openExternalLink } from '@/lib/external-link'
import {
  $inAppOfferDialog,
  claimInAppOffer,
  dismissInAppOffer,
  inAppOfferClaim,
  listenInAppOfferStatus,
  markInAppOfferShown,
  releaseInAppOffer,
  reportInAppOfferEvent,
  startInAppOfferRefresh
} from '@/store/in-app-offer'

/**
 * The Nous Portal's in-app offer as one dismissible dialog. Copy is the
 * server's; the renderer only decides presentation (store/in-app-offer.ts).
 *
 * Mounted once per main window at the shell, beside the free-tier sign-in
 * dialog. The first mount claims the offer; any other window's mount claims
 * nothing and renders null, so two windows never show two dialogs.
 */
export function InAppOfferDialog() {
  const id = useId()
  const claim = useStore(inAppOfferClaim())
  const offer = useStore($inAppOfferDialog)
  const { requestGateway } = useGatewayRequest()
  const { t }: I18nContextValue = useI18n()
  const owned = claim === id

  useEffect(() => {
    claimInAppOffer(id)

    return () => releaseInAppOffer(id)
  }, [id])

  useEffect(() => {
    if (!owned) {
      return
    }

    const stopListening = listenInAppOfferStatus()
    const stopRefresh = startInAppOfferRefresh(requestGateway)

    return () => {
      stopListening()
      stopRefresh()
    }
  }, [owned, requestGateway])

  useEffect(() => {
    // Presenting starts the frequency cap, so an offer closed without a
    // dismissal (the app quit) can return tomorrow but not sooner.
    if (owned && offer) {
      markInAppOfferShown()
      reportInAppOfferEvent(requestGateway, offer.id, 'shown')
    }
  }, [owned, offer, requestGateway])

  if (!owned || !offer) {
    return null
  }

  const claimBy = offer.claimable_until ? formatClaimDeadline(offer.claimable_until) : null

  return (
    <Dialog
      onOpenChange={open => {
        if (!open) {
          dismissInAppOffer(offer.id, requestGateway)
        }
      }}
      open
    >
      {/* No autofocus: the dialog opens unprompted, and focusing the close
          button would let a stray Space or Enter dismiss it unread. */}
      <DialogContent onOpenAutoFocus={preventCloseButtonAutoFocus}>
        <DialogHeader>
          <DialogTitle>{offer.title}</DialogTitle>
          <DialogDescription className="sr-only">{t.inAppOffer.describedBy}</DialogDescription>
        </DialogHeader>
        <p className="text-sm whitespace-pre-line break-words">{offer.body}</p>
        {claimBy && <p className="text-xs text-muted-foreground">{t.inAppOffer.claimBy(claimBy)}</p>}
        <DialogFooter>
          <Button onClick={() => dismissInAppOffer(offer.id, requestGateway)} size="sm" variant="outline">
            {t.inAppOffer.dismiss}
          </Button>
          <Button
            onClick={() => {
              openExternalLink(offer.cta_url)
              reportInAppOfferEvent(requestGateway, offer.id, 'clicked')
              dismissInAppOffer(offer.id, requestGateway)
            }}
            size="sm"
          >
            {offer.cta_label}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/** "Dec 15, 2026, 12:00 PM UTC": the Portal's own deadline wording (claimDeadlineLabel there), in
 * UTC so the dialog, the Portal's Billing card and its claim page name the same instant. */
export function formatClaimDeadline(iso: string): string | null {
  const at = new Date(iso)

  if (Number.isNaN(at.getTime())) {
    return null
  }

  const date = new Intl.DateTimeFormat('en-US', {
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
    month: 'short',
    timeZone: 'UTC',
    year: 'numeric'
  }).format(at)

  return `${date} UTC`
}
