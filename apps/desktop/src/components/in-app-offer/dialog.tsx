import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { type I18nContextValue, useI18n } from '@/i18n'
import { openExternalLink } from '@/lib/external-link'
import {
  $inAppOfferDialog,
  dismissInAppOffer,
  listenInAppOfferStatus,
  markInAppOfferShown
} from '@/store/in-app-offer'

/**
 * The Nous Portal's in-app offer as one dismissible dialog. Copy is the
 * server's; the renderer only decides presentation. Mounted once at the
 * shell (beside the free-tier sign-in dialog), where it rides the ambient
 * free-tier status cadence.
 */
export function InAppOfferDialog() {
  const offer = useStore($inAppOfferDialog)
  const { t }: I18nContextValue = useI18n()

  useEffect(() => listenInAppOfferStatus(), [])

  useEffect(() => {
    // The presentation itself starts the frequency cap, so any way this
    // dialog leaves the screen without a dismissal can still let a live
    // campaign return tomorrow.
    if (offer) {
      markInAppOfferShown()
    }
  }, [offer])

  if (!offer) {
    return null
  }

  return (
    <Dialog
      onOpenChange={open => {
        if (!open) {
          dismissInAppOffer(offer.id)
        }
      }}
      open
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{offer.title}</DialogTitle>
          <DialogDescription>{t.inAppOffer.describedBy}</DialogDescription>
        </DialogHeader>
        <p className="text-sm">{offer.body}</p>
        <DialogFooter>
          <Button
            onClick={() => dismissInAppOffer(offer.id)}
            size="sm"
            variant="outline"
          >
            {t.inAppOffer.dismiss}
          </Button>
          <Button
            onClick={() => {
              openExternalLink(offer.cta_url)
              dismissInAppOffer(offer.id)
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
