import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { AGENTS_ROUTE, CAPABILITIES_ROUTE, SETTINGS_ROUTE } from '@/app/routes'
import { appIsQuiet, SETTLE_MIN_MS, SETTLE_SPREAD_MS, TICK_MS } from '@/components/tips/use-tip-rotation'
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
import { useI18n } from '@/i18n'
import { isMacPlatform } from '@/lib/platform'
import { readKey, writeKey } from '@/lib/storage'
import { openCommandPalette } from '@/store/command-palette'
import { $localModelOfferPending, LOCAL_MODEL_OFFERED_KEY, LOCAL_MODEL_PENDING_KEY } from '@/store/local-model-offer'
import { readConnectedLocalSetupEligibility, readLocalSetupEligibility } from '@/store/local-setup-offer'
import { $onboardingSettleFrom } from '@/store/onboarding-run'
import { requestRoute } from '@/store/recovery-requests'
import type { LocalCatalogModel } from '@/types/hermes'

/** Shown once per install after first-run setup ends: the three places worth opening next. */
const MOVES_SEEN_KEY = 'hermes:onboarding-moves-seen'

function settleStart(settleFrom: null | number): null | number {
  if (settleFrom !== null) {
    writeKey(LOCAL_MODEL_PENDING_KEY, readKey(LOCAL_MODEL_PENDING_KEY) ?? String(settleFrom))
  }

  const pending = Number(readKey(LOCAL_MODEL_PENDING_KEY))

  return Number.isFinite(pending) && pending > 0 ? pending : null
}

/**
 * The local-model offer after first-run onboarding: armed when a run ends or the user first leaves
 * the onboarding chat, then shown at the first quiet moment once a settle window has passed.
 */
export function OnboardingEnd() {
  const { t } = useI18n()
  const settleFrom = useStore($onboardingSettleFrom)
  const [offer, setOffer] = useState<LocalCatalogModel | null>(null)
  const [movesOpen, setMovesOpen] = useState(false)

  // The run just ended: show the next-moves checklist once per install, then the
  // local-model effect below waits for its quiet moment.
  useEffect(() => {
    if (settleFrom !== null && !readKey(MOVES_SEEN_KEY)) {
      writeKey(MOVES_SEEN_KEY, '1')
      setMovesOpen(true)
    }
  }, [settleFrom])

  useEffect(() => {
    const from = settleStart(settleFrom)

    if (from === null || readKey(LOCAL_MODEL_OFFERED_KEY)) {
      return
    }

    $localModelOfferPending.set(true)
    // A machine that cannot run a local model (or cannot be read) gets no offer, so the sign-in offer
    // stops waiting. Before the connection is up nothing is known yet, so it keeps waiting until it is.
    void readConnectedLocalSetupEligibility().then(
      ({ fit }) => !fit && $localModelOfferPending.set(false),
      () => $localModelOfferPending.set(false)
    )

    const settledAt = from + SETTLE_MIN_MS + Math.random() * SETTLE_SPREAD_MS
    let lastTypedAt = 0
    let checking = false
    let cancelled = false

    const noteTyping = () => {
      lastTypedAt = Date.now()
    }

    const tick = async () => {
      if (checking || Date.now() < settledAt || readKey(LOCAL_MODEL_OFFERED_KEY) || !appIsQuiet(lastTypedAt)) {
        return
      }

      checking = true
      const { fit } = await readLocalSetupEligibility()
      checking = false

      // The read is async: the moment may have stopped being quiet while it ran.
      if (cancelled || !fit || readKey(LOCAL_MODEL_OFFERED_KEY) || !appIsQuiet(lastTypedAt)) {
        return
      }

      writeKey(LOCAL_MODEL_OFFERED_KEY, '1')
      window.clearInterval(timer)
      setOffer(fit.model)
    }

    const timer = window.setInterval(() => void tick(), TICK_MS)
    window.addEventListener('keydown', noteTyping, true)

    return () => {
      cancelled = true
      window.clearInterval(timer)
      window.removeEventListener('keydown', noteTyping, true)
    }
  }, [settleFrom])

  const copy = t.firstRun.localModel

  const close = () => {
    setOffer(null)
    $localModelOfferPending.set(false)
  }

  const moves = t.firstRun.nextMoves

  const go = (path: string) => {
    setMovesOpen(false)
    requestRoute(path)
  }

  const movesDialog = (
    <Dialog onOpenChange={open => !open && setMovesOpen(false)} open={movesOpen}>
      <DialogContent onOpenAutoFocus={preventCloseButtonAutoFocus}>
        <DialogHeader>
          <DialogTitle>{moves.title}</DialogTitle>
          <DialogDescription>{moves.subtitle}</DialogDescription>
        </DialogHeader>
        <div className="grid gap-1">
          <Button
            className="justify-start"
            onClick={() => {
              setMovesOpen(false)
              openCommandPalette()
            }}
            size="lg"
            variant="ghost"
          >
            {moves.palette}
            <span className="text-muted-foreground/70 text-xs">{isMacPlatform() ? '⌘K' : 'Ctrl K'}</span>
          </Button>
          <Button className="justify-start" onClick={() => go(AGENTS_ROUTE)} size="lg" variant="ghost">
            {moves.bots}
          </Button>
          <Button className="justify-start" onClick={() => go(CAPABILITIES_ROUTE)} size="lg" variant="ghost">
            {moves.plugins}
          </Button>
        </div>
        <DialogFooter>
          <Button onClick={() => setMovesOpen(false)} size="sm" type="button" variant="text">
            {moves.done}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )

  return (
    <>
      {movesDialog}
      <Dialog onOpenChange={open => !open && close()} open={offer !== null}>
        {offer && (
          <DialogContent onOpenAutoFocus={preventCloseButtonAutoFocus}>
            <DialogHeader>
              <DialogTitle>{copy.title}</DialogTitle>
              <DialogDescription>{copy.text(offer.display_name, offer.size_label)}</DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button onClick={close} size="sm" type="button" variant="text">
                {copy.notNow}
              </Button>
              <Button
                onClick={() => {
                  close()
                  requestRoute(`${SETTINGS_ROUTE}?tab=providers&pview=local`)
                }}
                type="button"
              >
                {copy.setUp}
              </Button>
            </DialogFooter>
          </DialogContent>
        )}
      </Dialog>
    </>
  )
}
