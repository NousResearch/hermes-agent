import './usage-stats-card.css'

import { useStore } from '@nanostores/react'
import { useId, useRef } from 'react'

import { useUsageStatsCopy } from '@/app/discovery/usage-stats-copy'
import { useUsageStatsAnswer } from '@/app/discovery/use-usage-stats-answer'
import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { X } from '@/lib/icons'
import { $freeTierRoute, $freeTierStatus, freeTierStripPending } from '@/store/free-tier'
import type { SharedMetricsChoice } from '@/store/shared-metrics'
import { $usageStatsOfferPlace, deferUsageStatsOffer, openUsageStatsDetails } from '@/store/usage-stats-offer'

/** Where focus goes when the card leaves from under it: the sidebar's New chat, the panel's first
 *  control. Read before the card goes; null when focus was elsewhere (nothing to move). */
function landingAfter(card: HTMLElement | null): HTMLElement | null {
  if (!card?.contains(document.activeElement)) {
    return null
  }

  return (
    card
      .closest<HTMLElement>('[data-slot="sidebar"]')
      ?.querySelector<HTMLElement>('[data-slot="new-session-split"] button') ?? null
  )
}

/**
 * The usage-stats offer at the foot of the chat sidebar (Figma 1335:32952, 1370:34567): the redesign's
 * place for classic's composer strip, so it shows wherever the sidebar does and nowhere else. Two
 * answers, neither focused; Details opens the short explainer; × puts it in the notifications tray
 * unanswered. One offer at a time: it waits while the free-tier notice is up, as the strip does.
 */
export function UsageStatsCard() {
  const copy = useUsageStatsCopy().card
  const place = useStore($usageStatsOfferPlace)
  const freeTierNotice = freeTierStripPending(useStore($freeTierStatus), useStore($freeTierRoute))
  const { answer, saving } = useUsageStatsAnswer()
  const titleId = useId()
  const card = useRef<HTMLElement>(null)

  if (place !== 'card' || freeTierNotice) {
    return null
  }

  const choose = async (choice: SharedMetricsChoice) => {
    const landing = landingAfter(card.current)

    if (await answer(choice)) {
      landing?.focus()
    }
  }

  const putAway = () => {
    const landing = landingAfter(card.current)
    deferUsageStatsOffer()
    landing?.focus()
  }

  return (
    <section aria-labelledby={titleId} className="portal-usage-offer" data-slot="usage-stats-offer" ref={card}>
      <h2 className="portal-usage-offer-title" id={titleId}>
        {copy.title}
      </h2>
      <p className="portal-usage-offer-body">
        {copy.body}{' '}
        <button className="portal-usage-offer-link" onClick={openUsageStatsDetails} type="button">
          {copy.details}
        </button>
      </p>
      <div className="portal-usage-offer-actions">
        <button
          className="portal-usage-offer-action"
          data-tone="primary"
          disabled={saving}
          onClick={() => void choose('share')}
          type="button"
        >
          {copy.share}
        </button>
        <button className="portal-usage-offer-action" disabled={saving} onClick={() => void choose('off')} type="button">
          {copy.decline}
        </button>
      </div>
      <Tip label={copy.later}>
        <Button aria-label={copy.later} className="portal-usage-offer-close" onClick={putAway} size="icon-xs" variant="ghost">
          <X />
        </Button>
      </Tip>
    </section>
  )
}
