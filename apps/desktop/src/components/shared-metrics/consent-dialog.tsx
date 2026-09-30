import { useStore } from '@nanostores/react'
import { useEffect, useId, useState } from 'react'

import { DocsLink } from '@/components/onboarding/flow'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  preventCloseButtonAutoFocus
} from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { ChevronDown } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import { $desktopOnboarding } from '@/store/onboarding'
import { $onboardingSurfaces } from '@/store/onboarding-presence'
import {
  $sharedMetricsConsent,
  $sharedMetricsDetailsOpen,
  answerSharedMetricsOffer,
  readSharedMetricsConsent,
  SHARED_METRICS_DOCS_URL,
  type SharedMetricsChoice,
  sharedMetricsOfferPending,
  type SharedMetricsRequester
} from '@/store/shared-metrics'

interface SharedMetricsConsentDialogProps {
  /** The focused gateway is open. */
  enabled: boolean
  /** Re-ask per profile: each profile keeps its own opt-ins and install ID. */
  profile: string
  requestGateway: SharedMetricsRequester
}

const CHOICE_ORDER: readonly SharedMetricsChoice[] = ['share', 'local', 'off']

/**
 * Owner of the one-time shared-metrics question, the Desktop twin of `hermes
 * setup`'s Shared Metrics section. It reads the focused profile's answer once
 * first-run onboarding is out of the way; an unanswered profile gets an OFFER in
 * the composer status stack (`SharedMetricsConsentStrip`), never a modal — a
 * healthy install opens straight to chat. An answer given in the CLI (or here,
 * or in Settings) is the same config keys, so nobody is asked twice. This host
 * also paints the "What is collected" details the strip opens on request:
 * three equal choices, none preselected or focused, and closing it decides
 * nothing (the strip stays until answered).
 */
export function SharedMetricsConsentDialog({ enabled, profile, requestGateway }: SharedMetricsConsentDialogProps) {
  const { t } = useI18n()
  const copy = t.sharedMetrics
  const onboarding = useStore($desktopOnboarding)
  const surfaces = useStore($onboardingSurfaces)
  const detailsId = useId()
  const consent = useStore($sharedMetricsConsent)
  const detailsOpen = useStore($sharedMetricsDetailsOpen)
  const [expanded, setExpanded] = useState(true)
  const [saving, setSaving] = useState(false)

  // Never over the provider picker, the free-tier welcome or the guided chat:
  // the question belongs to the moment after setup.
  const onboardingSettled =
    (onboarding.configured === true || onboarding.firstRunSkipped) &&
    !onboarding.manual &&
    !onboarding.freeTierReady &&
    surfaces.size === 0

  const ready = enabled && onboardingSettled

  useEffect(() => {
    $sharedMetricsConsent.set(null)

    if (!ready) {
      return
    }

    let cancelled = false

    void readSharedMetricsConsent(requestGateway).then(next => {
      if (!cancelled) {
        $sharedMetricsConsent.set(next)
      }
    })

    return () => void (cancelled = true)
  }, [ready, profile, requestGateway])

  if (!ready || !detailsOpen || !sharedMetricsOfferPending(consent)) {
    return null
  }

  const choose = async (choice: SharedMetricsChoice) => {
    if (saving) {
      return
    }

    setSaving(true)

    try {
      await answerSharedMetricsOffer(requestGateway, choice)
    } catch (err) {
      notifyError(err, copy.saveFailed)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog onOpenChange={open => $sharedMetricsDetailsOpen.set(open)} open>
      <DialogContent className="max-w-md" onOpenAutoFocus={preventCloseButtonAutoFocus}>
        <DialogHeader>
          <DialogTitle>{copy.consentTitle}</DialogTitle>
          <DialogDescription>{copy.consentBody}</DialogDescription>
        </DialogHeader>

        <div className="grid gap-2">
          <button
            aria-controls={detailsId}
            aria-expanded={expanded}
            className="flex w-fit items-center gap-1 text-sm font-medium text-(--ui-text-secondary) hover:text-(--ui-text-primary)"
            onClick={() => setExpanded(value => !value)}
            type="button"
          >
            <ChevronDown className={cn('size-3.5 transition-transform', expanded ? 'rotate-0' : '-rotate-90')} />
            {copy.whatIsCollected}
          </button>
          {expanded ? (
            <div
              className="grid gap-2 rounded-lg bg-(--ui-bg-tertiary)/40 px-3 py-2.5 text-[0.8125rem] leading-5 text-(--ui-text-secondary)"
              id={detailsId}
            >
              <p>{copy.collectedIntro}</p>
              <ul className="list-disc space-y-0.5 pl-5">
                <li>{copy.collectedActivity}</li>
                <li>{copy.collectedModels}</li>
                <li>{copy.collectedNames}</li>
                <li>{copy.collectedMilestones}</li>
                <li>{copy.collectedReliability}</li>
                <li>{copy.collectedUsage}</li>
                <li>{copy.collectedMachine}</li>
              </ul>
              <p>{copy.installId}</p>
              <p>{copy.consentWindow}</p>
              <div>
                <DocsLink href={SHARED_METRICS_DOCS_URL}>{copy.readDocs}</DocsLink>
              </div>
            </div>
          ) : null}
        </div>

        <div className="grid gap-2">
          {CHOICE_ORDER.map(choice => (
            <Button disabled={saving} key={choice} onClick={() => void choose(choice)} type="button" variant="outline">
              {copy[choice]}
            </Button>
          ))}
        </div>
        <p className="text-xs text-(--ui-text-tertiary)">{copy.changeLater}</p>
      </DialogContent>
    </Dialog>
  )
}
