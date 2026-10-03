import { useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { StatusRow } from '@/components/chat/status-row'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { $sharedMetricsDetailsOpen, answerSharedMetricsOffer, type SharedMetricsChoice } from '@/store/shared-metrics'

const CHOICES: readonly SharedMetricsChoice[] = ['share', 'local', 'off']

/**
 * The first-run shared-metrics offer, in the composer status stack beside the
 * free-tier strip: it never blocks the composer or takes focus, and it stays
 * until one of the three equal answers is saved (the backend's `decided` is
 * the only latch). "Details" opens the full explainer the host paints.
 */
export function SharedMetricsConsentStrip() {
  const { requestGateway } = useGatewayRequest()
  const { t } = useI18n()
  const copy = t.sharedMetrics
  const [saving, setSaving] = useState(false)

  const choose = async (choice: SharedMetricsChoice) => {
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
    <StatusRow
      leading={<Codicon aria-hidden className="text-(--ui-text-tertiary)" name="graph" size="0.8rem" />}
      trailing={
        <>
          {CHOICES.map(choice => (
            <Button
              className="text-foreground/90 hover:text-foreground"
              disabled={saving}
              key={choice}
              onClick={() => void choose(choice)}
              size="micro"
              type="button"
              variant="text"
            >
              {copy.stripChoices[choice]}
            </Button>
          ))}
          <Button
            className="text-muted-foreground/75 hover:text-foreground/90"
            onClick={() => $sharedMetricsDetailsOpen.set(true)}
            size="micro"
            type="button"
            variant="text"
          >
            {copy.stripDetails}
          </Button>
        </>
      }
      trailingVisible
    >
      <span className="min-w-0 truncate text-[0.73rem] leading-4 text-foreground/92">
        <span className="font-medium">{copy.consentTitle}</span>
        <span className="text-muted-foreground/80"> {copy.stripBody}</span>
      </span>
    </StatusRow>
  )
}
