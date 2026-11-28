import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

import { $apiRequestScope } from '@/api/client'
import { StatusRow } from '@/components/chat/status-row'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { requestGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import {
  $sharedMetricsConsent,
  $sharedMetricsDetailsOpen,
  answerSharedMetricsOffer,
  type SharedMetricsChoice,
  sharedMetricsProfileRequester
} from '@/store/shared-metrics'

const CHOICES: readonly SharedMetricsChoice[] = ['share', 'off']

/**
 * The first-run shared-metrics offer, in the composer status stack beside the
 * free-tier strip: it never blocks the composer or takes focus, and it stays
 * until one of the two equal answers is saved (the backend's `decided` is
 * the only latch). "Details" opens the full explainer the host paints.
 */
export function SharedMetricsConsentStrip() {
  const scope = useStore($apiRequestScope)
  const activeProfile = useStore($activeGatewayProfile)
  const profile = normalizeProfileKey(scope.profile ?? activeProfile)

  return <SharedMetricsConsentStripForOwner connectionId={scope.connectionId} key={JSON.stringify([scope.connectionId, profile])} profile={profile} />
}

interface SharedMetricsConsentStripOwnerProps {
  connectionId: string | null
  profile: string
}

function SharedMetricsConsentStripForOwner({ connectionId, profile }: SharedMetricsConsentStripOwnerProps) {
  const { t } = useI18n()
  const copy = t.sharedMetrics
  const reask = useStore($sharedMetricsConsent)?.reask === true
  const [saving, setSaving] = useState(false)

  const scopedRequest = useMemo(
    () => sharedMetricsProfileRequester(
      <T,>(method: string, params: Record<string, unknown> = {}) =>
        requestGatewayForAgent<T>(connectionId, profile, method, params),
      profile
    ),
    [connectionId, profile]
  )

  const choose = async (choice: SharedMetricsChoice) => {
    setSaving(true)

    try {
      await answerSharedMetricsOffer(scopedRequest, choice, `${connectionId ?? ''}|${profile}`)
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
        <span className="text-muted-foreground/80"> {reask ? copy.stripReaskBody : copy.stripBody}</span>
      </span>
    </StatusRow>
  )
}
