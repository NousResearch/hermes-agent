import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { answerSharedMetricsOffer, type SharedMetricsChoice, sharedMetricsProfileRequester } from '@/store/shared-metrics'

/** Answer the shared-metrics offer for the focused agent the way classic's strip does: the same
 *  profile-scoped RPC, the same metrics gate, and a failure toast that leaves the offer up. */
export function useUsageStatsAnswer() {
  const { requestGateway } = useGatewayRequest()
  const { t } = useI18n()
  const profile = normalizeProfileKey(useStore($activeGatewayProfile))
  const [saving, setSaving] = useState(false)

  const request = useMemo(() => sharedMetricsProfileRequester(requestGateway, profile), [profile, requestGateway])

  const answer = async (choice: SharedMetricsChoice): Promise<boolean> => {
    setSaving(true)

    try {
      await answerSharedMetricsOffer(request, choice)

      return true
    } catch (err) {
      notifyError(err, t.sharedMetrics.saveFailed)

      return false
    } finally {
      setSaving(false)
    }
  }

  return { answer, saving }
}
