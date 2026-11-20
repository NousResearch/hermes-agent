import { useStore } from '@nanostores/react'
import { useMemo, useState } from 'react'

import { $apiRequestScope } from '@/api/client'
import { useI18n } from '@/i18n'
import { requestGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { answerSharedMetricsOffer, type SharedMetricsChoice, sharedMetricsProfileRequester } from '@/store/shared-metrics'

/** Answer the shared-metrics offer for the focused agent the way classic's strip does: the same
 *  owner-pinned, profile-scoped RPC, the same captured metrics gate, and a failure toast that
 *  leaves the offer up. */
export function useUsageStatsAnswer() {
  const { t } = useI18n()
  const scope = useStore($apiRequestScope)
  const activeProfile = useStore($activeGatewayProfile)
  const connectionId = scope.connectionId
  const profile = normalizeProfileKey(scope.profile ?? activeProfile)
  const [saving, setSaving] = useState(false)

  const request = useMemo(
    () =>
      sharedMetricsProfileRequester(
        <T>(method: string, params: Record<string, unknown> = {}) =>
          requestGatewayForAgent<T>(connectionId, profile, method, params),
        profile
      ),
    [connectionId, profile]
  )

  const answer = async (choice: SharedMetricsChoice): Promise<boolean> => {
    setSaving(true)

    try {
      await answerSharedMetricsOffer(request, choice, `${connectionId ?? ''}|${profile}`)

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
