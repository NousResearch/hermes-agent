/**
 * The catalog plugins the onboarding card offers beside the hosted connectors (NS-960 D1, D4).
 *
 * The backend decides which entries are curated (`onboarding: true`) and which this OS runs, and judges
 * each app from the plugin's pinned declaration (`plugins.manage action=onboarding`). The card only
 * orders and draws them. A failed or missing RPC is an empty list: the connectors half still works.
 */
import type { OnboardingCatalogPlugin } from '@hermes/shared'
import { useEffect, useState } from 'react'

import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { isSessionOwnerRoute } from '@/store/session-request-router'

export type OnboardingPlugin = OnboardingCatalogPlugin

/** A plugin whose app is not on this machine stays pickable; the row says what is missing (D5). */
export const pluginNeedsApp = (plugin: OnboardingPlugin): boolean => plugin.app_state === 'missing_app'

export function useOnboardingPlugins(storedId: null | string): OnboardingPlugin[] {
  const [plugins, setPlugins] = useState<OnboardingPlugin[]>([])

  useEffect(() => {
    if (!storedId) {
      return
    }

    let cancelled = false
    const ambientProfile = $activeGatewayProfile.get()

    void resolveSessionOwner(storedId)
      .then(scope => {
        const connectionId = isSessionOwnerRoute(scope) ? scope.connectionId : null
        const profile = isSessionOwnerRoute(scope) ? scope.profile : scope || ambientProfile

        return requestGatewayForAgent<{ onboarding?: OnboardingPlugin[] | null }>(
          connectionId,
          profile,
          'plugins.manage',
          { action: 'onboarding' },
          20000
        )
      })
      .then(response => {
        if (!cancelled) {
          setPlugins(response.onboarding ?? [])
        }
      })
      .catch(() => {
        if (!cancelled) {
          setPlugins([])
        }
      })

    return () => {
      cancelled = true
    }
  }, [storedId])

  return plugins
}
