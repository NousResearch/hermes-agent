/**
 * The catalog plugins the onboarding card offers beside the hosted connectors (NS-960 D1, D4).
 *
 * The backend decides which entries are curated (`onboarding: true`) and which this OS runs, and judges
 * each app from the plugin's pinned declaration (`plugins.manage action=onboarding`). The card only
 * orders and draws them. A failed or missing RPC is an empty list: the connectors half still works.
 */
import type { OnboardingCatalogPlugin } from '@hermes/shared'
import { useQuery } from '@tanstack/react-query'

import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import { queryClient } from '@/lib/query-client'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { isSessionOwnerRoute } from '@/store/session-request-router'

export type OnboardingPlugin = OnboardingCatalogPlugin

/** A plugin whose app is not on this machine stays pickable; the row says what is missing (D5). */
export const pluginNeedsApp = (plugin: OnboardingPlugin): boolean => plugin.app_state === 'missing_app'

async function readOnboardingPlugins(storedId: string): Promise<OnboardingPlugin[]> {
  try {
    const scope = await resolveSessionOwner(storedId)
    const connectionId = isSessionOwnerRoute(scope) ? scope.connectionId : null
    const profile = isSessionOwnerRoute(scope) ? scope.profile : scope || $activeGatewayProfile.get()

    const response = await requestGatewayForAgent<{ onboarding?: OnboardingPlugin[] | null }>(
      connectionId,
      profile,
      'plugins.manage',
      { action: 'onboarding' },
      20000
    )

    return response.onboarding ?? []
  } catch {
    return []
  }
}

const pluginsKey = (storedId: null | string) => ['onboarding', 'plugins.manage:onboarding', storedId] as const

/** Started with the guide session, like the connector read. */
export function prefetchOnboardingPlugins(storedId: string): void {
  void queryClient.prefetchQuery({
    queryFn: () => readOnboardingPlugins(storedId),
    queryKey: pluginsKey(storedId),
    staleTime: Infinity
  })
}

/** Cached per session like the connector read, so the card's remounts keep the rows. */
export function useOnboardingPlugins(storedId: null | string): OnboardingPlugin[] {
  const query = useQuery({
    enabled: Boolean(storedId),
    queryFn: () => readOnboardingPlugins(storedId!),
    queryKey: pluginsKey(storedId),
    staleTime: Infinity
  })

  return query.data ?? []
}
