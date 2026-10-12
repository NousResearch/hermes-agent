import type { OnboardingCatalogPlugin } from '@hermes/shared'
import { useQuery } from '@tanstack/react-query'

import { resolveSessionOwner } from '@/app/session/hooks/use-session-actions/utils'
import { queryClient } from '@/lib/query-client'
import { requestGatewayForAgent } from '@/store/gateway'
import { $activeGatewayProfile } from '@/store/profile'
import { isSessionOwnerRoute } from '@/store/session-request-router'

export type OnboardingPlugin = OnboardingCatalogPlugin

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

export function prefetchOnboardingPlugins(storedId: string): void {
  void queryClient.prefetchQuery({
    queryFn: () => readOnboardingPlugins(storedId),
    queryKey: pluginsKey(storedId),
    staleTime: Infinity
  })
}

export function useOnboardingPluginList(storedId: null | string): null | OnboardingPlugin[] {
  const query = useQuery({
    enabled: Boolean(storedId),
    queryFn: () => readOnboardingPlugins(storedId!),
    queryKey: pluginsKey(storedId),
    staleTime: Infinity
  })

  return query.data ?? null
}
