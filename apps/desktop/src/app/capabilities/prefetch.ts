import { getProfiles, getSkills, getToolsets } from '@/hermes'
import { queryClient } from '@/lib/query-client'
import { $activeGatewayProfile } from '@/store/profile'

import { CAPABILITY_PROFILES_KEY } from './scope-selector'
import { skillsQueryKey } from './skills/skills-data'
import { toolsetsQueryKey } from './toolsets/toolsets-data'

/** Warm what the Capabilities page reads before it can lay out (the installed Skills and Tools lists
 *  its tabs count and the Tools tab waits on, and the scope roster) for the scope it opens on, under
 *  the page's own keys, so a rail click paints its final layout from the cache. The ~100k-row skill
 *  feed is not warmed: parsing it would block the idle window it runs in. */
export async function prefetchCapabilitiesInventory(): Promise<void> {
  const scope = $activeGatewayProfile.get() ?? null

  await Promise.all([
    queryClient.prefetchQuery({ queryKey: skillsQueryKey(scope), queryFn: () => getSkills(scope) }),
    queryClient.prefetchQuery({ queryKey: toolsetsQueryKey(scope), queryFn: () => getToolsets(scope) }),
    queryClient.prefetchQuery({ queryKey: CAPABILITY_PROFILES_KEY, queryFn: () => getProfiles() })
  ])
}
