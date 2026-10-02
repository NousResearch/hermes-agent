import { useQuery } from '@tanstack/react-query'

import { getToolsets, type ProfileScope, profileScopeKey } from '@/hermes'
import { isDesktopToolsetVisible } from '@/lib/desktop-toolsets'
import { normalize } from '@/lib/text'
import type { ToolsetInfo } from '@/types/hermes'

import { includesQuery, toolNames, toolsetDisplayLabel } from '../../settings/helpers'

// Toolsets live in the RQ cache so switching tabs/pages paints the cached list
// instantly (no reload flash) and mount only fires a deduped background
// refetch. A profile swap globally invalidates (see store/profile), so this
// plain key refetches against the new backend automatically.
export const TOOLSETS_QUERY_KEY = ['toolsets-list'] as const

/** The list key for one scope: the plain key plus the Capabilities scope key,
 *  so every scoped profile keeps its own cached copy (prefix invalidations
 *  still match). */
export const toolsetsQueryKey = (profile: ProfileScope) => [...TOOLSETS_QUERY_KEY, profileScopeKey(profile)]

export function useToolsetsQuery(profile: ProfileScope) {
  return useQuery({
    queryKey: toolsetsQueryKey(profile),
    queryFn: () => getToolsets(profile),
    staleTime: 0
  })
}

export const toolsetCalls = (toolset: ToolsetInfo, toolCalls: Record<string, number>): number =>
  toolNames(toolset).reduce((sum, name) => sum + (toolCalls[name] ?? 0), 0)

export function filteredToolsets(
  toolsets: ToolsetInfo[],
  query: string,
  toolCalls: Record<string, number>,
  desc: boolean
): ToolsetInfo[] {
  const q = normalize(query)
  const sign = desc ? 1 : -1

  return toolsets
    .filter(toolset => {
      if (!isDesktopToolsetVisible(toolset.name)) {
        return false
      }

      if (!q) {
        return true
      }

      return (
        includesQuery(toolset.name, q) ||
        includesQuery(toolsetDisplayLabel(toolset), q) ||
        includesQuery(toolset.description, q) ||
        toolNames(toolset).some(name => includesQuery(name, q))
      )
    })
    .sort(
      (a, b) =>
        sign * (toolsetCalls(b, toolCalls) - toolsetCalls(a, toolCalls)) ||
        toolsetDisplayLabel(a).localeCompare(toolsetDisplayLabel(b))
    )
}

export const visibleToolsetCount = (toolsets: ToolsetInfo[]) =>
  toolsets.filter(ts => isDesktopToolsetVisible(ts.name)).length

/** Tool names for the search field's rotating placeholder nudges — they teach
 *  that search understands tool names, not just titles. */
export function toolsetSearchTerms(toolsets: ToolsetInfo[]): string[] {
  return toolsets
    .filter(ts => isDesktopToolsetVisible(ts.name) && toolNames(ts).length > 0)
    .slice(0, 5)
    .map(ts => toolNames(ts)[0])
}
