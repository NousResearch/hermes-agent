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

/** Browsing groups for the Tools page, by what a toolset lets the agent do.
 *  Toolsets not named here land in the trailing "More tools" group. */
export type ToolsetThemeId = 'work' | 'create' | 'recall' | 'connect' | 'more'

/** Browsing-group membership; labels and blurbs are i18n (`catalog.toolsets.themes`). */
export const TOOLSET_THEMES: ReadonlyArray<{ id: Exclude<ToolsetThemeId, 'more'>; names: readonly string[] }> = [
  {
    id: 'work',
    names: [
      'browser',
      'computer_use',
      'web',
      'terminal',
      'file',
      'code_execution',
      'cronjob',
      'kanban',
      'delegation',
      'todo',
      'clarify'
    ]
  },
  {
    id: 'create',
    names: ['vision', 'video', 'stt', 'tts', 'image_gen', 'video_gen']
  },
  {
    id: 'recall',
    names: ['memory', 'session_search', 'skills']
  },
  {
    id: 'connect',
    names: ['connections', 'a2a', 'homeassistant', 'x_search', 'messaging', 'spotify']
  }
]

export const TOOLSET_MORE_THEME: { id: ToolsetThemeId } = { id: 'more' }

/** Toolsets in the feature row, and the fewest a profile must have before the row shows. */
export const TOOLSET_FEATURE_COUNT = 4
export const TOOLSET_FEATURE_MIN = 8

/** The feature row before there's usage to rank by. */
export const TOOLSET_ESSENTIALS: readonly string[] = ['browser', 'file', 'code_execution', 'memory']

export const toolsetTheme = (name: string): { id: ToolsetThemeId } =>
  TOOLSET_THEMES.find(theme => theme.names.includes(name)) ?? TOOLSET_MORE_THEME

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
