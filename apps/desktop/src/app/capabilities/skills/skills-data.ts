import { useQuery } from '@tanstack/react-query'

import { getSkills, type ProfileScope, profileScopeKey } from '@/hermes'
import { normalize } from '@/lib/text'
import type { SkillInfo } from '@/types/hermes'

import { asText, includesQuery } from '../../settings/helpers'

// Skills live in the RQ cache so switching tabs/pages paints the cached list
// instantly (no reload flash) and mount only fires a deduped background
// refetch. A profile swap globally invalidates (see store/profile), so this
// plain key refetches against the new backend automatically.
// `store/hub-actions` invalidates it after a hub (un)install — it imports this
// constant rather than re-spelling the key.
export const SKILLS_QUERY_KEY = ['skills-list'] as const

/** The list key for one scope: the plain key plus the Capabilities scope key,
 *  so every scoped profile keeps its own cached copy (prefix invalidations
 *  still match). */
export const skillsQueryKey = (profile: ProfileScope) => [...SKILLS_QUERY_KEY, profileScopeKey(profile)]

export function useSkillsQuery(profile: ProfileScope) {
  return useQuery({
    queryKey: skillsQueryKey(profile),
    queryFn: () => getSkills(profile),
    staleTime: 0
  })
}

export const usageOf = (skill: SkillInfo): number => (typeof skill.usage === 'number' ? skill.usage : 0)

export const categoryFor = (skill: SkillInfo): string => asText(skill.category) || 'general'

export function filteredSkills(skills: SkillInfo[], query: string, desc: boolean): SkillInfo[] {
  const q = normalize(query)
  const sign = desc ? 1 : -1

  return skills
    .filter(
      skill =>
        !q || includesQuery(skill.name, q) || includesQuery(skill.description, q) || includesQuery(skill.category, q)
    )
    .sort((a, b) => sign * (usageOf(b) - usageOf(a)) || asText(a.name).localeCompare(asText(b.name)))
}

/** The user's busiest categories, for the search field's rotating placeholder
 *  nudges — they teach that search understands categories, not just titles. */
export function skillSearchTerms(skills: SkillInfo[]): string[] {
  const counts = new Map<string, number>()

  for (const skill of skills) {
    const key = categoryFor(skill)
    counts.set(key, (counts.get(key) || 0) + 1)
  }

  return [...counts.entries()]
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5)
    .map(([category]) => category.toLowerCase())
}
