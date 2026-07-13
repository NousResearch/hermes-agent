import { atom, computed } from 'nanostores'

import type { StudyReviewItem, StudyReviewStatsResponse, StudyProfile } from '@/types/hermes'

export type ReviewLoadState = 'idle' | 'loading' | 'ready' | 'error'

export const $reviewDueItems = atom<StudyReviewItem[]>([])
export const $reviewSubjects = atom<string[]>([])
export const $reviewStats = atom<StudyReviewStatsResponse | null>(null)
export const $reviewLoadState = atom<ReviewLoadState>('idle')
export const $reviewError = atom<null | string>(null)
export const $reviewProfile = atom<StudyProfile>({
  daily_review_limit: 20,
  review_level_filter: null,
  subject_filter: null
})
export const $reviewSubjectFilter = atom<null | string>(null)
export const $reviewLevelFilter = atom<null | number>(null)
export const $reviewTab = atom<'due' | 'queue' | 'stats'>('due')
export const $reviewCompletedToday = atom<number>(0)

export const $reviewProgress = computed(
  [$reviewDueItems, $reviewProfile, $reviewCompletedToday],
  (due, profile, completed) => {
    const limit = profile.daily_review_limit
    return {
      completed,
      limit,
      percent: limit > 0 ? Math.min(100, Math.round((completed / limit) * 100)) : 0
    }
  }
)

export const $filteredDueItems = computed(
  [$reviewDueItems, $reviewSubjectFilter, $reviewLevelFilter],
  (items, subject, level) =>
    items.filter(item => {
      if (subject) {
        const s = subject.toLowerCase()
        const matchesSubject = item.subject?.toLowerCase() === s
        const matchesTag = item.tags?.some(t => t.toLowerCase() === s)
        const matchesConcept = item.concepts?.some(c => c.toLowerCase().includes(s))
        if (!matchesSubject && !matchesTag && !matchesConcept) return false
      }
      if (level !== null && item.review_level !== level) return false
      return true
    })
)

export const $allTags = computed([$reviewDueItems, $reviewSubjects], (items, subjects) => {
  const tagSet = new Set<string>()
  for (const subject of subjects) tagSet.add(subject)
  for (const item of items) {
    if (item.subject) tagSet.add(item.subject)
    for (const tag of item.tags || []) tagSet.add(tag)
    for (const concept of item.concepts || []) tagSet.add(concept)
  }
  return [...tagSet].sort()
})

export function resetReviewState(): void {
  $reviewDueItems.set([])
  $reviewSubjects.set([])
  $reviewStats.set(null)
  $reviewLoadState.set('idle')
  $reviewError.set(null)
  $reviewCompletedToday.set(0)
}
