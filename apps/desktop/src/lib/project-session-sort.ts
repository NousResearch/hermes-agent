export const PROJECT_SESSION_SORT_VALUES = ['recent', 'title-asc', 'title-desc'] as const

export type ProjectSessionSort = (typeof PROJECT_SESSION_SORT_VALUES)[number]

export const PROJECT_RECENT_PREVIEW_LIMIT = 3
export const PROJECT_TITLE_PREVIEW_LIMIT = 2_000

export function isProjectSessionSort(value: string): value is ProjectSessionSort {
  return PROJECT_SESSION_SORT_VALUES.includes(value as ProjectSessionSort)
}

export function projectPreviewLimit(sort: ProjectSessionSort): number {
  return sort === 'recent' ? PROJECT_RECENT_PREVIEW_LIMIT : PROJECT_TITLE_PREVIEW_LIMIT
}
