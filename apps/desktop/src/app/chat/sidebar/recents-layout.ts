import { cn } from '@/lib/utils'

export interface RecentsFillInput {
  inProject: boolean
  showSessionSkeletons: boolean
  sessionCount: number
  projectCount: number
  projectTreeLoading: boolean
  worktreeGroupingActive: boolean
  groupedSessionCount: number
}

/** Recents should absorb leftover sidebar height only when it has something to scroll. */
export function recentsFillsSidebar(input: RecentsFillInput): boolean {
  return (
    input.inProject ||
    input.showSessionSkeletons ||
    input.sessionCount > 0 ||
    input.projectCount > 0 ||
    (input.worktreeGroupingActive && input.projectTreeLoading) ||
    input.groupedSessionCount > 0
  )
}

const SCROLL_Y = 'overflow-y-auto overflow-x-hidden overscroll-contain scrollbar-fade'
const COMPACT_FLAT = 'compact:max-h-none compact:overflow-visible'

// The section itself switches a virtualized list to overflow-visible. Keep
// SCROLL_Y on a populated list so the two don't disagree and leave recents
// with no scroller. Compact mode flattens that scroller unless the list is
// long enough to virtualize.

export function recentsSectionClassNames(fills: boolean, virtualizes: boolean): {
  rootClassName: string
  contentClassName: string
} {
  return {
    rootClassName: cn(
      'overflow-hidden p-0',
      fills ? 'min-h-32 flex-1' : 'shrink-0',
      fills && !virtualizes && 'compact:min-h-0 compact:flex-none compact:overflow-visible'
    ),
    contentClassName: cn(
      'flex flex-col gap-px pb-1.75',
      fills && 'min-h-0 flex-1',
      fills && SCROLL_Y,
      fills && !virtualizes && COMPACT_FLAT
    )
  }
}
