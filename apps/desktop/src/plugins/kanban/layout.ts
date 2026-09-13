/** Responsive layout contracts shared by the Kanban board and drawer. */
export const KANBAN_LAYOUT = {
  page: 'relative flex h-full min-w-0 flex-col overflow-hidden bg-(--ui-surface-background)',
  toolbar: 'flex shrink-0 flex-wrap items-center gap-2 px-3 py-2 md:px-4',
  toolbarSummary: 'flex min-w-0 items-center gap-2',
  toolbarFilters: 'order-3 flex w-full min-w-0 items-center gap-2 md:order-none md:w-auto',
  toolbarSearch: 'flex-1 md:flex-none',
  toolbarActions: 'ml-auto flex shrink-0 items-center gap-1',
  laneRail:
    'flex min-w-0 flex-1 snap-x snap-mandatory gap-2 overflow-x-auto overscroll-x-contain scroll-px-3 px-3 pt-1 pb-3 md:snap-none md:scroll-px-4 md:px-4',
  lane: 'group/col flex h-full w-full shrink-0 snap-start snap-always flex-col rounded-lg p-2 transition-colors md:w-64 md:snap-none',
  collapsedLane:
    'flex h-full w-full shrink-0 snap-start snap-always flex-row items-center gap-1.5 rounded-lg p-2 transition-colors hover:bg-(--ui-bg-quinary) md:w-8 md:snap-none md:flex-col',
  drawer:
    'absolute inset-y-0 right-0 z-20 flex w-full max-w-full flex-col border-l border-(--ui-stroke-tertiary) bg-(--ui-bg-elevated) pb-[env(safe-area-inset-bottom)] pl-[env(safe-area-inset-left)] pr-[env(safe-area-inset-right)] duration-150 ease-out animate-in fade-in slide-in-from-right-4 md:w-[26rem] md:p-0'
} as const
