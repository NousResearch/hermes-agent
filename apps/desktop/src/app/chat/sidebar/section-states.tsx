import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Skeleton } from '@/components/ui/skeleton'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

export function SidebarSessionSkeletons() {
  return (
    <div aria-hidden="true" className="grid gap-px">
      {['w-32', 'w-40', 'w-28', 'w-36', 'w-24'].map((width, i) => (
        <div
          className="grid min-h-[1.625rem] grid-cols-[minmax(0,1fr)_1.375rem] items-center rounded-md pl-2"
          key={`${width}-${i}`}
        >
          <Skeleton className={cn('h-3 rounded-sm', width)} />
          <Skeleton className="mx-auto size-3.5 rounded-sm opacity-60" />
        </div>
      ))}
    </div>
  )
}

export function SidebarBlankState({ onNewProject }: { onNewProject: () => void }) {
  const { t } = useI18n()
  const s = t.sidebar

  return (
    <div className="grid min-h-0 flex-1 place-items-center px-4 text-center">
      <div className="flex flex-col items-center gap-2">
        <Codicon className="text-(--ui-text-quaternary)" name="root-folder" size="1.25rem" />
        <p className="text-xs text-(--ui-text-tertiary)">{s.noSessions}</p>
        <Button className="mt-0.5 text-(--ui-text-secondary)" onClick={onNewProject} size="sm" variant="ghost">
          <Codicon name="add" size="0.75rem" />
          {s.projects.newButton}
        </Button>
      </div>
    </div>
  )
}

export function SidebarPinnedEmptyState() {
  const { t } = useI18n()

  return (
    <div className="flex min-h-7 items-center gap-1.5 rounded-lg pl-2 text-[0.75rem] text-(--ui-text-tertiary)">
      <span className="grid w-3.5 shrink-0 place-items-center text-(--ui-text-quaternary)">
        <Codicon name="pin" size="0.75rem" />
      </span>
      <span>{t.sidebar.shiftClickHint}</span>
    </div>
  )
}
