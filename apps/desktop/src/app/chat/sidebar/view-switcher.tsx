import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'

export type SidebarView = 'projects' | 'sessions'

interface SidebarViewSwitcherProps {
  active: SidebarView
  ariaLabel: string
  onChange: (view: SidebarView) => void
  projectsLabel: string
  sessionsLabel: string
}

/**
 * The session pane has two deliberately explicit top-level views. Project
 * grouping remains a layout preference underneath, but users should not have
 * to discover it through the Filters menu just to reach their projects.
 */
export function SidebarViewSwitcher({
  active,
  ariaLabel,
  onChange,
  projectsLabel,
  sessionsLabel
}: SidebarViewSwitcherProps) {
  const itemClass = (view: SidebarView) =>
    cn(
      'h-7 min-w-0 flex-1 gap-1 rounded-[5px] px-2 text-[0.6875rem] font-medium',
      active === view
        ? 'bg-(--ui-control-active-background) text-foreground shadow-none'
        : 'text-(--ui-text-tertiary) hover:bg-(--ui-control-hover-background) hover:text-foreground'
    )

  return (
    <div
      aria-label={ariaLabel}
      className="flex w-full gap-0.5 rounded-md border border-(--ui-stroke-tertiary) p-0.5"
      data-sidebar-view-switcher="true"
      role="group"
    >
      <Button
        aria-pressed={active === 'sessions'}
        className={itemClass('sessions')}
        onClick={() => onChange('sessions')}
        size="xs"
        type="button"
        variant="ghost"
      >
        <Codicon name="comment-discussion" size="0.75rem" />
        <span className="truncate">{sessionsLabel}</span>
      </Button>
      <Button
        aria-pressed={active === 'projects'}
        className={itemClass('projects')}
        onClick={() => onChange('projects')}
        size="xs"
        type="button"
        variant="ghost"
      >
        <Codicon name="root-folder" size="0.75rem" />
        <span className="truncate">{projectsLabel}</span>
      </Button>
    </div>
  )
}
