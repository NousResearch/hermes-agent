import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { isProjectSessionSort } from '@/lib/project-session-sort'
import { $projectSessionSort, setProjectSessionSort } from '@/store/layout'
import { refreshProjectTree } from '@/store/projects'

/** Compact, persisted sort picker for session rows rendered inside Projects. */
export function ProjectSessionSortMenu() {
  const { t } = useI18n()
  const sort = useStore($projectSessionSort)
  const copy = t.sidebar.projects

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={copy.sortSessions}
          className="text-(--ui-text-tertiary) opacity-70 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground hover:opacity-100 focus-visible:opacity-100 data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground data-[state=open]:opacity-100"
          size="icon-xs"
          title={copy.sortSessions}
          variant="ghost"
        >
          <Codicon name="sort-precedence" size="0.75rem" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-40" sideOffset={6}>
        <DropdownMenuLabel>{copy.sortSessions}</DropdownMenuLabel>
        <DropdownMenuRadioGroup
          onValueChange={value => {
            if (isProjectSessionSort(value)) {
              setProjectSessionSort(value)
              void refreshProjectTree()
            }
          }}
          value={sort}
        >
          <DropdownMenuRadioItem value="recent">{copy.sortRecent}</DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="title-asc">{copy.sortTitleAsc}</DropdownMenuRadioItem>
          <DropdownMenuRadioItem value="title-desc">{copy.sortTitleDesc}</DropdownMenuRadioItem>
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
