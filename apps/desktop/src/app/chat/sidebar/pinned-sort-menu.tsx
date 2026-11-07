import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $pinnedSortKey, PINNED_SORT_KEYS, type PinnedSortKey, setPinnedSortKey } from '@/store/layout'
import { $sessionsHaveCost } from '@/store/sidebar-archive'

// Pinned's own ordering control, sitting in that section's header. The section
// is the one place a conversation's order is hand-built (drag a pin to the top)
// and the one place the flat list's Ordering does not reach, so it needs its own
// switch instead of inheriting the sidebar-wide sort.
const OPTIONS: Record<PinnedSortKey, string> = {
  cost: 'credit-card',
  created: 'add',
  manual: 'list-ordered',
  name: 'case-sensitive',
  status: 'pulse',
  tokens: 'symbol-numeric',
  updated: 'clock'
}

export function PinnedSortMenu() {
  const { t } = useI18n()
  const f = t.sidebar.filter
  const ordering = useStore($pinnedSortKey)
  const hasCost = useStore($sessionsHaveCost)

  const labels: Record<PinnedSortKey, string> = {
    cost: f.cost,
    created: f.created,
    manual: f.manual,
    name: f.name,
    status: f.status,
    tokens: f.tokens,
    updated: f.updated
  }

  // Same rule as the filter menu: a cost sort over a list of zeroes is a dead
  // option, so it waits until some session actually reports spend.
  const options = PINNED_SORT_KEYS.filter(id => id !== 'cost' || hasCost || ordering === 'cost')

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          aria-label={f.ordering}
          className={cn(
            'text-(--ui-text-tertiary) opacity-0 transition-opacity hover:bg-(--ui-control-hover-background) hover:text-foreground focus-visible:opacity-100 group-hover/section:opacity-100 data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground data-[state=open]:opacity-100',
            // A sort that is not the shipped pin order is a state, not a hover:
            // keep its control on screen so the order has a visible reason.
            ordering !== 'manual' && 'opacity-100'
          )}
          size="icon-xs"
          type="button"
          variant="ghost"
        >
          <Codicon name="list-ordered" size="0.75rem" />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end">
        <DropdownMenuRadioGroup
          onValueChange={value => setPinnedSortKey(value as PinnedSortKey)}
          value={ordering}
        >
          {options.map(id => (
            <DropdownMenuRadioItem key={id} value={id}>
              <Codicon name={OPTIONS[id]} size="0.875rem" />
              {labels[id]}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
