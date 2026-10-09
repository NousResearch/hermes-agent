import { type ReactNode, useId } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'

interface CompactTabPickerProps {
  activeId: string
  tabs: { id: string; title: string; label?: ReactNode }[]
  onSelect: (id: string) => void
  onClose?: () => void
  newTab: { label: string; onSelect: () => void } | null
}

/** A phone's strip shows its current tab and a count; every open tab stays
 * reachable without a hidden horizontal drag gesture. */
export function CompactTabPicker({ activeId, tabs, onSelect, onClose, newTab }: CompactTabPickerProps) {
  const { t } = useI18n()
  const labelId = useId()
  const countId = useId()
  const active = tabs.find(tab => tab.id === activeId)
  const count = t.zones.tabCount(tabs.length)

  return (
    <div className="flex min-w-0 flex-1 items-stretch" onPointerDown={event => event.stopPropagation()}>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            aria-labelledby={`${countId} ${labelId}`}
            className="min-w-0 flex-1 justify-between"
            data-slot="compact-tab-picker"
            size="touch"
            variant="ghost"
          >
            <span className="truncate" id={labelId}>{active?.label ?? active?.title}</span>
            <span className="shrink-0 text-xs text-muted-foreground" id={countId}>{count}</span>
            <Codicon name="chevron-down" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="max-w-[calc(100vw-1rem)] min-w-56" side="bottom">
          <DropdownMenuRadioGroup onValueChange={onSelect} value={activeId}>
            {tabs.map(tab => (
              <DropdownMenuRadioItem className="min-h-11" key={tab.id} value={tab.id}>
                <span className="min-w-0 whitespace-normal break-words">{tab.label ?? tab.title}</span>
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
          {onClose && (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="min-h-11" onSelect={onClose}>{t.common.close}</DropdownMenuItem>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
      {newTab && (
        <Button aria-label={newTab.label} onClick={newTab.onSelect} size="touch" variant="ghost">
          <Codicon name="add" />
        </Button>
      )}
    </div>
  )
}
