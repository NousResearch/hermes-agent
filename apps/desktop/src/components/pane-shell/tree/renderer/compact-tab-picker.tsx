import { Fragment, type ReactNode, useId, useRef, useState } from 'react'

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

import type { TabActionsMenu } from './track-model'

interface CompactTabPickerProps {
  activeId: string
  tabs: { id: string; title: string; label?: ReactNode }[]
  onSelect: (id: string) => void
  onClose?: () => void
  newTab: { label: string; onSelect: () => void } | null
  /** A tab's own menu (the strip's right-click one), resolved per row when
   *  the picker opens; undefined = that tab has none. */
  tabActions?: (id: string) => TabActionsMenu | undefined
}

interface PickerRowProps {
  /** Called when the row MOUNTS (each time the picker opens): the main tab
   *  gains its menu once it holds a session, and the picker's own props do
   *  not change when that happens. */
  hasActions: () => boolean
  onActions: () => void
  tab: CompactTabPickerProps['tabs'][number]
}

function PickerRow({ hasActions, onActions, tab }: PickerRowProps) {
  const { t } = useI18n()

  return (
    <div className="flex items-stretch">
      <DropdownMenuRadioItem className="min-h-11 min-w-0 flex-1" value={tab.id}>
        <span className="min-w-0 whitespace-normal break-words">{tab.label ?? tab.title}</span>
      </DropdownMenuRadioItem>
      {hasActions() && (
        <DropdownMenuItem
          aria-label={t.zones.tabActions(tab.title)}
          className="min-h-11 min-w-11 shrink-0 justify-center"
          onSelect={onActions}
        >
          <Codicon name="ellipsis" />
        </DropdownMenuItem>
      )}
    </div>
  )
}

/** A phone's strip shows its current tab and a count; every open tab stays
 * reachable without a hidden horizontal drag gesture, and so does every tab's
 * menu (archive, delete, rename…) — a phone has no strip to right-click. */
export function CompactTabPicker({ activeId, tabs, onSelect, onClose, newTab, tabActions }: CompactTabPickerProps) {
  const { t } = useI18n()
  const labelId = useId()
  const countId = useId()
  // The tab whose menu is showing. Kept after the menu closes: the dialog an
  // item opened (rename, delete confirm) lives in that menu's owner.
  const [menu, setMenu] = useState<{ id: string; open: boolean } | null>(null)
  // A row's ⋯ hands focus to the tab menu; the closing picker must not pull
  // it back to its trigger (that reads as an outside focus and shuts the menu).
  const handingOff = useRef(false)
  const pickerRef = useRef<HTMLButtonElement>(null)
  const active = tabs.find(tab => tab.id === activeId)
  const count = t.zones.tabCount(tabs.length)
  const menuFor = menu ? tabActions?.(menu.id) : undefined

  return (
    <div className="relative flex min-w-0 flex-1 items-stretch" onPointerDown={event => event.stopPropagation()}>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            aria-labelledby={`${countId} ${labelId}`}
            className="min-w-0 flex-1 justify-between"
            data-slot="compact-tab-picker"
            ref={pickerRef}
            size="touch"
            variant="ghost"
          >
            <span className="truncate" id={labelId}>{active?.label ?? active?.title}</span>
            <span className="shrink-0 text-xs text-muted-foreground" id={countId}>{count}</span>
            <Codicon name="chevron-down" />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent
          align="start"
          className="max-w-[calc(100vw-1rem)] min-w-56"
          onCloseAutoFocus={event => {
            if (handingOff.current) {
              handingOff.current = false
              event.preventDefault()
            }
          }}
          side="bottom"
        >
          <DropdownMenuRadioGroup onValueChange={onSelect} value={activeId}>
            {tabs.map(tab => (
              <PickerRow
                hasActions={() => Boolean(tabActions?.(tab.id))}
                key={tab.id}
                onActions={() => {
                  handingOff.current = true
                  setMenu({ id: tab.id, open: true })
                }}
                tab={tab}
              />
            ))}
          </DropdownMenuRadioGroup>
          {onClose && (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem className="min-h-11" onSelect={onClose}>
                {t.common.close}
              </DropdownMenuItem>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>
      {menu && menuFor && (
        <Fragment key={menu.id}>
          {menuFor({
            onOpenChange: open => {
              setMenu({ id: menu.id, open })

              // The menu's own trigger is an inert anchor, so a plain close
              // (Escape, a non-dialog verb) would drop focus on the page.
              // Hand it back to the picker unless a dialog took it.
              if (!open) {
                requestAnimationFrame(() => {
                  const doc = pickerRef.current?.ownerDocument

                  if (!doc?.activeElement || doc.activeElement === doc.body) {
                    pickerRef.current?.focus()
                  }
                })
              }
            },
            open: menu.open,
            // The menu drops from the picker row; the row's ⋯ that asked for
            // it went away with the picker.
            trigger: <span aria-hidden className="pointer-events-none absolute inset-x-0 bottom-0" />
          })}
        </Fragment>
      )}
      {newTab && (
        <Button aria-label={newTab.label} onClick={newTab.onSelect} size="touch" variant="ghost">
          <Codicon name="add" />
        </Button>
      )}
    </div>
  )
}
