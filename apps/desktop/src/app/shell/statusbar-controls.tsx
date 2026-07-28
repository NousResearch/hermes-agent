import { useStore } from '@nanostores/react'
import { type ComponentProps, memo, type ReactNode, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import {
  ContextMenu,
  ContextMenuCheckboxItem,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuLabel,
  ContextMenuSeparator,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Tip, TipKeybindLabel, Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { useKeybindHint } from '@/lib/keybinds/use-keybind-hint'
import { cn } from '@/lib/utils'
import { $statusbarHiddenIds, setStatusbarItemVisible, toggleStatusbarVisible } from '@/store/statusbar-prefs'

// Shared chrome styling for interactive statusbar items (button / link / menu
// trigger). The 'text' variant intentionally omits hover/transition/disabled.
const STATUSBAR_ACTION_CLASS =
  'inline-flex h-full items-center gap-1 rounded-none px-1.5 text-[0.6875rem] text-(--ui-text-tertiary) transition-colors hover:bg-(--chrome-action-hover) hover:text-foreground disabled:cursor-default disabled:opacity-45'

export interface StatusbarMenuItem {
  id: string
  icon?: ReactNode
  label: string
  className?: string
  disabled?: boolean
  hidden?: boolean
  href?: string
  onSelect?: () => void
  title?: string
  to?: string
}

export interface StatusbarItem {
  id: string
  /** Escape hatch: render an arbitrary node into the bar (own state, tooltip,
   *  events). When set, it OWNS the slot — label/variant/onSelect are ignored.
   *  This is how a plugin drops a full stateful React component into the bar. */
  render?: () => ReactNode
  label?: ReactNode
  detail?: ReactNode
  icon?: ReactNode
  className?: string
  disabled?: boolean
  hidden?: boolean
  href?: string
  menuAlign?: 'center' | 'end' | 'start'
  menuClassName?: string
  // A render fn receives a `close()` to dismiss the popover from inside the content.
  menuContent?: ((close: () => void) => ReactNode) | ReactNode
  menuItems?: readonly StatusbarMenuItem[]
  onSelect?: (modifiers: StatusbarSelectModifiers) => void
  /** Keybind action id — when set, the tooltip shows the label + keybind hint. */
  actionId?: string
  title?: ReactNode
  to?: string
  variant?: 'action' | 'link' | 'menu' | 'text'
  /** Plain-text name for the bar's right-click show/hide menu. An item without
   *  one is never listed there and always shows — the safe default for plugin
   *  contributions that don't opt in. */
  toggleLabel?: string
  /** Listed in the menu but not switchable: the bar's own affordances (command
   *  center, update/version pills) would strand the user if they could be
   *  hidden from the surface that hides them. */
  lockedVisible?: boolean
}

export interface StatusbarSelectModifiers {
  shiftKey: boolean
}

export type StatusbarItemSide = 'left' | 'right'
export type SetStatusbarItemGroup = (id: string, items: readonly StatusbarItem[], side?: StatusbarItemSide) => void

interface StatusbarControlsProps extends ComponentProps<'footer'> {
  leftItems?: readonly StatusbarItem[]
  items?: readonly StatusbarItem[]
}

export function StatusbarControls({ className, leftItems = [], items = [], ...props }: StatusbarControlsProps) {
  const navigate = useNavigate()
  const hiddenIds = useStore($statusbarHiddenIds)

  const visible = (item: StatusbarItem) =>
    !item.hidden && (item.lockedVisible || !item.toggleLabel || !hiddenIds.includes(item.id))

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <footer
          className={cn(
            'flex h-5 shrink-0 items-stretch justify-between gap-2 bg-(--ui-sidebar-surface-background) px-1 py-0 text-(--ui-text-tertiary) [-webkit-app-region:no-drag]',
            className
          )}
          data-slot="statusbar"
          {...props}
        >
          {/* `overflow-x-clip` (not `overflow-x-auto`) so a wide status item — for
              example "Connecting…" on a fresh/untitled session — can't paint a
              horizontal scrollbar across the bottom of the window. Items already
              `truncate` their labels, so clipping is the right behavior. */}
          <div className="flex min-w-0 items-stretch gap-0.5 overflow-x-clip">
            {leftItems.filter(visible).map(item => (
              <StatusbarItemView item={item} key={`left:${item.id}`} navigate={navigate} />
            ))}
          </div>
          <div className="flex min-w-0 items-stretch gap-0.5 overflow-x-clip">
            {items.filter(visible).map(item => (
              <StatusbarItemView item={item} key={`right:${item.id}`} navigate={navigate} />
            ))}
          </div>
        </footer>
      </ContextMenuTrigger>
      <StatusbarVisibilityMenu hiddenIds={hiddenIds} items={items} leftItems={leftItems} />
    </ContextMenu>
  )
}

/** Right-click the bar to choose what it shows. Lists every item that named
 *  itself with `toggleLabel`, in bar order (left cluster then right), so the
 *  menu reads like the surface it edits. Hiding the whole bar lives at the
 *  bottom — VS Code puts it on the same context menu. */
function StatusbarVisibilityMenu({
  hiddenIds,
  items,
  leftItems
}: {
  hiddenIds: readonly string[]
  items: readonly StatusbarItem[]
  leftItems: readonly StatusbarItem[]
}) {
  const { t } = useI18n()
  const copy = t.shell.statusbar

  // Deduped by id: an item can legitimately appear in both clusters across
  // renders (contributions move sides), and a repeated checkbox would let one
  // row's toggle silently contradict the other's.
  const toggles = useMemo(() => {
    const seen = new Set<string>()

    return [...leftItems, ...items].filter(item => {
      if (!item.toggleLabel || seen.has(item.id)) {
        return false
      }

      seen.add(item.id)

      return true
    })
  }, [items, leftItems])

  return (
    <ContextMenuContent className="w-52">
      {toggles.length > 0 && (
        <>
          <ContextMenuLabel>{copy.customizeTitle}</ContextMenuLabel>
          <ContextMenuSeparator />
          {toggles.map(item => (
            <ContextMenuCheckboxItem
              checked={item.lockedVisible || !hiddenIds.includes(item.id)}
              disabled={item.lockedVisible}
              key={item.id}
              onCheckedChange={checked => setStatusbarItemVisible(item.id, checked)}
              // Radix closes the menu on select; keep it open so several items can
              // be toggled in one pass (this is a preferences surface, not a
              // command list).
              onSelect={event => event.preventDefault()}
            >
              <span className="truncate">{item.toggleLabel}</span>
            </ContextMenuCheckboxItem>
          ))}
          <ContextMenuSeparator />
        </>
      )}
      <ContextMenuItem onSelect={toggleStatusbarVisible}>
        <span className="truncate">{copy.hideStatusbar}</span>
        <StatusbarHideHint />
      </ContextMenuItem>
    </ContextMenuContent>
  )
}

/** The live ⌘⇧S hint on the hide row — the way back once the bar is gone. */
function StatusbarHideHint() {
  const hint = useKeybindHint('view.toggleStatusbar')

  return hint ? <span className="ml-auto pl-2 text-(--ui-text-quaternary)">{hint}</span> : null
}

/** Memoized: `useStatusbarItems` rebuilds the item array whenever ANY of its
 *  inputs change, but each individual item object is usually identical across
 *  those rebuilds. Without this, one changed item (the running timer, say)
 *  re-rendered every other item in the bar — measured at 1,446 wasted renders
 *  of 2,174 during a five-tab streaming run. `navigate` is stable for the
 *  router's lifetime, so item identity is the only real input. */
const StatusbarItemView = memo(function StatusbarItemView({
  item,
  navigate
}: {
  item: StatusbarItem
  navigate: ReturnType<typeof useNavigate>
}) {
  const [menuOpen, setMenuOpen] = useState(false)

  // Render escape hatch: the contribution owns its own chrome/state/tooltip.
  if (item.render) {
    return <>{item.render()}</>
  }

  const tooltipLabel = item.actionId ? (
    <TipKeybindLabel actionId={item.actionId} text={typeof item.title === 'string' ? item.title : undefined} />
  ) : (
    item.title
  )

  const content = (
    <>
      {item.icon}
      {item.label && <span className="truncate">{item.label}</span>}
      {item.detail && <span className="truncate text-muted-foreground/80">{item.detail}</span>}
    </>
  )

  if (item.variant === 'menu' && (item.menuContent || !!item.menuItems?.length)) {
    // The `Tip` helper can't wrap a menu: its TooltipTrigger needs a DOM child,
    // but DropdownMenu's Root renders no element, so the hover listeners never
    // land on the button and the tooltip silently never shows. Compose the two
    // trigger Slots directly onto the same <button> instead (both asChild), the
    // way profile-switcher.tsx stacks Popover/ContextMenu/Tooltip triggers.
    const trigger = (
      <DropdownMenuTrigger asChild>
        <button className={cn(STATUSBAR_ACTION_CLASS, item.className)} disabled={item.disabled} type="button">
          {content}
        </button>
      </DropdownMenuTrigger>
    )

    return (
      <DropdownMenu onOpenChange={setMenuOpen} open={menuOpen}>
        {item.title ? (
          <TooltipProvider delayDuration={0}>
            <Tooltip>
              <TooltipTrigger asChild>{trigger}</TooltipTrigger>
              <TooltipContent>{tooltipLabel}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        ) : (
          trigger
        )}
        <DropdownMenuContent
          align={item.menuAlign ?? 'start'}
          className={cn('w-56', item.menuContent && 'p-0', item.menuClassName)}
          side="top"
          sideOffset={8}
        >
          {item.menuContent
            ? typeof item.menuContent === 'function'
              ? item.menuContent(() => setMenuOpen(false))
              : item.menuContent
            : (item.menuItems ?? [])
                .filter(menuItem => !menuItem.hidden)
                .map(menuItem => (
                  <DropdownMenuItem
                    className={cn('gap-2 text-foreground focus:bg-accent [&_svg]:size-4', menuItem.className)}
                    disabled={menuItem.disabled}
                    key={menuItem.id}
                    onSelect={() => {
                      if (menuItem.to) {
                        navigate(menuItem.to)
                      }

                      menuItem.onSelect?.()
                    }}
                  >
                    {menuItem.href ? (
                      <a
                        className="inline-flex w-full items-center gap-2"
                        href={menuItem.href}
                        rel="noreferrer"
                        target="_blank"
                      >
                        {menuItem.icon}
                        <span className="truncate">{menuItem.label}</span>
                      </a>
                    ) : (
                      <>
                        {menuItem.icon}
                        <span className="truncate">{menuItem.label}</span>
                      </>
                    )}
                  </DropdownMenuItem>
                ))}
        </DropdownMenuContent>
      </DropdownMenu>
    )
  }

  if (item.variant === 'text' && !item.onSelect && !item.to && !item.href) {
    return (
      <Tip label={tooltipLabel}>
        <div
          className={cn(
            'inline-flex h-full items-center gap-1 px-1.5 text-[0.6875rem] text-(--ui-text-tertiary)',
            item.className
          )}
        >
          {content}
        </div>
      </Tip>
    )
  }

  if (item.href || item.variant === 'link') {
    return (
      <Tip label={tooltipLabel}>
        <a className={cn(STATUSBAR_ACTION_CLASS, item.className)} href={item.href} rel="noreferrer" target="_blank">
          {content}
        </a>
      </Tip>
    )
  }

  return (
    <Tip label={tooltipLabel}>
      <button
        className={cn(STATUSBAR_ACTION_CLASS, item.className)}
        disabled={item.disabled}
        onClick={event => {
          if (item.to) {
            navigate(item.to)
          }

          item.onSelect?.({ shiftKey: event.shiftKey })
        }}
        type="button"
      >
        {content}
      </button>
    </Tip>
  )
})
