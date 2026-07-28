import type * as React from 'react'

import { Codicon } from '@/components/ui/codicon'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuLabel,
  ContextMenuSeparator,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
  ContextMenuTrigger
} from '@/components/ui/context-menu'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger
} from '@/components/ui/dropdown-menu'

// One place to define a set of actions and get BOTH a kebab dropdown and a
// matching right-click context menu — so a row's ⋯ menu and its right-click menu
// never drift. The dropdown and context primitives share an identical item
// surface (Item / Separator / Sub…), so a caller writes `items={kit => …}` once
// and hands the render function to both wrappers.
//
// The pattern originated inline in the session row menu; it lives here so every
// kebab in the app can add right-click parity in one line.

/** A menu flavour (dropdown / context) — the item + separator + submenu parts. */
export interface MenuKit {
  Item: typeof DropdownMenuItem | typeof ContextMenuItem
  Label: typeof DropdownMenuLabel | typeof ContextMenuLabel
  Separator: typeof DropdownMenuSeparator | typeof ContextMenuSeparator
  Sub: typeof DropdownMenuSub | typeof ContextMenuSub
  SubTrigger: typeof DropdownMenuSubTrigger | typeof ContextMenuSubTrigger
  SubContent: typeof DropdownMenuSubContent | typeof ContextMenuSubContent
  /** `CopyButton`'s `appearance` for this flavour — pass to a menu-item copy. */
  copyAppearance: 'context-menu-item' | 'menu-item'
}

export const DROPDOWN_KIT: MenuKit = {
  Item: DropdownMenuItem,
  Label: DropdownMenuLabel,
  Separator: DropdownMenuSeparator,
  Sub: DropdownMenuSub,
  SubContent: DropdownMenuSubContent,
  SubTrigger: DropdownMenuSubTrigger,
  copyAppearance: 'menu-item'
}

export const CONTEXT_KIT: MenuKit = {
  Item: ContextMenuItem,
  Label: ContextMenuLabel,
  Separator: ContextMenuSeparator,
  Sub: ContextMenuSub,
  SubContent: ContextMenuSubContent,
  SubTrigger: ContextMenuSubTrigger,
  copyAppearance: 'context-menu-item'
}

/** A single action row. Provide `icon` (codicon name) or `iconNode` (any node). */
export interface ActionItemSpec {
  className?: string
  disabled?: boolean
  icon?: string
  iconNode?: React.ReactNode
  /** Stable key; defaults to `label` when it's a string. */
  key?: string
  label: React.ReactNode
  onSelect: (event: Event) => void
  variant?: 'default' | 'destructive'
}

/** Render one `ActionItemSpec` with the given kit's Item component. */
export function renderActionItem(
  kit: MenuKit,
  { className, disabled, icon, iconNode, key, label, onSelect, variant }: ActionItemSpec
) {
  return (
    <kit.Item
      className={className}
      disabled={disabled}
      key={key ?? (typeof label === 'string' ? label : undefined)}
      onSelect={onSelect}
      variant={variant}
    >
      {iconNode ?? (icon ? <Codicon name={icon} size="0.875rem" /> : null)}
      {typeof label === 'string' ? <span>{label}</span> : label}
    </kit.Item>
  )
}

interface ActionsMenuProps extends Pick<
  React.ComponentProps<typeof DropdownMenuContent>,
  'align' | 'side' | 'sideOffset'
> {
  /** The trigger (a kebab button). Wrapped in `DropdownMenuTrigger asChild`. */
  children: React.ReactNode
  /** The action rows, rendered with `DROPDOWN_KIT`. Share this with `ActionsContextMenu`. */
  items: (kit: MenuKit) => React.ReactNode
  ariaLabel?: string
  contentClassName?: string
  open?: boolean
  onOpenChange?: (open: boolean) => void
}

/**
 * A kebab dropdown menu. Pair it with `ActionsContextMenu` using the same
 * `items` render function so the two menus stay identical. No tip on the
 * trigger — `aria-label` on the button is enough (see DESIGN.md).
 */
export function ActionsMenu({
  align = 'end',
  ariaLabel,
  children,
  contentClassName,
  items,
  onOpenChange,
  open,
  side,
  sideOffset = 6
}: ActionsMenuProps) {
  return (
    <DropdownMenu onOpenChange={onOpenChange} open={open}>
      <DropdownMenuTrigger asChild>{children}</DropdownMenuTrigger>
      <DropdownMenuContent
        align={align}
        aria-label={ariaLabel}
        className={contentClassName}
        side={side}
        sideOffset={sideOffset}
      >
        {items(DROPDOWN_KIT)}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

interface ActionsContextMenuProps {
  /** The area that receives right-click. Wrapped in `ContextMenuTrigger asChild`. */
  children: React.ReactNode
  /** The action rows, rendered with `CONTEXT_KIT`. Share this with `ActionsMenu`. */
  items: (kit: MenuKit) => React.ReactNode
  ariaLabel?: string
  contentClassName?: string
  /** Skip the wrapper (render children bare) — e.g. nothing is actionable yet. */
  disabled?: boolean
}

/**
 * Wrap a row so right-clicking it opens the same menu as its kebab. Pass the
 * kebab's `items` render function so both surfaces mirror each other.
 */
export function ActionsContextMenu({
  ariaLabel,
  children,
  contentClassName,
  disabled,
  items
}: ActionsContextMenuProps) {
  if (disabled) {
    return <>{children}</>
  }

  return (
    <ContextMenu>
      <ContextMenuTrigger asChild>{children}</ContextMenuTrigger>
      <ContextMenuContent aria-label={ariaLabel} className={contentClassName}>
        {items(CONTEXT_KIT)}
      </ContextMenuContent>
    </ContextMenu>
  )
}
