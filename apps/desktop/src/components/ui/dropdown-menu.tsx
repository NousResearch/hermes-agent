import { DropdownMenu as DropdownMenuPrimitive } from 'radix-ui'
import * as React from 'react'

import { Codicon } from '@/components/ui/codicon'
import { usePopoverPortalContainer } from '@/components/ui/dialog-portal-context'
import {
  menuItemClass,
  menuItemFocusClass,
  menuLabelClass,
  menuMotionClass,
  menuSurfaceClass
} from '@/components/ui/menu'
import { cn } from '@/lib/utils'

// Shared class tokens for edge-to-edge menus (use with `p-0` content): rows go
// full-width, square, and compact so the highlight spans the whole surface.
// Reuse these instead of re-deriving per menu so every searchable/compact menu
// reads identically.
export const dropdownMenuRow = 'gap-2 rounded-none px-2.5 py-1 text-xs'
export const dropdownMenuSectionLabel = 'px-2.5 pt-1 pb-0.5 text-[0.625rem] font-medium uppercase tracking-wide'

// Keys that must reach Radix's menu handler (navigation/close). Everything else
// is a filter keystroke and is stopped so the menu's typeahead doesn't hijack it.
const DROPDOWN_NAV_KEYS = new Set(['ArrowDown', 'ArrowUp', 'Enter', 'Escape', 'Tab'])

// Radix highlights a row by FOCUSING it on hover (MenuItemImpl's pointermove
// calls item.focus()). In a menu with a DropdownMenuSearch, that pulls the
// caret out of the field mid-word (#53980). So while the field holds focus,
// rows cancel the pointermove (Radix skips its handler when the event is
// defaultPrevented) and set `data-highlighted` themselves. Once focus is
// elsewhere, e.g. arrowed onto a row, Radix hover applies as usual.
function searchHoldsFocus(row: HTMLElement): boolean {
  const active = row.ownerDocument.activeElement

  return Boolean(
    active?.closest('[data-slot="dropdown-menu-search"]') && row.closest('[role="menu"]')?.contains(active)
  )
}

const searchHoverClass = 'data-[highlighted]:bg-(--ui-control-active-background) data-[highlighted]:text-foreground'

type RowPointerHandlers = Pick<React.HTMLAttributes<HTMLDivElement>, 'onPointerLeave' | 'onPointerMove'>

function useSearchSafeHover(
  { onPointerLeave, onPointerMove }: RowPointerHandlers,
  { enter, leave }: { enter?: () => void; leave?: () => void } = {}
) {
  const [hovered, setHovered] = React.useState(false)

  return {
    ...(hovered ? { 'data-highlighted': '' } : {}),
    onPointerLeave: (event: React.PointerEvent<HTMLDivElement>) => {
      onPointerLeave?.(event)

      if (event.defaultPrevented || event.pointerType !== 'mouse') {
        return
      }

      setHovered(false)

      // Radix's leave handler focuses the menu surface: same theft.
      if (searchHoldsFocus(event.currentTarget)) {
        event.preventDefault()
        leave?.()
      }
    },
    onPointerMove: (event: React.PointerEvent<HTMLDivElement>) => {
      onPointerMove?.(event)

      if (event.defaultPrevented || event.pointerType !== 'mouse') {
        return
      }

      const held = searchHoldsFocus(event.currentTarget)

      setHovered(held)

      if (held) {
        event.preventDefault()
        enter?.()
      }
    }
  }
}

// Cancelling the pointermove also skips Radix's submenu hover-open, and Radix
// closes an open submenu only when focus moves to another row, which no longer
// happens. The menu re-creates both with Radix's delay as the travel grace:
// one hover-opened submenu at a time, closed when another row is hovered
// unless the pointer reaches the submenu first.
const SUBMENU_HOVER_DELAY_MS = 100

type SetSubOpen = (open: boolean) => void

function createHoverSubmenus() {
  let current: SetSubOpen | null = null
  let pending: SetSubOpen | null = null
  let openTimer = 0
  let closeTimer = 0

  const cancelOpen = () => {
    window.clearTimeout(openTimer)
    pending = null
  }

  const keep = () => {
    window.clearTimeout(closeTimer)
    closeTimer = 0
  }

  return {
    cancelOpen,
    dispose: () => {
      cancelOpen()
      keep()
    },
    enterSub: (setOpen: SetSubOpen, open: boolean) => {
      keep()

      if (open) {
        cancelOpen()
        current = setOpen

        return
      }

      if (pending === setOpen) {
        return
      }

      cancelOpen()
      pending = setOpen
      openTimer = window.setTimeout(() => {
        pending = null

        if (current !== setOpen) {
          current?.(false)
        }

        current = setOpen
        setOpen(true)
      }, SUBMENU_HOVER_DELAY_MS)
    },
    enterRow: () => {
      cancelOpen()

      if (current && !closeTimer) {
        closeTimer = window.setTimeout(() => {
          closeTimer = 0
          current?.(false)
          current = null
        }, SUBMENU_HOVER_DELAY_MS)
      }
    },
    keep
  }
}

const HoverSubmenusContext = React.createContext<ReturnType<typeof createHoverSubmenus> | null>(null)
const SubOpenContext = React.createContext<{ open: boolean; setOpen: SetSubOpen } | null>(null)
// The parent Content's resolved portal target (dialog content or an explicit
// portalContainer), so its submenus portal into the same node.
const MenuPortalContainerContext = React.createContext<HTMLElement | undefined>(undefined)

function useRowSearchHover(props: RowPointerHandlers) {
  const hoverSubmenus = React.useContext(HoverSubmenusContext)

  return useSearchSafeHover(props, { enter: () => hoverSubmenus?.enterRow() })
}

function DropdownMenu({ ...props }: React.ComponentProps<typeof DropdownMenuPrimitive.Root>) {
  return <DropdownMenuPrimitive.Root data-slot="dropdown-menu" {...props} />
}

function DropdownMenuPortal({ ...props }: React.ComponentProps<typeof DropdownMenuPrimitive.Portal>) {
  return <DropdownMenuPrimitive.Portal data-slot="dropdown-menu-portal" {...props} />
}

function DropdownMenuTrigger({ ...props }: React.ComponentProps<typeof DropdownMenuPrimitive.Trigger>) {
  return <DropdownMenuPrimitive.Trigger data-slot="dropdown-menu-trigger" {...props} />
}

/**
 * Borderless filter input for a searchable dropdown. Autofocuses, keeps the
 * menu's typeahead from eating keystrokes, and still lets arrow/enter/escape
 * drive the list. Drop it in as the first child of a `DropdownMenuContent`.
 */
function DropdownMenuSearch({
  className,
  onChange,
  onKeyDown,
  onValueChange,
  ...props
}: Omit<React.ComponentProps<'input'>, 'type'> & {
  onValueChange?: (value: string) => void
}) {
  const hoverSubmenus = React.useContext(HoverSubmenusContext)

  return (
    <div className="px-2.5 py-1.5" data-slot="dropdown-menu-search">
      <input
        autoFocus
        className={cn(
          'h-4 w-full bg-transparent text-xs leading-none text-foreground placeholder:text-(--ui-text-tertiary) focus:outline-none',
          className
        )}
        onChange={event => {
          onChange?.(event)
          onValueChange?.(event.target.value)
        }}
        onKeyDown={event => {
          if (!DROPDOWN_NAV_KEYS.has(event.key)) {
            event.stopPropagation()
          }

          // Radix focuses a submenu that opens after a keypress; a hover-open
          // still pending while the user types would take the caret with it.
          hoverSubmenus?.cancelOpen()

          onKeyDown?.(event)
        }}
        // Search fields here filter ids, slugs, and model names — dictionary
        // squiggles under them are noise (matching the composer/settings
        // inputs, which already disable spellcheck).
        spellCheck={false}
        type="text"
        {...props}
      />
    </div>
  )
}

function DropdownMenuContent({
  className,
  collisionPadding = 8,
  portalContainer,
  sideOffset = 4,
  ...props
}: React.ComponentProps<typeof DropdownMenuPrimitive.Content> & {
  portalContainer?: HTMLElement | null
}) {
  // An explicit target-owned container supports global menus whose component
  // lives outside the dialog's React context. Nested menus still inherit the
  // enclosing dialog; everything else falls back to document.body.
  const container = usePopoverPortalContainer(portalContainer)
  const [hoverSubmenus] = React.useState(createHoverSubmenus)

  React.useEffect(() => hoverSubmenus.dispose, [hoverSubmenus])

  return (
    <DropdownMenuPrimitive.Portal container={container}>
      <MenuPortalContainerContext.Provider value={container}>
        <HoverSubmenusContext.Provider value={hoverSubmenus}>
          <DropdownMenuPrimitive.Content
            className={cn(
              menuSurfaceClass,
              menuMotionClass,
              'z-50 max-h-(--radix-dropdown-menu-content-available-height) min-w-36 origin-(--radix-dropdown-menu-content-transform-origin) overflow-x-hidden overflow-y-auto',
              className
            )}
            // Keep the menu inside the viewport: Radix flips/shifts away from edges
            // (avoidCollisions defaults on); the padding stops it kissing the edge.
            collisionPadding={collisionPadding}
            data-slot="dropdown-menu-content"
            sideOffset={sideOffset}
            {...props}
          />
        </HoverSubmenusContext.Provider>
      </MenuPortalContainerContext.Provider>
    </DropdownMenuPrimitive.Portal>
  )
}

function DropdownMenuGroup({ ...props }: React.ComponentProps<typeof DropdownMenuPrimitive.Group>) {
  return <DropdownMenuPrimitive.Group data-slot="dropdown-menu-group" {...props} />
}

function DropdownMenuItem({
  className,
  inset,
  variant = 'default',
  ...props
}: React.ComponentProps<typeof DropdownMenuPrimitive.Item> & {
  inset?: boolean
  variant?: 'default' | 'destructive'
}) {
  const hoverProps = useRowSearchHover(props)

  return (
    <DropdownMenuPrimitive.Item
      className={cn(
        menuItemClass,
        menuItemFocusClass,
        searchHoverClass,
        "data-[disabled]:pointer-events-none data-[disabled]:opacity-50 data-[inset]:pl-7 data-[variant=destructive]:text-destructive data-[variant=destructive]:focus:bg-destructive/10 data-[variant=destructive]:focus:text-destructive dark:data-[variant=destructive]:focus:bg-destructive/20 [&_svg:not([class*='text-'])]:text-(--ui-text-tertiary) data-[variant=destructive]:*:[svg]:text-destructive!",
        className
      )}
      data-inset={inset}
      data-slot="dropdown-menu-item"
      data-variant={variant}
      {...props}
      {...hoverProps}
    />
  )
}

function DropdownMenuCheckboxItem({
  className,
  children,
  checked,
  ...props
}: React.ComponentProps<typeof DropdownMenuPrimitive.CheckboxItem>) {
  const hoverProps = useRowSearchHover(props)

  return (
    <DropdownMenuPrimitive.CheckboxItem
      checked={checked}
      className={cn(
        menuItemClass,
        menuItemFocusClass,
        searchHoverClass,
        'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
        className
      )}
      data-slot="dropdown-menu-checkbox-item"
      {...props}
      {...hoverProps}
    >
      {children}
      <DropdownMenuPrimitive.ItemIndicator className="ml-auto flex items-center pl-2 text-foreground">
        <Codicon name="check" size="0.75rem" />
      </DropdownMenuPrimitive.ItemIndicator>
    </DropdownMenuPrimitive.CheckboxItem>
  )
}

function DropdownMenuRadioGroup({ ...props }: React.ComponentProps<typeof DropdownMenuPrimitive.RadioGroup>) {
  return <DropdownMenuPrimitive.RadioGroup data-slot="dropdown-menu-radio-group" {...props} />
}

function DropdownMenuRadioItem({
  className,
  children,
  ...props
}: React.ComponentProps<typeof DropdownMenuPrimitive.RadioItem>) {
  const hoverProps = useRowSearchHover(props)

  return (
    <DropdownMenuPrimitive.RadioItem
      className={cn(
        menuItemClass,
        menuItemFocusClass,
        searchHoverClass,
        'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
        className
      )}
      data-slot="dropdown-menu-radio-item"
      {...props}
      {...hoverProps}
    >
      {children}
      <DropdownMenuPrimitive.ItemIndicator className="ml-auto flex items-center pl-2 text-foreground">
        <Codicon name="check" size="0.75rem" />
      </DropdownMenuPrimitive.ItemIndicator>
    </DropdownMenuPrimitive.RadioItem>
  )
}

function DropdownMenuLabel({
  className,
  inset,
  ...props
}: React.ComponentProps<typeof DropdownMenuPrimitive.Label> & {
  inset?: boolean
}) {
  return (
    <DropdownMenuPrimitive.Label
      className={cn(menuLabelClass, 'data-[inset]:pl-7', className)}
      data-inset={inset}
      data-slot="dropdown-menu-label"
      {...props}
    />
  )
}

function DropdownMenuSeparator({ className, ...props }: React.ComponentProps<typeof DropdownMenuPrimitive.Separator>) {
  return (
    <DropdownMenuPrimitive.Separator
      className={cn('-mx-1 my-1 h-px bg-(--ui-stroke-tertiary)', className)}
      data-slot="dropdown-menu-separator"
      {...props}
    />
  )
}

function DropdownMenuShortcut({ className, ...props }: React.ComponentProps<'span'>) {
  return (
    <span
      className={cn('ml-auto text-xs tracking-widest text-muted-foreground', className)}
      data-slot="dropdown-menu-shortcut"
      {...props}
    />
  )
}

function DropdownMenuSub({
  defaultOpen = false,
  onOpenChange,
  open: openProp,
  ...props
}: React.ComponentProps<typeof DropdownMenuPrimitive.Sub>) {
  // Owned here (still honouring a controlled `open`) so a sub trigger can
  // hover-open it while a menu search keeps focus; see createHoverSubmenus.
  const [openState, setOpenState] = React.useState(defaultOpen)
  const open = openProp ?? openState
  const onOpenChangeRef = React.useRef(onOpenChange)

  React.useEffect(() => {
    onOpenChangeRef.current = onOpenChange
  })

  const setOpen = React.useCallback((next: boolean) => {
    setOpenState(next)
    onOpenChangeRef.current?.(next)
  }, [])

  const sub = React.useMemo(() => ({ open, setOpen }), [open, setOpen])

  return (
    <SubOpenContext.Provider value={sub}>
      <DropdownMenuPrimitive.Sub data-slot="dropdown-menu-sub" onOpenChange={setOpen} open={open} {...props} />
    </SubOpenContext.Provider>
  )
}

function DropdownMenuSubTrigger({
  className,
  inset,
  hideChevron = false,
  children,
  ...props
}: React.ComponentProps<typeof DropdownMenuPrimitive.SubTrigger> & {
  inset?: boolean
  /** Suppress the trailing caret — for triggers that own their right-side affordance. */
  hideChevron?: boolean
}) {
  const sub = React.useContext(SubOpenContext)
  const hoverSubmenus = React.useContext(HoverSubmenusContext)

  const hoverProps = useSearchSafeHover(props, {
    enter: () => sub && hoverSubmenus?.enterSub(sub.setOpen, sub.open),
    leave: () => hoverSubmenus?.cancelOpen()
  })

  return (
    <DropdownMenuPrimitive.SubTrigger
      className={cn(
        searchHoverClass,
        "flex cursor-pointer items-center gap-2 rounded-md px-2 py-1 text-xs outline-hidden select-none focus:bg-(--ui-control-active-background) focus:text-foreground data-[inset]:pl-7 data-[state=open]:bg-(--ui-control-active-background) data-[state=open]:text-foreground [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-3.5 [&_svg:not([class*='text-'])]:text-(--ui-text-tertiary)",
        className
      )}
      data-inset={inset}
      data-slot="dropdown-menu-sub-trigger"
      {...props}
      {...hoverProps}
    >
      {children}
      {!hideChevron && <Codicon className="ml-auto text-(--ui-text-tertiary)" name="chevron-right" size="1rem" />}
    </DropdownMenuPrimitive.SubTrigger>
  )
}

function DropdownMenuSubContent({
  className,
  collisionPadding = 8,
  onPointerMove,
  style,
  ...props
}: React.ComponentProps<typeof DropdownMenuPrimitive.SubContent>) {
  const hoverSubmenus = React.useContext(HoverSubmenusContext)
  // A body portal would sit under a dialog's modal overlay: blurred, and the
  // overlay eats its clicks (#80798).
  const container = React.useContext(MenuPortalContainerContext)

  return (
    // Portal the submenu out of the parent Content so it escapes that Content's
    // `overflow` clip. Radix Popper still anchors it to the SubTrigger and
    // handles collision/flip. React events still bubble through the portal, so
    // the parent menu doesn't treat a press here as an outside click.
    //
    // `updatePositionStrategy="always"` makes Floating UI's autoUpdate use a
    // continuous rAF loop instead of the default "optimized" passive scroll
    // listeners. Without it, a portaled submenu visibly lags behind its trigger
    // while the parent Content scrolls: the scroll → getBoundingClientRect →
    // reposition pipeline cannot keep pace with fast wheel events.
    <DropdownMenuPrimitive.Portal container={container}>
      <DropdownMenuPrimitive.SubContent
        // Fixed `max-h-80` rather than the Radix available-height variable:
        // that variable is only published on Content, NOT SubContent — using
        // it here collapses the submenu to 0px height.
        className={cn(
          menuSurfaceClass,
          menuMotionClass,
          // Sharing the parent's container means sharing its stacking context,
          // so step above the parent's z-50.
          container ? 'z-(--z-modal-popover)' : 'z-50',
          'max-h-80 min-w-36 origin-(--radix-dropdown-menu-content-transform-origin) overflow-y-auto',
          className
        )}
        // Flip to the other side / shift vertically when near a viewport edge
        // (e.g. the status bar menu opening from the bottom-right corner) so
        // the submenu never gets clipped.
        collisionPadding={collisionPadding}
        data-slot="dropdown-menu-sub-content"
        onPointerMove={event => {
          // The pointer made it into the submenu: cancel a pending hover close.
          hoverSubmenus?.keep()
          onPointerMove?.(event)
        }}
        // The modal parent menu disables pointer events on dismissable layers
        // registered before it. A submenu that mounts with the menu registers
        // first (child effects run first), so Radix marks it `none`.
        style={{ ...(container ? { pointerEvents: 'auto' } : null), ...style }}
        {...props}
        // Placed after the spread so callers cannot override it: the rAF loop
        // is load-bearing for scroll-tracked submenus and must never revert to
        // the default passive-listener strategy.
        updatePositionStrategy="always"
      />
    </DropdownMenuPrimitive.Portal>
  )
}

export {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuPortal,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSearch,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger
}
