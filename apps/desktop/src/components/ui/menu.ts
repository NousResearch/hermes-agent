// The one floating-list look. DropdownMenu, Select, and every Popover + cmdk
// picker (`<PopoverContent variant="menu">` + `<Command variant="menu">`) paint
// through these, so a list reads the same wherever it opens. Change the menu
// here, never per call site.

// `dt-portal-scrollbar` reproduces the thin themed scrollbar from
// `.scrollbar-dt` for portaled overlays (Radix renders under document.body,
// outside #root's scope). See styles.css.
export const menuSurfaceClass =
  'dt-portal-scrollbar rounded-lg border border-(--ui-stroke-secondary) bg-[color-mix(in_srgb,var(--ui-bg-elevated)_96%,transparent)] p-1 text-[length:var(--conversation-text-font-size)] text-popover-foreground shadow-md backdrop-blur-md'

export const menuMotionClass =
  'data-[side=bottom]:slide-in-from-top-1 data-[side=left]:slide-in-from-right-1 data-[side=right]:slide-in-from-left-1 data-[side=top]:slide-in-from-bottom-1 data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=closed]:zoom-out-95 data-[state=open]:animate-in data-[state=open]:fade-in-0 data-[state=open]:zoom-in-95'

// Row shape only. Each primitive adds its own highlight/disabled selectors:
// Radix rows take real focus (`focus:`), cmdk rows stay behind the input and
// mark `data-selected="true"`.
export const menuItemClass =
  "relative flex items-center gap-2 rounded-md px-2 py-1 text-xs outline-hidden select-none [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-3.5"

export const menuItemFocusClass = 'focus:bg-(--ui-control-active-background) focus:text-foreground'

export const menuLabelClass = 'px-2 py-1 text-xs font-medium text-(--ui-text-tertiary)'
