import * as React from 'react'

import { type MenuKit, renderActionItem } from '@/components/ui/actions-menu'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import { translateNow } from '@/i18n'
import { isMetaClose, middleClickHandlers } from '@/lib/middle-click'
import { cn } from '@/lib/utils'

/** Inset stroke for a vertical tab rail — content-facing edge. */
export const PANE_TAB_STRIP_LINE_LEFT = 'shadow-[inset_1px_0_0_var(--ui-stroke-tertiary)]'
export const PANE_TAB_STRIP_LINE_RIGHT = 'shadow-[inset_-1px_0_0_var(--ui-stroke-tertiary)]'

// `--tab-face` is the tab's EFFECTIVE surface color — what actually sits under
// the label after every wash lands. The hover close-button gradient fades into
// it, so the fade is seamless on any theme. Idle hover repaints it below with
// the same color-mix the darkening wash applies.
const TAB =
  'group/tab relative flex shrink-0 items-center border-transparent bg-(--tab-bg) text-[0.6875rem] font-medium [-webkit-app-region:no-drag] [--tab-face:var(--tab-bg)]'

// Full height: with the strip's rule removed there is no last-pixel row to
// leave uncovered, so tabs fill the bar and no sliver of gutter shows through.
const TAB_HORIZONTAL = 'h-full min-w-0 max-w-48 not-first:border-l not-first:border-l-(--ui-stroke-quaternary)'

const TAB_VERTICAL =
  'w-full max-h-48 justify-center not-first:border-t not-first:border-t-(--ui-stroke-quaternary) [writing-mode:vertical-rl]'

const TAB_ACTIVE = 'h-full text-foreground [--tab-bg:var(--pane-tab-active-bg,var(--ui-editor-surface-background))]'

// Horizontal only: the active tab is the sole seam on the strip — a
// theme-primary underline drawn as an inset shadow in its own last pixel row,
// so it costs no layout and can't shift the tab.
const TAB_ACTIVE_UNDERLINE = 'shadow-[inset_0_-2px_0_var(--pane-tab-active-accent,var(--theme-primary))]'

// Inactive = gutter, defaulting to the shared chrome surface so a strip that
// sets no vars still matches the sidebar/titlebar instead of falling through to
// the raw (unmixed) card seed. Hover DARKENS: surfaces this close in value need
// a darkening wash to register at all. `--tab-face` tracks the wash — the same
// mix flattened onto `--tab-bg` — so the close gradient matches what shows.
const TAB_IDLE =
  'text-(--ui-text-tertiary) [--tab-bg:var(--pane-tab-strip-bg,var(--ui-sidebar-surface-background))] hover:shadow-[inset_0_0_0_100vmax_color-mix(in_srgb,#000_var(--ui-tab-hover-darken),transparent)] hover:[--tab-face:color-mix(in_srgb,#000_var(--ui-tab-hover-darken),var(--tab-bg))] hover:text-(--ui-text-secondary)'

// A tab riding a multi-tab selection: an accent wash over whatever surface the
// tab sits on. A background-image gradient (not a shadow) so it stacks cleanly
// over `--tab-bg` without fighting the active underline / hover shadows.
// `--tab-face` gets the same wash flattened in, keeping the close fade honest.
const TAB_SELECTED =
  '[background-image:linear-gradient(color-mix(in_srgb,var(--ui-accent)_14%,transparent),color-mix(in_srgb,var(--ui-accent)_14%,transparent))] [--tab-face:color-mix(in_srgb,var(--ui-accent)_14%,var(--tab-bg))] text-foreground'

interface PaneTabProps extends React.ComponentProps<'div'> {
  active?: boolean
  dirty?: boolean
  /** Close verb. Horizontal tabs reveal a hover ✕ on the right (a `--tab-face`
   *  gradient fades it over the label); middle-click and ⌘-click always work,
   *  and stay the only gestures on vertical rails (no room for a chip ✕).
   *  There is no way to take the ✕ off a tab that HAS this verb: the chip and
   *  the pointer gestures are one affordance, so a closeable tab always says
   *  so. Omit `onClose` to make a tab uncloseable. */
  onClose?: () => void
  /** Part of a multi-tab selection (⌥/Ctrl-click, Shift-click) — an accent
   *  wash marks every tab that a drag would carry, Chrome-style. */
  selected?: boolean
  /** Vertical rail form (collapsed sidebar zones). */
  vertical?: boolean
  /** Content-facing edge of a vertical rail — the strip line the active tab cuts. */
  side?: 'left' | 'right'
}

/**
 * Editor tab shell — preview rail + zone headers + collapsed vertical rails.
 *
 * Defaults need no vars: the active tab takes the editor surface, inactive the
 * sidebar one. Override `--pane-tab-active-bg` to change what the active tab
 * merges into, `--pane-tab-strip-bg` for a gutter unlike the bar around it.
 */
export const PaneTab = React.forwardRef<HTMLDivElement, PaneTabProps>(function PaneTab(
  {
    active = false,
    dirty = false,
    onClose,
    onMouseDown,
    onPointerDown,
    onPointerUp,
    onClickCapture,
    selected = false,
    vertical = false,
    side = 'left',
    children,
    className,
    ...props
  },
  ref
) {
  // Vertical rails only. Horizontal tabs draw no bottom border — the strip owns
  // that rule, and a per-tab border stacked a second translucent line over it.
  const edge = vertical ? (side === 'right' ? 'border-l' : 'border-r') : undefined
  const middle = middleClickHandlers(onClose)

  return (
    <div
      className={cn(
        TAB,
        vertical ? TAB_VERTICAL : TAB_HORIZONTAL,
        edge,
        active
          ? cn(TAB_ACTIVE, !vertical && TAB_ACTIVE_UNDERLINE)
          : cn(TAB_IDLE, edge && `${edge}-(--ui-stroke-tertiary)`),
        selected && TAB_SELECTED,
        className
      )}
      data-active={active}
      data-selected={selected || undefined}
      data-vertical={vertical || undefined}
      onClickCapture={event => {
        // Sites whose tab activates on the label's own onClick (the preview
        // rail) fire it AFTER our pointerdown close — swallow that stray click
        // in the capture phase so it can't re-select the just-closed tab.
        if (onClose && isMetaClose(event)) {
          event.preventDefault()
          event.stopPropagation()
        }

        onClickCapture?.(event)
      }}
      onMouseDown={event => {
        middle.onMouseDown(event)
        onMouseDown?.(event)
      }}
      onPointerDown={event => {
        middle.onPointerDown(event)

        // ⌘-click closes. Preempt here — the tab strips activate/drag on
        // pointerdown (drag-session onTap), so we must claim the press before
        // the shell's own handler starts a drag, and skip it entirely.
        if (onClose && isMetaClose(event)) {
          event.preventDefault()
          event.stopPropagation()
          onClose()

          return
        }

        onPointerDown?.(event)
      }}
      onPointerUp={event => {
        middle.onPointerUp(event)
        onPointerUp?.(event)
      }}
      ref={ref}
      {...props}
    >
      {children}
      {dirty && (
        <span
          aria-hidden
          className={cn(
            'pointer-events-none absolute grid size-4 place-items-center',
            vertical ? 'bottom-1.5 left-1/2 -translate-x-1/2' : 'right-1.5 top-1/2 -translate-y-1/2'
          )}
        >
          <span className="size-2 rounded-full bg-amber-500 shadow-[0_0_0_2px_var(--tab-bg),0_1px_2px_rgba(0,0,0,0.45)] dark:bg-amber-400" />
        </span>
      )}
      {onClose && !vertical && (
        // Hover ✕, painted OVER the label's right edge as an overlay (no
        // layout shift, tab width never jumps on hover). The runway is a tiny
        // transparent→`--tab-face` gradient, so the button melts into the
        // tab's effective surface instead of hard-clipping the text under it.
        // Rendered after the dirty dot: on hover the ✕ takes the dot's spot,
        // VS Code-style.
        <span className="pointer-events-none absolute inset-y-0 right-0 flex items-stretch opacity-0 transition-opacity group-hover/tab:pointer-events-auto group-hover/tab:opacity-100">
          {/* Both pieces re-draw the active underline: they paint over the
              tab's own last-pixel row, so without it the ✕ would bite a
              notch out of the accent line on the active tab. */}
          <span
            aria-hidden
            className="w-4 bg-linear-to-r from-transparent to-(--tab-face) group-data-[active=true]/tab:shadow-[inset_0_-2px_0_var(--pane-tab-active-accent,var(--theme-primary))]"
          />
          <button
            aria-label={translateNow('common.close')}
            className="grid cursor-pointer place-items-center bg-(--tab-face) pr-1.5 pl-0.5 text-(--ui-text-tertiary) outline-none hover:text-foreground group-data-[active=true]/tab:shadow-[inset_0_-2px_0_var(--pane-tab-active-accent,var(--theme-primary))]"
            onClick={event => {
              event.preventDefault()
              event.stopPropagation()
              onClose()
            }}
            onPointerDown={event => {
              // Claim a plain left press so the shell can't also activate or
              // drag the tab. Middle/⌘ presses bubble on purpose — the tab's
              // own close gestures already route them.
              if (event.button === 0 && !isMetaClose(event)) {
                event.stopPropagation()
              }
            }}
            tabIndex={-1}
            type="button"
          >
            <Codicon name="close" size="0.6875rem" />
          </button>
        </span>
      )}
    </div>
  )
})

interface PaneTabLabelProps extends React.ComponentProps<'button'> {
  /** `button` when the label is the activation target (preview rail);
   *  default `span` defers to the shell (zone drag/activate). */
  as?: 'button' | 'span'
}

/** Truncating label inside a `PaneTab`. `className` merges into the text span
 *  (e.g. `normal-case tracking-normal` for filenames). */
export const PaneTabLabel = React.forwardRef<HTMLElement, PaneTabLabelProps>(function PaneTabLabel(
  { as = 'span', className, children, ...props },
  ref
) {
  const Comp = as as React.ElementType

  return (
    <Comp
      className="flex h-full min-w-0 max-w-full items-center overflow-hidden px-2 text-left outline-none group-data-[vertical]/tab:h-auto group-data-[vertical]/tab:w-full group-data-[vertical]/tab:justify-center group-data-[vertical]/tab:py-2"
      ref={ref}
      {...props}
    >
      <span className={cn('block min-w-0 truncate text-[9px] font-medium tracking-wide uppercase', className)}>
        {children}
      </span>
    </Comp>
  )
})

interface PaneTabStripProps extends React.ComponentProps<'div'> {
  /** The scrolling tab list — receives `role="tablist"`. */
  children: React.ReactNode
  /** Ref on the scroller itself, for `useActiveTabVisible`. */
  listRef?: React.Ref<HTMLDivElement>
  /** Non-scrolling trailing chrome pinned to the right (the minimize chevron). */
  trailing?: React.ReactNode
}

/**
 * The horizontal tab bar every strip in the app sits in. Owns the bar's height
 * and surface, the scroll behaviour (hidden scrollbars, contained overscroll),
 * and the pinned trailing slot — so a new strip inherits all of it instead of
 * re-deriving the geometry and drifting out of alignment.
 *
 * Tabs go in `children` as `PaneTab`s; per-strip extras (drag handlers,
 * `data-zone-tabstrip`, drop carets) ride on the usual div props.
 */
export const PaneTabStrip = React.forwardRef<HTMLDivElement, PaneTabStripProps>(function PaneTabStrip(
  { children, className, listRef, trailing, ...props },
  ref
) {
  return (
    <div
      // Strip and active tab both sit on the sidebar surface, so the bar reads
      // as one piece of chrome with the titlebar above it. No bottom rule — the
      // active tab's primary underline is the only seam.
      className={cn(
        'group/pane-header relative flex h-7 shrink-0 select-none bg-(--ui-sidebar-surface-background) [-webkit-app-region:no-drag] [--pane-tab-active-bg:var(--ui-sidebar-surface-background)]',
        className
      )}
      ref={ref}
      {...props}
    >
      <div
        className="flex min-w-0 flex-1 overflow-x-auto overflow-y-hidden overscroll-x-contain [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
        ref={listRef}
        role="tablist"
      >
        {children}
      </div>
      {trailing}
    </div>
  )
})

/** A glyph button on a tab strip: the "+" and anything a pane contributes (a
 *  preview's console / DevTools). Callers pass DATA, never classes — the same
 *  contract as `TitlebarTool`, so every glyph on every strip matches. */
export interface PaneStripTool {
  active?: boolean
  disabled?: boolean
  icon: React.ReactNode
  id: string
  /** Tooltip text and accessible name. */
  label: string
  onSelect: () => void
}

/**
 * Renders one `PaneStripTool` through the app's `Button` + `Tip` primitives, the
 * way `TitlebarToolButton` does: ghost variant, no active background — state
 * reads from the glyph's own opacity, with `aria-pressed` carrying it for a11y.
 *
 * Pointerdown is claimed here so a click can never also activate or drag the
 * zone behind the strip.
 */
export function PaneStripGlyph({ active, disabled, icon, label, onSelect }: Omit<PaneStripTool, 'id'>) {
  return (
    <Tip label={label}>
      <Button
        aria-label={label}
        aria-pressed={active ?? undefined}
        className={cn(
          'self-center bg-transparent select-none',
          active ? 'opacity-100' : 'opacity-60 hover:opacity-100'
        )}
        disabled={disabled}
        onClick={onSelect}
        onPointerDown={event => event.stopPropagation()}
        size="icon-xs"
        type="button"
        variant="ghost"
      >
        {icon}
      </Button>
    </Tip>
  )
}

/** Close-verb enablement for `paneTabCloseItems` — how many tabs each verb hits. */
export interface PaneTabCloseCounts {
  all: number
  others: number
  right: number
}

interface PaneTabCloseItemsOptions {
  counts: PaneTabCloseCounts
  /** Omit to hide Close entirely (an uncloseable tab shows no dead action). */
  onClose?: () => void
  onCloseAll: () => void
  onCloseOthers: () => void
  onCloseToRight: () => void
}

/**
 * The four close verbs every tab menu offers — Close / others / to the right /
 * all — so a tab answers a right-click the same way wherever it lives. No ⌘W
 * hint on Close: the keybind closes the FOCUSED zone's active tab, so it would
 * be a lie on the inactive tab the user actually right-clicked.
 */
export function paneTabCloseItems(
  kit: MenuKit,
  { counts, onClose, onCloseAll, onCloseOthers, onCloseToRight }: PaneTabCloseItemsOptions
) {
  return (
    <>
      {onClose &&
        renderActionItem(kit, {
          icon: 'close',
          label: translateNow('common.close'),
          onSelect: onClose
        })}
      {renderActionItem(kit, {
        disabled: !counts.others,
        icon: 'close-all',
        label: translateNow('zones.closeOthers'),
        onSelect: onCloseOthers
      })}
      {renderActionItem(kit, {
        disabled: !counts.right,
        icon: 'arrow-right',
        label: translateNow('zones.closeToRight'),
        onSelect: onCloseToRight
      })}
      {renderActionItem(kit, {
        disabled: !counts.all,
        icon: 'clear-all',
        label: translateNow('zones.closeAll'),
        onSelect: onCloseAll
      })}
    </>
  )
}
