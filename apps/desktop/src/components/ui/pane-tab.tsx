import * as React from 'react'

import { isMetaClose, middleClickHandlers } from '@/lib/middle-click'
import { cn } from '@/lib/utils'

/** Inset stroke for a vertical tab rail — content-facing edge. */
export const PANE_TAB_STRIP_LINE_LEFT = 'shadow-[inset_1px_0_0_var(--ui-stroke-tertiary)]'
export const PANE_TAB_STRIP_LINE_RIGHT = 'shadow-[inset_-1px_0_0_var(--ui-stroke-tertiary)]'

const TAB =
  'group/tab relative flex shrink-0 items-center border-transparent bg-(--tab-bg) text-[0.6875rem] font-medium [-webkit-app-region:no-drag]'

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
// a darkening wash to register at all.
const TAB_IDLE =
  'text-(--ui-text-tertiary) [--tab-bg:var(--pane-tab-strip-bg,var(--ui-sidebar-surface-background))] hover:shadow-[inset_0_0_0_100vmax_color-mix(in_srgb,#000_var(--ui-tab-hover-darken),transparent)] hover:text-(--ui-text-secondary)'

interface PaneTabProps extends React.ComponentProps<'div'> {
  active?: boolean
  dirty?: boolean
  /** Close gesture, no hover X (too easy to hit on small tabs): middle-click,
   *  or ⌘-click as the trackpad-friendly Mac equivalent. */
  onClose?: () => void
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
        className
      )}
      data-active={active}
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
