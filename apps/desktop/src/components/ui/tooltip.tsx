import './tooltip.css'

import { Tooltip as TooltipPrimitive } from 'radix-ui'
import * as React from 'react'

import { useI18n } from '@/i18n'
import { type InputModality, lastInputModality } from '@/lib/input-modality'
import { useKeybindHint } from '@/lib/keybinds/use-keybind-hint'
import { cn } from '@/lib/utils'

import { TOOLTIP_PLACEMENTS, type TooltipPlacement } from './tooltip-placement'

/** Default hover-open delay for `Tip`. Below 150ms a passing cursor still
 *  opens the tip; above 250ms an intentional hover feels broken. Call sites
 *  that need an instant tip pass `delayDuration={0}`. */
const TIP_DELAY_MS = 200

/** After a tip closes, this window stays warm: the next trigger opens
 *  instantly (Radix `skipDelayDuration`). Long enough to cover the move
 *  between adjacent chrome, short enough that a hover a second later waits
 *  again. */
const TIP_SKIP_DELAY_MS = 300

/** True inside `RootTooltipProvider`. `Tip` uses this to decide whether it
 *  needs to supply its own provider — see the note on `Tip`. */
const HasTooltipProvider = React.createContext(false)
const TooltipAnchor = React.createContext<React.RefObject<HTMLButtonElement | null> | null>(null)

function TooltipProvider({
  delayDuration = 0,
  // First hover waits `delayDuration` so a sweep across chrome does not
  // flash a trail. After one tip has opened, the page is warm: every
  // trigger entered within this window skips the delay. The cooldown
  // starts on close; a hover a second later waits again.
  skipDelayDuration = TIP_SKIP_DELAY_MS,
  // Tips are labels, not interactive surfaces. Hoverable content + Radix's
  // pointer-grace bridge is what leaves tips stuck open — especially over
  // Electron `-webkit-app-region: drag` chrome where pointermove never fires
  // to clear the grace area. Default off so open state tracks the trigger only.
  disableHoverableContent = true,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Provider>) {
  return (
    <TooltipPrimitive.Provider
      data-slot="tooltip-provider"
      delayDuration={delayDuration}
      disableHoverableContent={disableHoverableContent}
      skipDelayDuration={skipDelayDuration}
      {...props}
    />
  )
}

function Tooltip({ ...props }: React.ComponentProps<typeof TooltipPrimitive.Root>) {
  const anchor = React.useRef<HTMLButtonElement | null>(null)

  return (
    <TooltipAnchor value={anchor}>
      <TooltipPrimitive.Root data-slot="tooltip" {...props} />
    </TooltipAnchor>
  )
}

// Radix opens a tooltip on ANY trigger focus (its pointer-down guard only
// covers clicks on the trigger itself). Menus and dialogs return focus to
// their trigger when they close, so "open the model menu, pick a model" left
// the trigger's tip stuck open over the fresh selection. Gate focus-opens to
// KEYBOARD focus so a mouse pick's focus restore is suppressed while Tab-focus
// still shows the tip for a11y. preventDefault doesn't cancel the focus itself
// — Radix's composed handler just skips its onOpen when defaultPrevented.
//
// `:focus-visible` ALONE is not that gate. Radix menus autofocus their content
// and keyboard-navigate their items, so Chromium is in keyboard modality by the
// time a mouse pick restores focus and matches `:focus-visible` — the model
// pill's tip reopened over every selection. Qualify it with the device behind
// the last real interaction, which a mouse pick reports as `pointer`.
export function suppressNonKeyboardFocusOpen(
  event: React.FocusEvent<HTMLElement>,
  modality: InputModality = lastInputModality()
): void {
  let keyboardFocus = modality === 'keyboard'

  try {
    keyboardFocus &&= event.currentTarget.matches(':focus-visible')
  } catch {
    // Selector unsupported (older jsdom) — fall back to the modality alone.
  }

  if (!keyboardFocus) {
    event.preventDefault()
  }
}

function TooltipTrigger({ onFocus, ...props }: React.ComponentProps<typeof TooltipPrimitive.Trigger>) {
  const anchor = React.useContext(TooltipAnchor)
  const { ref, ...triggerProps } = props

  const setRef = React.useCallback(
    (node: HTMLButtonElement | null) => {
      if (anchor) {
        anchor.current = node
      }

      const cleanup = typeof ref === 'function' ? ref(node) : undefined

      if (ref && typeof ref !== 'function') {
        ref.current = node
      }

      return () => {
        if (anchor) {
          anchor.current = null
        }

        if (typeof cleanup === 'function') {
          cleanup()
        } else if (typeof ref === 'function') {
          ref(null)
        } else if (ref) {
          ref.current = null
        }
      }
    },
    [anchor, ref]
  )

  return (
    <TooltipPrimitive.Trigger
      data-slot="tooltip-trigger"
      onFocus={event => {
        onFocus?.(event)
        suppressNonKeyboardFocusOpen(event)
      }}
      {...triggerProps}
      ref={setRef}
    />
  )
}

interface TooltipContentProps extends React.ComponentProps<typeof TooltipPrimitive.Content> {
  placement?: TooltipPlacement
  /** Row descriptions may extend beyond a pane; local controls stay inside. */
  boundary?: 'pane' | 'viewport'
}

/** `display: contents` (and detached) elements report an all-zero rect. */
function hasLayout(element: Element | null): boolean {
  const rect = element?.getBoundingClientRect()

  return !!rect && (rect.width > 0 || rect.height > 0)
}

function TooltipContent(props: TooltipContentProps) {
  return (
    <TooltipPrimitive.Portal>
      <PaneClippedContent {...props} />
    </TooltipPrimitive.Portal>
  )
}

// Rendered inside the Portal, which Radix mounts only while the tip is open —
// so the pane is resolved at OPEN time, on every open. Resolving it once at
// `Tip` mount clipped composer tips against the zero-rect floating host for
// good: the trigger had no layout yet when the effect ran, the host was kept
// "as before", and nothing ever re-resolved it (#114602, live pass).
function PaneClippedContent({
  align,
  arrowPadding = 6,
  children,
  className,
  collisionBoundary,
  collisionPadding = 12,
  hideWhenDetached,
  placement = 'control',
  boundary = placement === 'control' || placement === 'toolbar' ? 'pane' : 'viewport',
  side,
  sideOffset = 5,
  ...props
}: TooltipContentProps) {
  const preferred = TOOLTIP_PLACEMENTS[placement]
  const anchor = React.useContext(TooltipAnchor)
  const [pane, setPane] = React.useState<Element | null>(null)

  React.useLayoutEffect(() => {
    if (boundary !== 'pane') {
      setPane(null)

      return
    }

    // A boundary without geometry (a `display: contents` host, e.g. the
    // floating-composer tree-group) zeroes every clipping rect, so `hide()`
    // detaches a fully visible trigger and the tip mounts straight into
    // `visibility: hidden`. Only a pane that has layout of its own may clip:
    // skip layout-less hosts up to the enclosing pane, or the viewport when
    // none has layout. A trigger without geometry cannot be judged (jsdom),
    // so the nearest pane stands as before.
    const trigger = anchor?.current ?? null
    let candidate = trigger?.closest('[data-tree-group]') ?? null

    if (hasLayout(trigger)) {
      while (candidate && !hasLayout(candidate)) {
        candidate = candidate.parentElement?.closest('[data-tree-group]') ?? null
      }
    }

    setPane(candidate)
  }, [anchor, boundary])

  return (
    <TooltipPrimitive.Content
      align={align ?? preferred.align}
      arrowPadding={arrowPadding}
      className={cn(
        'tooltip-bubble pointer-events-none z-(--z-over-modal) w-fit select-none bg-foreground px-2 py-1 text-[0.6875rem] font-medium leading-[1.4] text-background',
        className
      )}
      collisionBoundary={collisionBoundary ?? pane ?? undefined}
      collisionPadding={collisionPadding}
      data-slot="tooltip-content"
      // Radix's `collisionPadding` also insets the hide middleware's clip box, so
      // a 7px rail tick drawn flush against the strip's edge reads as scrolled
      // out and mounts hidden — a mark that looks dead on hover (#115723). Rail
      // ticks unmount when scrolled away, so the middleware has nothing to hide
      // there; leave it to triggers that stay put inside scrolling lists.
      hideWhenDetached={hideWhenDetached ?? (placement !== 'left-rail' && placement !== 'right-rail')}
      side={side ?? preferred.side}
      sideOffset={sideOffset}
      {...props}
    >
      <div className="tooltip-bubble-label" data-slot="tooltip-label">
        {children}
      </div>
      <TooltipPrimitive.Arrow asChild height={5} width={10}>
        <svg aria-hidden data-slot="tooltip-arrow" viewBox="0 0 10 5">
          <path d="M0 0h10L5.7 4.3a1 1 0 0 1-1.4 0Z" />
        </svg>
      </TooltipPrimitive.Arrow>
    </TooltipPrimitive.Content>
  )
}

interface TipProps extends Omit<TooltipContentProps, 'content'> {
  label: React.ReactNode
  children: React.ReactNode
  delayDuration?: number
}

// Drop-in replacement for native `title=`: wrap any single element. Instant,
// position-aware, themed. Renders the child untouched when label is falsy.
// Open state is trigger-hover only — never sticky, never click-blocking.
//
// NO per-instance `TooltipProvider`. There are ~107 `Tip` call sites, and each
// private provider is another subtree that re-renders whenever anything above
// it does. Measured on a sash drag with five mounted tiles: 52,784
// TooltipProvider renders and 18.3s of component time in a single gesture.
//
// Radix's provider holds only refs and stable callbacks (no reactive state), so
// hoisting one to the app root is exactly what it is designed for — see
// `RootTooltipProvider`, mounted in main.tsx. `Tooltip` still reads
// `delayDuration`/`disableHoverableContent` from context, and the per-Tip
// overrides below keep the previous behavior for anything that passed them.
//
// Deliberately NOT lazy-mounted: deferring the Radix subtree until hover was
// tried and reverted. `asChild` puts `data-slot="tooltip-trigger"` on the
// child element itself, so arming REPLACES that node — which broke 18 tests
// encoding that contract, and risks focus/ref identity at every call site.
function Tip({ label, children, delayDuration = TIP_DELAY_MS, ...props }: TipProps) {
  // A component rendered in isolation (every unit test, and any surface
  // mounted outside the app root) has no provider above it, and Radix throws
  // "`Tooltip` must be used within `TooltipProvider`". Fall back to a local
  // one there. Inside the app this is always false, so the common path is a
  // bare Tooltip and the ~107 providers collapse to one.
  const provided = React.useContext(HasTooltipProvider)

  if (!label) {
    return <>{children}</>
  }

  const tip = (
    <Tooltip delayDuration={delayDuration} disableHoverableContent>
      <TooltipTrigger asChild>{children}</TooltipTrigger>
      <TooltipContent {...props}>{label}</TooltipContent>
    </Tooltip>
  )

  return provided ? tip : <TooltipProvider delayDuration={delayDuration}>{tip}</TooltipProvider>
}

/** Hover-open delay for `OverflowTip`. Longer than `TIP_DELAY_MS`: the trigger
 *  is a row's own content (not a control), so the tip should only appear on a
 *  deliberate, lingering hover — a cursor travelling the list must not pop a
 *  trail of titles. */
const OVERFLOW_TIP_DELAY_MS = 600

/**
 * A `Tip` that only opens when the trigger's content is actually truncated
 * (its `scrollWidth` exceeds its `clientWidth` at pointerenter). A tooltip that
 * repeats a fully visible label is noise, and Radix's uncontrolled hover-open
 * can't see overflow — so this owns `open` and arms its own timer after
 * measuring. Pointer-only by design: keyboard focus keeps the child's existing
 * a11y affordances (the full text is already in the accessible name).
 *
 * Measurement happens on the CHILD element (`asChild` puts the trigger props on
 * it), so wrap the element that carries the truncation/overflow styling.
 */
function OverflowTip({ label, children, delayDuration = OVERFLOW_TIP_DELAY_MS, ...props }: TipProps) {
  const provided = React.useContext(HasTooltipProvider)
  const [open, setOpen] = React.useState(false)
  const timer = React.useRef<number | undefined>(undefined)

  const cancel = React.useCallback(() => {
    if (timer.current !== undefined) {
      window.clearTimeout(timer.current)
      timer.current = undefined
    }
  }, [])

  // A row unmounting mid-hover (list refresh, filter) must not fire a stale
  // timer into a torn-down tooltip.
  React.useEffect(() => cancel, [cancel])

  if (!label) {
    return <>{children}</>
  }

  const close = () => {
    cancel()
    setOpen(false)
  }

  const tip = (
    // Controlled: only closes are honored from Radix (Escape, pointer-down
    // grace); opens are ours, gated on the measured overflow below.
    <Tooltip onOpenChange={next => !next && close()} open={open}>
      <TooltipTrigger
        asChild
        // Clicking the row means the user is acting on it, not reading the tip.
        onPointerDown={close}
        onPointerEnter={event => {
          const el = event.currentTarget

          cancel()

          // Same 2px slack the sidebar marquee uses: sub-pixel rounding can
          // report a 1px "overflow" on a title that fully fits.
          if (el.scrollWidth - el.clientWidth > 2) {
            timer.current = window.setTimeout(() => setOpen(true), delayDuration)
          }
        }}
        onPointerLeave={close}
      >
        {children}
      </TooltipTrigger>
      <TooltipContent {...props}>{label}</TooltipContent>
    </Tooltip>
  )

  return provided ? tip : <TooltipProvider delayDuration={delayDuration}>{tip}</TooltipProvider>
}

/** The app's single tooltip provider. Mounted once at the root so every
 *  `Tip` shares one delay + warm-window. */
function RootTooltipProvider({ children }: { children: React.ReactNode }) {
  return (
    <HasTooltipProvider value>
      <TooltipProvider delayDuration={0} disableHoverableContent>
        {children}
      </TooltipProvider>
    </HasTooltipProvider>
  )
}

interface TipHintLabelProps {
  text: string
  hint?: string
}

/** Tooltip label with an optional trailing hotkey hint. */
function TipHintLabel({ text, hint }: TipHintLabelProps) {
  if (!hint) {
    return <>{text}</>
  }

  return (
    <>
      {text}
      <span className="ms-2 opacity-55">{hint}</span>
    </>
  )
}

interface TipKeybindLabelProps {
  /** Keybind action id — pulls the label from i18n AND the combo from the store. */
  actionId: string
  /** Override the i18n label (for context-dependent text like "Show"/"Hide"). */
  text?: string
}

/** TipHintLabel that auto-reads both its label and keybind from the action
 *  registry. Pass only `actionId` for the common case; pass `text` to override
 *  when the button's tooltip is context-dependent. */
function TipKeybindLabel({ actionId, text }: TipKeybindLabelProps) {
  const { t } = useI18n()
  const hint = useKeybindHint(actionId)

  const label = text ?? t.keybinds.actions[actionId] ?? actionId

  return <TipHintLabel hint={hint ?? undefined} text={label} />
}

export {
  OverflowTip,
  RootTooltipProvider,
  Tip,
  TipHintLabel,
  TipKeybindLabel,
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger
}
