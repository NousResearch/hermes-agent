import { type CSSProperties, type ReactNode, useEffect, useLayoutEffect, useRef } from 'react'

import { TITLEBAR_HEIGHT } from '@/app/shell/titlebar'
import { TitlebarIcon } from '@/app/shell/titlebar-icon'
import { useTitlebarHeight } from '@/app/shell/use-touch-titlebar'
import { Button } from '@/components/ui/button'
import { translateNow } from '@/i18n'
import { ESCAPE_PRIORITY, isTopEscapeLayer, pushEscapeLayer } from '@/lib/escape-layers'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'
import { usePortalShell } from '@/store/shell-variant'

import { PageToolbar } from './page-toolbar'

// Shared top clearance for overlay content that sits *beside* the floating
// close button (which is absolute at `0.1875rem + titlebar/2`, -translate-y-1/2,
// so it costs no layout space): a Panel's header and the split layout's left
// sidebar links. They ride up next to the X on the same line across every
// overlay (settings, system, agents, cron, …) — change it here, not per-surface.
// Main content sits *under* the X (top-right) and keeps its own taller pad.
export const OVERLAY_TOP_CLEARANCE = 'pt-[calc(var(--titlebar-height)/2-0.4375rem)]'

/** A redesign page overlay takes keyboard focus on mount so Tab enters the page's own controls. The
 *  live global-nav rail sits earlier in the DOM (the layout tree beside it is inert), so without
 *  this the first Tab after a background element is blurred would land on the rail behind the page.
 *  Focus goes back to whatever held it outside the overlay when the page closes, unless that element
 *  is gone or now inert. Classic's floating card dims the shell it leaves interactive, so it must not
 *  move focus — gate the whole thing on `page`. */
function usePageOverlayFocus(page: boolean) {
  const ref = useRef<HTMLDivElement>(null)
  const restoreTo = useRef<HTMLElement | null>(null)

  useLayoutEffect(() => {
    const root = ref.current

    if (!page || !root) {
      return
    }

    const active = document.activeElement

    if (active instanceof HTMLElement && active !== document.body && !root.contains(active)) {
      restoreTo.current = active
    }

    // A descendant that already autofocused itself owns focus; don't yank it back to the container.
    if (!root.contains(document.activeElement)) {
      root.focus({ preventScroll: true })
    }

    return () => {
      const target = restoreTo.current
      restoreTo.current = null

      // Hand focus back unless the user has since moved it somewhere real. The overlay's own
      // container counts as ours to give up: browsers drop focus to `<body>` when the focused node is
      // removed, and jsdom may leave the (now detached) node named.
      const active = document.activeElement
      const ours = !active || !active.isConnected || active === document.body || root.contains(active)

      if (target?.isConnected && !target.closest('[inert]') && ours) {
        target.focus({ preventScroll: true })
      }
    }
  }, [page])

  return ref
}

interface OverlayViewProps {
  children: ReactNode
  onClose: () => void
  closeLabel?: string
  contentClassName?: string
  /** Chrome pinned to the card's top edge, horizontally centered and riding
   *  the border half-in half-out (e.g. the Settings search pill). Rendered
   *  beside the card, not inside it — the card clips its own overflow. */
  edgeBadge?: ReactNode
  headerContent?: ReactNode
  rootClassName?: string
  /** Controls rendered on the close button's row, to its left. They ride the
   *  titlebar strip, so keep them titlebar-sized and quiet. */
  titlebarActions?: ReactNode
}

export function OverlayView({
  children,
  onClose,
  closeLabel = translateNow('common.close'),
  contentClassName,
  edgeBadge,
  headerContent,
  rootClassName,
  titlebarActions
}: OverlayViewProps) {
  const page = usePortalShell()
  const titlebarHeight = useTitlebarHeight()
  const surfaceRef = usePageOverlayFocus(page)

  const closeOverlay = () => {
    triggerHaptic('close')
    onClose()
  }

  // Esc dismisses every OverlayView-based overlay. Nested Radix dialogs
  // stop propagation themselves, so opening (e.g.) the model picker inside
  // Settings still closes the picker first instead of the underlying overlay.
  useEffect(() => {
    const releaseLayer = pushEscapeLayer(ESCAPE_PRIORITY.overlay)

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented || !isTopEscapeLayer(ESCAPE_PRIORITY.overlay)) {
        return
      }

      event.preventDefault()
      triggerHaptic('close')
      onClose()
    }

    window.addEventListener('keydown', onKeyDown)

    return () => {
      window.removeEventListener('keydown', onKeyDown)
      releaseLayer()
    }
  }, [onClose])

  return (
    <div
      className={cn(
        'fixed z-50',
        // Equidistant inset on every side. The top value is driven by the
        // titlebar height so the card clears the OS traffic-lights vertically;
        // since the card top already sits below them, the left needs no extra
        // inset — keeping all sides equal so the card is ~full-width at any size.
        page
          ? 'bottom-0 start-(--global-nav-width) end-0 top-(--titlebar-height) bg-background outline-none'
          : 'inset-0 bg-black/22 p-[calc(var(--titlebar-height)+0.625rem)] backdrop-blur-[0.125rem] sm:p-[calc(var(--titlebar-height)+0.875rem)]'
      )}
      // Every OverlayView-based overlay (settings, command-center, agents, cron,
      // profiles, star map, …) covers the chat while the composer stays mounted
      // beneath it. This marker tells `composerFocusBlockedBySurface` to stand
      // the global type-to-focus / soft `/` / Enter down, so keystrokes don't
      // leak into the hidden composer (and the overlay's own bare-key shortcuts,
      // e.g. star map's Space, keep working).
      data-overlay-surface=""
      data-page-surface={page || undefined}
      onClick={event => {
        if (!page && event.target === event.currentTarget) {
          closeOverlay()
        }
      }}
      ref={surfaceRef}
      role="presentation"
      // Window-level chrome: overlays always clear the real titlebar. The
      // contrib shell zeroes --titlebar-height for CONTENT areas (panes sit
      // below its in-flow title bar), and CSS vars inherit through the DOM —
      // so a fixed overlay mounted inside a zone would read 0 and bleed to
      // the edges. Re-pin the real height at the overlay root.
      // Classic's card insets from its compact band on every host, as on main.
      style={{ '--titlebar-height': `${page ? titlebarHeight : TITLEBAR_HEIGHT}px` } as CSSProperties}
      // A redesign page's container is programmatically focused (usePageOverlayFocus) so Tab enters
      // the page; it is not a control, so it stays out of the tab order and draws no focus ring.
      tabIndex={page ? -1 : undefined}
    >
      {page && (
        <div className="absolute inset-x-0 bottom-full h-(--titlebar-height) bg-background [-webkit-app-region:drag]" data-slot="page-toolbar">
          <PageToolbar actions={titlebarActions} onBack={closeOverlay}>
            {headerContent}
            {edgeBadge}
          </PageToolbar>
        </div>
      )}
      <div className="relative h-full min-h-0">
        <div
          className={cn(
            'relative flex h-full min-h-0 flex-col overflow-hidden bg-(--ui-chat-surface-background)',
            !page && 'rounded-xl border border-(--ui-stroke-secondary) shadow-md',
            rootClassName
          )}
          // Marks the card as a RAISED surface for window glass: while the field
          // behind it thins to show the desktop, this card stays near-opaque
          // (see the [data-glass-raised] rules in styles.css). Inert otherwise.
          data-glass-raised=""
        >
          {!page && (
            <div className="pointer-events-none absolute inset-x-0 top-0 z-10 h-[calc(var(--titlebar-height)+0.1875rem)] [-webkit-app-region:drag]">
              {headerContent && (
                <div className="pointer-events-auto absolute left-1/2 top-[calc(0.5rem+var(--titlebar-height)/2)] -translate-x-1/2 -translate-y-1/2 [-webkit-app-region:no-drag]">
                  {headerContent}
                </div>
              )}

              <div className="pointer-events-auto absolute right-[var(--titlebar-tools-right,0.75rem)] top-[calc(0.1875rem+var(--titlebar-height)/2)] flex -translate-y-1/2 items-center gap-1.5 [-webkit-app-region:no-drag]">
                {titlebarActions}

                <Button
                  aria-label={closeLabel}
                  className="text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground"
                  onClick={closeOverlay}
                  size="icon-titlebar"
                  variant="ghost"
                >
                  <TitlebarIcon name="close" />
                </Button>
              </div>
            </div>
          )}

          {/* No top padding here: the split-layout columns own their own
              titlebar clearance so their backgrounds run flush to the card top
              (otherwise the card surface shows as a gap above the sidebar). */}
          <div className={cn('min-h-0 flex flex-1 flex-col', contentClassName)}>{children}</div>
        </div>

        {/* Sibling of the card, not a child: the card clips its own overflow
            (rounded corners), and the badge deliberately straddles the top
            border — half above, half below. */}
        {!page && edgeBadge && (
          <div className="absolute left-1/2 top-0 z-20 -translate-x-1/2 -translate-y-1/2">{edgeBadge}</div>
        )}
      </div>
    </div>
  )
}
