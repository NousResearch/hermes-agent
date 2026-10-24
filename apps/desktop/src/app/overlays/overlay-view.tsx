import { type CSSProperties, type ReactNode, useEffect } from 'react'

import { TITLEBAR_HEIGHT } from '@/app/shell/titlebar'
import { TitlebarIcon } from '@/app/shell/titlebar-icon'
import { Button } from '@/components/ui/button'
import { translateNow } from '@/i18n'
import { ESCAPE_PRIORITY, isTopEscapeLayer, pushEscapeLayer } from '@/lib/escape-layers'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'

// Shared top clearance for overlay content that sits *beside* the floating
// close button (which is absolute at `0.1875rem + titlebar/2`, -translate-y-1/2,
// so it costs no layout space): a Panel's header and the split layout's left
// sidebar links. They ride up next to the X on the same line across every
// overlay (settings, system, agents, cron, …) — change it here, not per-surface.
// Main content sits *under* the X (top-right) and keeps its own taller pad.
export const OVERLAY_TOP_CLEARANCE = 'pt-[calc(var(--titlebar-height)/2-0.4375rem)]'

interface OverlayViewProps {
  children: ReactNode
  presentation?: 'overlay' | 'page'
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
  presentation = 'overlay',
  closeLabel = translateNow('common.close'),
  contentClassName,
  edgeBadge,
  headerContent,
  rootClassName,
  titlebarActions
}: OverlayViewProps) {
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
        'fixed inset-0 z-50',
        presentation === 'page' ? 'bg-(--ui-bg-chrome) pt-(--titlebar-height)' : 'bg-black/22 backdrop-blur-[0.125rem]',
        // Equidistant inset on every side. The top value is driven by the
        // titlebar height so the card clears the OS traffic-lights vertically;
        // since the card top already sits below them, the left needs no extra
        // inset — keeping all sides equal so the card is ~full-width at any size.
        presentation === 'overlay' &&
          'p-[calc(var(--titlebar-height)+0.625rem)] sm:p-[calc(var(--titlebar-height)+0.875rem)]'
      )}
      // Every OverlayView-based overlay (settings, command-center, agents, cron,
      // profiles, star map, …) covers the chat while the composer stays mounted
      // beneath it. This marker tells `composerFocusBlockedBySurface` to stand
      // the global type-to-focus / soft `/` / Enter down, so keystrokes don't
      // leak into the hidden composer (and the overlay's own bare-key shortcuts,
      // e.g. star map's Space, keep working).
      data-overlay-surface=""
      onClick={event => {
        if (event.target === event.currentTarget) {
          closeOverlay()
        }
      }}
      role="presentation"
      // Window-level chrome: overlays always clear the real titlebar. The
      // contrib shell zeroes --titlebar-height for CONTENT areas (panes sit
      // below its in-flow title bar), and CSS vars inherit through the DOM —
      // so a fixed overlay mounted inside a zone would read 0 and bleed to
      // the edges. Re-pin the real height at the overlay root.
      style={
        {
          '--titlebar-height': `${TITLEBAR_HEIGHT}px`,
          ...(presentation === 'page' ? { '--overlay-close-space': '45vw' } : {})
        } as CSSProperties
      }
    >
      <div className="relative h-full min-h-0">
        <div
          className={cn(
            'relative flex h-full min-h-0 flex-col overflow-hidden bg-(--ui-chat-surface-background)',
            presentation === 'overlay' && 'rounded-xl border border-(--ui-stroke-secondary) shadow-md',
            rootClassName
          )}
          // Marks the card as a RAISED surface for window glass: while the field
          // behind it thins to show the desktop, this card stays near-opaque
          // (see the [data-glass-raised] rules in styles.css). Inert otherwise.
          data-glass-raised=""
        >
          <div className="pointer-events-none absolute inset-x-0 top-0 z-10 h-[calc(var(--titlebar-height)+0.1875rem)] [-webkit-app-region:drag]">
            {headerContent && (
              <div className="pointer-events-auto absolute left-1/2 top-[calc(0.5rem+var(--titlebar-height)/2)] -translate-x-1/2 -translate-y-1/2 [-webkit-app-region:no-drag]">
                {headerContent}
              </div>
            )}

            <div
              className={cn(
                'pointer-events-auto absolute top-[calc(0.1875rem+var(--titlebar-height)/2)] flex -translate-y-1/2 items-center gap-1.5 [-webkit-app-region:no-drag]',
                presentation === 'page' ? 'left-3' : 'right-[var(--titlebar-tools-right,0.75rem)]'
              )}
            >
              {titlebarActions}

              <Button
                aria-label={closeLabel}
                className={cn(
                  'text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground',
                  presentation === 'page' && 'max-w-[45vw] text-foreground'
                )}
                onClick={closeOverlay}
                size={presentation === 'page' ? 'sm' : 'icon-titlebar'}
                variant="ghost"
              >
                {presentation === 'page' ? (
                  <>
                    <TitlebarIcon name="arrow-left" />
                    <span className="truncate">{closeLabel}</span>
                  </>
                ) : (
                  <TitlebarIcon name="close" />
                )}
              </Button>
            </div>
          </div>

          {/* No top padding here: the split-layout columns own their own
              titlebar clearance so their backgrounds run flush to the card top
              (otherwise the card surface shows as a gap above the sidebar). */}
          <div className={cn('min-h-0 flex flex-1 flex-col', contentClassName)}>{children}</div>
        </div>

        {/* Sibling of the card, not a child: the card clips its own overflow
            (rounded corners), and the badge deliberately straddles the top
            border — half above, half below. */}
        {edgeBadge && (
          <div
            className={cn(
              'absolute left-1/2 z-20 -translate-x-1/2 -translate-y-1/2',
              presentation === 'page' ? '-top-[calc(var(--titlebar-height)/2)] [-webkit-app-region:no-drag]' : 'top-0'
            )}
          >
            {edgeBadge}
          </div>
        )}
      </div>
    </div>
  )
}
