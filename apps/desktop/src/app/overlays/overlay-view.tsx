import { type CSSProperties, type ReactNode, useEffect } from 'react'

import { TITLEBAR_HEIGHT } from '@/app/shell/titlebar'
import { SurfaceCloseButton, SurfaceCloseHeader } from '@/components/ui/surface-close'
import { translateNow } from '@/i18n'
import { ESCAPE_PRIORITY, isTopEscapeLayer, pushEscapeLayer } from '@/lib/escape-layers'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'

interface OverlayViewProps {
  children: ReactNode
  onClose: () => void
  closeLabel?: string
  contentClassName?: string
  /** Search/badge chrome shares the reserved header row, never the body. */
  edgeBadge?: ReactNode
  headerContent?: ReactNode
  rootClassName?: string
  /** Controls rendered before the close button in the reserved header row. */
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
        'fixed inset-0 z-50 bg-black/22 backdrop-blur-[0.125rem]',
        // Equidistant inset on every side. The top value is driven by the
        // titlebar height so the card clears the OS traffic-lights vertically;
        // since the card top already sits below them, the left needs no extra
        // inset — keeping all sides equal so the card is ~full-width at any size.
        'p-[calc(var(--titlebar-height)+0.625rem)]',
        'sm:p-[calc(var(--titlebar-height)+0.875rem)]'
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
      style={{ '--titlebar-height': `${TITLEBAR_HEIGHT}px` } as CSSProperties}
    >
      <div className="relative h-full min-h-0">
        <div
          className={cn(
            'relative flex h-full min-h-0 flex-col overflow-hidden rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-chat-surface-background) shadow-md',
            rootClassName
          )}
          // Marks the card as a RAISED surface for window glass: while the field
          // behind it thins to show the desktop, this card stays near-opaque
          // (see the [data-glass-raised] rules in styles.css). Inert otherwise.
          data-glass-raised=""
        >
          <SurfaceCloseHeader
            className="[-webkit-app-region:drag]"
            close={<SurfaceCloseButton aria-label={closeLabel} onClick={closeOverlay} />}
          >
            {edgeBadge}
            {headerContent}
            {titlebarActions}
          </SurfaceCloseHeader>
          <div className={cn('min-h-0 min-w-0 flex flex-1 flex-col', contentClassName)}>{children}</div>
        </div>
      </div>
    </div>
  )
}
