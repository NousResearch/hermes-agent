import { type ReactNode, useEffect } from 'react'
import { createPortal } from 'react-dom'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { translateNow } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'

interface OverlayViewProps {
  children: ReactNode
  onClose: () => void
  closeLabel?: string
  contentClassName?: string
  headerContent?: ReactNode
  /**
   * Render into `document.body` instead of in-place. REQUIRED for any overlay
   * mounted inside a subtree that establishes a containing block for fixed
   * positioning — `transform`, `filter`, `backdrop-filter`, `contain`,
   * `will-change`, etc. (CSS spec). Without it, this overlay's `position:
   * fixed; inset: 0` resolves against that ancestor's box, not the viewport,
   * so the "full-screen" overlay collapses into the ancestor's rect (e.g. the
   * composer's glassy `backdrop-filter` surface at the bottom of the window).
   * Root-mounted overlays (settings, command center) don't need it; overlays
   * docked under the composer/thread (which use backdrop-blur + contain:paint)
   * do.
   */
  portal?: boolean
  rootClassName?: string
}

export function OverlayView({
  children,
  onClose,
  closeLabel = translateNow('common.close'),
  contentClassName,
  headerContent,
  portal = false,
  rootClassName
}: OverlayViewProps) {
  const closeOverlay = () => {
    triggerHaptic('close')
    onClose()
  }

  // Esc dismisses every OverlayView-based overlay. Nested Radix dialogs
  // stop propagation themselves, so opening (e.g.) the model picker inside
  // Settings still closes the picker first instead of the underlying overlay.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || event.defaultPrevented) {
        return
      }

      event.preventDefault()
      triggerHaptic('close')
      onClose()
    }

    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  const overlay = (
    <div
      className="fixed inset-0 z-50 bg-black/22 p-3 backdrop-blur-[0.125rem] sm:p-6"
      onClick={event => {
        if (event.target === event.currentTarget) {
          closeOverlay()
        }
      }}
      role="presentation"
    >
      <div
        className={cn(
          'relative flex h-full min-h-0 flex-col overflow-hidden rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-chat-surface-background) shadow-md',
          rootClassName
        )}
      >
        <div className="pointer-events-none absolute inset-x-0 top-0 z-10 h-[calc(var(--titlebar-height)+0.1875rem)] [-webkit-app-region:drag]">
          {headerContent && (
            <div className="pointer-events-auto absolute left-1/2 top-[calc(0.5rem+var(--titlebar-height)/2)] -translate-x-1/2 -translate-y-1/2 [-webkit-app-region:no-drag]">
              {headerContent}
            </div>
          )}

          <Button
            aria-label={closeLabel}
            className="pointer-events-auto absolute right-3 top-[calc(0.1875rem+var(--titlebar-height)/2)] -translate-y-1/2 text-(--ui-text-tertiary) hover:bg-(--chrome-action-hover) hover:text-foreground [-webkit-app-region:no-drag]"
            onClick={closeOverlay}
            size="icon-titlebar"
            variant="ghost"
          >
            <Codicon name="close" size="1rem" />
          </Button>
        </div>

        {/* No top padding here: the split-layout columns own their own
            titlebar clearance so their backgrounds run flush to the card top
            (otherwise the card surface shows as a gap above the sidebar). */}
        <div className={cn('min-h-0 flex flex-1 flex-col', contentClassName)}>{children}</div>
      </div>
    </div>
  )

  // Portal to body so `position: fixed` resolves against the viewport rather
  // than a transformed/filtered/contained ancestor (see the `portal` prop).
  if (portal && typeof document !== 'undefined') {
    return createPortal(overlay, document.body)
  }

  return overlay
}
