import { COMPOSER_HEART_CONFIG, HeartField } from '@/components/chat/vibe-hearts'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import { COMPOSER_FADE_BACKGROUND } from './composer-utils'

interface ComposerDockGlowProps {
  dockProximity: number
  dragging: boolean
  poppedOut: boolean
}

/**
 * Radial dock-target glow shown while a popped-out composer is dragged.
 * Intensity tracks how close the composer is to the dock (1 = peak).
 */
export function ComposerDockGlow({ dockProximity, dragging, poppedOut }: ComposerDockGlowProps) {
  if (!dragging || !poppedOut) {
    return null
  }

  return (
    <div
      aria-hidden
      // `absolute`, not `fixed`: anchor to the chat-column root (the same
      // `relative isolate` container the docked composer centers in) so the
      // glow spans the thread area only — never the full viewport / under the
      // sidebar. The dock target IS the docked position, so they must share
      // a containing block.
      className="pointer-events-none absolute inset-x-0 bottom-0 z-20 h-32"
      style={{
        // A bottom-centered radial glow — soft on every side by construction,
        // so it reads as the dock target without any hard band edges. Its
        // intensity tracks how close the composer is to the dock (1 = peak).
        background:
          'radial-gradient(64% 130% at 50% 100%, color-mix(in srgb, var(--color-primary) 26%, transparent) 0%, transparent 70%)',
        // Scaled by --dock-glow-scale (lower in light mode — see styles.css).
        opacity: `calc(${0.1 + dockProximity * 0.57} * var(--dock-glow-scale, 1))`
      }}
    />
  )
}

interface ComposerPopoutChromeProps {
  dragging: boolean
  onToggle: () => void
  popoutAllowed: boolean
  poppedOut: boolean
}

/** The fade and grab margin sit behind the surface, never over its controls. */
export function ComposerPopoutChrome({
  dragging,
  onToggle,
  popoutAllowed,
  poppedOut
}: ComposerPopoutChromeProps) {
  return (
    <>
      {!poppedOut && (
        <div
          className="pointer-events-none absolute inset-0 rounded-[inherit]"
          style={{ background: COMPOSER_FADE_BACKGROUND }}
        />
      )}
      {/* The surface's z-4 leaves only its transparent margin draggable.
          Hover hatching and opacity are owned by styles.css. */}
      {popoutAllowed && (
        <div
          aria-hidden
          className={cn('pointer-events-auto absolute inset-0', dragging ? 'cursor-grabbing' : 'cursor-grab')}
          data-dragging={dragging ? '' : undefined}
          data-slot="composer-drag-region"
          onDoubleClick={onToggle}
        />
      )}
    </>
  )
}

/** "Editing a queued prompt" banner with Cancel / Save actions. */
export function QueuedEditBanner({ onExit }: { onExit: (action: 'cancel' | 'save') => unknown }) {
  const { t } = useI18n()

  return (
    <div className="flex items-center justify-between gap-2 rounded-lg border border-[color-mix(in_srgb,var(--dt-composer-ring)_32%,transparent)] bg-accent/18 px-2 py-1">
      <div className="min-w-0 text-[0.7rem] text-muted-foreground/88">{t.composer.editingQueuedInComposer}</div>
      <div className="flex shrink-0 items-center gap-1">
        <Button
          className="h-6 rounded-md px-2 text-[0.68rem]"
          onClick={() => onExit('cancel')}
          type="button"
          variant="ghost"
        >
          {t.common.cancel}
        </Button>
        <Button className="h-6 rounded-md px-2 text-[0.68rem]" onClick={() => onExit('save')} type="button">
          {t.common.save}
        </Button>
      </div>
    </div>
  )
}

export function ComposerHeartField() {
  return (
    <HeartField
      className="absolute inset-x-0 z-30"
      config={COMPOSER_HEART_CONFIG}
      style={{ top: 0, bottom: 'calc(var(--composer-measured-height) + 0.25rem)' }}
    />
  )
}
