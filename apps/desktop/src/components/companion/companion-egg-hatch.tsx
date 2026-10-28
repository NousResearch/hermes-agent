/**
 * Egg-hatch visuals for the companion generation flow (Cmd-K → Companions → Generate).
 *
 * `CompanionEggHatch` is the incubation beat shown while `companion.hatch` runs: a wobbling
 * egg that reads as "something is about to hatch" instead of a bare spinner. The
 * reveal celebration is the canvas `CompanionStarShower`. Motion is disabled under
 * `prefers-reduced-motion`.
 */

import { PixelEggSprite } from '@/components/companion/pixel-egg-sprite'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'

interface CompanionEggHatchProps {
  subtitle?: string
  onCancel?: () => void
  cancelLabel?: string
}

/**
 * Thin progress bar. Determinate when given done/total (hatch rows stream one by
 * one, so a real percentage is meaningful); indeterminate otherwise (drafts
 * return together, so a count would just snap 0→100).
 */
export function CompanionProgress({ done, total }: { done?: number; total?: number }) {
  const determinate = typeof done === 'number' && typeof total === 'number' && total > 0

  return (
    <Progress
      animated
      aria-label="Hatching progress"
      className="bg-[color-mix(in_srgb,var(--ui-accent)_15%,transparent)]"
      fillClassName="bg-(--ui-accent)"
      indeterminate={!determinate}
      size="sm"
      value={determinate ? done / total : 0}
    />
  )
}

export function CompanionEggHatch({ subtitle, onCancel, cancelLabel }: CompanionEggHatchProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3">
      <div className="flex flex-col items-center">
        <PixelEggSprite mode="bounce" size={88} />
        {/* The egg sprite has transparent canvas below the art, so pull the
            shadow up ~a fifth of its size to sit at the egg's base. */}
        <span className="companion-egg-shadow" style={{ marginTop: '-0.55rem' }} />
      </div>

      {subtitle && (
        <p className="shimmer shimmer-color-primary whitespace-nowrap text-center text-[length:var(--conversation-caption-font-size)] leading-snug text-(--ui-text-tertiary)">
          {subtitle}
        </p>
      )}

      {onCancel && (
        <Button onClick={onCancel} size="xs" variant="text">
          {cancelLabel ?? 'Cancel'}
        </Button>
      )}
    </div>
  )
}
