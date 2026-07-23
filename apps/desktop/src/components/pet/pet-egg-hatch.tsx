/**
 * Egg-hatch visuals for the pet generation flow (Cmd-K → Pets → Generate).
 *
 * `PetEggHatch` is the incubation beat shown while `pet.hatch` runs: a wobbling
 * egg that reads as "something is about to hatch" instead of a bare spinner. The
 * reveal celebration is the canvas `PetStarShower`. Motion is disabled under
 * `prefers-reduced-motion`.
 */

import { PixelEggSprite } from '@/components/pet/pixel-egg-sprite'
import { Button } from '@/components/ui/button'

interface PetEggHatchProps {
  subtitle?: string
  onCancel?: () => void
  cancelLabel?: string
}

/**
 * Thin progress bar. Determinate when given done/total (hatch rows stream one by
 * one, so a real percentage is meaningful); indeterminate otherwise (drafts
 * return together, so a count would just snap 0→100).
 */
export function PetProgress({ done, total }: { done?: number; total?: number }) {
  const determinate = typeof done === 'number' && typeof total === 'number' && total > 0
  const pct = determinate ? Math.min(100, Math.round((done / total) * 100)) : 0

  return (
    <div
      aria-valuemax={100}
      aria-valuemin={0}
      aria-valuenow={determinate ? pct : undefined}
      className="pet-progress"
      role="progressbar"
    >
      {determinate ? (
        <div className="pet-progress__fill" style={{ width: `${pct}%` }} />
      ) : (
        <div className="pet-progress__indeterminate" />
      )}
    </div>
  )
}

export function PetEggHatch({ subtitle, onCancel, cancelLabel }: PetEggHatchProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3">
      <div className="flex flex-col items-center">
        <PixelEggSprite mode="bounce" size={88} />
        {/* The egg sprite has transparent canvas below the art, so pull the
            shadow up ~a fifth of its size to sit at the egg's base. */}
        <span className="pet-egg-shadow" style={{ marginTop: '-0.55rem' }} />
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
