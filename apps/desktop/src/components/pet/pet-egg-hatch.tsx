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
