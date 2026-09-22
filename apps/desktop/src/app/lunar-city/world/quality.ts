import type { EntityKey, QualityTier } from '../model'

export type DynamicShadowMode = 'near' | 'none'
export type AnimationDistance = 'long' | 'normal' | 'short'

export interface QualitySettings {
  animationDistance: AnimationDistance
  decorations: boolean
  dynamicShadows: DynamicShadowMode
  lodAdvance: number
  renderScale: number
  tier: QualityTier
}

export interface HardwareScalingEngine {
  setHardwareScalingLevel?(level: number): void
}

const ANIMATION_DISTANCE_UNITS: Readonly<Record<AnimationDistance, number>> = Object.freeze({
  long: 48,
  normal: 28,
  short: 14
})

/** Bounded camera distance for genuine rig animation; farther workers instance statically. */
export function animationDistanceUnits(distance: AnimationDistance): number {
  return ANIMATION_DISTANCE_UNITS[distance]
}

const QUALITY_BY_TIER: Readonly<Record<QualityTier, QualitySettings>> = Object.freeze({
  balanced: {
    animationDistance: 'normal',
    decorations: true,
    dynamicShadows: 'near',
    lodAdvance: 0,
    renderScale: 0.85,
    tier: 'balanced'
  },
  detailed: {
    animationDistance: 'long',
    decorations: true,
    dynamicShadows: 'near',
    lodAdvance: 0,
    renderScale: 1,
    tier: 'detailed'
  },
  efficient: {
    animationDistance: 'short',
    decorations: false,
    dynamicShadows: 'none',
    // Keep the authored near silhouettes for the overview.  The efficient
    // tier already lowers pixel work and removes dynamic detail; jumping one
    // LOD at boot made every district collapse into the placeholder blocks
    // from the approved asset pack's far representation.  The governor can
    // still advance this value after sustained frame pressure.
    lodAdvance: 0,
    renderScale: 0.7,
    tier: 'efficient'
  }
})

export function qualitySettings(tier: QualityTier): QualitySettings {
  return { ...QUALITY_BY_TIER[tier] }
}

/** Render scale is output pixels divided by display pixels. */
export function applyQualitySettings(engine: HardwareScalingEngine, settings: QualitySettings): void {
  engine.setHardwareScalingLevel?.(1 / settings.renderScale)
}

function degradationSteps(tier: QualityTier): readonly ((settings: QualitySettings) => QualitySettings)[] {
  const detailedSteps = [
    (settings: QualitySettings) => ({ ...settings, renderScale: 0.85 }),
    (settings: QualitySettings) => ({ ...settings, renderScale: 0.7 }),
    (settings: QualitySettings) => ({ ...settings, dynamicShadows: 'none' as const }),
    (settings: QualitySettings) => ({ ...settings, animationDistance: 'short' as const }),
    (settings: QualitySettings) => ({ ...settings, decorations: false }),
    (settings: QualitySettings) => ({ ...settings, lodAdvance: settings.lodAdvance + 1 })
  ]

  if (tier === 'detailed') {
    return detailedSteps
  }

  if (tier === 'balanced') {
    return detailedSteps.slice(1)
  }

  return [(settings: QualitySettings) => ({ ...settings, lodAdvance: settings.lodAdvance + 1 })]
}

function isOverBudget(elapsedMs: number, interactive: boolean): boolean {
  return elapsedMs > (interactive ? 1_000 / 30 : 1_000 / 15)
}

export interface QualityControllerOptions {
  selectedEntityKey?: EntityKey
}

/**
 * A bounded, purely visual governor.  It cannot alter identity, selection,
 * commands, conversations, or the authoritative population.
 */
export function createQualityController(tier: QualityTier, options: QualityControllerOptions = {}) {
  let selectedTier = tier
  let degradation = 0
  let overBudgetFrames = 0
  let underBudgetFrames = 0

  const resolved = (): QualitySettings => {
    let value = qualitySettings(selectedTier)
    const steps = degradationSteps(selectedTier)

    for (let step = 0; step < degradation; step += 1) {
      value = steps[step]!(value)
    }

    return value
  }

  return {
    noteFrame({ elapsedMs, interactive }: { elapsedMs: number; interactive: boolean }): boolean {
      if (!Number.isFinite(elapsedMs) || elapsedMs < 0) {
        return false
      }

      if (isOverBudget(elapsedMs, interactive)) {
        overBudgetFrames += 1
        underBudgetFrames = 0

        if (overBudgetFrames >= 120 && degradation < degradationSteps(selectedTier).length) {
          degradation += 1
          overBudgetFrames = 0

          return true
        }

        return false
      }

      underBudgetFrames += 1
      overBudgetFrames = 0

      if (underBudgetFrames >= 600 && degradation > 0) {
        degradation -= 1
        underBudgetFrames = 0

        return true
      }

      return false
    },
    selectedEntityKey(): EntityKey | undefined {
      return options.selectedEntityKey
    },
    setTier(nextTier: QualityTier): void {
      selectedTier = nextTier
      degradation = 0
      overBudgetFrames = 0
      underBudgetFrames = 0
    },
    settings(): QualitySettings {
      return resolved()
    }
  }
}
