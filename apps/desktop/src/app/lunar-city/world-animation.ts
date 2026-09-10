import type { NpcActivity, WorldPresentation } from './world-presentation'

export type WorldAnimationClip =
  | 'idle'
  | 'walk'
  | 'work'
  | 'carry'
  | 'inspect'
  | 'repair'
  | 'talk'
  | 'wait'
  | 'panic'
  | 'celebrate'
  | 'rest'
  | 'return'

export interface ResolvedWorldAnimation {
  clip: WorldAnimationClip
  loop: boolean
  intensity: 0 | 1 | 2 | 3
  tags: string[]
}

const CLIP_BY_STATE: Record<NpcActivity['state'], WorldAnimationClip> = {
  idle: 'idle',
  walking: 'walk',
  working: 'work',
  carrying: 'carry',
  inspecting: 'inspect',
  repairing: 'repair',
  talking: 'talk',
  waiting: 'wait',
  panicking: 'panic',
  celebrating: 'celebrate',
  resting: 'rest',
  returning: 'return'
}

const NON_LOOPING = new Set<WorldAnimationClip>(['panic', 'celebrate', 'inspect', 'repair', 'talk'])

export function resolveWorldAnimation(
  activity: Pick<NpcActivity, 'state' | 'animationTags'>,
  presentation: Pick<WorldPresentation, 'cosmetic'>
): ResolvedWorldAnimation {
  const clip = CLIP_BY_STATE[activity.state] ?? 'idle'

  return {
    clip,
    intensity: presentation.cosmetic.intensity,
    loop: !NON_LOOPING.has(clip),
    tags: [...activity.animationTags]
  }
}
