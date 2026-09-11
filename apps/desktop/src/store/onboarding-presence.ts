/**
 * Onboarding presence — one tiny dependency-free flag the first-run surfaces
 * raise while they own the screen (intro cinematic and guided solo chat).
 *
 * Exists so leaf infrastructure (the update toast, future ambient
 * notifications) can ask "is onboarding on screen?" WITHOUT importing the
 * feature stores — updates.ts pulling intro-reveal/assembly
 * dragged their whole import chains into every test that mocks around it.
 * Surfaces push state in; consumers read a boolean out.
 */

import { atom } from 'nanostores'

export type OnboardingSurface = 'intro' | 'solo-chat'

const EMPTY: ReadonlySet<OnboardingSurface> = new Set()

export const $onboardingSurfaces = atom<ReadonlySet<OnboardingSurface>>(EMPTY)

export function setOnboardingSurfaceActive(surface: OnboardingSurface, active: boolean): void {
  const current = $onboardingSurfaces.get()

  if (current.has(surface) === active) {
    return
  }

  const next = new Set(current)

  if (active) {
    next.add(surface)
  } else {
    next.delete(surface)
  }

  $onboardingSurfaces.set(next.size === 0 ? EMPTY : next)
}

/** True while any first-run surface owns the screen. */
export function onboardingSurfaceActive(): boolean {
  return $onboardingSurfaces.get().size > 0
}

/** Hard reset for tests. */
export function resetOnboardingPresenceForTests(): void {
  $onboardingSurfaces.set(EMPTY)
}
