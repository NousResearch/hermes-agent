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
