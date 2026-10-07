import { atom } from 'nanostores'

export type OnboardingSurface = 'solo-chat'

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

export function onboardingSurfaceActive(): boolean {
  return $onboardingSurfaces.get().size > 0
}
