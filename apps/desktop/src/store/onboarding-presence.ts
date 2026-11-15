import { atom, computed } from 'nanostores'

import { setModeContext } from '@/store/interface-mode'

type OnboardingSurface = 'intro' | 'onboarding'

const EMPTY: ReadonlySet<OnboardingSurface> = new Set()

export const $onboardingSurfaces = atom<ReadonlySet<OnboardingSurface>>(EMPTY)

/**
 * First-run onboarding's due check has answered (or the person went on without it, or it does not
 * apply in this window). Until it has, every "not during setup" surface waits: onboarding may still open.
 */
export const $onboardingDecided = atom(false)

/** How long after `gateway.ready` the boot overlay waits on the due check before it offers Continue.
 *  The backend waits up to 8 s on its boot record, then works the answer out live. */
export const ONBOARDING_DECIDE_OFFER_MS = 15_000

export function markOnboardingDecided(): void {
  $onboardingDecided.set(true)
}

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

/** Nothing first-run owns the screen: the due check answered and no onboarding surface is open. */
export const $onboardingSurfaceClear = computed(
  [$onboardingDecided, $onboardingSurfaces],
  (decided, surfaces) => decided && surfaces.size === 0
)

/** A first-run surface is open, or may still open. */
export function onboardingSurfaceActive(): boolean {
  return !$onboardingSurfaceClear.get()
}

/** A first-run surface is open now; an undecided due check does not count. */
export function onboardingSurfaceOpen(): boolean {
  return $onboardingSurfaces.get().size > 0
}

/** Run once nothing first-run owns the screen (now, when it already doesn't). */
export function afterOnboardingSurfaceClear(run: () => void): void {
  if ($onboardingSurfaceClear.get()) {
    run()

    return
  }

  const stop = $onboardingSurfaceClear.listen(clear => {
    if (clear) {
      stop()
      run()
    }
  })
}

// Simple mode keeps the status bar up while onboarding is open: account and download progress live there.
$onboardingSurfaces.subscribe(surfaces => setModeContext({ onboardingOpen: surfaces.size > 0 }))
