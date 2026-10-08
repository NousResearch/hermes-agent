import { atom, computed } from 'nanostores'

import { setModeContext } from '@/store/interface-mode'
import { isMainWindow } from '@/store/windows'

type OnboardingSurface = 'questionnaire'

const EMPTY: ReadonlySet<OnboardingSurface> = new Set()

export const $onboardingSurfaces = atom<ReadonlySet<OnboardingSurface>>(EMPTY)

/**
 * The first-run questionnaire's due check has answered (or timed out). Until it has, every "not
 * during setup" surface waits: the questionnaire may still open. Only the main window runs it, so
 * every other window (secondary, HUD, peer, watch) starts decided.
 */
export const $questionnaireDecided = atom(!isMainWindow())

/** How long anything waits on the due check before treating the questionnaire as not due. */
export const QUESTIONNAIRE_DECIDE_DEADLINE_MS = 4_000

export function markQuestionnaireDecided(): void {
  $questionnaireDecided.set(true)
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
  [$questionnaireDecided, $onboardingSurfaces],
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

// Simple mode keeps the status bar up while the questionnaire is open: account and download progress live there.
$onboardingSurfaces.subscribe(surfaces => setModeContext({ onboardingOpen: surfaces.size > 0 }))
