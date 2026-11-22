import { useStore } from '@nanostores/react'
import { computed, type ReadableAtom } from 'nanostores'
import { type ComponentType, type ReactNode, useEffect } from 'react'

import { existingUserVerdict, type ExistingUserVerdict } from '@/lib/existing-user'
import { $cronJobs } from '@/store/cron'
import { $firstTimeSplash, closeFirstTimeSplash, considerFirstTimeSplash } from '@/store/first-time-splash'
import { $onboardingPhase, $setupGuidePhase } from '@/store/onboarding-phase'
import { $profiles } from '@/store/profile'
import { $renameSplash, considerRenameSplash, dismissRenameSplash } from '@/store/rename-splash'
import { $sessions, $sessionsLoadError, $sessionsLoading } from '@/store/session'
import { $signInFailed, dismissSignInRecovery } from '@/store/visitor-access'

import { KeptChatRoutes } from '../contrib/route-cover'

import { FirstTimeSplash } from './first-time-splash'
import { RenameSplash } from './rename-splash'
import { SignInRecovery } from './sign-in-recovery'

interface SplashDefinition {
  /** Why it is on screen (`reopened` from the tray, else `shown`), or null while closed. */
  $open: ReadableAtom<null | 'reopened' | 'shown'>
  consider: (verdict: ExistingUserVerdict) => void
  closeByNav: () => void
  Splash: ComponentType<{ reopened: boolean }>
}

/** The splashes that cover the chat area, first come first. A sign up that came back without signing
 *  the visitor in says so before anything else. The verdict decides which of the other two a person
 *  gets: the rename splash for someone who used Hermes before, the first-time splash for someone new. */
const SPLASHES: readonly SplashDefinition[] = [
  {
    $open: computed($signInFailed, failed => (failed ? 'shown' : null)),
    // Read from the page once (store/visitor-access.ts), not from the machine's verdict.
    consider: () => undefined,
    closeByNav: dismissSignInRecovery,
    Splash: SignInRecovery
  },
  {
    $open: $renameSplash,
    consider: considerRenameSplash,
    closeByNav: () => dismissRenameSplash('nav'),
    Splash: RenameSplash
  },
  {
    $open: $firstTimeSplash,
    consider: considerFirstTimeSplash,
    closeByNav: () => closeFirstTimeSplash('nav'),
    Splash: FirstTimeSplash
  }
]

/** Settle every splash on `verdict`, the rename splash first: a `fresh` verdict records the newcomer
 *  the first-time splash reads. Repeated calls are no-ops once each is decided. */
export function considerSplashes(verdict: ExistingUserVerdict): void {
  SPLASHES.forEach(splash => splash.consider(verdict))
}

/** The splash on screen and why it is there, or null. */
const $shownSplash = computed(
  SPLASHES.map(splash => splash.$open),
  (...reasons): null | { definition: SplashDefinition; reason: 'reopened' | 'shown' } => {
    const index = reasons.findIndex(reason => reason !== null)
    const definition = SPLASHES[index]
    const reason = reasons[index]

    return definition && reason ? { definition, reason } : null
  }
)

/** The navigation a splash gives way to: the rail (docked, or in the compact menu sheet) and the
 *  chat sidebar (docked, or as a sheet). */
const NAV_REGIONS = '[data-slot="global-nav"], [data-zone-pane="sessions"], [data-narrow-overlay]'

/** Settles the splashes from what the machine has told this window, as it arrives, and closes the
 *  open one on any click in the nav (the click still goes where it was going). Renders nothing, so
 *  the session list's churn re-renders only this. */
function SplashTriggers() {
  const sessions = useStore($sessions)
  const loading = useStore($sessionsLoading)
  const failed = useStore($sessionsLoadError)
  const profiles = useStore($profiles)
  const routines = useStore($cronJobs)
  const onboarding = useStore($onboardingPhase)
  const setupGuide = useStore($setupGuidePhase)
  const open = useStore($shownSplash) !== null

  const verdict = existingUserVerdict({
    sessionsSettled: !loading && !failed,
    setupRunning: onboarding === 'shown' || setupGuide === 'shown',
    chats: sessions.length,
    agents: profiles.length,
    routines: routines.length
  })

  useEffect(() => considerSplashes(verdict), [verdict])

  useEffect(() => {
    if (!open) {
      return
    }

    // Capture: a nav control may stop propagation, or unmount what was clicked.
    const onClick = (event: MouseEvent) => {
      if (event.target instanceof Element && event.target.closest(NAV_REGIONS)) {
        $shownSplash.get()?.definition.closeByNav()
      }
    }

    document.addEventListener('click', onClick, true)

    return () => document.removeEventListener('click', onClick, true)
  }, [open])

  return null
}

interface SplashCoveredRoutesProps {
  chat: ReactNode
  pages: ReactNode
}

/**
 * The redesigned workspace pane's routes with the splash over them: the splash fills the chat area
 * while the nav around it stays live, and whatever it covers (the kept chat, a page) is inert under
 * it and gets its focus back when it closes (route-cover.tsx).
 */
export function SplashCoveredRoutes({ chat, pages }: SplashCoveredRoutesProps) {
  const shown = useStore($shownSplash)

  return (
    <>
      <SplashTriggers />
      <KeptChatRoutes
        chat={chat}
        cover={
          shown && (
            <div className="discovery-splash-layer" data-slot="discovery-splash-layer">
              <shown.definition.Splash reopened={shown.reason === 'reopened'} />
            </div>
          )
        }
        pages={pages}
      />
    </>
  )
}
