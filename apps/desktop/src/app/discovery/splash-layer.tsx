import { useStore } from '@nanostores/react'
import { type ReactNode, useEffect } from 'react'

import { existingUserVerdict } from '@/lib/existing-user'
import { $cronJobs } from '@/store/cron'
import { $onboardingPhase, $setupGuidePhase } from '@/store/onboarding-phase'
import { $profiles } from '@/store/profile'
import { $renameSplash, considerRenameSplash, dismissRenameSplash } from '@/store/rename-splash'
import { $sessions, $sessionsLoadError, $sessionsLoading } from '@/store/session'

import { KeptChatRoutes } from '../contrib/route-cover'

import { RenameSplash } from './rename-splash'

/** The navigation a splash gives way to: the rail (docked, or in the compact menu sheet) and the
 *  chat sidebar (docked, or as a sheet). */
const NAV_REGIONS = '[data-slot="global-nav"], [data-zone-pane="sessions"], [data-narrow-overlay]'

/** Settles the splash from what the machine has told this window, as it arrives, and closes it on
 *  any click in the nav (the click still goes where it was going). Renders nothing, so the session
 *  list's churn re-renders only this. */
function SplashTriggers() {
  const sessions = useStore($sessions)
  const loading = useStore($sessionsLoading)
  const failed = useStore($sessionsLoadError)
  const profiles = useStore($profiles)
  const routines = useStore($cronJobs)
  const onboarding = useStore($onboardingPhase)
  const setupGuide = useStore($setupGuidePhase)
  const open = useStore($renameSplash) !== null

  const verdict = existingUserVerdict({
    sessionsSettled: !loading && !failed,
    setupRunning: onboarding === 'shown' || setupGuide === 'shown',
    chats: sessions.length,
    agents: profiles.length,
    routines: routines.length
  })

  useEffect(() => considerRenameSplash(verdict), [verdict])

  useEffect(() => {
    if (!open) {
      return
    }

    // Capture: a nav control may stop propagation, or unmount what was clicked.
    const onClick = (event: MouseEvent) => {
      if (event.target instanceof Element && event.target.closest(NAV_REGIONS)) {
        dismissRenameSplash('nav')
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
  const reason = useStore($renameSplash)

  return (
    <>
      <SplashTriggers />
      <KeptChatRoutes
        chat={chat}
        cover={
          reason && (
            <div className="discovery-splash-layer" data-slot="discovery-splash-layer">
              <RenameSplash reopened={reason === 'reopened'} />
            </div>
          )
        }
        pages={pages}
      />
    </>
  )
}
