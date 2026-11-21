import { useStore } from '@nanostores/react'
import { type ReactNode, useMemo } from 'react'

import { LandingGreeting, StarterIdeas } from '@/app/discovery/starter-ideas'
import { $introHoldsThread } from '@/components/onboarding/first-run/intro'
import { useStoreSelector } from '@/lib/use-session-slice'
import { $introView } from '@/store/onboarding-intro'
import { $setupGuideOnboarding } from '@/store/shell-policy'
import { usePortalShell } from '@/store/shell-variant'
import { $signUpRequired } from '@/store/visitor-access'
import { isMainWindow } from '@/store/windows'

import type { ComposerTarget } from './composer/focus'

/** Classic's first-run intro on the main window's primary chat; the redesign's setup guide
 *  replaces it, so none of it holds the thread or marks the guide as arrived there. */
export function useClassicIntro(isPrimary: boolean) {
  const setupGuide = useStore($setupGuideOnboarding)
  const classicPrimary = !setupGuide && isPrimary && isMainWindow()
  const introHoldsThread = useStore($introHoldsThread) && classicPrimary
  const guideStarted = useStoreSelector($introView, view => view === 'intro' || view === 'ended') && !setupGuide

  return { classicPrimary, guideStarted, introHoldsThread }
}

export function useChatLanding(
  showIntro: boolean,
  onboarding: boolean,
  setup: boolean,
  personality?: string,
  seed?: number
) {
  const enabled = usePortalShell()
  const centered = enabled && showIntro && !onboarding && !setup

  return { centered, intro: showIntro && !centered ? { personality, seed } : undefined }
}

interface ComposerLanding {
  /** What the landing lays above and below the composer (a newcomer's greeting and starter ideas). */
  around?: { above: ReactNode; below: ReactNode }
  /** The composer waits for a connection, except a visitor's who must sign up: what they send is
   *  held, never sent. */
  disabled: boolean
}

/** The composer's part in the primary chat's centred landing. */
export function useComposerLanding(
  centered: boolean,
  primary: boolean,
  target: ComposerTarget,
  gatewayOpen: boolean
): ComposerLanding {
  const signUpRequired = useStore($signUpRequired)
  const landing = centered && primary

  const around = useMemo(
    () =>
      landing ? { above: <LandingGreeting target={target} />, below: <StarterIdeas target={target} /> } : undefined,
    [landing, target]
  )

  return { around, disabled: !gatewayOpen && !signUpRequired }
}
