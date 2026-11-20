import { useStore } from '@nanostores/react'
import { type ReactNode, useMemo } from 'react'

import { LandingGreeting, StarterIdeas } from '@/app/discovery/starter-ideas'
import { usePortalShell } from '@/store/shell-variant'
import { $signUpRequired } from '@/store/visitor-access'

import type { ComposerTarget } from './composer/focus'

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
