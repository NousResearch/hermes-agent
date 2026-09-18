import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { takeGuideShape } from '@/components/onboarding-chat/assembly'
import {
  $introReveal,
  finishIntroReveal,
  installIntroRevealBridgeListeners,
  isIntroRevealEnabled,
  isIntroRevealSkipped,
  leaveIntroReveal,
  shouldPlayFirstRunIntro,
  startIntroReveal
} from '@/store/intro-reveal'
import { $desktopOnboarding } from '@/store/onboarding'
import { beginOnboardingFlow, beginOnboardingFlowWithoutIntro, queueGuideAfterIntro } from '@/store/onboarding-gate'

import { INTRO_DEADMAN_MS, INTRO_EXIT_MS } from './timeline'

interface IntroRevealGateProps {
  enabled: boolean
}

export function IntroRevealGate({ enabled }: IntroRevealGateProps) {
  const onboarding = useStore($desktopOnboarding)
  const intro = useStore($introReveal)
  const reduceMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

  useEffect(() => {
    if (enabled && isIntroRevealEnabled()) {
      return installIntroRevealBridgeListeners()
    }
  }, [enabled])

  useEffect(() => {
    if (!enabled || !isIntroRevealEnabled()) {
      return
    }

    // Observe the store edge directly: a failed native open can finish before
    // React renders the playing phase. Take the guide's shape on the same
    // tick: finishIntroReveal shows the main window right after this fires.
    return $introReveal.listen((state, previous) => {
      if (state.phase === 'hidden' && previous?.phase !== 'hidden') {
        queueGuideAfterIntro()
        takeGuideShape()
      }
    })
  }, [enabled])

  useEffect(() => {
    if (!enabled) {
      return
    }

    // skipIntro turns the film off; the guided chat behind it must still run.
    // Take the guide's shape on this tick, exactly like the film's completion
    // edge, so no full-size shell paints while the guide session comes up.
    if (isIntroRevealSkipped()) {
      if (intro.phase === 'hidden') {
        beginOnboardingFlowWithoutIntro(onboarding.firstRunSkipped)
        takeGuideShape()
      }

      return
    }

    if (intro.phase === 'hidden' && shouldPlayFirstRunIntro(onboarding.firstRunSkipped)) {
      beginOnboardingFlow()
      startIntroReveal()
    }
  }, [enabled, intro.phase, onboarding.firstRunSkipped])

  // The native surface runs the frame loop: the hidden main renderer's animation frames are throttled.
  useEffect(() => {
    if (intro.phase === 'hidden') {
      return
    }

    const total = reduceMotion ? 2600 : INTRO_DEADMAN_MS

    const id = window.setTimeout(
      intro.phase === 'leaving' ? finishIntroReveal : leaveIntroReveal,
      intro.phase === 'leaving' ? INTRO_EXIT_MS : total
    )

    return () => window.clearTimeout(id)
  }, [intro.phase, reduceMotion])

  return null
}
