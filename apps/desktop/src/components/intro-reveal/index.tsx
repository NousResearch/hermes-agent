import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import {
  $introReveal,
  finishIntroReveal,
  installIntroRevealBridgeListeners,
  isIntroRevealEnabled,
  leaveIntroReveal,
  shouldPlayFirstRunIntro,
  startIntroReveal
} from '@/store/intro-reveal'
import { $desktopOnboarding } from '@/store/onboarding'
import { beginOnboardingFlow, queueGuideAfterIntro } from '@/store/onboarding-gate'

import { IntroRevealSurface } from './intro-reveal-surface'
import { INTRO_DEADMAN_MS, INTRO_EXIT_MS, INTRO_WALL_MS } from './timeline'

interface IntroRevealGateProps {
  enabled: boolean
}

export function IntroRevealGate({ enabled }: IntroRevealGateProps) {
  const onboarding = useStore($desktopOnboarding)
  const intro = useStore($introReveal)
  const nativeSurface = Boolean(window.hermesDesktop?.introReveal)
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
    // React renders the playing phase.
    return $introReveal.listen((state, previous) => {
      if (state.phase === 'hidden' && previous?.phase !== 'hidden') {
        queueGuideAfterIntro()
      }
    })
  }, [enabled])

  useEffect(() => {
    if (enabled && intro.phase === 'hidden' && shouldPlayFirstRunIntro(onboarding.firstRunSkipped)) {
      beginOnboardingFlow()
      startIntroReveal()
    }
  }, [enabled, intro.phase, onboarding.firstRunSkipped])

  // The native surface owns rAF: the hidden main renderer's clock is throttled.
  useEffect(() => {
    if (intro.phase === 'hidden') {
      return
    }

    const total = reduceMotion ? 2600 : nativeSurface ? INTRO_DEADMAN_MS : INTRO_WALL_MS

    const id = window.setTimeout(
      intro.phase === 'leaving' ? finishIntroReveal : leaveIntroReveal,
      intro.phase === 'leaving' ? INTRO_EXIT_MS : total
    )

    return () => window.clearTimeout(id)
  }, [intro.phase, nativeSurface, reduceMotion])

  if (!enabled || !isIntroRevealEnabled() || intro.phase === 'hidden' || nativeSurface) {
    return null
  }

  return (
    <div className="fixed inset-0 z-(--z-onboarding)">
      <IntroRevealSurface onSkip={leaveIntroReveal} />
    </div>
  )
}
