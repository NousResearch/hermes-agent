import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { ackFreeTierNotice, type FreeTierRequester } from '@/store/free-tier'
import { $introReveal } from '@/store/intro-reveal'
import { clearFreeTierIntro } from '@/store/onboarding'
import { $onboardingGate, runGuideKickoff } from '@/store/onboarding-gate'

interface OnboardingChatGateProps {
  enabled: boolean
  onKickoff: () => Promise<boolean>
  requestGateway: FreeTierRequester
}

export function OnboardingChatGate({ enabled, onKickoff, requestGateway }: OnboardingChatGateProps) {
  const gate = useStore($onboardingGate)
  const intro = useStore($introReveal)

  useEffect(() => {
    if (!enabled || !isOnboardingEnabled()) {
      return
    }

    // subscribe also sees an intro started by the preceding sibling's effect.
    return $introReveal.subscribe(state => {
      if (state.phase === 'playing') {
        clearFreeTierIntro()
        void ackFreeTierNotice(requestGateway).then(acked => {
          if (acked) {
            clearFreeTierIntro()
          }
        })
      }
    })
  }, [enabled, requestGateway])

  useEffect(() => {
    if (enabled && gate.guideQueued && intro.phase === 'hidden') {
      void runGuideKickoff(onKickoff)
    }
  }, [enabled, gate.guideQueued, intro.phase, onKickoff])

  return null
}
