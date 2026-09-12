import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { takeGuideShape } from '@/components/onboarding-chat/assembly'
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

  // A guide is owed the moment the renderer knows it (cinematic with the film
  // seen, or a relaunch mid-guide). Take the solo shape now, before the
  // gateway opens. Otherwise the normal shell paints at full size for the
  // seconds the backend takes to come up, and then snaps down to the guide.
  useEffect(() => {
    if (gate.guideQueued && intro.phase === 'hidden') {
      takeGuideShape()
    }
    // Once, on mount: the queued flag is a boot fact, not a live signal.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!enabled || !isOnboardingEnabled()) {
      return
    }

    // The guide is the free tier's introduction, whichever way it opens: the
    // film, or the guided chat directly when the film is skipped. Ack the
    // one-time notice as soon as either takes the screen, or a readiness
    // round mid-guide raises the ready screen over the conversation.
    const ack = () => {
      clearFreeTierIntro()
      void ackFreeTierNotice(requestGateway).then(acked => {
        if (acked) {
          clearFreeTierIntro()
        }
      })
    }

    // subscribe also sees an intro started by the preceding sibling's effect.
    const offIntro = $introReveal.subscribe(state => {
      if (state.phase === 'playing') {
        ack()
      }
    })

    const offGate = $onboardingGate.subscribe(state => {
      if (state.phase === 'guided') {
        ack()
      }
    })

    return () => {
      offIntro()
      offGate()
    }
  }, [enabled, requestGateway])

  useEffect(() => {
    if (enabled && gate.guideQueued && intro.phase === 'hidden') {
      void runGuideKickoff(onKickoff)
    }
  }, [enabled, gate.guideQueued, intro.phase, onKickoff])

  return null
}
