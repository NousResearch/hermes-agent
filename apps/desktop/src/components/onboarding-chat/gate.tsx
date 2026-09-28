import { useStore } from '@nanostores/react'
import { useEffect, useLayoutEffect } from 'react'

import { endChatOnboardingSolo, takeGuideShape } from '@/components/onboarding-chat/assembly'
import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { ackFreeTierNotice, type FreeTierRequester } from '@/store/free-tier'
import { $desktopOnboarding, clearFreeTierIntro } from '@/store/onboarding'
import {
  $guideOpening,
  $onboardingGate,
  beginOnboardingFlow,
  runGuideKickoff,
  skipGuide
} from '@/store/onboarding-gate'

import { GuideLoading } from './guide-loading'

interface OnboardingChatGateProps {
  enabled: boolean
  onKickoff: () => Promise<boolean>
  requestGateway: FreeTierRequester
}

export function OnboardingChatGate({ enabled, onKickoff, requestGateway }: OnboardingChatGateProps) {
  const gate = useStore($onboardingGate)
  const opening = useStore($guideOpening)

  // A guide is owed the moment the renderer knows it (a first launch, or a
  // relaunch mid-guide). Take the solo shape before first paint and before the
  // gateway opens. Otherwise the normal shell paints at full size for the
  // seconds the backend takes to come up, and then snaps down to the guide.
  // Once, on mount: first-launch eligibility is a boot fact, not a live signal.
  useLayoutEffect(() => {
    beginOnboardingFlow($desktopOnboarding.get().firstRunSkipped)

    if ($onboardingGate.get().guideQueued) {
      takeGuideShape()
    }
  }, [])

  useEffect(() => {
    if (!enabled || !isOnboardingEnabled()) {
      return
    }

    // The guide is the free tier's introduction. Ack the one-time notice as
    // soon as it takes the screen, or a readiness round mid-guide raises the
    // ready screen over the conversation.
    const ack = () => {
      clearFreeTierIntro()
      void ackFreeTierNotice(requestGateway).then(acked => {
        if (acked) {
          clearFreeTierIntro()
        }
      })
    }

    return $onboardingGate.subscribe(state => {
      if (state.phase === 'guided') {
        ack()
      }
    })
  }, [enabled, requestGateway])

  useEffect(() => {
    if (enabled && gate.guideQueued) {
      const recover = () => {
        endChatOnboardingSolo()
        skipGuide()
      }

      void runGuideKickoff(onKickoff).then(started => {
        if (!started) {
          recover()
        }
      }, recover)
    }
  }, [enabled, gate.guideQueued, onKickoff])

  return opening ? <GuideLoading /> : null
}
