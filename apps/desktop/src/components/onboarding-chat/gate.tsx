import type { OnboardingStateResult } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { endChatOnboardingSolo, takeGuideShape } from '@/components/onboarding-chat/assembly'
import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { ackFreeTierNotice, type FreeTierRequester } from '@/store/free-tier'
import { $desktopOnboarding, clearFreeTierIntro } from '@/store/onboarding'
import {
  $guideOpening,
  $onboardingGate,
  abandonGuide,
  beginOnboardingFlow,
  type GuideKickoffResult,
  markOnboardingStateRead,
  runGuideKickoff
} from '@/store/onboarding-gate'

import { GuideLoading } from './guide-loading'

interface OnboardingChatGateProps {
  enabled: boolean
  onKickoff: () => Promise<GuideKickoffResult>
  requestGateway: FreeTierRequester
}

export function OnboardingChatGate({ enabled, onKickoff, requestGateway }: OnboardingChatGateProps) {
  const gate = useStore($onboardingGate)
  const opening = useStore($guideOpening)

  useEffect(() => {
    if (!enabled || !isOnboardingEnabled()) {
      return
    }

    void requestGateway<OnboardingStateResult>('onboarding.state')
      .then(
        state => {
          beginOnboardingFlow(state, $desktopOnboarding.get().firstRunSkipped)

          if ($onboardingGate.get().guideQueued) {
            takeGuideShape()
          }
        },
        error => console.warn('[onboarding] state could not be read', error)
      )
      .finally(markOnboardingStateRead)
  }, [enabled, requestGateway])

  useEffect(() => {
    if (!enabled || !isOnboardingEnabled()) {
      return
    }

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
      const recover = (result: Exclude<GuideKickoffResult, 'started'>) => {
        endChatOnboardingSolo()
        abandonGuide(result)
      }

      void runGuideKickoff(onKickoff).then(
        result => {
          if (result !== 'started') {
            recover(result)
          }
        },
        () => recover('failed')
      )
    }
  }, [enabled, gate.guideQueued, onKickoff])

  return opening ? <GuideLoading /> : null
}
