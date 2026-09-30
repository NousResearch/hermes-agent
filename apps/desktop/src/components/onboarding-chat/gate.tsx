import type { OnboardingStateResult } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { ackFreeTierNotice, type FreeTierRequester } from '@/store/free-tier'
import { clearFreeTierIntro } from '@/store/onboarding'
import {
  $onboardingGate,
  $setupProfileName,
  abandonGuide,
  beginOnboardingFlow,
  type GuideKickoffResult,
  markOnboardingStateRead,
  runGuideKickoff
} from '@/store/onboarding-gate'
import { $introView } from '@/store/onboarding-intro'

import { failIntro, startIntro } from './intro'

interface OnboardingChatGateProps {
  enabled: boolean
  onKickoff: () => Promise<GuideKickoffResult>
  requestGateway: FreeTierRequester
}

/** Mounted in the main window only: secondary windows never run the intro. */
export function OnboardingChatGate({ enabled, onKickoff, requestGateway }: OnboardingChatGateProps) {
  const gate = useStore($onboardingGate)

  useEffect(() => {
    if (!enabled || !isOnboardingEnabled()) {
      return
    }

    void requestGateway<OnboardingStateResult>('onboarding.state')
      .then(
        state => {
          $setupProfileName.set(state.profile ?? null)
          beginOnboardingFlow(state)

          if ($onboardingGate.get().guideQueued) {
            startIntro()
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

    return $introView.subscribe(view => {
      if (view === 'intro') {
        ack()
      }
    })
  }, [enabled, requestGateway])

  useEffect(() => {
    if (enabled && gate.guideQueued) {
      const recover = (result: Exclude<GuideKickoffResult, 'started'>) => {
        failIntro()
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

  return null
}
