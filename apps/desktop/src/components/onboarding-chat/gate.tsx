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

// `starting` is never left waiting on a hung kickoff: past this it counts as a failed start. The kickoff
// notices on its next step and undoes what it did.
const INTRO_START_DEADLINE_MS = 45_000

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

      let settled = false

      const settle = (result: GuideKickoffResult) => {
        if (settled) {
          return
        }

        settled = true
        window.clearTimeout(deadline)

        if (result !== 'started') {
          recover(result)
        }
      }

      const deadline = window.setTimeout(() => settle('failed'), INTRO_START_DEADLINE_MS)

      void runGuideKickoff(onKickoff).then(settle, () => settle('failed'))

      return () => window.clearTimeout(deadline)
    }
  }, [enabled, gate.guideQueued, onKickoff])

  return null
}
