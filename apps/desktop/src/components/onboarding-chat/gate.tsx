import { useStore } from '@nanostores/react'
import { useEffect, useLayoutEffect } from 'react'

import { endChatOnboardingSolo, takeGuideShape } from '@/components/onboarding-chat/assembly'
import { $desktopOnboarding } from '@/store/onboarding'
import {
  $guideOpening,
  $onboardingGate,
  beginOnboardingFlow,
  runGuideKickoff,
  skipGuide
} from '@/store/onboarding-gate'

import { GuideLoading } from './guide-loading'

type GatewayRequester = <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>

interface OnboardingChatGateProps {
  enabled: boolean
  onKickoff: () => Promise<boolean>
  requestGateway: GatewayRequester
}

export function OnboardingChatGate({ enabled, onKickoff, requestGateway }: OnboardingChatGateProps) {
  const gate = useStore($onboardingGate)
  const opening = useStore($guideOpening)

  useLayoutEffect(() => {
    beginOnboardingFlow($desktopOnboarding.get().firstRunSkipped)

    if ($onboardingGate.get().guideQueued) {
      takeGuideShape()
    }
  }, [])

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
