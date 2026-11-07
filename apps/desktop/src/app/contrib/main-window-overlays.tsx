import type { ComponentProps } from 'react'

import { DesktopInstallOverlay } from '@/components/desktop-install-overlay'
import { FreeTierSignInDialog } from '@/components/free-tier/sign-in-dialog'
import { DesktopOnboardingOverlay } from '@/components/onboarding'
import { OnboardingEnd } from '@/components/onboarding/onboarding-end'
import { SharedMetricsConsentDialog } from '@/components/shared-metrics/consent-dialog'
import { isAuxiliaryWindow } from '@/store/windows'

import { SimplifiedAgentOffer } from '../chat/sidebar/simplified-agent-offer'

type MainWindowOverlaysProps = Pick<
  ComponentProps<typeof DesktopOnboardingOverlay>,
  'enabled' | 'profile' | 'requestGateway' | 'onCompleted'
> &
  Pick<ComponentProps<typeof FreeTierSignInDialog>, 'onSelectModel'>

/** App-level prompts stay available even when the user hides all shell chrome. */
export function MainWindowOverlays({
  enabled,
  profile,
  requestGateway,
  onSelectModel,
  onCompleted
}: MainWindowOverlaysProps) {
  if (isAuxiliaryWindow()) {
    return null
  }

  return (
    <>
      <DesktopInstallOverlay />
      <DesktopOnboardingOverlay
        enabled={enabled}
        onCompleted={onCompleted}
        profile={profile}
        requestGateway={requestGateway}
      />
      <OnboardingEnd />
      <SimplifiedAgentOffer />
      <SharedMetricsConsentDialog enabled={enabled} profile={profile} requestGateway={requestGateway} />
      <FreeTierSignInDialog onSelectModel={onSelectModel} />
    </>
  )
}
