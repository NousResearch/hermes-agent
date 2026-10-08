import type { ComponentProps } from 'react'

import { OnboardingSkip } from '@/components/onboarding-chat/skip'

import { ActionBadges } from './micro-actions'
import { PreparedImageRecovery } from './prepared-image-recovery'
import { SuggestionPills } from './suggestion-pills'

interface Props extends ComponentProps<typeof PreparedImageRecovery> {
  className: string
  statusSessionId: string | null
}

export function ComposerDraftNotices({ className, statusSessionId, ...recovery }: Props) {
  return (
    <div className={className}>
      <ActionBadges sessionId={statusSessionId} />
      <SuggestionPills sessionId={statusSessionId} />
      <PreparedImageRecovery {...recovery} />
      <OnboardingSkip />
    </div>
  )
}
