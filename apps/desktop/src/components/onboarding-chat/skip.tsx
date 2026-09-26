/**
 * Skips the guided setup. Rendered in the composer's floating strip, the same row as the action badges and the
 * suggestion pills, so it aligns with the composer's edges. Skipping assembles the basic layout, sets the onboarding
 * phase to skipped, and leaves the user in the full app; the guided chat stays in the transcript. Shown from guide
 * kickoff until the layout pick assembles the app ($chatOnboardingSolo).
 */

import { useStore } from '@nanostores/react'

import { $chatOnboardingSolo, skipChatOnboarding } from '@/components/onboarding-chat/assembly'

export function OnboardingSkip() {
  const solo = useStore($chatOnboardingSolo)

  if (!solo) {
    return null
  }

  return (
    <button
      className="ml-auto text-[11px] text-(--ui-text-quaternary) transition-colors hover:text-(--ui-text-secondary)"
      onClick={skipChatOnboarding}
      type="button"
    >
      Skip setup
    </button>
  )
}
