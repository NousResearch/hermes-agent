/**
 * The guided setup's escape hatch. Rides the composer's floating strip — the
 * same band the action badges and suggestion pills use — so it shares the
 * composer's edges instead of floating at an arbitrary offset. Skip assembles
 * the default layout, marks onboarding done, and drops the user in the full
 * app; the guided chat stays in the transcript. Visible from guide kickoff
 * until the layout pick assembles ($chatOnboardingSolo).
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
