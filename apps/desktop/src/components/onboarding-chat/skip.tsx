import { useStore } from '@nanostores/react'

import { $introView } from '@/store/onboarding-intro'

import { skipIntro } from './intro'

export function OnboardingSkip() {
  const intro = useStore($introView) === 'intro'

  if (!intro) {
    return null
  }

  return (
    <button
      className="ml-auto text-[11px] text-(--ui-text-quaternary) transition-colors hover:text-(--ui-text-secondary)"
      onClick={skipIntro}
      title="Switching you over to your default profile"
      type="button"
    >
      Skip setup
    </button>
  )
}
