import { useStore } from '@nanostores/react'

import { Tip } from '@/components/ui/tooltip'
import { $introView } from '@/store/onboarding-intro'

import { skipIntro } from './intro'

export function OnboardingSkip() {
  const intro = useStore($introView) === 'intro'

  if (!intro) {
    return null
  }

  return (
    <Tip label="Switching you over to your default profile">
      <button
        className="ml-auto text-[11px] text-(--ui-text-quaternary) transition-colors hover:text-(--ui-text-secondary)"
        onClick={skipIntro}
        type="button"
      >
        Skip setup
      </button>
    </Tip>
  )
}
