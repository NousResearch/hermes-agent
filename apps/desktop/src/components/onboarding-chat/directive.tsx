import { useEffect } from 'react'

import { FirstBuildCard, HandoffCard, ProgressCard } from '@/components/onboarding-chat/cards/build'
import type { CardProps } from '@/components/onboarding-chat/cards/frame'
import { ConnectorsCard, LayoutCard, LookCard } from '@/components/onboarding-chat/cards/setup'
import { $onboardingAnswers, setOnboardingAnswers } from '@/store/onboarding-answers'

/** Steps that only carry data — the model handing the renderer what the user
 *  said. Each maps to the answer field it writes ('working' is the guided
 *  flow's name for the context answer: same storage, same consumers). */
const DATA_STEPS = {
  name: 'name',
  working: 'context'
} as const

/** Unrecognized steps are silent, including the greeting acknowledgement. */
const STEP_CARDS = {
  connectors: ConnectorsCard,
  first: FirstBuildCard,
  handoff: HandoffCard,
  layout: LayoutCard,
  look: LookCard,
  progress: ProgressCard
} satisfies Record<string, (props: CardProps) => React.ReactNode>

type DataStep = keyof typeof DATA_STEPS

/** Writing an answer is an EFFECT, not a render fact. Doing it inline in the
 *  directive's render triggered React's cross-component setState warning and
 *  re-entrant renders (live desktop.log). */
function DataDirective({ step, value }: { step: DataStep; value: string }) {
  const field = DATA_STEPS[step]

  useEffect(() => {
    if (!value || $onboardingAnswers.get()[field] === value) {
      return
    }

    setOnboardingAnswers({ [field]: value })
  }, [field, value])

  return null
}

export function OnboardingChatDirective({ attrs, streaming }: { attrs: Record<string, string>; streaming: boolean }) {
  const step = attrs.step ?? ''

  if (step in DATA_STEPS) {
    return <DataDirective step={step as DataStep} value={(attrs.value ?? '').trim()} />
  }

  const Card = STEP_CARDS[step as keyof typeof STEP_CARDS]

  // Mount as soon as the directive is parsed — returning null until settle
  // grows the transcript by a card when the turn finishes. Keep it inert
  // mid-stream so the growing paragraph can't be clicked through.
  return Card ? <Card attrs={attrs} locked={streaming} /> : null
}
