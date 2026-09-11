/**
 * In-chat onboarding cards — the `::onboarding{step="…"}` transcript
 * directive. Hermes walks the user through setup in the transcript, and each
 * step's paragraph renders as an interactive picker with a shared option
 * catalog and persistence.
 *
 * This module is only the dispatcher. Two tables say what a step means — one
 * writes an answer, the other renders a card — and a step in neither renders
 * nothing, which is the right answer for the model's invisible acks. The cards
 * themselves live in ./cards.
 */

import { useEffect } from 'react'

import { FirstBuildCard, HandoffCard, ProgressCard } from '@/components/onboarding-chat/cards/build'
import type { CardProps } from '@/components/onboarding-chat/cards/frame'
import { ConnectorsCard, LayoutCard, LookCard } from '@/components/onboarding-chat/cards/setup'
import { $onboardingAnswers, setOnboardingAnswers } from '@/store/onboarding-answers'

/** Steps that only carry data — the model handing the renderer what the user
 *  said. Each maps to the answer field it writes ('working' is the guided
 *  flow's name for the context answer: same storage, same consumers). */
type AnswerField = 'name' | 'context'

const DATA_STEPS = new Map<string, AnswerField>([
  ['name', 'name'],
  ['working', 'context']
])

/** Unrecognized steps are silent, including the greeting acknowledgement. */
const STEP_CARDS = new Map<string, (props: CardProps) => React.ReactNode>([
  ['connectors', ConnectorsCard],
  ['first', FirstBuildCard],
  ['handoff', HandoffCard],
  ['layout', LayoutCard],
  ['look', LookCard],
  ['progress', ProgressCard]
])

/** Writing an answer is an EFFECT, not a render fact. Doing it inline in the
 *  directive's render triggered React's cross-component setState warning and
 *  re-entrant renders (live desktop.log). */
function DataDirective({ field, value }: { field: AnswerField; value: string }) {
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

  const field = DATA_STEPS.get(step)

  if (field) {
    return <DataDirective field={field} value={(attrs.value ?? '').trim()} />
  }

  const Card = STEP_CARDS.get(step)

  // Mount as soon as the directive is parsed — returning null until settle
  // grows the transcript by a card when the turn finishes. Keep it inert
  // mid-stream so the growing paragraph can't be clicked through.
  return Card ? <Card attrs={attrs} locked={streaming} /> : null
}
