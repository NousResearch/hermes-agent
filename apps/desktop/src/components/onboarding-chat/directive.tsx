/**
 * Dispatcher for the `::onboarding{step="…"}` transcript directive, which turns a setup step into an interactive
 * picker in the transcript. Two tables decide what a step does: one writes an answer to the store, the other renders
 * a card. A step in neither table renders nothing. The cards live in ./cards.
 */

import { useEffect } from 'react'

import { FirstBuildCard, HandoffCard, ProgressCard } from '@/components/onboarding-chat/cards/build'
import type { CardProps } from '@/components/onboarding-chat/cards/frame'
import { ConnectorsCard, LayoutCard, LookCard } from '@/components/onboarding-chat/cards/setup'
import { $onboardingAnswers, setOnboardingAnswers } from '@/store/onboarding-answers'

/** Steps that only carry data, mapped to the answer field each one writes. The runbook names the context step
 *  'working' (store/onboarding-script.ts), so the step name and the field name differ. */
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

/** Writes the answer from an effect. Writing it during the directive's render triggered React's cross-component
 *  setState warning and re-entrant renders. */
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

  // Mount as soon as the directive is parsed. Returning null until the turn settles would grow the transcript by a
  // card when the turn finishes. The card stays inert while streaming so the growing paragraph cannot be clicked.
  return Card ? <Card attrs={attrs} locked={streaming} /> : null
}
