import { useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { FirstBuildCard, HandoffCard, ProgressCard } from '@/components/onboarding-chat/cards/build'
import type { CardProps } from '@/components/onboarding-chat/cards/frame'
import { ConnectorsCard, LayoutCard, LookCard } from '@/components/onboarding-chat/cards/setup'
import { type DesktopOnboardingStep, recordOnboarding } from '@/store/desktop-metrics'
import { $onboardingAnswers, setOnboardingAnswers } from '@/store/onboarding-answers'
import { $onboardingGate } from '@/store/onboarding-gate'

type AnswerField = 'name' | 'context'

const DATA_STEPS = new Map<string, AnswerField>([
  ['name', 'name'],
  ['working', 'context']
])

/** Guided-setup cards counted as first-run funnel steps (hermes.desktop.onboarding). */
const FUNNEL_STEPS = new Map<string, DesktopOnboardingStep>([
  ['connectors', 'guide_connectors'],
  ['first', 'guide_first_build'],
  ['layout', 'guide_layout'],
  ['look', 'guide_look']
])

const STEP_CARDS = new Map<string, (props: CardProps) => React.ReactNode>([
  ['connectors', ConnectorsCard],
  ['first', FirstBuildCard],
  ['handoff', HandoffCard],
  ['layout', LayoutCard],
  ['look', LookCard],
  ['progress', ProgressCard]
])

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
  const view = useSessionView()
  const storedId = useStore(view.$storedId)
  const runtimeId = useStore(view.$runtimeId)
  const messageId = useAuiState(state => state.message.id)
  const identity = JSON.stringify([storedId ?? runtimeId, messageId])
  const step = attrs.step ?? ''

  useEffect(() => {
    const funnelStep = FUNNEL_STEPS.get(step)

    // Only while the guide is live — a finished setup transcript re-renders its cards on every visit.
    if (funnelStep && $onboardingGate.get().phase === 'guided') {
      recordOnboarding(funnelStep, 'reached')
    }
  }, [step])

  const field = DATA_STEPS.get(step)

  if (field) {
    return <DataDirective field={field} value={(attrs.value ?? '').trim()} />
  }

  const Card = STEP_CARDS.get(step)

  return Card ? <Card attrs={attrs} locked={streaming} messageId={identity} /> : null
}
