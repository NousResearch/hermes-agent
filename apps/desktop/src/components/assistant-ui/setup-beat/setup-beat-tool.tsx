'use client'

import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import type { SetupBeat } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { Puzzle } from 'lucide-react'
import { type ComponentType, type ReactNode, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { CLARIFY_ICON_CLASS } from '@/components/assistant-ui/clarify/core/shell'
import { ConnectorDirectoryCard } from '@/components/assistant-ui/connector-directory-card'
import { AccentPickerCard } from '@/components/assistant-ui/theme-picker/accent-picker-card'
import { LayoutPickerCard } from '@/components/assistant-ui/theme-picker/layout-picker-card'
import { LayoutDashboard, MessageQuestion, Palette, Plug } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { applyAppearance } from '@/store/appearance-apply'
import {
  $setupBeatRequests,
  answerSetupBeat,
  type SetupBeatAnswer,
  type SetupBeatRequest,
  setupBeatResult
} from '@/store/setup-beat'

import { NameCard, OptionsCard } from './beat-cards'
import { CARD_LABEL_CLASS } from './card-type'
import { PluginsCard } from './plugins-card'
import {
  SETUP_BEAT_SHELL_CLASS,
  type SetupBeatOutcome,
  setupBeatOutcome,
  SetupBeatReceipt,
  useTourLabel
} from './receipt'

/** The card body for one beat; `answer` sends the beat's picks and settles the card. */
function BeatBody({ answer, request }: { answer: (picks: SetupBeatAnswer) => boolean; request: SetupBeatRequest }) {
  const { card } = request
  const tourLabel = useTourLabel()

  const commit = (applied: boolean, picks: SetupBeatAnswer): boolean => applied && answer(picks)

  switch (request.beat) {
    case 'name':
      return (
        <NameCard
          onName={name => answer({ name })}
          suggested={Array.isArray(card.suggested) ? undefined : card.suggested}
        />
      )

    case 'theme':
      return (
        <AccentPickerCard
          onContinue={accent => commit(applyAppearance({ accent: accent ?? 'theme' }).success, { accent })}
        />
      )

    case 'layout':
      return <LayoutPickerCard onContinue={layout => commit(applyAppearance({ layout }).success, { layout })} />

    case 'plugins':
      return (
        <PluginsCard
          names={card.plugins ?? []}
          onContinue={(installed, titles) => answer({ installed, label: titles.join(', ') })}
        />
      )

    case 'tour':
      return (
        <OptionsCard
          onPick={option => answer({ label: option.label, tour: option.id })}
          options={(card.options ?? []).map(option => ({ ...option, label: tourLabel(option.id) ?? option.label }))}
        />
      )

    case 'build':
      return (
        <OptionsCard
          onPick={option => answer({ build: option.id, label: option.label })}
          options={card.options ?? []}
        />
      )

    case 'connectors':
      return null
  }
}

// The theme beat picks the accent, so it carries the accent's palette icon.
const BEAT_ICONS = {
  build: MessageQuestion,
  connectors: Plug,
  layout: LayoutDashboard,
  name: MessageQuestion,
  plugins: Puzzle,
  theme: Palette,
  tour: MessageQuestion
} satisfies Record<SetupBeat, ComponentType<{ className?: string }>>

function BeatShell({ beat, children, question }: { beat: SetupBeat; children: ReactNode; question: string }) {
  const Icon = BEAT_ICONS[beat]

  return (
    <div className={cn(SETUP_BEAT_SHELL_CLASS, 'grid gap-3')} data-setup-beat={beat}>
      <div className="flex items-start gap-2">
        <span className={cn(CARD_LABEL_CLASS, 'flex-1')}>{question}</span>
        <Icon aria-hidden className={CLARIFY_ICON_CLASS} />
      </div>
      {children}
    </div>
  )
}

/** The setup_beat tool row: the beat's card while its request waits, then a one-line receipt where the
 *  choice happened. A card closed by words in the composer or by the deadline collapses the same way. */
export function SetupBeatTool(props: ToolCallMessagePartProps) {
  const view = useSessionView()
  const sessionId = useStore(view.$runtimeId)
  const busy = useStore(view.$busy)
  const requests = useStore($setupBeatRequests)
  const [settled, setSettled] = useState<null | SetupBeatOutcome>(null)

  if (props.result !== undefined) {
    const outcome = setupBeatOutcome(setupBeatResult(props.result))

    return outcome ? <SetupBeatReceipt outcome={outcome} /> : null
  }

  // Answered: the card is gone from the store a beat before tool.complete brings the result.
  if (settled) {
    return <SetupBeatReceipt outcome={settled} />
  }

  const pending = sessionId ? requests[sessionId] : undefined
  const request = pending && (pending.toolCallId ?? props.toolCallId) === props.toolCallId ? pending : undefined

  if (!sessionId || !request) {
    // A row left without a result and no card for it once the turn is over (a crash, a restart).
    return busy ? null : <SetupBeatReceipt outcome={{ beat: null, kind: 'timed_out' }} />
  }

  const answer = (picks: SetupBeatAnswer): boolean => {
    if (!answerSetupBeat(sessionId, picks)) {
      return false
    }

    setSettled({ answer: picks, beat: request.beat, kind: 'answer' })

    return true
  }

  // The directory card has its own header and plug icon; a frame label above it would be a second header.
  if (request.beat === 'connectors') {
    return (
      <div data-setup-beat="connectors">
        <ConnectorDirectoryCard
          onContinue={({ connected, newly_connected }) => answer({ connected, newly_connected })}
          suggested={Array.isArray(request.card.suggested) ? request.card.suggested : []}
        />
      </div>
    )
  }

  return (
    <BeatShell beat={request.beat} question={request.question}>
      <BeatBody answer={answer} request={request} />
    </BeatShell>
  )
}
