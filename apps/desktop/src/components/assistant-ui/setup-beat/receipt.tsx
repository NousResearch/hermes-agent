'use client'

import type { SetupBeat } from '@hermes/shared'

import { FIRST_RUN_LAYOUTS } from '@/components/assistant-ui/theme-picker/layout-picker-card'
import { TypedAnswer } from '@/components/assistant-ui/typed-answer'
import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import { LAYOUTS_AREA } from '@/components/pane-shell/tree/presets'
import { useContributions } from '@/contrib/react/use-contributions'
import { useI18n } from '@/i18n'
import { connectorTitle } from '@/lib/connector-tools'
import { cn } from '@/lib/utils'
import type { SetupBeatAnswerFields, SetupBeatToolResult } from '@/store/setup-beat'
import { accentName } from '@/themes/accents'

export const SETUP_BEAT_SHELL_CLASS = cn(
  WIDGET_SHELL_CLASS,
  'my-1.5 text-[length:var(--conversation-text-font-size)] text-(--ui-text-primary)'
)

/** What a settled card shows: the beat's pick, or how the card closed without one. */
export type SetupBeatOutcome =
  | { answer: SetupBeatAnswerFields; beat: SetupBeat; kind: 'answer' }
  | { beat: SetupBeat; kind: 'said'; said: string }
  | { beat: SetupBeat | null; kind: 'refused' | 'skipped_all' | 'timed_out' }

/** The receipt a setup_beat result calls for; null for a result that showed no card (a value given in
 *  words, a skip, `done`, a refused call), which draws nothing. An error naming the beat is a card
 *  answer the backend refused. */
export function setupBeatOutcome(result: SetupBeatToolResult): null | SetupBeatOutcome {
  const beat = result.beat ?? null

  if (result.skipped_all) {
    return { beat, kind: 'skipped_all' }
  }

  if (!beat) {
    return null
  }

  if (result.error !== undefined) {
    return { beat, kind: 'refused' }
  }

  // Words typed in the composer show as the user's bubble, also when they answered the card (the name).
  if (result.said?.trim()) {
    return { beat, kind: 'said', said: result.said.trim() }
  }

  if (result.answer) {
    return { answer: result.answer, beat, kind: 'answer' }
  }

  // A message sent instead: it shows as its own bubble in the next turn, so the card reads unanswered.
  return result.timed_out || result.wrote_instead ? { beat, kind: 'timed_out' } : null
}

interface AnswerNames {
  layout: (id: string) => string
  none: string
  themeAccent: string
  tour: (answer: SetupBeatAnswerFields) => string
}

/** The tour card's label for an option id, in the app's language. */
export function useTourLabel(): (id: string) => string | undefined {
  const { t } = useI18n()

  return id => Object.entries(t.firstRun.tours).find(([key]) => key === id)?.[1]
}

function answerText(beat: SetupBeat, answer: SetupBeatAnswerFields, names: AnswerNames): string {
  switch (beat) {
    case 'theme':
      return answer.accent ? accentName(answer.accent) : names.themeAccent

    case 'layout':
      return names.layout(answer.layout ?? '')
    case 'connectors': {
      const connected = answer.connected ?? []

      return connected.length > 0 ? connected.map(connectorTitle).join(', ') : names.none
    }

    case 'plugins':
      return answer.label || (answer.installed?.length ? answer.installed.join(', ') : names.none)

    case 'name':
      return answer.name ?? ''

    case 'tour':
      return names.tour(answer)

    case 'build':
      return answer.label ?? answer.build ?? ''
  }
}

/** One line where the choice happened: the beat, then what was picked or how the card closed. Words typed
 *  in the composer replace the line with the user's bubble. */
export function SetupBeatReceipt({ outcome }: { outcome: SetupBeatOutcome }) {
  return outcome.kind === 'said' ? <TypedAnswer text={outcome.said} /> : <ReceiptLine outcome={outcome} />
}

function ReceiptLine({ outcome }: { outcome: Exclude<SetupBeatOutcome, { kind: 'said' }> }) {
  const { t } = useI18n()
  const copy = t.firstRun
  const presets = useContributions(LAYOUTS_AREA)
  const tour = useTourLabel()
  const themeCopy = t.assistant.themePicker

  const closed = { refused: copy.answerRefused, skipped_all: copy.skippedAll, timed_out: copy.noAnswer }

  // The first-run layouts read as the card named them (Basic, Elite), not by their preset titles.
  const layout = (id: string) => {
    const choice = FIRST_RUN_LAYOUTS.find(row => row.id === id)

    return choice ? themeCopy[choice.name] : (presets.find(preset => preset.id === id)?.title ?? id)
  }

  const text =
    outcome.kind === 'answer'
      ? answerText(outcome.beat, outcome.answer, {
          layout,
          none: copy.noneChosen,
          themeAccent: themeCopy.themeAccent,
          tour: answer => tour(answer.tour ?? '') ?? answer.label ?? ''
        })
      : closed[outcome.kind]

  return (
    <div className={cn(SETUP_BEAT_SHELL_CLASS, 'flex flex-wrap items-center gap-x-2')} data-setup-beat-settled="">
      {outcome.beat && outcome.kind !== 'skipped_all' && (
        <span className="text-(--ui-text-tertiary)">{copy.beats[outcome.beat]}</span>
      )}
      <span className={cn(outcome.kind !== 'answer' && 'italic text-(--ui-text-tertiary)')}>{text}</span>
    </div>
  )
}
