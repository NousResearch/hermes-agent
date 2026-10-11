'use client'

import { useI18n } from '@/i18n'
import { type ClarifyQuestion, type ClarifyRequest, SETUP_CHOOSE_QID } from '@/store/clarify'

import type { useClarifyKeys } from './core/use-clarify-keys'
import { isSetupPickerKind, QuestionPills, SETUP_PICKERS, type SetupPickerKind } from './setup-pickers'
import type { SetupRow } from './setup-rows'

type SetupSource = Pick<ClarifyRequest, 'questions' | 'setup'>

/** What the card draws from: the live request once it can be answered, else the tool-call args preview. */
export function setupChooseSource(request: ClarifyRequest | null, fromArgs: null | SetupSource) {
  const ready = Boolean(request?.requestId && request.setup)
  const source = request ?? fromArgs
  const setup = source?.setup ?? null
  const kind = setup?.kind ?? 'question'

  return {
    kind,
    pickerKind: isSetupPickerKind(kind) ? kind : null,
    ready,
    requestId: ready ? (request?.requestId ?? null) : null,
    setup,
    source
  }
}

interface SetupChooseBodyProps {
  cursor: null | number
  draft: string
  keys: ReturnType<typeof useClarifyKeys>
  onDraft: (value: string) => void
  onStage: (id: string) => void
  picked: string[]
  pickerKind: null | SetupPickerKind
  question: ClarifyQuestion
  ready: boolean
  rows: null | SetupRow[]
}

function PillsLoading({ question }: { question: string }) {
  const setupCopy = useI18n().t.assistant.setupChoose

  return (
    <div className="grid gap-2">
      <span className="whitespace-pre-wrap font-medium leading-(--conversation-line-height)">{question}</span>
      <div className="flex flex-wrap gap-2 p-1" role="status">
        <span className="sr-only">{setupCopy.loading}</span>
        {Array.from({ length: 3 }, (_, index) => (
          <div className="h-7 w-28 animate-pulse rounded-full bg-muted/40" key={index} />
        ))}
      </div>
    </div>
  )
}

function PickerRows({
  cursor,
  keys,
  onStage,
  picked,
  pickerKind,
  rows
}: Pick<SetupChooseBodyProps, 'cursor' | 'keys' | 'onStage' | 'picked' | 'rows'> & { pickerKind: SetupPickerKind }) {
  const setupCopy = useI18n().t.assistant.setupChoose
  const Picker = SETUP_PICKERS[pickerKind]

  if (rows === null) {
    return (
      <div className="grid grid-cols-3 gap-2" role="status">
        <span className="sr-only">{setupCopy.loading}</span>
        {Array.from({ length: 6 }, (_, index) => (
          <div className="h-10 animate-pulse rounded-lg bg-muted/40" key={index} />
        ))}
      </div>
    )
  }

  if (rows.length === 0) {
    return <p className="text-(--ui-text-tertiary)">{setupCopy.unavailable}</p>
  }

  return <Picker cursor={cursor} onPick={index => keys.pick(0, index)} onStage={onStage} picked={picked} rows={rows} />
}

export function SetupChooseBody(props: SetupChooseBodyProps) {
  const { cursor, draft, keys, onDraft, picked, pickerKind, question, ready, rows } = props

  if (pickerKind === null && rows === null) {
    return <PillsLoading question={question.question} />
  }

  if (pickerKind === null) {
    return (
      <QuestionPills
        cursor={cursor}
        details={(rows ?? []).map(row => row.detail)}
        disabled={!ready}
        onActivate={() => keys.focusQuestion(0)}
        onDraft={onDraft}
        onOtherFocus={() => keys.onOtherFocus(0)}
        onPick={index => keys.pick(0, index)}
        onRowFocus={index => keys.focusRow(0, index)}
        question={question}
        staged={{
          choices: (rows ?? []).filter(row => picked.includes(row.id)).map(row => row.label),
          draft
        }}
      />
    )
  }

  return (
    <fieldset
      className="m-0 grid min-w-0 gap-2 border-0 p-0"
      data-clarify-batch-question={SETUP_CHOOSE_QID}
      disabled={!ready}
    >
      <span className="whitespace-pre-wrap font-medium leading-(--conversation-line-height)">{question.question}</span>
      <PickerRows {...props} pickerKind={pickerKind} />
    </fieldset>
  )
}
