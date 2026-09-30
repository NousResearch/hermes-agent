'use client'

import type { SetupChooseIntent, SetupChooseKind } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { Puzzle } from 'lucide-react'
import { type ComponentType, type FormEvent, useCallback, useMemo, useRef, useState } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { LayoutDashboard, MessageQuestion, Moon, Palette, Plug } from '@/lib/icons'
import { type ClarifyQuestion, type ClarifyRequest, clearClarifyRequest, SETUP_CHOOSE_QID } from '@/store/clarify'
import { notifyError } from '@/store/notifications'
import { respondToServerRequest } from '@/store/server-requests'
import { useTheme } from '@/themes'

import { ClarifyConfirmBar } from './core/confirm-bar'
import { QuestionBlock } from './core/question-block'
import { CLARIFY_ICON_CLASS, ClarifyShell } from './core/shell'
import { useClarifyKeys } from './core/use-clarify-keys'
import { SETUP_PICKERS, SetupIntentRows } from './setup-pickers'
import { LIVE_APPLY, useSetupRows } from './setup-rows'
import { handleClarifySubmitShortcut } from './submit-shortcut'
import { UndeliveredNotice } from './undelivered-notice'

type SetupSource = Pick<ClarifyRequest, 'questions' | 'setup'>

const KIND_ICONS: Record<SetupChooseKind, ComponentType<{ className?: string }>> = {
  accent: Palette,
  connectors: Plug,
  layout: LayoutDashboard,
  plugins: Puzzle,
  question: MessageQuestion,
  theme: Moon
}

export function SetupChoosePending({
  fromArgs,
  onAnswered,
  request,
  undelivered
}: {
  fromArgs: null | SetupSource
  onAnswered: () => void
  request: ClarifyRequest | null
  undelivered: boolean
}) {
  const { t } = useI18n()
  const copy = t.assistant.clarify
  const setupCopy = t.assistant.setupChoose
  const storedId = useStore(useSessionView().$storedId)
  const { setMode } = useTheme()

  const ready = Boolean(request?.requestId && request.setup)
  const source = request ?? fromArgs
  const setup = source?.setup ?? null
  const kind = setup?.kind ?? 'question'
  const pickerKind = kind === 'question' ? null : kind
  const freeText = pickerKind === null
  const rows = useSetupRows(setup, storedId)

  const [picked, setPicked] = useState<string[]>([])
  const [draft, setDraft] = useState('')
  const [intents, setIntents] = useState<Record<string, SetupChooseIntent>>({})

  const question: ClarifyQuestion = useMemo(
    () => ({
      choices: rows && rows.length > 0 ? rows.map(row => row.label) : null,
      multiSelect: Boolean(setup?.multiSelect && rows && rows.length > 0),
      qid: SETUP_CHOOSE_QID,
      question: source?.questions[0]?.question ?? ''
    }),
    [rows, setup?.multiSelect, source]
  )

  const answer = useMemo((): null | string | string[] => {
    const text = draft.trim()

    if (question.multiSelect) {
      const all = [...picked, ...(text ? [text] : [])]

      return all.length > 0 ? all : null
    }

    return picked[0] ?? (text || null)
  }, [draft, picked, question.multiSelect])

  const stage = useCallback(
    (id: string) => {
      const removing = question.multiSelect && picked.includes(id)

      setPicked(current =>
        question.multiSelect ? (removing ? current.filter(value => value !== id) : [...current, id]) : [id]
      )

      if (!question.multiSelect) {
        setDraft('')
      }

      if (!removing) {
        LIVE_APPLY[kind]?.(id, setMode)
      }
    },
    [kind, picked, question.multiSelect, setMode]
  )

  const toggle = useCallback(
    (_question: ClarifyQuestion, choice: string) => {
      const row = rows?.[question.choices?.indexOf(choice) ?? -1]

      if (row) {
        stage(row.id)
      }
    },
    [question.choices, rows, stage]
  )

  const onDraft = useCallback(
    (value: string) => {
      setDraft(value)

      if (!question.multiSelect) {
        setPicked([])
      }
    },
    [question.multiSelect]
  )

  const confirm = useCallback(() => {
    if (!request || answer === null) {
      return
    }

    const ids = picked.filter(id => rows?.some(row => row.id === id))
    const intent = setup?.intent ? Object.fromEntries(ids.map(id => [id, intents[id] ?? 'later'])) : undefined

    if (!respondToServerRequest(request.requestId, { intent, picked: answer })) {
      notifyError(new Error(copy.notReady), copy.sendFailed)

      return
    }

    triggerHaptic('submit')
    onAnswered()
    clearClarifyRequest(request.requestId, request.sessionId)
  }, [answer, copy, intents, onAnswered, picked, request, rows, setup?.intent])

  const skip = useCallback(() => {
    if (!request) {
      return
    }

    onAnswered()
    clearClarifyRequest(request.requestId, request.sessionId)
    respondToServerRequest(request.requestId, {})
  }, [onAnswered, request])

  const handleSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()

      if (ready) {
        confirm()
      }
    },
    [confirm, ready]
  )

  const formRef = useRef<HTMLFormElement | null>(null)
  const questions = useMemo(() => [question], [question])

  const keys = useClarifyKeys({
    enabled: ready,
    formRef,
    isStaged: () => answer !== null,
    onClear: () => setPicked([]),
    onConfirm: confirm,
    onToggle: toggle,
    other: freeText,
    questions
  })

  const cursor = ready ? keys.cursorRow : null
  const Picker = pickerKind === null ? null : SETUP_PICKERS[pickerKind]
  const Icon = KIND_ICONS[kind]
  const intentRows = setup?.intent && rows ? rows.filter(row => picked.includes(row.id)) : []

  return (
    <form
      aria-busy={ready || undelivered ? undefined : 'true'}
      className="my-1.5 grid gap-4"
      data-clarify-batch={1}
      data-clarify-batch-preview={ready ? undefined : ''}
      data-clarify-choices={ready ? question.choices?.length || undefined : undefined}
      data-clarify-other={freeText ? undefined : 'false'}
      data-setup-choose={kind}
      onKeyDownCapture={handleClarifySubmitShortcut}
      onSubmit={handleSubmit}
      ref={formRef}
    >
      {ready || undelivered ? null : (
        <span className="sr-only" role="status">
          {copy.loadingQuestion}
        </span>
      )}
      <ClarifyShell className="grid gap-3">
        <div className="flex items-start gap-2">
          <span className="flex-1 text-[0.6875rem] leading-4 text-(--ui-text-tertiary)">
            {pickerKind === null ? copy.questionProgress(answer === null ? 0 : 1, 1) : setupCopy.kinds[pickerKind]}
          </span>
          <Icon aria-hidden className={CLARIFY_ICON_CLASS} />
        </div>
        {undelivered ? <UndeliveredNotice /> : null}
        {Picker === null ? (
          <QuestionBlock
            cursor={cursor}
            disabled={!ready}
            onActivate={() => keys.focusQuestion(0)}
            onDraft={onDraft}
            onOtherFocus={() => keys.onOtherFocus(0)}
            onPick={index => keys.pick(0, index)}
            question={question}
            staged={{
              choices: (rows ?? []).filter(row => picked.includes(row.id)).map(row => row.label),
              draft
            }}
          />
        ) : (
          <fieldset
            className="m-0 grid min-w-0 gap-2 border-0 p-0"
            data-clarify-batch-question={SETUP_CHOOSE_QID}
            disabled={!ready}
          >
            <span className="whitespace-pre-wrap font-medium leading-(--conversation-line-height)">
              {question.question}
            </span>
            {rows === null ? (
              <div className="grid grid-cols-3 gap-2" role="status">
                <span className="sr-only">{setupCopy.loading}</span>
                {Array.from({ length: 6 }, (_, index) => (
                  <div className="h-10 animate-pulse rounded-lg bg-muted/40" key={index} />
                ))}
              </div>
            ) : rows.length === 0 ? (
              <p className="text-(--ui-text-tertiary)">{setupCopy.unavailable}</p>
            ) : (
              <Picker
                cursor={cursor}
                onPick={index => keys.pick(0, index)}
                onStage={stage}
                picked={picked}
                rows={rows}
              />
            )}
            <SetupIntentRows
              intents={intents}
              onIntent={(id, intent) => setIntents(current => ({ ...current, [id]: intent }))}
              rows={intentRows}
            />
          </fieldset>
        )}
      </ClarifyShell>

      {undelivered ? null : (
        <ClarifyConfirmBar canConfirm={answer !== null} disabled={!ready} onSkip={skip} submitting={false} />
      )}
    </form>
  )
}
