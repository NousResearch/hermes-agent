'use client'

import type { SetupChooseKind } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { Puzzle } from 'lucide-react'
import { type ComponentType, type FormEvent, useCallback, useEffect, useMemo, useRef } from 'react'

import { useSessionView } from '@/app/chat/session-view'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { LayoutDashboard, MessageQuestion, Moon, Palette, Plug } from '@/lib/icons'
import {
  $setupChooseStages,
  type ClarifyQuestion,
  type ClarifyRequest,
  clearClarifyRequest,
  commitSetupChoose,
  EMPTY_SETUP_STAGE,
  SETUP_CHOOSE_QID,
  setupChooseAnswer,
  setupChooseStage,
  skipClarify,
  stageSetupChoose
} from '@/store/clarify'
import { notifyError } from '@/store/notifications'
import { respondToServerRequest } from '@/store/server-requests'
import { useTheme } from '@/themes'

import { ClarifyConfirmBar } from './core/confirm-bar'
import { CLARIFY_ICON_CLASS, ClarifyShell } from './core/shell'
import { useClarifyKeys } from './core/use-clarify-keys'
import { SetupChooseBody, setupChooseSource } from './setup-pending-parts'
import { PICKER_COLUMNS } from './setup-pickers'
import { LIVE_LOOK, useSetupRows } from './setup-rows'
import { handleClarifySubmitShortcut } from './submit-shortcut'
import { UndeliveredNotice } from './undelivered-notice'

type SetupSource = Pick<ClarifyRequest, 'questions' | 'setup'>

function sameLabels(a: Record<string, string>, b: Record<string, string>): boolean {
  const ids = Object.keys(a)

  return ids.length === Object.keys(b).length && ids.every(id => a[id] === b[id])
}

const KIND_ICONS: Record<SetupChooseKind, ComponentType<{ className?: string }>> = {
  accent: Palette,
  connectors: Plug,
  fork: MessageQuestion,
  layout: LayoutDashboard,
  machine_use: MessageQuestion,
  plugins: Puzzle,
  question: MessageQuestion,
  theme: Moon,
  tour: MessageQuestion
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
  const { mode, setMode } = useTheme()

  const { kind, pickerKind, ready, requestId, setup, source } = setupChooseSource(request, fromArgs)
  const freeText = pickerKind === null
  const rows = useSetupRows(setup, storedId)

  const stages = useStore($setupChooseStages)
  const { draft, picked } = (requestId && stages[requestId]) || EMPTY_SETUP_STAGE
  const preselected = setup?.preselected
  const rowLabels = useMemo(() => Object.fromEntries((rows ?? []).map(row => [row.id, row.label])), [rows])

  // Once the list is known, record its names (composer text is matched against them) and start the card with
  // the rows the scan saw in use; the card may not list them all.
  useEffect(() => {
    if (!requestId || !rows) {
      return
    }

    const current = $setupChooseStages.get()[requestId]

    if (current && sameLabels(current.labels, rowLabels)) {
      return
    }

    const ids = new Set(rows.map(row => row.id))
    const scanned = !current && preselected?.length ? { picked: preselected.filter(id => ids.has(id)) } : {}

    stageSetupChoose(requestId, { labels: rowLabels, ...scanned })
  }, [preselected, requestId, rowLabels, rows])

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
      if (!requestId) {
        return
      }

      const current = setupChooseStage(requestId)
      const removing = question.multiSelect && current.picked.includes(id)
      const live = removing ? undefined : LIVE_LOOK[kind]

      stageSetupChoose(requestId, {
        picked: question.multiSelect
          ? removing
            ? current.picked.filter(value => value !== id)
            : [...current.picked, id]
          : [id],
        revert: current.revert ?? live?.snapshot(mode, setMode) ?? null,
        ...(question.multiSelect ? {} : { draft: '' })
      })
      live?.apply(id, setMode)
    },
    // Value-stable deps only: the effect below writes this into the store the card reads, so a dep that
    // changes identity every render redraws the card forever (React #185). The labels effect stages the row names.
    [kind, mode, question.multiSelect, requestId, setMode]
  )

  // Composer text that names a row picks it through this card, so the typed pick applies its look too.
  const liveRequestId = request?.requestId

  useEffect(() => {
    if (liveRequestId) {
      stageSetupChoose(liveRequestId, { preview: stage })
    }
  }, [liveRequestId, stage])

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
      if (requestId) {
        stageSetupChoose(requestId, { draft: value, ...(question.multiSelect ? {} : { picked: [] }) })
      }
    },
    [question.multiSelect, requestId]
  )

  const confirm = useCallback(() => {
    if (!request || answer === null) {
      return
    }

    if (!respondToServerRequest(request.requestId, setupChooseAnswer(answer, rowLabels))) {
      notifyError(new Error(copy.notReady), copy.sendFailed)

      return
    }

    triggerHaptic('submit')
    onAnswered()
    commitSetupChoose(request.requestId)
    clearClarifyRequest(request.requestId, request.sessionId)
  }, [answer, copy, onAnswered, request, rowLabels])

  const skip = useCallback(() => {
    if (!request) {
      return
    }

    onAnswered()
    skipClarify(request)
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
    columns: pickerKind === null ? undefined : PICKER_COLUMNS[pickerKind](rows ?? []),
    enabled: ready,
    formRef,
    initialRow: Math.max(0, rows?.findIndex(row => row.id === picked[0]) ?? 0),
    isStaged: (_question, row) =>
      answer !== null &&
      (pickerKind === null || question.multiSelect || row === undefined || picked.includes(rows?.[row]?.id ?? '')),
    onClear: pickerKind === null && requestId ? () => stageSetupChoose(requestId, { picked: [] }) : undefined,
    onConfirm: confirm,
    onToggle: toggle,
    other: freeText,
    questions,
    shortcuts: false
  })

  const cursor = ready ? keys.cursorRow : null
  const Icon = KIND_ICONS[kind]

  return (
    <form
      aria-busy={ready || undelivered ? undefined : 'true'}
      className="my-1.5 grid gap-4"
      data-clarify-batch={1}
      data-clarify-batch-preview={ready ? undefined : ''}
      data-clarify-choices={ready ? 0 : undefined}
      data-clarify-other="false"
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
            {pickerKind === null ? copy.oneQuestion : setupCopy.kinds[pickerKind]}
          </span>
          <Icon aria-hidden className={CLARIFY_ICON_CLASS} />
        </div>
        {undelivered ? <UndeliveredNotice /> : null}
        <SetupChooseBody
          cursor={cursor}
          draft={draft}
          keys={keys}
          onDraft={onDraft}
          onStage={stage}
          picked={picked}
          pickerKind={pickerKind}
          question={question}
          ready={ready}
          rows={rows}
        />
      </ClarifyShell>

      {undelivered ? null : (
        <ClarifyConfirmBar canConfirm={answer !== null} disabled={!ready} onSkip={skip} submitting={false} />
      )}
    </form>
  )
}
