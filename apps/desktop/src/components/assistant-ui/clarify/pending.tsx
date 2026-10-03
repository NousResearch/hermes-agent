'use client'

import { useStore } from '@nanostores/react'
import { type FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Loader } from '@/components/ui/loader'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Loader2, MessageQuestion } from '@/lib/icons'
import {
  bareChoice,
  type ClarifyQuestion,
  type ClarifyRequest,
  type ClarifyStagedAnswer,
  clearClarifyRequest,
  stageClarifyAnswer
} from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { reconnectAction } from '@/store/gateway-reconnect'
import { notifyError } from '@/store/notifications'
import { forgetServerRequest, respondToServerRequest } from '@/store/server-requests'
import { requestForOwnedSession } from '@/store/session-states'

import { emptyStage, QuestionBlock } from './core/question-block'
import { CLARIFY_ICON_CLASS, ClarifyShell } from './core/shell'
import { useClarifyKeys } from './core/use-clarify-keys'
import type { ClarifyArgs } from './parse'
import { handleClarifySubmitShortcut } from './submit-shortcut'
import { UndeliveredNotice } from './undelivered-notice'

const EMPTY_STAGED: Record<string, ClarifyStagedAnswer> = {}

/** Live batch card: all questions at once, staged on the store, ONE confirm.
 * Picks and drafts stay in component state — nothing reaches the server
 * until the user presses the single
 * "Confirm and continue" button, which sends the per-question locks
 * back-to-back and completes the batch. Staged answers stay editable up to
 * that moment. The per-question wire protocol is unchanged (the TUI/CLI
 * still lock incrementally); this card just batches its locks at the end. */
export function ClarifyToolPending({
  fromArgs,
  onAnswered,
  request,
  undelivered
}: {
  fromArgs?: ClarifyArgs
  onAnswered: () => void
  request: ClarifyRequest | null
  undelivered: boolean
}) {
  const { t } = useI18n()
  const copy = t.assistant.clarify
  const gateway = useStore($gateway)

  // qids only exist on the gateway request — args are a hydration-race
  // fallback for display, never answerable (no ids to respond with).
  const liveQuestions = request?.questions ?? []
  const ready = Boolean(request?.requestId) && liveQuestions.length > 0

  // Preview items from the tool args: same question text/choices, synthetic
  // qids, shown disabled until the live request lands (or indefinitely when
  // the caller has no gateway request at all — e.g. an external tool call —
  // so the user sees the question instead of an endless spinner). ONE form
  // renders both states: every control is disabled while !ready, so nothing
  // is ever staged under a synthetic qid and the swap to live qids is clean.
  const previewQuestions: ClarifyQuestion[] = useMemo(
    () =>
      (fromArgs?.questions ?? []).map((entry, index) => ({
        choices: entry.choices ?? null,
        multiSelect: entry.multiSelect ?? false,
        qid: `args-${index}`,
        question: entry.question
      })),
    [fromArgs]
  )

  const questions = ready ? liveQuestions : previewQuestions

  const [submitting, setSubmitting] = useState(false)

  // In-progress answers live on the parked request (#58783): a card remount
  // (stream update, reconnect reconciliation, expiry repaint) restores what
  // was typed instead of destroying it. The store is the source of truth; the
  // locked-answers replay below seeds into the SAME staging so a confirmed
  // server-side answer and a locally typed one read through one path.
  const stagedFromStore = request?.stagedAnswers ?? EMPTY_STAGED

  const stageQuestion = useCallback(
    (qid: string, stage: { choices: string[]; draft: string } | null) => {
      if (request) {
        stageClarifyAnswer(request.requestId, request.sessionId, qid, stage)
      }
    },
    [request]
  )

  // Reconnect replay: answers the server already locked (an earlier window's
  // partial progress) pre-stage their questions so the restored card shows
  // them selected instead of blank.
  useEffect(() => {
    const lockedAnswers = request?.lockedAnswers

    if (!lockedAnswers) {
      return
    }

    for (const question of questions) {
      const answer = lockedAnswers[question.qid]

      if (answer === undefined || answer === null || stagedFromStore[question.qid]) {
        continue
      }

      const options = question.choices ?? []
      let replayedAnswers = [answer]

      if (question.multiSelect) {
        try {
          const parsed = JSON.parse(answer)

          if (Array.isArray(parsed) && parsed.every(value => typeof value === 'string')) {
            replayedAnswers = parsed
          }
        } catch {
          // Older/non-JSON replies remain a one-value replay below.
        }
      }

      const matchedChoices = options.filter(choice => replayedAnswers.includes(bareChoice(choice)))
      stageQuestion(
        question.qid,
        matchedChoices.length > 0 ? { choices: matchedChoices, draft: '' } : { choices: [], draft: answer }
      )
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the replay map only
  }, [request?.lockedAnswers])

  const stageFor = (qid: string) => stagedFromStore[qid] ?? emptyStage

  const stagedAnswer = useCallback(
    (question: ClarifyQuestion): string | null => {
      const stage = stagedFromStore[question.qid] ?? emptyStage
      const draft = stage.draft.trim()

      if (question.multiSelect) {
        // The typed text is an additional answer, not a replacement for the
        // staged choices.
        const combined = [...stage.choices.map(bareChoice), ...(draft ? [draft] : [])]

        return combined.length > 0 ? JSON.stringify(combined) : null
      }

      if (stage.choices.length > 0) {
        return bareChoice(stage.choices[0])
      }

      return draft ? draft : null
    },
    [stagedFromStore]
  )

  const answeredCount = questions.filter(q => stagedAnswer(q) !== null).length
  const canConfirm = answeredCount > 0

  const confirmAll = useCallback(async () => {
    if (!request || !gateway) {
      notifyError(
        new Error(request ? copy.gatewayDisconnected : copy.notReady),
        copy.sendFailed,
        request ? { action: reconnectAction() } : {}
      )

      return
    }

    setSubmitting(true)

    try {
      // Sequential, not Promise.all: the LAST lock resolves the blocked
      // server request, so every earlier lock must already be accepted when
      // it lands — a reordered burst could complete the batch with a missing
      // answer. `clarify.lock` is a normal RPC; it rides the session's OWNER
      // socket (a profile / Bot Chat switch re-points ambient elsewhere).
      for (const question of questions) {
        const answer = stagedAnswer(question)

        await requestForOwnedSession<{ remaining?: string[]; status?: string }>(
          request.sessionId,
          gateway.request.bind(gateway) as typeof gateway.request,
          'clarify.lock',
          {
            answer,
            question_id: question.qid,
            request_id: request.requestId
          }
        )
      }

      forgetServerRequest(request.requestId)

      triggerHaptic('submit')
      onAnswered()
      // tool.complete lands next → ClarifyToolSettled.
      clearClarifyRequest(request.requestId, request.sessionId)
    } catch (error) {
      notifyError(error, copy.sendFailed)
      setSubmitting(false)
    }
  }, [copy, gateway, onAnswered, questions, request, stagedAnswer])

  const toggleChoice = useCallback(
    (question: ClarifyQuestion, choice: string) => {
      const stage = stagedFromStore[question.qid] ?? emptyStage

      const next = question.multiSelect
        ? stage.choices.includes(choice)
          ? stage.choices.filter(value => value !== choice)
          : [...stage.choices, choice]
        : [choice]

      // Multi-select keeps the typed text alongside the toggled choices;
      // single-select stays mutually exclusive.
      stageQuestion(question.qid, { choices: next, draft: question.multiSelect ? stage.draft : '' })
    },
    [stageQuestion, stagedFromStore]
  )

  const draftFor = useCallback(
    (question: ClarifyQuestion, value: string) => {
      const stage = stagedFromStore[question.qid] ?? emptyStage

      // Multi-select keeps the staged choices while the free-text field is
      // edited; single-select stays mutually exclusive.
      stageQuestion(question.qid, {
        choices: question.multiSelect ? stage.choices : [],
        draft: value
      })
    },
    [stageQuestion, stagedFromStore]
  )

  const cancelAll = useCallback(async () => {
    if (!request) {
      return
    }

    onAnswered()
    clearClarifyRequest(request.requestId, request.sessionId)

    // A response with no `answers` is the cancel-all (the plain Esc path).
    respondToServerRequest(request.requestId, {})
  }, [onAnswered, request])

  const handleSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()

      if (ready && canConfirm) {
        void confirmAll()
      }
    },
    [canConfirm, confirmAll, ready]
  )

  const clearStage = useCallback(
    (question: ClarifyQuestion) => {
      stageQuestion(question.qid, null)
    },
    [stageQuestion]
  )

  const isStaged = useCallback((question: ClarifyQuestion) => stagedAnswer(question) !== null, [stagedAnswer])

  const formRef = useRef<HTMLFormElement | null>(null)

  const keys = useClarifyKeys({
    enabled: ready && !submitting,
    formRef,
    isStaged,
    onClear: clearStage,
    onConfirm: () => void confirmAll(),
    onToggle: toggleChoice,
    questions
  })

  const disabled = submitting || !ready

  if (questions.length === 0) {
    return (
      <ClarifyShell aria-label={copy.loadingQuestion} className="my-1.5 grid min-h-12 place-items-center" role="status">
        <Loader aria-hidden="true" className="size-6 text-(--ui-text-tertiary)" role="presentation" type="rose-curve" />
      </ClarifyShell>
    )
  }

  return (
    <form
      aria-busy={ready || undelivered ? undefined : 'true'}
      className="my-1.5 grid gap-4"
      data-clarify-batch={questions.length}
      data-clarify-batch-preview={ready ? undefined : ''}
      data-clarify-choices={ready ? questions[keys.activeQuestion]?.choices?.length || undefined : undefined}
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
            {copy.questionProgress(answeredCount, questions.length)}
          </span>
          <MessageQuestion aria-hidden className={CLARIFY_ICON_CLASS} />
        </div>
        {undelivered ? <UndeliveredNotice /> : null}
        {questions.map((question, index) => (
          <QuestionBlock
            cursor={ready && keys.activeQuestion === index ? keys.cursorRow : null}
            disabled={disabled}
            key={question.qid}
            onActivate={() => keys.focusQuestion(index)}
            onDraft={value => draftFor(question, value)}
            onOtherFocus={() => keys.onOtherFocus(index)}
            onPick={choiceIndex => keys.pick(index, choiceIndex)}
            question={question}
            staged={stageFor(question.qid)}
          />
        ))}
      </ClarifyShell>

      {undelivered ? null : (
        <div className="flex items-center justify-end gap-1">
          <Button disabled={disabled} onClick={() => void cancelAll()} size="xs" type="button" variant="text">
            {copy.skip}
          </Button>
          <Button disabled={disabled || !canConfirm} size="xs" type="submit">
            {submitting ? (
              <Loader2 className="animate-spin" />
            ) : (
              <>
                {copy.confirmAndContinueLabel}
                <span aria-hidden className="ml-0.5 text-[0.625rem] opacity-70">
                  ⏎
                </span>
              </>
            )}
          </Button>
        </div>
      )}
    </form>
  )
}
