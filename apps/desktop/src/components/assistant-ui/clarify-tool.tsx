'use client'

import { type ToolCallMessagePartProps, useAuiState } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import {
  type ComponentProps,
  type FormEvent,
  type KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState
} from 'react'

import { requestComposerFocus, requestComposerInsert } from '@/app/chat/composer/focus'
import { useSessionView } from '@/app/chat/session-view'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { WIDGET_SHELL_CLASS } from '@/components/chat/widget-shell'
import { Button } from '@/components/ui/button'
import { Kbd } from '@/components/ui/kbd'
import { Textarea } from '@/components/ui/textarea'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { CircleLetterA, Loader2, MessageQuestion } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  bareChoice,
  type ClarifyQuestion,
  type ClarifyRequest,
  clearClarifyRequest,
  normalizeChoices,
  RECOMMENDED_LABEL,
  sessionClarifyRequest,
  warnDroppedChoices
} from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'

import { selectMessageRunning } from './tool/fallback-model'
import { parseMaybeObject } from './tool/fallback-model/format'

interface ClarifyArgs {
  question?: string
  choices?: string[] | null
  multiSelect?: boolean
  questions?: { question: string; choices?: string[] | null; multiSelect?: boolean }[]
}

interface ClarifyResult {
  question?: string
  answer?: string
  error?: string
}

function stringField(row: Record<string, unknown>, ...keys: string[]): string | undefined {
  for (const key of keys) {
    const value = row[key]

    if (typeof value === 'string') {
      return value
    }
  }
}

function readClarifyArgs(args: unknown): ClarifyArgs {
  const row = parseMaybeObject(args)
  const rawChoices = row.choices
  const choices = normalizeChoices(rawChoices)

  const question = stringField(row, 'question')

  if (rawChoices != null && choices.length === 0 && question) {
    warnDroppedChoices('tool_args', question, rawChoices)
  }

  // Batch form: tool args carry the model's questions array. Entries are
  // normalized leniently here (qid comes from the gateway request, not args).
  let questions: ClarifyArgs['questions']

  if (Array.isArray(row.questions)) {
    const parsed = row.questions
      .map(entry => {
        const item = parseMaybeObject(entry)
        const text = stringField(item, 'question')

        if (!text) {
          return null
        }

        const itemChoices = normalizeChoices(item.choices)

        return {
          choices: itemChoices.length > 0 ? itemChoices : null,
          multiSelect: item.multi_select === true && itemChoices.length > 0,
          question: text
        }
      })
      .filter((entry): entry is NonNullable<typeof entry> => entry !== null)

    if (parsed.length > 0) {
      questions = parsed
    }
  }

  return {
    question,
    choices: choices.length > 0 ? choices : null,
    multiSelect: row.multi_select === true,
    questions
  }
}

interface ClarifyBatchResponse {
  id?: string
  question?: string
  answer?: string | string[]
}

/** Parse batch clarify tool JSON (`responses` array + optional timed_out). */
export function readClarifyBatchResult(result: unknown): {
  responses: ClarifyBatchResponse[]
  timedOut: boolean
} {
  const row = parseMaybeObject(result)

  if (!Array.isArray(row.responses)) {
    return { responses: [], timedOut: false }
  }

  const responses = row.responses.map((entry): ClarifyBatchResponse => {
    const item = parseMaybeObject(entry)
    const answer = item.user_response

    return {
      answer: Array.isArray(answer) ? answer.map(String) : typeof answer === 'string' ? answer : undefined,
      id: stringField(item, 'id'),
      question: stringField(item, 'question')
    }
  })

  return { responses, timedOut: row.timed_out === true }
}

/** Parse clarify tool JSON (`question` + `user_response`). */
export function readClarifyResult(result: unknown): ClarifyResult {
  const row = parseMaybeObject(result)

  if (Object.keys(row).length === 0) {
    return typeof result === 'string' && result.trim() ? { answer: result.trim() } : {}
  }

  return {
    question: stringField(row, 'question'),
    answer: stringField(row, 'user_response', 'answer'),
    error: stringField(row, 'error')
  }
}

const letterFor = (index: number): string => String.fromCharCode(65 + index)

// The backend tags the agent's preferred option (`mark_recommended`); the card
// renders the label in tertiary text so the option itself still reads first.
function ChoiceLabel({ choice }: { choice: string }) {
  const bare = bareChoice(choice)

  if (bare === choice) {
    return <>{choice}</>
  }

  return (
    <>
      {bare} <span className="text-(--ui-text-tertiary)">{RECOMMENDED_LABEL}</span>
    </>
  )
}

const OPTION_ROW_CLASS =
  'flex w-full items-start gap-2 rounded-[0.25rem] px-1.5 py-1 text-left disabled:cursor-not-allowed disabled:opacity-50'

// field-sizing on top of Textarea's shared chrome; kill min-h-16 for one-liners.
const CLARIFY_TEXTAREA_CLASS = 'field-sizing-content max-h-40 min-h-0 resize-none'

const CLARIFY_SHELL_CLASS = `${WIDGET_SHELL_CLASS} text-[length:var(--conversation-text-font-size)] text-(--ui-text-primary)`

const CLARIFY_ICON_CLASS = 'mt-px size-4 shrink-0 text-(--ui-text-tertiary)'

function ClarifyShell({ children, className, ...props }: ComponentProps<'div'>) {
  return (
    <div className={cn(CLARIFY_SHELL_CLASS, className)} data-slot="clarify-inline" {...props}>
      {children}
    </div>
  )
}

function ClarifyLine({
  children,
  className,
  icon: Icon,
  ...props
}: ComponentProps<'div'> & { icon: typeof MessageQuestion }) {
  return (
    <div className={cn('flex items-start gap-2', className)} {...props}>
      <div className="min-w-0 flex-1">{children}</div>
      <Icon aria-hidden className={CLARIFY_ICON_CLASS} />
    </div>
  )
}

function KeyBadge({ char, preview, selected }: { char: string; preview?: boolean; selected: boolean }) {
  return (
    <Kbd
      className={cn(
        'mt-px',
        selected && 'border-primary bg-primary text-white shadow-none',
        !selected && preview && 'border-primary text-primary shadow-none'
      )}
      size="sm"
    >
      {char}
    </Kbd>
  )
}

/** A letter-badged option row. Shared by the live pending card (where a click
 * selects an answer) and the settled skip card (where a click drafts a
 * follow-up), so both stay visually identical. */
function ChoiceButton({
  active = false,
  char,
  choice,
  disabled,
  keyShortcuts,
  onClick,
  selected,
  title
}: {
  active?: boolean
  char: string
  choice: string
  disabled?: boolean
  keyShortcuts?: string
  onClick: () => void
  selected?: boolean
  title?: string
}) {
  // `Tip` is the repo's themed replacement for native `title=` (a native
  // tooltip on a <button> is banned by the no-native-title guard). It renders
  // the child untouched when `label` is falsy, so the live card (no tip) is
  // unaffected and only the settled skip card gets the hover hint.
  //
  // `active` is the keyboard cursor on the live card (arrow-key navigation);
  // it highlights the row and previews its key badge. The settled skip card
  // never passes it, so its rows stay plain.
  return (
    <Tip label={title}>
      <button
        aria-current={active || undefined}
        aria-keyshortcuts={keyShortcuts}
        aria-pressed={selected}
        className={cn(
          OPTION_ROW_CLASS,
          'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-(--ui-text-primary)',
          active && 'bg-(--chrome-action-hover) text-(--ui-text-primary)',
          selected && 'text-(--ui-text-primary)'
        )}
        data-choice
        data-highlighted={active || undefined}
        disabled={disabled}
        onClick={onClick}
        type="button"
      >
        <KeyBadge char={char} preview={active} selected={Boolean(selected)} />
        <span className="flex-1 wrap-anywhere">
          <ChoiceLabel choice={choice} />
        </span>
      </button>
    </Tip>
  )
}

export const ClarifyTool = (props: ToolCallMessagePartProps) => {
  // Answered → settled Q&A (ToolFallback collapsed the answer away).
  if (props.result !== undefined) {
    return <ClarifyToolSettled {...props} />
  }

  return <ClarifyToolLive {...props} />
}

function ClarifyToolLive(props: ToolCallMessagePartProps) {
  const messageRunning = useAuiState(selectMessageRunning)

  // Stopped mid-prompt with no result — don't leave a dead interactive panel.
  if (!messageRunning) {
    return <ToolFallback {...props} />
  }

  return <ClarifyToolPending {...props} />
}

function ClarifyToolSettled(props: ToolCallMessagePartProps) {
  const batch = readClarifyBatchResult(props.result)

  if (batch.responses.length > 0) {
    return <ClarifyToolBatchSettled responses={batch.responses} />
  }

  return <ClarifyToolSingleSettled {...props} />
}

function ClarifyToolSingleSettled({ args, result }: ToolCallMessagePartProps) {
  const { t } = useI18n()
  const copy = t.assistant.clarify
  const fromArgs = useMemo(() => readClarifyArgs(args), [args])
  const fromResult = useMemo(() => readClarifyResult(result), [result])

  const question = fromResult.question || fromArgs.question || ''
  const answer = fromResult.answer
  const error = fromResult.error
  const skipped = !error && answer !== undefined && !answer.trim()
  const answerText = error || (skipped ? copy.skipped : (answer ?? '').trim())
  const choices = fromArgs.choices ?? []

  // A skipped (timed-out) clarify keeps its choices on screen and actionable.
  // The blocking request is long gone — the tool already returned empty — so a
  // pick can't resolve it retroactively. Instead it drafts a quoted follow-up
  // into the composer (Enter sends; if the agent is mid-turn it queues like
  // any other prompt). Without this the card collapsed to just "Skipped" and
  // the options were unrecoverable.
  const followUp = useCallback(
    (choice: string) => {
      requestComposerInsert(copy.lateAnswer(question, choice), { mode: 'block' })
      requestComposerFocus()
      triggerHaptic('selection')
    },
    [copy, question]
  )

  return (
    <ClarifyShell className="my-1.5 grid gap-1.5" data-clarify-settled="">
      {question ? (
        <ClarifyLine icon={MessageQuestion}>
          <span className="whitespace-pre-wrap font-medium leading-(--conversation-line-height)">{question}</span>
        </ClarifyLine>
      ) : null}
      {answerText ? (
        <ClarifyLine icon={CircleLetterA}>
          <p
            className={cn(
              'whitespace-pre-wrap leading-(--conversation-line-height)',
              error ? 'text-destructive' : 'text-(--ui-text-secondary)',
              skipped && 'italic text-(--ui-text-tertiary)'
            )}
            data-clarify-answer=""
          >
            {answerText}
          </p>
        </ClarifyLine>
      ) : null}
      {skipped && choices.length > 0 ? (
        <div className="grid gap-px" data-clarify-late-choices="" role="group">
          {choices.map((choice, index) => (
            <ChoiceButton
              char={letterFor(index)}
              choice={choice}
              key={`${index}-${choice}`}
              onClick={() => followUp(choice)}
              title={copy.lateAnswerTip}
            />
          ))}
          <p className="px-1.5 pt-0.5 text-[0.6875rem] leading-4 text-(--ui-text-tertiary)">{copy.lateAnswerHint}</p>
        </div>
      ) : null}
    </ClarifyShell>
  )
}

function ClarifyToolPending(props: ToolCallMessagePartProps) {
  // The tool row is in whichever session's transcript rendered it — read THAT
  // session's clarify (primary or tile), not the globally-active one.
  const sessionId = useStore(useSessionView().$runtimeId)
  const $request = useMemo(() => sessionClarifyRequest(sessionId), [sessionId])
  const request = useStore($request)
  const fromArgs = useMemo(() => readClarifyArgs(props.args), [props.args])

  // Batch: the gateway request carries qid-keyed questions. Args alone can't
  // drive the form (no qids to respond with), so batch waits for the request.
  if (request?.questions?.length || fromArgs.questions) {
    return <ClarifyToolBatchPending request={request} />
  }

  return <ClarifyToolSinglePending fromArgs={fromArgs} request={request} />
}

function ClarifyToolSinglePending({
  fromArgs,
  request
}: {
  fromArgs: ClarifyArgs
  request: ClarifyRequest | null
}) {
  const { t } = useI18n()
  const copy = t.assistant.clarify
  const gateway = useStore($gateway)

  const matchingRequest = useMemo(() => {
    if (!request || request.questions?.length) {
      return null
    }

    if (fromArgs.question && request.question && fromArgs.question !== request.question) {
      return null
    }

    return request
  }, [fromArgs.question, request])

  const question = fromArgs.question || matchingRequest?.question || ''

  const choices = useMemo(
    // Prefer the gateway request's choices over the raw tool args: the backend
    // labels the recommended option there (`mark_recommended`), and the card
    // only renders once `matchingRequest` exists, so the args are a fallback
    // for a hydration race, not the normal path.
    () => matchingRequest?.choices ?? fromArgs.choices ?? [],
    [fromArgs.choices, matchingRequest?.choices]
  )

  const hasChoices = choices.length > 0
  const multiSelect = hasChoices && Boolean(matchingRequest?.multiSelect ?? fromArgs.multiSelect)

  const [draft, setDraft] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [selectedChoices, setSelectedChoices] = useState<string[]>([])
  // The keyboard cursor. Indices 0..choices.length-1 are the options; the
  // trailing index (=== choices.length) is the "Other" free-text row.
  const [activeIndex, setActiveIndex] = useState(0)
  const [otherFocused, setOtherFocused] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  // Race: tool.start fires a tick before clarify.request, so request_id
  // arrives slightly after the tool block mounts. Hold the whole panel on a
  // spinner until the gateway request is wired — showing disabled choices or
  // a "loading question" stub is worse than a brief wait.
  const ready = Boolean(matchingRequest?.requestId)
  const loading = !ready && !submitting

  const respond = useCallback(
    async (answer: string) => {
      if (!ready || !matchingRequest) {
        notifyError(new Error(copy.notReady), copy.sendFailed)

        return
      }

      if (!gateway) {
        notifyError(new Error(copy.gatewayDisconnected), copy.sendFailed)

        return
      }

      setSubmitting(true)

      try {
        await gateway.request<{ ok?: boolean }>('clarify.respond', {
          request_id: matchingRequest.requestId,
          answer
        })
        triggerHaptic('submit')
        clearClarifyRequest(matchingRequest.requestId, matchingRequest.sessionId)
        // tool.complete lands next → ClarifyToolSettled.
      } catch (error) {
        notifyError(error, copy.sendFailed)
        setSubmitting(false)
      }
    },
    [copy.gatewayDisconnected, copy.notReady, copy.sendFailed, gateway, matchingRequest, ready]
  )

  const trimmedDraft = draft.trim()
  // The answer is whichever input is active: a picked choice, or typed text.
  // Picking a choice no longer fires immediately — it selects, then the user
  // confirms with Continue (or Enter from the field).

  const selectedAnswer = multiSelect
    ? selectedChoices.length > 0
      ? JSON.stringify(selectedChoices)
      : null
    : (selectedChoices[0] ?? null)

  const pendingAnswer = selectedAnswer ?? (trimmedDraft || null)

  const selectChoice = useCallback(
    (choice: string, index: number) => {
      // Picking a choice and typing are mutually exclusive answers.
      setDraft('')
      setSelectedChoices(selected => {
        if (!multiSelect) {
          return [choice]
        }

        return selected.includes(choice) ? selected.filter(value => value !== choice) : [...selected, choice]
      })
      setActiveIndex(index)
    },
    [multiSelect]
  )

  // Keep the cursor in range when the choice set changes (never past "Other").
  useEffect(() => {
    setActiveIndex(index => Math.min(index, choices.length))
  }, [choices.length])

  const moveActive = useCallback(
    (delta: number) => {
      const itemCount = choices.length + 1

      // Arrow navigation is a move, not a pick. Multi-select keeps staged
      // choices while the cursor moves so the user can build a set; the
      // single-select path retains its existing clear-on-navigation behaviour.
      setDraft('')

      if (!multiSelect) {
        setSelectedChoices([])
      }

      setActiveIndex(index => (index + delta + itemCount) % itemCount)
    },
    [choices.length, multiSelect]
  )

  const submitAnswer = useCallback(() => {
    if (pendingAnswer) {
      void respond(pendingAnswer)
    }
  }, [pendingAnswer, respond])

  const activateActive = useCallback(() => {
    const choice = choices[activeIndex]

    // Multi-select Enter toggles the highlighted choice. The user confirms the
    // staged set explicitly with Continue so this path never submits a scalar.
    if (multiSelect && choice) {
      selectChoice(choice, activeIndex)

      return
    }

    // A staged answer (picked choice or typed text) wins — confirm it.
    if (pendingAnswer) {
      submitAnswer()

      return
    }

    // Otherwise act on the highlighted row: a choice responds immediately, and
    // the trailing "Other" row focuses the free-text field.
    if (choice) {
      void respond(choice)

      return
    }

    textareaRef.current?.focus()
  }, [activeIndex, choices, multiSelect, pendingAnswer, respond, selectChoice, submitAnswer])

  const handleTextareaKey = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.nativeEvent.isComposing) {
        return
      }

      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault()
        submitAnswer()
      }
    },
    [submitAnswer]
  )

  const handleSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      submitAnswer()
    },
    [submitAnswer]
  )

  // Arrow keys move a visual cursor, 1-9 and A/B/C… pick directly, and Enter
  // confirms the current answer (or acts on the highlighted row). Stands down
  // whenever a focusable control (a field, a choice button, the action bar) is
  // focused, so it never eats keystrokes meant for the composer, the Other box,
  // or a button the user tabbed to.
  useEffect(() => {
    if (!ready || !hasChoices || submitting) {
      return
    }

    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || event.defaultPrevented) {
        return
      }

      const active = document.activeElement as HTMLElement | null

      if (
        active &&
        (active.isContentEditable || active.matches('a[href], button, input, select, textarea, [role="button"]'))
      ) {
        return
      }

      if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
        event.preventDefault()
        moveActive(event.key === 'ArrowDown' ? 1 : -1)

        return
      }

      if (/^[1-9]$/.test(event.key)) {
        const index = Number(event.key) - 1

        if (index < choices.length) {
          event.preventDefault()
          selectChoice(choices[index], index)
        } else if (index === choices.length) {
          event.preventDefault()
          setActiveIndex(index)
          textareaRef.current?.focus()
        }

        return
      }

      const key = event.key.toLowerCase()

      // Only the letters this card actually renders a row for. Anything past
      // the last row belongs to the composer — the user is typing a message
      // instead of picking an option, and swallowing the keystroke here would
      // make the first letter of it vanish.
      if (key.length === 1 && key >= 'a' && key <= 'z') {
        const index = key.charCodeAt(0) - 97

        if (index < choices.length) {
          event.preventDefault()
          selectChoice(choices[index], index)
        } else if (index === choices.length) {
          event.preventDefault()
          setActiveIndex(index)
          textareaRef.current?.focus()
        }

        return
      }

      if (event.key === 'Enter') {
        event.preventDefault()
        activateActive()
      }
    }

    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  }, [activateActive, choices, hasChoices, moveActive, ready, selectChoice, submitting])

  if (loading) {
    return (
      <ClarifyShell aria-label={copy.loadingQuestion} className="my-1.5 grid min-h-12 place-items-center" role="status">
        <Loader2 aria-hidden className="size-4 animate-spin text-(--ui-text-tertiary)" />
      </ClarifyShell>
    )
  }

  const onDraftChange = (value: string) => {
    setDraft(value)

    // Typing is its own answer — drop any picked choice so the two inputs can't
    // both look selected.
    if (value.trim()) {
      setSelectedChoices([])
    }
  }

  return (
    // `data-clarify-choices` marks the panel as owning its OWN shortcut keys
    // (Enter, and 1..N+1 / A.. for the N choices plus "Other") while they're
    // live, so the global type-to-focus listener (`clarifyCardOwnsKey`) yields
    // exactly those and lets every other printable through to the composer —
    // typing a real message instead of picking an option stays possible. The
    // value is the choice count so the check needs no store access.
    //
    // The form is the outer element so the actions can sit OUTSIDE the card and
    // still submit it — the panel holds the question, the buttons ride below it.
    <form
      className="my-1.5 grid gap-4"
      data-clarify-choices={hasChoices ? choices.length : undefined}
      onSubmit={handleSubmit}
    >
      <ClarifyShell className="grid gap-2">
        <div className="flex items-start gap-2">
          <span className="flex-1 whitespace-pre-wrap font-medium leading-(--conversation-line-height)">
            {question}
          </span>
          <MessageQuestion aria-hidden className="mt-px size-4 shrink-0 text-(--ui-text-tertiary)" />
        </div>

        {hasChoices ? (
          <div className="grid gap-px" role="group">
            {choices.map((choice, index) => (
              <ChoiceButton
                active={activeIndex === index}
                char={letterFor(index)}
                choice={choice}
                disabled={submitting}
                key={`${index}-${choice}`}
                keyShortcuts={`${letterFor(index)} ${index + 1}`}
                onClick={() => selectChoice(choice, index)}
                selected={selectedChoices.includes(choice)}
              />
            ))}
            <label
              className={cn(
                OPTION_ROW_CLASS,
                'items-center',
                activeIndex === choices.length && 'bg-(--chrome-action-hover)'
              )}
              data-highlighted={activeIndex === choices.length || undefined}
            >
              <KeyBadge
                char={letterFor(choices.length)}
                preview={otherFocused || activeIndex === choices.length}
                selected={Boolean(trimmedDraft)}
              />
              <Textarea
                aria-current={activeIndex === choices.length || undefined}
                aria-keyshortcuts={`${letterFor(choices.length)} ${choices.length + 1}`}
                className={CLARIFY_TEXTAREA_CLASS}
                disabled={submitting}
                onBlur={() => setOtherFocused(false)}
                onChange={event => onDraftChange(event.target.value)}
                onFocus={() => {
                  setSelectedChoices([])
                  setActiveIndex(choices.length)
                  setOtherFocused(true)
                }}
                onKeyDown={handleTextareaKey}
                placeholder={copy.other}
                ref={textareaRef}
                rows={1}
                size="sm"
                value={draft}
              />
            </label>
          </div>
        ) : (
          <Textarea
            className={CLARIFY_TEXTAREA_CLASS}
            disabled={submitting}
            onChange={event => onDraftChange(event.target.value)}
            onKeyDown={handleTextareaKey}
            placeholder={copy.placeholder}
            ref={textareaRef}
            rows={1}
            size="sm"
            value={draft}
          />
        )}
      </ClarifyShell>

      <div className="flex items-center justify-end gap-1">
        <Button disabled={submitting} onClick={() => void respond('')} size="xs" type="button" variant="text">
          {copy.skip}
        </Button>
        <Button disabled={submitting || !pendingAnswer} size="xs" type="submit">
          {submitting ? (
            <Loader2 className="size-3 animate-spin" />
          ) : (
            <>
              {copy.continueLabel}
              <span aria-hidden className="ml-0.5 text-[0.625rem] opacity-70">
                ⏎
              </span>
            </>
          )}
        </Button>
      </div>
    </form>
  )
}

// ─── Batch (multi-question) clarify ─────────────────────────────────────────

/** Settled batch card: every question with its locked (or absent) answer. */
function ClarifyToolBatchSettled({ responses }: { responses: { question?: string; answer?: string | string[] }[] }) {
  const { t } = useI18n()
  const copy = t.assistant.clarify

  return (
    <ClarifyShell className="my-1.5 grid gap-2.5" data-clarify-settled="">
      {responses.map((row, index) => {
        const answer = Array.isArray(row.answer) ? row.answer.join(', ') : (row.answer ?? '')
        const blank = !answer.trim()

        return (
          <div className="grid gap-1" key={`${index}-${row.question ?? ''}`}>
            {row.question ? (
              <ClarifyLine icon={MessageQuestion}>
                <span className="whitespace-pre-wrap font-medium leading-(--conversation-line-height)">
                  {row.question}
                </span>
              </ClarifyLine>
            ) : null}
            <ClarifyLine icon={CircleLetterA}>
              <p
                className={cn(
                  'whitespace-pre-wrap leading-(--conversation-line-height)',
                  blank ? 'italic text-(--ui-text-tertiary)' : 'text-(--ui-text-secondary)'
                )}
                data-clarify-answer=""
              >
                {blank ? copy.skipped : answer}
              </p>
            </ClarifyLine>
          </div>
        )
      })}
    </ClarifyShell>
  )
}

/** One question's interactive block inside the live batch card. */
function BatchQuestionBlock({
  disabled,
  locked,
  onDraft,
  onToggle,
  question,
  staged
}: {
  disabled: boolean
  locked: boolean
  onDraft: (value: string) => void
  onToggle: (choice: string) => void
  question: ClarifyQuestion
  staged: { choices: string[]; draft: string }
}) {
  const { t } = useI18n()
  const copy = t.assistant.clarify
  const choices = question.choices ?? []

  return (
    <div className="grid gap-1" data-clarify-batch-question={question.qid} data-locked={locked || undefined}>
      <div className="flex items-start gap-2">
        <span className="flex-1 whitespace-pre-wrap font-medium leading-(--conversation-line-height)">
          {question.question}
        </span>
        {locked ? (
          <span className="shrink-0 rounded-sm bg-(--chrome-action-hover) px-1 py-px text-[0.625rem] text-(--ui-text-tertiary)">
            ✓ {copy.answeredBadge}
          </span>
        ) : null}
      </div>

      {choices.length > 0 ? (
        <div className="grid gap-px" role="group">
          {choices.map((choice, index) => (
            <ChoiceButton
              char={letterFor(index)}
              choice={choice}
              disabled={disabled}
              key={`${index}-${choice}`}
              onClick={() => onToggle(choice)}
              selected={staged.choices.includes(choice)}
            />
          ))}
          <label className={cn(OPTION_ROW_CLASS, 'items-center')}>
            <KeyBadge char={letterFor(choices.length)} selected={Boolean(staged.draft.trim())} />
            <Textarea
              className={CLARIFY_TEXTAREA_CLASS}
              disabled={disabled}
              onChange={event => onDraft(event.target.value)}
              placeholder={copy.other}
              rows={1}
              size="sm"
              value={staged.draft}
            />
          </label>
        </div>
      ) : (
        <Textarea
          className={CLARIFY_TEXTAREA_CLASS}
          disabled={disabled}
          onChange={event => onDraft(event.target.value)}
          placeholder={copy.placeholder}
          rows={1}
          size="sm"
          value={staged.draft}
        />
      )}
    </div>
  )
}

const emptyStage = { choices: [] as string[], draft: '' }

/** Live batch card: all questions at once, staged locally, ONE confirm.
 * Picks and drafts stay in component state — nothing reaches the server
 * until every question has a staged answer and the user presses the single
 * "Confirm and continue" button, which sends the per-question locks
 * back-to-back and completes the batch. Staged answers stay editable up to
 * that moment. The per-question wire protocol is unchanged (the TUI/CLI
 * still lock incrementally); this card just batches its locks at the end. */
function ClarifyToolBatchPending({ request }: { request: ClarifyRequest | null }) {
  const { t } = useI18n()
  const copy = t.assistant.clarify
  const gateway = useStore($gateway)

  // qids only exist on the gateway request — args are a hydration-race
  // fallback for display, never answerable (no ids to respond with).
  const questions = request?.questions ?? []
  const ready = Boolean(request?.requestId) && questions.length > 0

  const [staged, setStaged] = useState<Record<string, { choices: string[]; draft: string }>>({})
  const [submitting, setSubmitting] = useState(false)

  // Reconnect replay: answers the server already locked (an earlier window's
  // partial progress) pre-stage their questions so the restored card shows
  // them selected instead of blank.
  useEffect(() => {
    const lockedAnswers = request?.lockedAnswers

    if (!lockedAnswers) {
      return
    }

    setStaged(current => {
      const next = { ...current }

      for (const question of questions) {
        const answer = lockedAnswers[question.qid]

        if (answer === undefined || next[question.qid]) {
          continue
        }

        const asChoice = (question.choices ?? []).find(choice => bareChoice(choice) === answer)
        next[question.qid] = asChoice ? { choices: [asChoice], draft: '' } : { choices: [], draft: answer }
      }

      return next
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps -- keyed by the replay map only
  }, [request?.lockedAnswers])

  const stageFor = (qid: string) => staged[qid] ?? emptyStage

  const stagedAnswer = useCallback(
    (question: ClarifyQuestion): string | null => {
      const stage = staged[question.qid] ?? emptyStage

      if (stage.choices.length > 0) {
        return question.multiSelect ? JSON.stringify(stage.choices.map(bareChoice)) : bareChoice(stage.choices[0])
      }

      const draft = stage.draft.trim()

      return draft ? draft : null
    },
    [staged]
  )

  const answeredCount = questions.filter(q => stagedAnswer(q) !== null).length
  const allStaged = answeredCount === questions.length

  const confirmAll = useCallback(async () => {
    if (!request || !gateway) {
      notifyError(new Error(request ? copy.gatewayDisconnected : copy.notReady), copy.sendFailed)

      return
    }

    setSubmitting(true)

    try {
      // Sequential, not Promise.all: the LAST lock resolves the blocked tool
      // server-side, so every earlier lock must already be accepted when it
      // lands — a reordered burst could complete the batch with a missing
      // answer.
      for (const question of questions) {
        const answer = stagedAnswer(question)

        await gateway.request<{ ok?: boolean }>('clarify.respond', {
          answer: answer ?? '',
          question_id: question.qid,
          request_id: request.requestId
        })
      }

      triggerHaptic('submit')
      // tool.complete lands next → ClarifyToolBatchSettled.
      clearClarifyRequest(request.requestId, request.sessionId)
    } catch (error) {
      notifyError(error, copy.sendFailed)
      setSubmitting(false)
    }
  }, [copy, gateway, questions, request, stagedAnswer])

  const toggleChoice = useCallback((question: ClarifyQuestion, choice: string) => {
    setStaged(current => {
      const stage = current[question.qid] ?? emptyStage
      const next = question.multiSelect
        ? stage.choices.includes(choice)
          ? stage.choices.filter(value => value !== choice)
          : [...stage.choices, choice]
        : [choice]

      return { ...current, [question.qid]: { choices: next, draft: '' } }
    })
  }, [])

  const draftFor = useCallback((question: ClarifyQuestion, value: string) => {
    setStaged(current => ({ ...current, [question.qid]: { choices: [], draft: value } }))
  }, [])

  const cancelAll = useCallback(async () => {
    if (!request) {
      return
    }

    clearClarifyRequest(request.requestId, request.sessionId)

    try {
      await gateway?.request('clarify.respond', { answer: '', request_id: request.requestId })
    } catch {
      // The tool times out on its own; a failed skip must never block the UI.
    }
  }, [gateway, request])

  const handleSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()

      if (allStaged) {
        void confirmAll()
      }
    },
    [allStaged, confirmAll]
  )

  if (!ready) {
    return (
      <ClarifyShell aria-label={copy.loadingQuestion} className="my-1.5 grid min-h-12 place-items-center" role="status">
        <Loader2 aria-hidden className="size-4 animate-spin text-(--ui-text-tertiary)" />
      </ClarifyShell>
    )
  }

  return (
    <form className="my-1.5 grid gap-4" data-clarify-batch={questions.length} onSubmit={handleSubmit}>
      <ClarifyShell className="grid gap-3">
        <div className="flex items-start gap-2">
          <span className="flex-1 text-[0.6875rem] leading-4 text-(--ui-text-tertiary)">
            {copy.questionProgress(answeredCount, questions.length)}
          </span>
          <MessageQuestion aria-hidden className={CLARIFY_ICON_CLASS} />
        </div>
        {questions.map(question => (
          <BatchQuestionBlock
            disabled={submitting}
            key={question.qid}
            locked={false}
            onDraft={value => draftFor(question, value)}
            onToggle={choice => toggleChoice(question, choice)}
            question={question}
            staged={stageFor(question.qid)}
          />
        ))}
      </ClarifyShell>

      <div className="flex items-center justify-end gap-1">
        <Button disabled={submitting} onClick={() => void cancelAll()} size="xs" type="button" variant="text">
          {copy.skip}
        </Button>
        <Button disabled={submitting || !allStaged} size="xs" type="submit">
          {submitting ? (
            <Loader2 className="size-3 animate-spin" />
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
    </form>
  )
}
