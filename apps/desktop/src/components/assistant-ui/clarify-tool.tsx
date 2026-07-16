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

import { useSessionView } from '@/app/chat/session-view'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { Button } from '@/components/ui/button'
import { Kbd } from '@/components/ui/kbd'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { CircleLetterA, Loader2, MessageQuestion } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { clearClarifyRequest, sessionClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'

import { selectMessageRunning } from './tool/fallback-model'
import { parseMaybeObject } from './tool/fallback-model/format'

interface ClarifyArgs {
  question?: string
  choices?: string[] | null
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
  const choices = Array.isArray(row.choices) ? row.choices.filter((c): c is string => typeof c === 'string') : null

  return {
    question: stringField(row, 'question'),
    choices: choices && choices.length > 0 ? choices : null
  }
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

const OPTION_ROW_CLASS =
  'flex w-full items-start gap-2 rounded-[0.25rem] px-1.5 py-1 text-left disabled:cursor-not-allowed disabled:opacity-50'

// field-sizing on top of Textarea's shared chrome; kill min-h-16 for one-liners.
const CLARIFY_TEXTAREA_CLASS = 'field-sizing-content max-h-40 min-h-0 resize-none'

const CLARIFY_SHELL_CLASS =
  'my-1.5 rounded-md border border-primary/20 bg-(--ui-chat-surface-background) text-[length:var(--conversation-text-font-size)] text-(--ui-text-primary)'

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

function ClarifyToolSettled({ args, result }: ToolCallMessagePartProps) {
  const { t } = useI18n()
  const copy = t.assistant.clarify
  const fromArgs = useMemo(() => readClarifyArgs(args), [args])
  const fromResult = useMemo(() => readClarifyResult(result), [result])

  const question = fromResult.question || fromArgs.question || ''
  const answer = fromResult.answer
  const error = fromResult.error
  const skipped = !error && answer !== undefined && !answer.trim()
  const answerText = error || (skipped ? copy.skipped : (answer ?? '').trim())

  return (
    <ClarifyShell className="grid gap-1.5 px-2.5 py-2" data-clarify-settled="">
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
    </ClarifyShell>
  )
}

function ClarifyToolPending({ args }: ToolCallMessagePartProps) {
  const { t } = useI18n()
  const copy = t.assistant.clarify
  // The tool row is in whichever session's transcript rendered it — read THAT
  // session's clarify (primary or tile), not the globally-active one.
  const sessionId = useStore(useSessionView().$runtimeId)
  const $request = useMemo(() => sessionClarifyRequest(sessionId), [sessionId])
  const request = useStore($request)
  const gateway = useStore($gateway)
  const fromArgs = useMemo(() => readClarifyArgs(args), [args])

  const matchingRequest = useMemo(() => {
    if (!request) {
      return null
    }

    if (fromArgs.question && request.question && fromArgs.question !== request.question) {
      return null
    }

    return request
  }, [fromArgs.question, request])

  const question = fromArgs.question || matchingRequest?.question || ''

  const choices = useMemo(
    () => fromArgs.choices ?? matchingRequest?.choices ?? [],
    [fromArgs.choices, matchingRequest?.choices]
  )

  const hasChoices = choices.length > 0

  const [draft, setDraft] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [selectedChoice, setSelectedChoice] = useState<string | null>(null)
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
  const pendingAnswer = selectedChoice ?? (trimmedDraft || null)

  const selectChoice = useCallback((choice: string) => {
    // Picking a choice and typing are mutually exclusive answers.
    setDraft('')
    setSelectedChoice(choice)
  }, [])

  const submitAnswer = useCallback(() => {
    if (selectedChoice !== null) {
      void respond(selectedChoice)

      return
    }

    if (trimmedDraft) {
      void respond(trimmedDraft)
    }
  }, [respond, selectedChoice, trimmedDraft])

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

  // Letter shortcuts: A/B/C… pick the matching option, the trailing letter jumps
  // into "Other", and Enter confirms the current pick. Stands down whenever a
  // field is focused (you're typing, not navigating) so it never eats keystrokes
  // meant for the composer or the Other box.
  useEffect(() => {
    if (!ready || !hasChoices || submitting) {
      return
    }

    const onKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || event.defaultPrevented) {
        return
      }

      const active = document.activeElement as HTMLElement | null

      if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA' || active.isContentEditable)) {
        return
      }

      const key = event.key.toLowerCase()

      if (key.length === 1 && key >= 'a' && key <= 'z') {
        const index = key.charCodeAt(0) - 97

        if (index < choices.length) {
          event.preventDefault()
          selectChoice(choices[index])
        } else if (index === choices.length) {
          event.preventDefault()
          textareaRef.current?.focus()
        }

        return
      }

      if (event.key === 'Enter' && pendingAnswer) {
        event.preventDefault()
        submitAnswer()
      }
    }

    window.addEventListener('keydown', onKeyDown)

    return () => window.removeEventListener('keydown', onKeyDown)
  }, [choices, hasChoices, pendingAnswer, ready, selectChoice, submitAnswer, submitting])

  if (loading) {
    return (
      <ClarifyShell
        aria-label={copy.loadingQuestion}
        className="grid min-h-12 place-items-center px-2.5 py-3"
        role="status"
      >
        <Loader2 aria-hidden className="size-4 animate-spin text-(--ui-text-tertiary)" />
      </ClarifyShell>
    )
  }

  const onDraftChange = (value: string) => {
    setDraft(value)

    // Typing is its own answer — drop any picked choice so the two inputs can't
    // both look selected.
    if (value.trim()) {
      setSelectedChoice(null)
    }
  }

  return (
    <ClarifyShell className="grid gap-2 px-2.5 py-2">
      <div className="flex items-start gap-2">
        <span className="flex-1 whitespace-pre-wrap font-medium leading-(--conversation-line-height)">{question}</span>
        <MessageQuestion aria-hidden className="mt-px size-4 shrink-0 text-(--ui-text-tertiary)" />
      </div>

      <form className="grid gap-2" onSubmit={handleSubmit}>
        {hasChoices ? (
          <div className="grid gap-px" role="group">
            {choices.map((choice, index) => (
              <button
                className={cn(
                  OPTION_ROW_CLASS,
                  'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-(--ui-text-primary)',
                  selectedChoice === choice && 'text-(--ui-text-primary)'
                )}
                data-choice
                disabled={submitting}
                key={`${index}-${choice}`}
                onClick={() => selectChoice(choice)}
                type="button"
              >
                <KeyBadge char={letterFor(index)} selected={selectedChoice === choice} />
                <span className="flex-1 wrap-anywhere">{choice}</span>
              </button>
            ))}
            <label className={cn(OPTION_ROW_CLASS, 'items-center')}>
              <KeyBadge char={letterFor(choices.length)} preview={otherFocused} selected={Boolean(trimmedDraft)} />
              <Textarea
                className={CLARIFY_TEXTAREA_CLASS}
                disabled={submitting}
                onBlur={() => setOtherFocused(false)}
                onChange={event => onDraftChange(event.target.value)}
                onFocus={() => {
                  setSelectedChoice(null)
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
    </ClarifyShell>
  )
}
