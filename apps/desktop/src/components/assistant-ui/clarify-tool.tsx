'use client'

import { type ToolCallMessagePartProps } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type ComponentProps, type FormEvent, type KeyboardEvent, useCallback, useMemo, useRef, useState } from 'react'

import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { Button } from '@/components/ui/button'
import { Kbd } from '@/components/ui/kbd'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Check, Loader2, MessageQuestion } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $clarifyRequest, clearClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { notify, notifyError } from '@/store/notifications'

import { selectMessageRunning } from './tool/fallback-model'

interface ClarifyArgs {
  allowOther?: boolean
  choices?: string[] | null
  maxSelections?: number | null
  minSelections?: number | null
  multiSelect?: boolean
  question?: string
}

function readNumber(row: Record<string, unknown>, camel: string, snake: string): number | null | undefined {
  const value = row[camel] ?? row[snake]

  return typeof value === 'number' ? value : value === null ? null : undefined
}

function readBoolean(row: Record<string, unknown>, camel: string, snake: string): boolean | undefined {
  const value = row[camel] ?? row[snake]

  return typeof value === 'boolean' ? value : undefined
}

const CLARIFY_RESPOND_TIMEOUT_MS = 120_000

function isClarifyRespondTimeoutError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /request timed out:\s*clarify\.respond/i.test(message)
}

const CLARIFY_RESPOND_TIMEOUT_MS = 120_000

function isClarifyRespondTimeoutError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /request timed out:\s*clarify\.respond/i.test(message)
}

function readClarifyArgs(args: unknown): ClarifyArgs {
  if (!args || typeof args !== 'object') {
    return {}
  }

  const row = args as Record<string, unknown>
  const choices = Array.isArray(row.choices) ? row.choices.filter((c): c is string => typeof c === 'string') : null

  return {
    allowOther: readBoolean(row, 'allowOther', 'allow_other'),
    choices: choices && choices.length > 0 ? choices : null,
    maxSelections: readNumber(row, 'maxSelections', 'max_selections'),
    minSelections: readNumber(row, 'minSelections', 'min_selections'),
    multiSelect: readBoolean(row, 'multiSelect', 'multi_select'),
    question: typeof row.question === 'string' ? row.question : undefined
  }
}

// Each option (and "Other") is keyed A, B, C… so it can be picked by pressing
// that letter — the badge doubles as the shortcut hint.
const letterFor = (index: number): string => String.fromCharCode(65 + index)

// Choice and "Other" rows share a layout; only color differs. Mirrors a tool
// row's compact rhythm so the panel reads as part of the transcript.
const OPTION_ROW_CLASS =
  'flex w-full items-start gap-2 rounded-[0.25rem] px-1.5 py-1 text-left disabled:cursor-not-allowed disabled:opacity-50'

// Content-sizing freeform field (CSS `field-sizing` — same primitive as the
// commit bar and search field): starts at one line, grows with what's typed,
// and never reflows the panel when focused. Bare so the "Other" row matches the
// choice rows above it.
const FREEFORM_INPUT_CLASS =
  'field-sizing-content max-h-40 min-h-0 w-full resize-none bg-transparent p-0 leading-(--conversation-line-height) text-(--ui-text-primary) outline-none placeholder:text-(--ui-text-tertiary) disabled:opacity-50'

// Quiet inline panel that matches the surrounding tool rows: a single hairline
// border in the shared stroke token, a soft surface fill, and a faint primary
// accent that signals "this one needs you" without the loud animated ring.
const CLARIFY_SHELL_CLASS =
  'my-1.5 rounded-md border border-primary/20 bg-(--ui-chat-surface-background) text-[length:var(--conversation-text-font-size)] text-(--ui-text-primary)'

function ClarifyShell({ children, className, ...props }: ComponentProps<'div'>) {
  return (
    <div className={cn(CLARIFY_SHELL_CLASS, className)} data-slot="clarify-inline" {...props}>
      {children}
    </div>
  )
}

// Selection lives on the letter badge alone — a solid primary fill — not the
// whole row, which stays a quiet hover target. `preview` is the focused-but-empty
// "Other" state: the badge outlines in primary to show it's armed, then fills
// once a value is actually typed.
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

function SelectToggle({ selected }: { selected: boolean }) {
  return (
    <span
      aria-hidden
      className={cn(
        'mt-px grid size-4 shrink-0 place-items-center rounded-full border transition-colors',
        selected ? 'border-primary bg-primary text-white' : 'border-(--ui-stroke-secondary) text-transparent'
      )}
    >
      {selected && <Check className="size-3" />}
    </span>
  )
}

function toggleChoice(choices: string[], choice: string, maxSelections: number | null): string[] {
  if (choices.includes(choice)) {
    return choices.filter(item => item !== choice)
  }

  if (maxSelections !== null && choices.length >= maxSelections) {
    return choices
  }

  return [...choices, choice]
}

export const ClarifyTool = (props: ToolCallMessagePartProps) => {
  // The clarify request itself is the source of truth for interactivity. Using
  // assistant-ui's messageRunning flag here can briefly flip false while the
  // backend is still blocked on clarify.respond, which makes the selection block
  // disappear even though the request is still live.
  const isPending = props.result === undefined

  if (!isPending) {
    return <ToolFallback {...props} />
  }

  return <ClarifyToolPending {...props} />
}

function ClarifyToolPending({ args }: ToolCallMessagePartProps) {
  const { t } = useI18n()
  const copy = t.assistant.clarify
  const request = useStore($clarifyRequest)
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
  const multiSelect = fromArgs.multiSelect ?? matchingRequest?.multiSelect ?? false
  const allowOther = fromArgs.allowOther ?? matchingRequest?.allowOther ?? true
  const minSelections = fromArgs.minSelections ?? matchingRequest?.minSelections ?? 0
  const maxSelections = fromArgs.maxSelections ?? matchingRequest?.maxSelections ?? null

  const [draft, setDraft] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [selectedChoice, setSelectedChoice] = useState<string | null>(null)
  const [selectedChoices, setSelectedChoices] = useState<string[]>([])
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
        await gateway.request<{ ok?: boolean }>(
          'clarify.respond',
          {
            request_id: matchingRequest.requestId,
            answer
          },
          CLARIFY_RESPOND_TIMEOUT_MS
        )
        triggerHaptic('submit')
        clearClarifyRequest(matchingRequest.requestId, matchingRequest.sessionId)
        // The matching tool.complete will land shortly after, swapping this
        // panel for the ToolFallback view above.
      } catch (error) {
        if (isClarifyRespondTimeoutError(error)) {
          notify({
            kind: 'warning',
            title: copy.responsePendingTitle,
            message: copy.responsePendingMessage,
            detail: error instanceof Error ? error.message : String(error),
            durationMs: 12_000
          })
        } else {
          notifyError(error, copy.sendFailed)
        }

        setSubmitting(false)
      }
    },
    [
      copy.gatewayDisconnected,
      copy.notReady,
      copy.responsePendingMessage,
      copy.responsePendingTitle,
      copy.sendFailed,
      gateway,
      matchingRequest,
      ready
    ]
  )

  const trimmedDraft = draft.trim()
  const selectedSummary = selectedChoices.join(', ')
  const customSummary = trimmedDraft ? `${copy.other}: ${trimmedDraft}` : ''
  const selectionSummary = selectedSummary || selectedChoice || customSummary
  const selectedChoiceCount = selectedChoices.length
  const canSubmitSelected = multiSelect && selectedChoiceCount > 0 && selectedChoiceCount >= minSelections
  const canSkip = minSelections <= 0

  const selectChoice = useCallback(
    (choice: string) => {
      setDraft('')
      setSelectedChoices([])
      setSelectedChoice(choice)
      void respond(choice)
    },
    [respond]
  )

  const toggleMultiChoice = useCallback(
    (choice: string) => {
      if (!multiSelect) {
        return
      }

      setDraft('')
      setSelectedChoice(null)
      setSelectedChoices(current => toggleChoice(current, choice, maxSelections))
    },
    [maxSelections, multiSelect]
  )

  const submitSelected = useCallback(() => {
    if (canSubmitSelected) {
      void respond(selectedChoices.join(', '))
    }
  }, [canSubmitSelected, respond, selectedChoices])

  const submitDraft = useCallback(() => {
    if (trimmedDraft) {
      void respond(trimmedDraft)
    }
  }, [respond, trimmedDraft])

  const handleTextareaKey = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.nativeEvent.isComposing) {
        return
      }

      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault()
        submitDraft()
      }
    },
    [submitDraft]
  )

  const handleSubmitDraft = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      submitDraft()
    },
    [submitDraft]
  )

  if (loading) {
    return (
      <ClarifyShell aria-label={copy.loadingQuestion} className="grid min-h-12 place-items-center px-2.5 py-3" role="status">
        <Loader2 aria-hidden className="size-4 animate-spin text-(--ui-text-tertiary)" />
      </ClarifyShell>
    )
  }

  const onDraftChange = (value: string) => {
    setDraft(value)

    // Typing is its own answer — drop any picked/staged choice so the inputs
    // can't both look selected.
    if (value.trim()) {
      setSelectedChoice(null)
      setSelectedChoices([])
    }
  }

  return (
    <ClarifyShell className="grid gap-2 px-2.5 py-2">
      <div className="flex items-start gap-2">
        <span className="flex-1 whitespace-pre-wrap font-medium leading-(--conversation-line-height)">{question}</span>
        <MessageQuestion aria-hidden className="mt-px size-4 shrink-0 text-(--ui-text-tertiary)" />
      </div>

      {selectionSummary && (
        <div className="rounded-[0.25rem] border border-primary/20 bg-primary/5 px-2 py-1 text-xs" role="status">
          <div className="flex items-center justify-between gap-2">
            <span className="font-medium text-primary">{copy.selected}</span>
            {selectedChoiceCount > 0 && (
              <span className="text-(--ui-text-tertiary)">{copy.selectedCount(selectedChoiceCount)}</span>
            )}
          </div>
          <div className="mt-0.5 wrap-anywhere text-(--ui-text-secondary)">{selectionSummary}</div>
        </div>
      )}

      {hasChoices && multiSelect && (
        <div className="rounded-[0.25rem] bg-(--chrome-action-hover) px-2 py-1 text-xs text-(--ui-text-secondary)" role="note">
          {copy.multiSelectHint}
        </div>
      )}

      {hasChoices && (
        <div className="grid gap-px" role="group">
          {choices.map((choice, index) => {
            const staged = selectedChoices.includes(choice)

            if (multiSelect) {
              return (
                <button
                  aria-label={`Toggle ${choice} for multi-select`}
                  aria-pressed={staged}
                  className={cn(
                    OPTION_ROW_CLASS,
                    'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-(--ui-text-primary)',
                    staged && 'text-(--ui-text-primary)'
                  )}
                  data-choice
                  disabled={submitting}
                  key={`${index}-${choice}`}
                  onClick={() => toggleMultiChoice(choice)}
                  type="button"
                >
                  <KeyBadge char={letterFor(index)} selected={staged} />
                  <span className="flex-1 wrap-anywhere">{choice}</span>
                  <SelectToggle selected={staged} />
                </button>
              )
            }

            return (
              <button
                aria-label={choice}
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
            )
          })}
          {allowOther && (
            <label className={cn(OPTION_ROW_CLASS, 'focus-within:bg-(--chrome-action-hover)')}>
              <KeyBadge char={letterFor(choices.length)} preview={otherFocused} selected={Boolean(trimmedDraft)} />
              <textarea
                aria-label={copy.other}
                className={FREEFORM_INPUT_CLASS}
                disabled={submitting}
                onBlur={() => setOtherFocused(false)}
                onChange={event => onDraftChange(event.target.value)}
                onFocus={() => {
                  setSelectedChoice(null)
                  setSelectedChoices([])
                  setOtherFocused(true)
                }}
                onKeyDown={handleTextareaKey}
                placeholder={copy.other}
                ref={textareaRef}
                rows={1}
                value={draft}
              />
            </label>
          )}
        </div>
      )}

      {!hasChoices && (
        <form className="grid gap-2" onSubmit={handleSubmitDraft}>
          <Textarea
            className={FREEFORM_INPUT_CLASS}
            disabled={submitting}
            onChange={event => onDraftChange(event.target.value)}
            onKeyDown={handleTextareaKey}
            placeholder={copy.placeholder}
            ref={textareaRef}
            rows={1}
            value={draft}
          />
        </form>
      )}

      <div className="flex items-center justify-end gap-1">
        {canSkip && (
          <Button disabled={submitting} onClick={() => void respond('')} size="xs" type="button" variant="text">
            {copy.skip}
          </Button>
        )}
        {multiSelect && !trimmedDraft ? (
          <Button disabled={submitting || !canSubmitSelected} onClick={submitSelected} size="xs" type="button">
            {submitting ? <Loader2 className="size-3" /> : copy.selectSelected}
          </Button>
        ) : (
          <Button disabled={submitting || !trimmedDraft} onClick={submitDraft} size="xs" type="button">
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
        )}
      </div>
    </ClarifyShell>
  )
}
