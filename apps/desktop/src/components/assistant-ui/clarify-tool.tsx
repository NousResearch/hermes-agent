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

import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { Button } from '@/components/ui/button'
import { Kbd } from '@/components/ui/kbd'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Check, CircleLetterA, Loader2, MessageQuestion } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $clarifyRequest, clearClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'

import { selectMessageRunning } from './tool/fallback-model'
import { parseMaybeObject } from './tool/fallback-model/format'

interface ClarifyOption {
  id: string
  index: number
  label: string
  description: string
  value: string
}

interface ClarifyArgs {
  question?: string
  context?: string
  recommendation?: string
  options?: ClarifyOption[] | null
}

interface ClarifyResult extends ClarifyArgs {
  selectedOption?: ClarifyOption | null
  answer?: string
  error?: string
}

const MAX_CLARIFY_CHOICES = 4

function stringField(row: Record<string, unknown>, ...keys: string[]): string | undefined {
  for (const key of keys) {
    const value = row[key]

    if (typeof value === 'string') {
      return value
    }
  }
}

function cleanText(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function cleanStringField(row: Record<string, unknown>, ...keys: string[]): string {
  for (const key of keys) {
    const value = cleanText(row[key])

    if (value) {
      return value
    }
  }

  return ''
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null
}

function scalarLabel(value: unknown): string {
  if (value === null || value === undefined) {
    return ''
  }

  return typeof value === 'string' ? value.trim() : String(value).trim()
}

function normalizeClarifyChoices(choices: unknown): ClarifyOption[] | null {
  if (!Array.isArray(choices)) {
    return null
  }

  const options: ClarifyOption[] = []
  const usedIds = new Set<string>()

  for (const raw of choices) {
    if (options.length >= MAX_CLARIFY_CHOICES) {
      break
    }

    const row = recordValue(raw)
    let id = ''
    let label = ''
    let description = ''
    let value = ''

    if (row) {
      id = cleanText(row.id)
      label = cleanStringField(row, 'label', 'text', 'title')
      description = cleanText(row.description)
      value = cleanText(row.value)

      if (!label) {
        label = description
        description = ''
      }
    } else {
      label = scalarLabel(raw)
    }

    if (!label) {
      continue
    }

    const index = options.length + 1
    const fallbackId = `option-${index}`
    let optionId = id || fallbackId

    if (usedIds.has(optionId)) {
      optionId = fallbackId

      let suffix = 2

      while (usedIds.has(optionId)) {
        optionId = `${fallbackId}-${suffix}`
        suffix += 1
      }
    }

    usedIds.add(optionId)
    options.push({
      id: optionId,
      index,
      label,
      description,
      value: value || label
    })
  }

  return options.length > 0 ? options : null
}

function normalizeSelectedOption(value: unknown): ClarifyOption | null {
  const row = recordValue(value)

  if (!row) {
    return null
  }

  const rawIndex = typeof row.index === 'number' && Number.isFinite(row.index) ? row.index : 1
  const index = Math.max(1, Math.floor(rawIndex))
  const description = cleanText(row.description)
  const label = cleanStringField(row, 'label', 'text', 'title') || description || cleanText(row.value) || cleanText(row.id)

  if (!label) {
    return null
  }

  return {
    id: cleanText(row.id) || `option-${index}`,
    index,
    label,
    description: label === description ? '' : description,
    value: cleanText(row.value) || label
  }
}

function parsePromptParts(value: unknown): Pick<ClarifyArgs, 'question' | 'context' | 'recommendation'> {
  const text = cleanText(value)

  if (!text) {
    return {}
  }

  const firstSection = text.search(/\n\s*\n(?:Context|Recommendation):/i)
  const question = (firstSection >= 0 ? text.slice(0, firstSection) : text).trim()
  const context = text.match(/(?:^|\n\s*\n)Context:\s*([\s\S]*?)(?=\n\s*\nRecommendation:|$)/i)?.[1]?.trim()
  const recommendation = text.match(/(?:^|\n\s*\n)Recommendation:\s*([\s\S]*)$/i)?.[1]?.trim()

  return {
    question: question || undefined,
    context: context || undefined,
    recommendation: recommendation || undefined
  }
}

function questionMatchValue(value: unknown): string {
  return (parsePromptParts(value).question ?? '').replace(/\s+/g, ' ').trim()
}

function questionsMatch(left: unknown, right: unknown): boolean {
  const leftQuestion = questionMatchValue(left)
  const rightQuestion = questionMatchValue(right)

  return Boolean(leftQuestion && rightQuestion && leftQuestion === rightQuestion)
}

function readClarifyArgs(args: unknown): ClarifyArgs {
  const row = parseMaybeObject(args)
  const prompt = parsePromptParts(row.question)

  return {
    question: prompt.question,
    context: cleanText(row.context) || prompt.context,
    recommendation: cleanText(row.recommendation) || prompt.recommendation,
    options: normalizeClarifyChoices(row.choices)
  }
}

/** Parse clarify tool JSON (`question` + `user_response`) plus structured option metadata. */
export function readClarifyResult(result: unknown): ClarifyResult {
  const row = parseMaybeObject(result)

  if (Object.keys(row).length === 0) {
    return typeof result === 'string' && result.trim() ? { answer: result.trim() } : {}
  }

  const prompt = parsePromptParts(row.question)

  return {
    question: prompt.question,
    context: cleanText(row.context) || prompt.context,
    recommendation: cleanText(row.recommendation) || prompt.recommendation,
    options: normalizeClarifyChoices(row.options) ?? normalizeClarifyChoices(row.choices_offered),
    selectedOption: normalizeSelectedOption(row.selected_option),
    answer: stringField(row, 'user_response', 'answer'),
    error: stringField(row, 'error')
  }
}

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

function ClarifyPromptBlock({
  context,
  question,
  recommendation
}: {
  context?: string
  question: string
  recommendation?: string
}) {
  return (
    <div className="grid gap-1.5">
      {question ? (
        <ClarifyLine icon={MessageQuestion}>
          <span className="whitespace-pre-wrap font-medium leading-(--conversation-line-height)">{question}</span>
        </ClarifyLine>
      ) : null}
      {context ? (
        <p className="whitespace-pre-wrap text-(--ui-text-secondary)" data-clarify-context="">
          {context}
        </p>
      ) : null}
      {recommendation ? (
        <p className="whitespace-pre-wrap font-medium text-primary" data-clarify-recommendation="">
          {recommendation}
        </p>
      ) : null}
    </div>
  )
}

function optionDisplayText(option: ClarifyOption): string {
  return option.description && option.description !== option.label
    ? `${option.label} - ${option.description}`
    : option.label
}

function optionMatchesAnswer(option: ClarifyOption, answer: string | undefined, selected?: ClarifyOption | null): boolean {
  if (selected) {
    return option.id === selected.id
  }

  const aliases = [answer].map(value => cleanText(value).toLowerCase()).filter(Boolean)

  if (aliases.length === 0) {
    return false
  }

  const optionAliases = [
    option.id,
    option.value,
    option.label,
    String(option.index),
    optionDisplayText(option),
    option.description && option.description !== option.label
      ? `${option.label} \u2013 ${option.description}`
      : ''
  ].map(value => cleanText(value).toLowerCase())

  return aliases.some(alias => optionAliases.includes(alias))
}

function ClarifyOptionContent({ option }: { option: ClarifyOption }) {
  return (
    <span className="flex min-w-0 flex-1 flex-col gap-0.5">
      <span className="wrap-anywhere font-medium">{option.label}</span>
      {option.description ? (
        <span className="wrap-anywhere text-[0.875em] text-(--ui-text-secondary)">{option.description}</span>
      ) : null}
    </span>
  )
}

export const ClarifyTool = (props: ToolCallMessagePartProps) => {
  // Answered -> settled Q&A (ToolFallback collapsed the answer away).
  if (props.result !== undefined) {
    return <ClarifyToolSettled {...props} />
  }

  return <ClarifyToolLive {...props} />
}

function ClarifyToolLive(props: ToolCallMessagePartProps) {
  const messageRunning = useAuiState(selectMessageRunning)

  // Stopped mid-prompt with no result: don't leave a dead interactive panel.
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
  const context = fromResult.context || fromArgs.context
  const recommendation = fromResult.recommendation || fromArgs.recommendation
  const options = fromResult.options ?? fromArgs.options ?? []
  const answer = fromResult.answer
  const error = fromResult.error
  const skipped = !error && answer !== undefined && !answer.trim()
  const selectedOption = options.find(option => optionMatchesAnswer(option, answer, fromResult.selectedOption))

  const answerText =
    error ||
    (skipped
      ? copy.skipped
      : selectedOption
        ? `${selectedOption.index}. ${selectedOption.label}`
        : (answer ?? '').trim())

  return (
    <ClarifyShell className="grid gap-1.5 px-2.5 py-2" data-clarify-settled="">
      <ClarifyPromptBlock context={context} question={question} recommendation={recommendation} />
      {options.length > 0 ? (
        <div className="grid gap-px" data-clarify-options="" role="list">
          {options.map(option => {
            const selected = selectedOption?.id === option.id

            return (
              <div
                className={cn(
                  OPTION_ROW_CLASS,
                  'text-(--ui-text-secondary)',
                  selected && 'bg-primary/10 text-(--ui-text-primary)'
                )}
                data-clarify-option={option.id}
                data-selected={selected ? 'true' : undefined}
                key={option.id}
                role="listitem"
              >
                <KeyBadge char={String(option.index)} selected={selected} />
                <ClarifyOptionContent option={option} />
                {selected ? <Check aria-hidden className="mt-0.5 size-3.5 shrink-0 text-primary" /> : null}
              </div>
            )
          })}
        </div>
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
  const request = useStore($clarifyRequest)
  const gateway = useStore($gateway)
  const fromArgs = useMemo(() => readClarifyArgs(args), [args])

  const matchingRequest = useMemo(() => {
    if (!request) {
      return null
    }

    if (fromArgs.question && request.question && !questionsMatch(fromArgs.question, request.question)) {
      return null
    }

    return request
  }, [fromArgs.question, request])

  const requestPrompt = useMemo(() => parsePromptParts(matchingRequest?.question), [matchingRequest?.question])
  const question = fromArgs.question || requestPrompt.question || ''
  const context = fromArgs.context || requestPrompt.context
  const recommendation = fromArgs.recommendation || requestPrompt.recommendation

  const options = useMemo(
    () => fromArgs.options ?? normalizeClarifyChoices(matchingRequest?.choices) ?? [],
    [fromArgs.options, matchingRequest?.choices]
  )

  const hasChoices = options.length > 0

  const [draft, setDraft] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [selectedOptionId, setSelectedOptionId] = useState<string | null>(null)
  const [otherFocused, setOtherFocused] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  // Race: tool.start fires a tick before clarify.request, so request_id
  // arrives slightly after the tool block mounts. Hold the whole panel on a
  // spinner until the gateway request is wired; showing disabled choices or
  // a "loading question" stub is worse than a brief wait.
  const ready = Boolean(matchingRequest?.requestId)
  const loading = !ready && !submitting

  const respond = useCallback(
    async (answer: string, optionId?: string, responseKind?: 'custom') => {
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
          answer,
          ...(optionId ? { option_id: optionId } : {}),
          ...(responseKind ? { response_kind: responseKind } : {})
        })
        triggerHaptic('submit')
        clearClarifyRequest(matchingRequest.requestId, matchingRequest.sessionId)
        // tool.complete lands next -> ClarifyToolSettled.
      } catch (error) {
        notifyError(error, copy.sendFailed)
        setSubmitting(false)
      }
    },
    [copy.gatewayDisconnected, copy.notReady, copy.sendFailed, gateway, matchingRequest, ready]
  )

  const trimmedDraft = draft.trim()

  const selectedOption = useMemo(
    () => options.find(option => option.id === selectedOptionId) ?? null,
    [options, selectedOptionId]
  )

  // The answer is whichever input is active: a picked choice, or typed text.
  // Picking a choice selects, then the user confirms with Continue or Enter.
  const pendingAnswer = selectedOption?.value ?? (trimmedDraft || null)

  const selectOption = useCallback((option: ClarifyOption) => {
    // Picking a choice and typing are mutually exclusive answers.
    setDraft('')
    setSelectedOptionId(option.id)
  }, [])

  const submitAnswer = useCallback(() => {
    if (selectedOption) {
      void respond(selectedOption.value, selectedOption.id)

      return
    }

    if (trimmedDraft) {
      void respond(trimmedDraft, undefined, 'custom')
    }
  }, [respond, selectedOption, trimmedDraft])

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

  // Numeric shortcuts: 1..N pick the matching option, N+1 jumps into Other,
  // and Enter confirms the current pick. Stands down whenever a field is focused.
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

      if (/^\d$/.test(event.key)) {
        const shortcut = Number(event.key)
        const index = shortcut - 1

        if (index >= 0 && index < options.length) {
          event.preventDefault()
          selectOption(options[index]!)
        } else if (shortcut === options.length + 1) {
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
  }, [hasChoices, options, pendingAnswer, ready, selectOption, submitAnswer, submitting])

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

    // Typing is its own answer; drop any picked choice so the two inputs can't
    // both look selected.
    if (value.trim()) {
      setSelectedOptionId(null)
    }
  }

  return (
    <ClarifyShell className="grid gap-2 px-2.5 py-2">
      <ClarifyPromptBlock context={context} question={question} recommendation={recommendation} />

      <form className="grid gap-2" onSubmit={handleSubmit}>
        {hasChoices ? (
          <div className="grid gap-px" role="group">
            {options.map(option => (
              <button
                aria-pressed={selectedOptionId === option.id}
                className={cn(
                  OPTION_ROW_CLASS,
                  'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-(--ui-text-primary)',
                  selectedOptionId === option.id && 'text-(--ui-text-primary)'
                )}
                data-choice
                disabled={submitting}
                key={option.id}
                onClick={() => selectOption(option)}
                type="button"
              >
                <KeyBadge char={String(option.index)} selected={selectedOptionId === option.id} />
                <ClarifyOptionContent option={option} />
              </button>
            ))}
            <label className={cn(OPTION_ROW_CLASS, 'items-center')}>
              <KeyBadge char={String(options.length + 1)} preview={otherFocused} selected={Boolean(trimmedDraft)} />
              <Textarea
                className={CLARIFY_TEXTAREA_CLASS}
                disabled={submitting}
                onBlur={() => setOtherFocused(false)}
                onChange={event => onDraftChange(event.target.value)}
                onFocus={() => {
                  setSelectedOptionId(null)
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
