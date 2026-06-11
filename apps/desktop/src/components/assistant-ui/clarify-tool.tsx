'use client'

import { type ToolCallMessagePartProps } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type FormEvent, type KeyboardEvent, useCallback, useMemo, useRef, useState } from 'react'

import { ToolFallback } from '@/components/assistant-ui/tool-fallback'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Check, HelpCircle, Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $clarifyRequest, clearClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'

interface ClarifyArgs {
  question?: string
  choices?: string[] | null
}

function readClarifyArgs(args: unknown): ClarifyArgs {
  if (!args || typeof args !== 'object') {
    return {}
  }

  const row = args as Record<string, unknown>
  const choices = Array.isArray(row.choices) ? row.choices.filter((c): c is string => typeof c === 'string') : null

  return {
    question: typeof row.question === 'string' ? row.question : undefined,
    choices: choices && choices.length > 0 ? choices : null
  }
}

// Choice and "Other" rows share a layout; only color/hover differs.
const OPTION_ROW_CLASS = 'flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-sm transition-colors'

function RadioDot({ selected }: { selected: boolean }) {
  return (
    <span
      aria-hidden
      className={cn(
        'grid size-3.5 shrink-0 place-items-center rounded-full border transition-colors',
        selected ? 'border-primary' : 'border-muted-foreground/40'
      )}
    >
      {selected && <span className="size-1.5 rounded-full bg-primary" />}
    </span>
  )
}

export const ClarifyTool = (props: ToolCallMessagePartProps) => {
  const isPending = props.result === undefined

  // Once Hermes records an answer, fall back to the standard tool block so
  // the past Q/A renders consistently with every other tool in the thread.
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

  const [typing, setTyping] = useState(false)
  const [draft, setDraft] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [selectedChoice, setSelectedChoice] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  // Race: tool.start fires a tick before clarify.request, so request_id
  // arrives slightly after the tool block mounts. Show the question (from
  // args) but disable submit until we have the request id from the gateway.
  const ready = Boolean(matchingRequest?.requestId)

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
        // The matching tool.complete will land shortly after, swapping this
        // panel for the ToolFallback view above.
      } catch (error) {
        notifyError(error, copy.sendFailed)
        setSubmitting(false)
      }
    },
    [gateway, matchingRequest, ready]
  )

  const handleTextareaKey = useCallback(
    (event: KeyboardEvent<HTMLTextAreaElement>) => {
      if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault()
        const trimmed = draft.trim()

        if (trimmed) {
          void respond(trimmed)
        }
      }
    },
    [draft, respond]
  )

  const handleSubmitFreeform = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      const trimmed = draft.trim()

      if (trimmed) {
        void respond(trimmed)
      }
    },
    [draft, respond]
  )

  return (
    <div
      className="relative mb-3 mt-2 grid gap-6 rounded-[0.5rem] border border-border/70 bg-card/40 px-3 py-2.5 text-sm shadow-[inset_0_1px_0_color-mix(in_srgb,var(--foreground)_3%,transparent)]"
      data-slot="clarify-inline"
    >
      <span aria-hidden className="arc-border" />
      <div className="flex items-start gap-2.5">
        <span
          aria-hidden
          className="mt-px grid size-6 shrink-0 place-items-center rounded-md bg-[color-mix(in_srgb,var(--dt-primary)_14%,transparent)] text-primary ring-1 ring-inset ring-primary/15"
        >
          <HelpCircle className="size-3.5" />
        </span>
        <span className="flex-1 whitespace-pre-wrap font-medium leading-snug text-foreground">
          {question || <em className="font-normal text-muted-foreground/70">{copy.loadingQuestion}</em>}
        </span>
      </div>

      {!typing && hasChoices && (
        <div className="grid gap-0.5" role="group">
          {choices.map((choice, index) => (
            <button
              className={cn(
                OPTION_ROW_CLASS,
                'text-foreground/95 hover:bg-accent/60 disabled:cursor-not-allowed disabled:opacity-55',
                selectedChoice === choice && 'bg-accent/60'
              )}
              data-choice
              disabled={!ready || submitting}
              key={`${index}-${choice}`}
              onClick={() => {
                setSelectedChoice(choice)
                void respond(choice)
              }}
              type="button"
            >
              <RadioDot selected={selectedChoice === choice} />
              <span className="flex-1 wrap-anywhere">{choice}</span>
              {selectedChoice === choice && <Check aria-hidden className="size-4 shrink-0 text-primary" />}
            </button>
          ))}
          <button
            className={cn(OPTION_ROW_CLASS, 'text-muted-foreground hover:bg-accent/40 hover:text-foreground')}
            disabled={submitting}
            onClick={() => {
              setTyping(true)
              window.setTimeout(() => textareaRef.current?.focus({ preventScroll: true }), 0)
            }}
            type="button"
          >
            <RadioDot selected={false} />
            <span className="flex-1">{copy.other}</span>
          </button>
        </div>
      )}

      {(typing || !hasChoices) && (
        <form className="grid gap-2" onSubmit={handleSubmitFreeform}>
          <Textarea
            className="min-h-20 resize-y rounded-lg border-transparent bg-accent/40 text-sm focus-visible:bg-background/60"
            disabled={submitting}
            onChange={event => setDraft(event.target.value)}
            onKeyDown={handleTextareaKey}
            placeholder={copy.placeholder}
            ref={textareaRef}
            value={draft}
          />
          <div className="flex items-center justify-between gap-2">
            <span className="text-[0.6875rem] text-muted-foreground/85">{copy.shortcut}</span>
            <div className="flex items-center gap-1.5">
              {hasChoices && (
                <Button
                  disabled={submitting}
                  onClick={() => {
                    setTyping(false)
                    setDraft('')
                  }}
                  size="sm"
                  type="button"
                  variant="ghost"
                >
                  {copy.back}
                </Button>
              )}
              <Button
                disabled={!ready || submitting}
                onClick={() => void respond('')}
                size="sm"
                type="button"
                variant="ghost"
              >
                {copy.skip}
              </Button>
              <Button disabled={!ready || submitting || !draft.trim()} size="sm" type="submit">
                {submitting ? <Loader2 className="size-3.5 animate-spin" /> : copy.send}
              </Button>
            </div>
          </div>
        </form>
      )}

      {!typing && hasChoices && (
        <div className="flex justify-end">
          <Button
            className="-mr-2"
            disabled={!ready || submitting}
            onClick={() => void respond('')}
            size="xs"
            type="button"
            variant="text"
          >
            {copy.skip}
          </Button>
        </div>
      )}
    </div>
  )
}
