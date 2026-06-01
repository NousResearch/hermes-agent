'use client'

import { type ToolCallMessagePartProps } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { type FormEvent, type KeyboardEvent, useCallback, useMemo, useRef, useState } from 'react'

import { ToolFallback } from '@/components/assistant-ui/tool-fallback'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { triggerHaptic } from '@/lib/haptics'
import { HelpCircle, Loader2, PencilLine } from '@/lib/icons'
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
  const textareaRef = useRef<HTMLTextAreaElement | null>(null)

  // Race: tool.start fires a tick before clarify.request, so request_id
  // arrives slightly after the tool block mounts. Show the question (from
  // args) but disable submit until we have the request id from the gateway.
  const ready = Boolean(matchingRequest?.requestId)

  const respond = useCallback(
    async (answer: string) => {
      if (!ready || !matchingRequest) {
        notifyError(new Error('Clarify request is not ready yet'), 'Could not send clarify response')

        return
      }

      if (!gateway) {
        notifyError(new Error('Hermes gateway is not connected'), 'Could not send clarify response')

        return
      }

      setSubmitting(true)

      try {
        await gateway.request<{ ok?: boolean }>('clarify.respond', {
          request_id: matchingRequest.requestId,
          answer
        })
        triggerHaptic('submit')
        clearClarifyRequest(matchingRequest.requestId)
        // The matching tool.complete will land shortly after, swapping this
        // panel for the ToolFallback view above.
      } catch (error) {
        notifyError(error, 'Could not send clarify response')
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

  const handleChoiceKey = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (typing || submitting) {
        return
      }

      const numeric = Number.parseInt(event.key, 10)

      if (Number.isFinite(numeric) && numeric >= 1 && numeric <= choices.length) {
        event.preventDefault()
        void respond(choices[numeric - 1]!)
      }
    },
    [choices, respond, submitting, typing]
  )

  return (
    <div
      className={cn(
        'mb-3 mt-2 grid gap-3 rounded-xl border border-border/70 bg-card/40 px-4 py-3.5 text-sm',
        'shadow-[inset_0_1px_0_color-mix(in_srgb,var(--foreground)_3%,transparent)]'
      )}
      data-slot="clarify-inline"
    >
      <div className="flex items-start gap-2.5">
        <span
          aria-hidden
          className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-md bg-[color-mix(in_srgb,var(--dt-primary)_14%,transparent)] text-primary ring-1 ring-inset ring-primary/15"
        >
          <HelpCircle className="size-3.5" />
        </span>
        <div className="grid flex-1 gap-0.5">
          <span className="text-[0.6875rem] font-medium uppercase tracking-wide text-muted-foreground/85">
            Hermes is asking
          </span>
          <span className="whitespace-pre-wrap leading-snug text-foreground">
            {question || <em className="text-muted-foreground/70">Loading question…</em>}
          </span>
        </div>
      </div>

      {!typing && hasChoices && (
        <div className="grid gap-1.5" onKeyDown={handleChoiceKey} role="group">
          {choices.map((choice, index) => (
            <button
              className={cn(
                'group/choice flex w-full items-center gap-3 rounded-lg border border-border/70 bg-background/60 px-3 py-2 text-left text-sm text-foreground/95',
                'transition-colors hover:border-border hover:bg-accent/60 disabled:cursor-not-allowed disabled:opacity-55'
              )}
              data-choice
              disabled={!ready || submitting}
              key={`${index}-${choice}`}
              onClick={() => void respond(choice)}
              type="button"
            >
              <span className="grid size-5 shrink-0 place-items-center rounded-md bg-muted text-[0.6875rem] font-mono tabular-nums text-muted-foreground group-hover/choice:bg-background">
                {index + 1}
              </span>
              <span className="flex-1 wrap-anywhere">{choice}</span>
            </button>
          ))}
          <button
            className={cn(
              'flex w-full items-center gap-3 rounded-lg border border-dashed border-border/60 bg-transparent px-3 py-2 text-left text-sm text-muted-foreground',
              'transition-colors hover:border-border hover:bg-accent/40 hover:text-foreground'
            )}
            disabled={submitting}
            onClick={() => {
              setTyping(true)
              window.setTimeout(() => textareaRef.current?.focus({ preventScroll: true }), 0)
            }}
            type="button"
          >
            <span
              aria-hidden
              className="grid size-5 shrink-0 place-items-center rounded-md bg-muted text-muted-foreground"
            >
              <PencilLine className="size-3" />
            </span>
            <span className="flex-1">Other (type your answer)</span>
          </button>
        </div>
      )}

      {(typing || !hasChoices) && (
        <form className="grid gap-2" onSubmit={handleSubmitFreeform}>
          <Textarea
            className="min-h-20 resize-y rounded-lg border-border/70 bg-background/60 text-sm"
            disabled={submitting}
            onChange={event => setDraft(event.target.value)}
            onKeyDown={handleTextareaKey}
            placeholder="Type your answer…"
            ref={textareaRef}
            value={draft}
          />
          <div className="flex items-center justify-between gap-2">
            <span className="text-[0.6875rem] text-muted-foreground/85">⌘/Ctrl + Enter to send</span>
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
                  Back
                </Button>
              )}
              <Button
                disabled={!ready || submitting}
                onClick={() => void respond('')}
                size="sm"
                type="button"
                variant="ghost"
              >
                Skip
              </Button>
              <Button disabled={!ready || submitting || !draft.trim()} size="sm" type="submit">
                {submitting ? <Loader2 className="size-3.5 animate-spin" /> : 'Send'}
              </Button>
            </div>
          </div>
        </form>
      )}

      {!typing && hasChoices && (
        <div className="flex items-center justify-between text-[0.6875rem] text-muted-foreground/85">
          <span>1–{choices.length} to pick</span>
          <button
            className="bg-transparent text-muted-foreground/85 underline-offset-4 decoration-current/20 hover:text-foreground hover:underline disabled:opacity-50"
            disabled={!ready || submitting}
            onClick={() => void respond('')}
            type="button"
          >
            Skip
          </button>
        </div>
      )}
    </div>
  )
}
