import { Button } from '@hermes/plugin-sdk'
import { useState } from 'react'

import type { DecomposeOutcome, SpecifyOutcome } from './api'

export interface TriageActionLabels {
  specify: string
  specifying: string
  specified: string
  specifiedRetitled: (title: string) => string
  decompose: string
  decomposing: string
  decomposed: (count: number) => string
  decomposedSingle: string
  failed: (action: string, reason: string) => string
  unknownError: string
}

interface TriageActionsProps {
  labels: TriageActionLabels
  onDecompose: () => Promise<DecomposeOutcome>
  onRefresh: () => void
  onSpecify: () => Promise<SpecifyOutcome>
  status: string
}

type Action = 'decompose' | 'specify'
type Feedback = null | { kind: 'error' | 'success'; message: string }

function errorText(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : String(error || fallback)
}

export function TriageActions({ labels, onDecompose, onRefresh, onSpecify, status }: TriageActionsProps) {
  const [completed, setCompleted] = useState(false)
  const [pending, setPending] = useState<null | Action>(null)
  const [feedback, setFeedback] = useState<Feedback>(null)

  if (status !== 'triage') {
    return null
  }

  const run = async (action: Action) => {
    if (completed || pending) {
      return
    }

    setPending(action)
    setFeedback(null)

    try {
      const outcome = action === 'specify' ? await onSpecify() : await onDecompose()

      if (!outcome.ok) {
        setFeedback({
          kind: 'error',
          message: labels.failed(
            action === 'specify' ? labels.specify : labels.decompose,
            outcome.reason || labels.unknownError
          )
        })

        return
      }

      let message: string

      if (action === 'specify') {
        message = outcome.new_title ? labels.specifiedRetitled(outcome.new_title) : labels.specified
      } else {
        const decomposed = outcome as DecomposeOutcome
        message = decomposed.fanout ? labels.decomposed(decomposed.child_ids.length) : labels.decomposedSingle
      }

      setCompleted(true)
      setFeedback({ kind: 'success', message })
      onRefresh()
    } catch (error) {
      setFeedback({
        kind: 'error',
        message: labels.failed(
          action === 'specify' ? labels.specify : labels.decompose,
          errorText(error, labels.unknownError)
        )
      })
    } finally {
      setPending(null)
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap gap-2">
        <Button
          aria-label={pending === 'specify' ? labels.specifying : labels.specify}
          disabled={completed || pending !== null}
          onClick={() => void run('specify')}
          size="sm"
          variant="secondary"
        >
          {pending === 'specify' ? labels.specifying : `✨ ${labels.specify}`}
        </Button>
        <Button
          aria-label={pending === 'decompose' ? labels.decomposing : labels.decompose}
          disabled={completed || pending !== null}
          onClick={() => void run('decompose')}
          size="sm"
          variant="secondary"
        >
          {pending === 'decompose' ? labels.decomposing : `⚗ ${labels.decompose}`}
        </Button>
      </div>
      {feedback && (
        <p
          aria-live="polite"
          className={
            feedback.kind === 'error' ? 'text-[0.71rem] text-destructive' : 'text-[0.71rem] text-(--ui-text-secondary)'
          }
        >
          {feedback.message}
        </p>
      )}
    </div>
  )
}
