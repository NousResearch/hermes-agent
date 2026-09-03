import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import {
  $promptCoachPreviewBySession,
  allowPromptCoachOriginal,
  closePromptCoachPreview,
  type PromptCoachPreview,
  recordPromptCoachAction
} from '@/store/prompt-coach'

interface PromptCoachPanelProps {
  onApply: (text: string) => void
  onSendOriginal: () => void
  sessionId: null | string
}

export function PromptCoachPanel({ onApply, onSendOriginal, sessionId }: PromptCoachPanelProps) {
  const previews = useStore($promptCoachPreviewBySession)
  const preview = previews[sessionId ?? '']

  if (!preview) {
    return null
  }

  return (
    <PromptCoachPreviewCard
      key={`${sessionId ?? ''}:${preview.original}`}
      onApply={onApply}
      onClose={() => closePromptCoachPreview(sessionId)}
      onSendOriginal={() => {
        allowPromptCoachOriginal(sessionId, preview.original)
        onSendOriginal()
      }}
      preview={preview}
    />
  )
}

export function PromptCoachPreviewCard({
  className,
  onApply,
  onClose,
  onSendOriginal,
  preview
}: {
  className?: string
  onApply: (text: string) => void
  onClose: () => void
  onSendOriginal: () => void
  preview: PromptCoachPreview
}) {
  const [editing, setEditing] = useState(false)
  const [edited, setEdited] = useState(preview.suggestedPrompt)

  useEffect(() => setEdited(preview.suggestedPrompt), [preview.suggestedPrompt])

  const apply = (action: 'edited' | 'replaced') => {
    recordPromptCoachAction(action)
    onApply(edited)
    onClose()
  }

  return (
    <section
      aria-label="Prompt Coach preview"
      className={cn(
        'mx-[5px] mb-1.5 w-[min(42rem,calc(100vw-1.5rem))] rounded-xl border border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) p-3 shadow-xl',
        className
      )}
      data-slot="prompt-coach-preview"
    >
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold">Prompt Coach</p>
          <p className="mt-0.5 text-xs text-(--ui-text-tertiary)">
            {preview.reason}. Nothing changes until you choose.
          </p>
          <p className="mt-1 text-[10px] text-(--ui-text-tertiary)">
            {preview.generatedBy === 'ai'
              ? 'Powered by the active Hermes AI model · wording preserved'
              : preview.generatedBy === 'pending'
                ? 'Hermes AI is tailoring the missing-information questions…'
                : 'Local fallback · wording preserved'}
          </p>
        </div>
        <span className="rounded-full bg-(--ui-bg-tertiary) px-2 py-1 text-[10px] text-(--ui-text-secondary)">
          Readiness {preview.score}%
        </span>
      </div>

      {preview.hasPotentialSecret && (
        <p className="mt-2 rounded-lg border border-amber-500/35 bg-amber-500/10 px-2.5 py-2 text-xs text-amber-300">
          Possible secret detected. The suggested copy redacts it; use the secure credential flow instead.
        </p>
      )}

      <div className="mt-3 grid gap-2 md:grid-cols-2">
        <div className="min-w-0">
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-(--ui-text-tertiary)">Original</p>
          <pre className="max-h-44 overflow-auto whitespace-pre-wrap rounded-lg bg-(--ui-bg-secondary) p-2 text-xs text-(--ui-text-secondary)">
            {preview.original}
          </pre>
        </div>
        <div className="min-w-0">
          <p className="mb-1 text-[10px] font-semibold uppercase tracking-wide text-(--ui-text-tertiary)">Suggested</p>
          {editing ? (
            <textarea
              aria-label="Edit improved prompt"
              autoFocus
              className="min-h-44 w-full resize-y rounded-lg border border-(--ui-stroke-secondary) bg-(--ui-bg-secondary) p-2 text-xs outline-none focus:border-primary"
              onChange={event => setEdited(event.target.value)}
              value={edited}
            />
          ) : (
            <pre className="max-h-44 overflow-auto whitespace-pre-wrap rounded-lg bg-(--ui-bg-secondary) p-2 text-xs">
              {edited}
            </pre>
          )}
        </div>
      </div>

      <div className="mt-3 flex flex-wrap justify-end gap-2">
        <Button
          onClick={() => {
            recordPromptCoachAction('dismissed')
            onClose()
          }}
          size="sm"
          type="button"
          variant="ghost"
        >
          Dismiss
        </Button>
        <Button
          onClick={() => {
            recordPromptCoachAction('sent-original')
            onClose()
            onSendOriginal()
          }}
          size="sm"
          type="button"
          variant="outline"
        >
          Send original
        </Button>
        {editing ? (
          <Button onClick={() => apply('edited')} size="sm" type="button">
            Apply edited
          </Button>
        ) : (
          <>
            <Button onClick={() => setEditing(true)} size="sm" type="button" variant="outline">
              Edit
            </Button>
            <Button onClick={() => apply('replaced')} size="sm" type="button">
              Replace
            </Button>
          </>
        )}
      </div>
    </section>
  )
}
