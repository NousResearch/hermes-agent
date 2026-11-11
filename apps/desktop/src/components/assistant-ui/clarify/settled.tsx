'use client'

import type { ToolCallMessagePartProps } from '@assistant-ui/react'

import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { useI18n } from '@/i18n'
import { CircleLetterA, MessageQuestion } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { ClarifyLine, ClarifyShell } from './core/shell'
import { readClarifyResult } from './parse'

/** Settled batch card: every question with its locked (or absent) answer. */
export function ClarifyToolSettled(props: ToolCallMessagePartProps) {
  const { t } = useI18n()
  const copy = t.assistant.clarify
  const { outcome, responses } = readClarifyResult(props.result)
  // Skip (the card's or the composer's) and a stopped turn both end the batch
  // as `cancelled`: the user chose not to answer, so it reads as skipped.
  const cancelled = outcome === 'cancelled'

  if (responses.length === 0) {
    return <ToolFallback {...props} />
  }

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
                {blank ? (row.unanswered && !cancelled ? copy.noAnswer : copy.skipped) : answer}
              </p>
            </ClarifyLine>
          </div>
        )
      })}
    </ClarifyShell>
  )
}
