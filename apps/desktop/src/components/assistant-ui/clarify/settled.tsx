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
  const { responses } = readClarifyResult(props.result)

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
                <span
                  className="block whitespace-pre-wrap text-start font-medium leading-(--conversation-line-height)"
                  dir="auto"
                >
                  {row.question}
                </span>
              </ClarifyLine>
            ) : null}
            <ClarifyLine icon={CircleLetterA}>
              <p
                className={cn(
                  'whitespace-pre-wrap text-start leading-(--conversation-line-height)',
                  blank ? 'italic text-(--ui-text-tertiary)' : 'text-(--ui-text-secondary)'
                )}
                data-clarify-answer=""
                dir="auto"
              >
                {blank ? (row.unanswered ? copy.noAnswer : copy.skipped) : answer}
              </p>
            </ClarifyLine>
          </div>
        )
      })}
    </ClarifyShell>
  )
}
