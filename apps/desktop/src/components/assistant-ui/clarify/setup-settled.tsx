'use client'

import type { ToolCallMessagePartProps } from '@assistant-ui/react'
import { useStore } from '@nanostores/react'

import { useSessionView } from '@/app/chat/session-view'
import { ToolFallback } from '@/components/assistant-ui/tool/fallback'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { normalizeSetupChoose } from '@/store/clarify'

import { parseMaybeObject } from '../tool/fallback-model/format'

import { ClarifyShell } from './core/shell'
import { useSetupLabel } from './setup-rows'

const OUTCOMES = new Set(['cancelled', 'no_answer', 'submitted'])

function readSetupChooseResult(result: unknown) {
  const row = parseMaybeObject(result)

  if (typeof row.outcome !== 'string' || !OUTCOMES.has(row.outcome)) {
    return null
  }

  const picked = Array.isArray(row.picked) ? row.picked.map(String) : typeof row.picked === 'string' ? [row.picked] : []

  return { outcome: row.outcome, picked }
}

export function SetupChooseSettled(props: ToolCallMessagePartProps) {
  const { t } = useI18n()
  const copy = t.assistant.clarify
  const storedId = useStore(useSessionView().$storedId)
  const source = normalizeSetupChoose(parseMaybeObject(props.args))
  const result = readSetupChooseResult(props.result)
  const label = useSetupLabel(source?.setup?.kind ?? 'question', source?.setup?.options ?? null, storedId)

  if (!result) {
    return <ToolFallback {...props} />
  }

  const answer = result.picked.map(label).join(', ')

  const question = source?.questions[0]?.question

  return (
    <ClarifyShell className="my-1.5 grid gap-1" data-clarify-settled="">
      {question ? (
        <span className="whitespace-pre-wrap font-medium leading-(--conversation-line-height)">{question}</span>
      ) : null}
      <p
        className={cn(
          'whitespace-pre-wrap leading-(--conversation-line-height)',
          answer ? 'text-(--ui-text-secondary)' : 'italic text-(--ui-text-tertiary)'
        )}
        data-clarify-answer=""
      >
        {answer || (result.outcome === 'no_answer' ? copy.noAnswer : copy.skipped)}
      </p>
    </ClarifyShell>
  )
}
