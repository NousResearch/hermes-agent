import { useId, useState } from 'react'

import { controlVariants } from '@/components/ui/control'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import type { StepId } from '../flow'

import { AnswerMark } from './motion'

const PILL_CLASS =
  'relative flex max-w-full shrink-0 items-center gap-1.5 rounded-full border px-3 py-1 text-left text-[12px] whitespace-normal wrap-anywhere transition-colors'

export interface PillChoice {
  detail?: string
  id: string
  label: string
}

/**
 * One question answered with a pill, plus an optional free-text "Other" pill (name and task only).
 * A pill with a detail line shows it on hover or focus, the way the clarify card does.
 */
export function QuestionPills({
  choices,
  onOther,
  onPick,
  other,
  otherPlaceholder,
  picked,
  question,
  stepId
}: {
  choices: PillChoice[]
  onOther?: (value: string) => void
  onPick: (id: string) => void
  /** The typed answer; `undefined` hides the Other pill. */
  other?: string
  /** With no pills to pick, Other is the whole answer: a plain field with this placeholder. */
  otherPlaceholder?: string
  picked: null | string
  question: string
  stepId: StepId
}) {
  const detailId = useId()
  const [focused, setFocused] = useState<null | string>(null)
  const detail = choices.find(choice => choice.id === focused)?.detail
  const typed = Boolean(other?.trim())

  const standalone = choices.length === 0

  return (
    <div className="grid gap-2">
      <span className="font-medium leading-(--conversation-line-height)">{question}</span>
      <div className={cn('flex min-w-0 flex-wrap gap-2', !standalone && 'p-1')} role="group">
        {choices.map(choice => {
          const selected = picked === choice.id && !typed

          return (
            <button
              aria-describedby={focused === choice.id && choice.detail ? detailId : undefined}
              aria-pressed={selected}
              className={cn(
                PILL_CLASS,
                selected
                  ? 'border-primary bg-primary text-primary-foreground'
                  : 'border-border bg-card hover:border-primary/50 hover:bg-primary/10'
              )}
              key={choice.id}
              onBlur={() => setFocused(null)}
              onClick={() => onPick(choice.id)}
              onFocus={() => setFocused(choice.id)}
              onPointerEnter={() => setFocused(choice.id)}
              onPointerLeave={() => setFocused(null)}
              type="button"
            >
              <span>{choice.label}</span>
              {choice.detail ? (
                <span aria-hidden className="size-1 shrink-0 rounded-full bg-current opacity-60" />
              ) : null}
              {selected ? <AnswerMark stepId={stepId} /> : null}
            </button>
          )
        })}
        {other !== undefined && onOther ? (
          <OtherField
            onChange={onOther}
            placeholder={otherPlaceholder}
            standalone={standalone}
            stepId={stepId}
            value={other}
          />
        ) : null}
      </div>
      {choices.some(choice => choice.detail) ? (
        <p className="h-4 truncate px-1 text-xs leading-4 text-(--ui-text-tertiary)" id={detailId}>
          {detail}
        </p>
      ) : null}
    </div>
  )
}

/**
 * The typed answer: an "Other" pill beside the choices, or the whole answer as a plain field when
 * there are none. One input either way, so the name field keeps focus when the OS name arrives.
 */
function OtherField({
  onChange,
  placeholder,
  standalone,
  stepId,
  value
}: {
  onChange: (value: string) => void
  placeholder?: string
  standalone: boolean
  stepId: StepId
  value: string
}) {
  const { t } = useI18n()
  const typed = Boolean(value.trim())
  const label = standalone ? (placeholder ?? t.assistant.clarify.other) : t.questionnaire.otherLabel

  return (
    <label
      className={
        standalone
          ? 'contents'
          : cn(PILL_CLASS, 'cursor-text', typed ? 'border-primary bg-primary/10' : 'border-border bg-card')
      }
    >
      <input
        aria-label={label}
        autoCapitalize={standalone ? 'words' : undefined}
        autoFocus={standalone}
        className={
          standalone
            ? cn(controlVariants(), 'max-w-72')
            : 'min-w-8 bg-transparent leading-5 outline-none [field-sizing:content] placeholder:text-muted-foreground focus:min-w-48'
        }
        onChange={event => onChange(event.target.value)}
        placeholder={standalone ? label : t.assistant.clarify.other}
        spellCheck={!standalone}
        value={value}
      />
      {typed && !standalone ? <AnswerMark stepId={stepId} /> : null}
    </label>
  )
}
