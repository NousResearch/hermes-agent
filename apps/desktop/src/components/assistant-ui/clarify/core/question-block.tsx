'use client'

import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import type { ClarifyQuestion } from '@/store/clarify'

import { ChoiceButton, KeyBadge, letterFor, OPTION_ROW_CLASS } from './choice-row'
import { CLARIFY_TEXTAREA_CLASS } from './shell'

interface QuestionBlockProps {
  cursor: null | number
  disabled: boolean
  onActivate: () => void
  onDraft: (value: string) => void
  onOtherFocus: () => void
  onPick: (index: number) => void
  question: ClarifyQuestion
  staged: { choices: string[]; draft: string }
}

/** One question's interactive block inside the live batch card. */
export function QuestionBlock({
  cursor,
  disabled,
  onActivate,
  onDraft,
  onOtherFocus,
  onPick,
  question,
  staged
}: QuestionBlockProps) {
  const { t } = useI18n()
  const copy = t.assistant.clarify
  const choices = question.choices ?? []
  const otherActive = cursor === choices.length

  return (
    <div
      className="grid gap-1"
      data-clarify-batch-question={question.qid}
      onFocus={onActivate}
      onPointerDown={onActivate}
    >
      <div className="flex items-start gap-2">
        <span
          className="flex-1 whitespace-pre-wrap text-start font-medium leading-(--conversation-line-height)"
          dir="auto"
        >
          {question.question}
        </span>
      </div>

      {choices.length > 0 ? (
        <div className="grid gap-px" role="group">
          {choices.map((choice, index) => (
            <ChoiceButton
              active={cursor === index}
              char={letterFor(index)}
              choice={choice}
              disabled={disabled}
              key={`${index}-${choice}`}
              keyShortcuts={cursor === null ? undefined : `${letterFor(index)} ${index + 1}`}
              onClick={() => onPick(index)}
              selected={staged.choices.includes(choice)}
            />
          ))}
          <label
            className={cn(OPTION_ROW_CLASS, 'items-center', otherActive && 'bg-(--chrome-action-hover)')}
            data-highlighted={otherActive || undefined}
          >
            <KeyBadge
              char={letterFor(choices.length)}
              disabled={disabled}
              preview={otherActive}
              selected={Boolean(staged.draft.trim())}
            />
            <Textarea
              aria-current={otherActive || undefined}
              aria-keyshortcuts={cursor === null ? undefined : `${letterFor(choices.length)} ${choices.length + 1}`}
              className={CLARIFY_TEXTAREA_CLASS}
              dir="auto"
              disabled={disabled}
              onChange={event => onDraft(event.target.value)}
              onFocus={onOtherFocus}
              placeholder={copy.other}
              rows={1}
              size="sm"
              value={staged.draft}
            />
          </label>
        </div>
      ) : (
        <Textarea
          className={CLARIFY_TEXTAREA_CLASS}
          dir="auto"
          disabled={disabled}
          onChange={event => onDraft(event.target.value)}
          onFocus={onOtherFocus}
          placeholder={copy.placeholder}
          rows={1}
          size="sm"
          value={staged.draft}
        />
      )}
    </div>
  )
}

export const emptyStage = { choices: [] as string[], draft: '' }
