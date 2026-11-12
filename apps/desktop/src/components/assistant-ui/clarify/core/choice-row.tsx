'use client'

import { Kbd } from '@/components/ui/kbd'
import { cn } from '@/lib/utils'
import { bareChoice, RECOMMENDED_LABEL } from '@/store/clarify'

export const letterFor = (index: number): string => String.fromCharCode(65 + index)

// The backend tags the agent's preferred option (`mark_recommended`); the card
// renders the label in tertiary text so the option itself still reads first.
export function ChoiceLabel({ choice }: { choice: string }) {
  const bare = bareChoice(choice)

  if (bare === choice) {
    return <>{choice}</>
  }

  return (
    <>
      {bare} <span className="text-(--ui-text-tertiary)">{RECOMMENDED_LABEL}</span>
    </>
  )
}

export const OPTION_ROW_CLASS =
  'flex w-full items-start gap-2 rounded-[0.25rem] px-1.5 py-1 text-left disabled:cursor-not-allowed disabled:opacity-50'

export function KeyBadge({
  char,
  disabled,
  preview,
  selected
}: {
  char: string
  disabled?: boolean
  preview?: boolean
  selected: boolean
}) {
  // The "Other" row is a <label>, which has no :disabled state of its own —
  // dim its badge alongside the disabled textarea so it matches the options.
  return (
    <Kbd
      className={cn(
        'mt-px',
        disabled && 'opacity-50',
        selected && 'border-primary bg-primary text-white shadow-none',
        !selected && preview && 'border-primary text-primary shadow-none'
      )}
      size="sm"
    >
      {char}
    </Kbd>
  )
}

/** A letter-badged option row. */
export function ChoiceButton({
  active = false,
  char,
  choice,
  disabled,
  keyShortcuts,
  onClick,
  selected
}: {
  active?: boolean
  char: string
  choice: string
  disabled?: boolean
  keyShortcuts?: string
  onClick: () => void
  selected?: boolean
}) {
  // `active` is the keyboard cursor on the live card (arrow-key navigation);
  // it highlights the row and previews its key badge.
  return (
    <button
      aria-current={active || undefined}
      aria-keyshortcuts={keyShortcuts}
      aria-pressed={selected}
      className={cn(
        OPTION_ROW_CLASS,
        'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-(--ui-text-primary)',
        active && 'bg-(--chrome-action-hover) text-(--ui-text-primary)',
        selected && 'text-(--ui-text-primary)'
      )}
      data-choice
      data-highlighted={active || undefined}
      disabled={disabled}
      onClick={onClick}
      type="button"
    >
      <KeyBadge char={char} preview={active} selected={Boolean(selected)} />
      <span className="flex-1 wrap-anywhere">
        <ChoiceLabel choice={choice} />
      </span>
    </button>
  )
}
