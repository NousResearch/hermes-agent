import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'

/** A local draft only: Apply uses the same owner/rollback path as effort edits. */
export function ReasoningBudgetInput({
  bounds,
  effort,
  disabled,
  onApply
}: {
  bounds: { min: number; max: number }
  effort: string
  disabled?: boolean
  onApply: (effort: string) => void
}) {
  const { t } = useI18n()
  const [value, setValue] = useState(/^budget:\d+$/.test(effort) ? effort.slice(7) : '')
  const tokens = Number(value)
  const valid = /^\d+$/.test(value) && Number.isSafeInteger(tokens) && tokens >= bounds.min && tokens <= bounds.max

  return (
    <form
      className="space-y-2 px-2.5 py-2"
      onKeyDown={event => event.stopPropagation()}
      onSubmit={event => {
        event.preventDefault()
        if (valid && !disabled) onApply(`budget:${tokens}`)
      }}
    >
      <label className="block space-y-1 text-xs text-(--ui-text-secondary)">
        <span>{t.shell.modelOptions.thinkingBudget}</span>
        <Input
          disabled={disabled}
          max={bounds.max}
          min={bounds.min}
          onChange={event => setValue(event.target.value)}
          placeholder={`${bounds.min.toLocaleString()}–${bounds.max.toLocaleString()}`}
          step={1}
          type="number"
          value={value}
        />
      </label>
      <Button disabled={disabled || !valid} size="sm" type="submit" variant="secondary">
        {t.common.apply}
      </Button>
    </form>
  )
}
