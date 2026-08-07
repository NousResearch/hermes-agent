import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { RefreshCw } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { CONTROL_TEXT } from './constants'

/**
 * One config row control for numeric ranges: a slider plus an editable text
 * box, with a live preview of what the value means in time terms. Dragging
 * the slider or typing a number both write straight through to `onChange`,
 * so the surrounding autosave flow persists it immediately.
 *
 * When `defaultValue` is provided, a reset button appears whenever the value
 * diverges from it — one click restores the shipped default.
 *
 * Schema fields consumed (see ConfigFieldSchema):
 *   min / max / step — slider bounds and granularity
 *   unit             — label shown next to the text box (e.g. "minutes")
 *   default          — the shipped default, target of the reset button
 */
export function RangeField({
  value,
  min = 0,
  max = 100,
  step = 1,
  unit,
  defaultValue,
  onChange
}: {
  value: unknown
  min?: number
  max?: number
  step?: number
  unit?: string
  defaultValue?: number
  onChange: (value: number) => void
}) {
  const numeric = typeof value === 'number' && Number.isFinite(value) ? value : min
  const [draft, setDraft] = useState<string>(String(numeric))

  // When the config value changes from outside (profile switch, save echo,
  // reset to defaults), re-sync the text box — but never while the user is
  // typing into it (draft would be clobbered mid-edit).
  const [focused, setFocused] = useState(false)
  useEffect(() => {
    if (!focused) {
      setDraft(String(numeric))
    }
  }, [numeric, focused])

  const clamped = Math.min(max, Math.max(min, numeric))
  const sliderValue = Math.round(clamped / step) * step

  const commit = (raw: string) => {
    setDraft(raw)
    const parsed = Number(raw)

    if (raw !== '' && Number.isFinite(parsed)) {
      onChange(Math.min(max, Math.max(min, parsed)))
    }
  }

  const hasDefault = typeof defaultValue === 'number' && Number.isFinite(defaultValue)
  const differsFromDefault = hasDefault && numeric !== defaultValue

  return (
    <div className="flex w-full flex-col items-end gap-1.5">
      <div className="flex w-full items-center justify-end gap-2">
        <input
          aria-label="slider"
          className={cn(
            'h-1.5 w-full max-w-44 cursor-pointer appearance-none rounded-full bg-(--ui-border)',
            'accent-(--ui-accent)'
          )}
          max={max}
          min={min}
          onChange={e => commit(e.target.value)}
          step={step}
          type="range"
          value={sliderValue}
        />
        <Input
          aria-label="value"
          className={cn('w-20 text-right', CONTROL_TEXT)}
          onBlur={() => setFocused(false)}
          onChange={e => commit(e.target.value)}
          onFocus={() => setFocused(true)}
          suffix={unit ? <span className="text-(--ui-text-tertiary)">{unit}</span> : undefined}
          type="text"
          value={draft}
        />
        {hasDefault && (
          <Button
            aria-label="reset to default"
            className="text-(--ui-text-tertiary)"
            disabled={!differsFromDefault}
            onClick={() => {
              setDraft(String(defaultValue))
              onChange(defaultValue)
            }}
            size="icon-xs"
            title="Reset to default"
            type="button"
            variant="ghost"
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </Button>
        )}
      </div>
      <div className="text-(--ui-text-tertiary) text-right text-[length:var(--conversation-caption-font-size)]">
        {formatTimeframe(clamped, unit)}
        {hasDefault && (
          <span className="ml-1">default {formatTimeframe(defaultValue, unit)}</span>
        )}
      </div>
    </div>
  )
}

/** Human-readable preview of the configured timeframe, e.g. "10 minutes". */
export function formatTimeframe(value: number, unit?: string): string {
  const n = Math.max(0, Math.round(value))

  if (unit === 'minutes') {
    if (n < 60) {
      return `${n} minute${n === 1 ? '' : 's'}`
    }

    const hours = n / 60
    const hoursLabel = Number.isInteger(hours) ? String(hours) : hours.toFixed(1)

    return `${hoursLabel} hour${hours === 1 ? '' : 's'} (${n} minutes)`
  }

  if (unit === 'seconds') {
    if (n < 60) {
      return `${n} second${n === 1 ? '' : 's'}`
    }

    const minutes = n / 60
    const minutesLabel = Number.isInteger(minutes) ? String(minutes) : minutes.toFixed(1)

    return `${minutesLabel} minute${minutes === 1 ? '' : 's'} (${n} seconds)`
  }

  if (unit === 'hours') {
    if (n < 24) {
      return `${n} hour${n === 1 ? '' : 's'}`
    }

    const days = n / 24
    const daysLabel = Number.isInteger(days) ? String(days) : days.toFixed(1)

    return `${daysLabel} day${days === 1 ? '' : 's'} (${n} hours)`
  }

  return unit ? `${n} ${unit}` : `${n}`
}
