import type { ReactNode } from 'react'
import { useEffect, useState } from 'react'

import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { prettyName } from '@/lib/text'
import { cn } from '@/lib/utils'
import type { ConfigFieldSchema } from '@/types/hermes'

import { ComboboxInput } from './combobox-input'
import {
  CONTROL_TEXT,
  EMPTY_SELECT_VALUE,
  FIELD_DESCRIPTIONS,
  FIELD_LABELS,
  FREE_INPUT_KEYS,
  NUMERIC_FIELD_CONFIG
} from './constants'
import { FallbackModelsField } from './fallback-models-field'
import { fieldCopyForSchemaKey } from './field-copy'
import { ListRow } from './primitives'
import { SearchableSelect } from './searchable-select'

export function roundToStepPrecision(n: number, step?: number): number {
  if (step !== undefined && step > 0) {
    const stepStr = step.toString()

    if (stepStr.includes('.')) {
      const decimals = Math.min(10, stepStr.split('.')[1]?.length ?? 2)

      return Number(n.toFixed(decimals))
    }
  }

  return n
}

export function NumberConfigInput({
  value,
  min,
  max,
  step,
  placeholder,
  className,
  onChange
}: {
  value: unknown
  min?: number
  max?: number
  step?: number
  placeholder?: string
  className?: string
  onChange: (value: number) => void
}) {
  const [draft, setDraft] = useState<string>(() => (value === undefined || value === null ? '' : String(value)))

  const [isFocused, setIsFocused] = useState(false)

  useEffect(() => {
    if (!isFocused) {
      setDraft(value === undefined || value === null ? '' : String(value))
    }
  }, [value, isFocused])

  const clampValue = (val: string): number => {
    const trimmed = val.trim()

    if (trimmed === '') {
      return min !== undefined && min > 0 ? min : 0
    }

    let n = Number(trimmed)

    if (Number.isNaN(n)) {
      return min !== undefined && min > 0 ? min : 0
    }

    if (min !== undefined && n < min) {
      n = min
    } else if (max !== undefined && n > max) {
      n = max
    }

    return roundToStepPrecision(n, step)
  }

  const commit = (rawStr: string) => {
    const clamped = clampValue(rawStr)

    setDraft(String(clamped))
    onChange(clamped)
  }

  return (
    <Input
      className={className}
      max={max}
      min={min}
      onBlur={() => {
        setIsFocused(false)
        commit(draft)
      }}
      onChange={e => {
        const raw = e.target.value

        setDraft(raw)

        if (raw === '') {
          return
        }

        const n = Number(raw)

        if (!Number.isNaN(n)) {
          if ((min === undefined || n >= min) && (max === undefined || n <= max)) {
            const rounded = roundToStepPrecision(n, step)

            onChange(rounded)
          }
        }
      }}
      onFocus={() => setIsFocused(true)}
      onKeyDown={e => {
        if (e.key === 'Enter') {
          commit(draft)
          e.currentTarget.blur()
        }
      }}
      placeholder={placeholder}
      step={step}
      type="number"
      value={draft}
    />
  )
}

/**
 * One generic config row: label + description resolved from the i18n field
 * copy (falling back to the schema description), and a control picked from the
 * field schema — Switch for booleans, Select for enums, free-input combobox
 * (Input + datalist) for FREE_INPUT_KEYS voice/model names, and Input/Textarea
 * for the rest. Shared by the Settings config sections and the Capabilities
 * TTS provider panel so both surfaces render identical fields.
 */
export function ConfigField({
  schemaKey,
  schema,
  value,
  enumOptions,
  optionLabels,
  onChange,
  descriptionExtra
}: {
  schemaKey: string
  schema: ConfigFieldSchema
  value: unknown
  enumOptions?: string[]
  optionLabels?: Record<string, string>
  onChange: (value: unknown) => void
  descriptionExtra?: ReactNode
}) {
  const { t } = useI18n()
  const c = t.settings.config

  const label =
    fieldCopyForSchemaKey(t.settings.fieldLabels, schemaKey) ??
    fieldCopyForSchemaKey(FIELD_LABELS, schemaKey) ??
    prettyName(schemaKey.split('.').pop() ?? schemaKey)

  const normalize = (v: string) => v.toLowerCase().replace(/[^a-z0-9]+/g, '')

  const rawDescription = (
    fieldCopyForSchemaKey(t.settings.fieldDescriptions, schemaKey) ??
    fieldCopyForSchemaKey(FIELD_DESCRIPTIONS, schemaKey) ??
    schema.description ??
    ''
  ).trim()

  const normalizedDesc = normalize(rawDescription)

  const description =
    rawDescription && normalizedDesc !== normalize(label) && normalizedDesc !== normalize(schemaKey)
      ? rawDescription
      : undefined

  const descriptionNode: ReactNode = descriptionExtra ? (
    <span className="inline-flex flex-wrap items-center gap-x-3 gap-y-1">
      {description}
      {descriptionExtra}
    </span>
  ) : (
    description
  )

  // Every config row is addressable by its canonical schema key, so a tour can
  // point at one setting (`[data-tour="field-model"]`) without hunting through
  // the section for an nth-child path. See lib/tour.
  const row = (action: ReactNode, wide = false) => (
    <ListRow action={action} data-tour={`field-${schemaKey}`} description={descriptionNode} title={label} wide={wide} />
  )

  // `fallback_providers` is a list of {provider, model} objects; the generic
  // `list` branch below would stringify them to "[object Object]". Render the
  // dedicated structured editor instead.
  if (schemaKey === 'fallback_providers') {
    return row(<FallbackModelsField onChange={onChange} value={value} />, true)
  }

  if (schema.type === 'boolean') {
    return row(
      <div className="flex items-center justify-end">
        <Switch checked={Boolean(value)} onCheckedChange={onChange} />
      </div>
    )
  }

  const selectOptions = enumOptions ?? (schema.type === 'select' ? (schema.options ?? []).map(String) : undefined)

  // Large closed-world lists (e.g. ~590 IANA timezones) get a searchable
  // Popover + cmdk combobox instead of a closed Select dropdown.  The schema
  // opt-in via `searchable: true` keeps this deterministic — no field
  // accidentally triggers based on dynamic option count.
  if (selectOptions && schema.searchable) {
    return row(
      <SearchableSelect
        clearLabel={schema.clearable ? c.systemDefault : undefined}
        emptyMessage={c.noResults}
        onChange={next => onChange(next)}
        options={selectOptions.filter(o => o !== '')}
        placeholder={c.searchPlaceholder}
        value={String(value ?? '')}
      />
    )
  }

  // Voice/model name fields are open-world (custom voice IDs, cloned voices,
  // brand-new model names) — render a free-input combobox where the known
  // options are dropdown suggestions instead of a closed Select gate. The old
  // native <datalist> filtered by the current value, so a field already set
  // to a valid option showed only that single suggestion.
  if (selectOptions && FREE_INPUT_KEYS.has(schemaKey)) {
    return row(
      <ComboboxInput
        className={CONTROL_TEXT}
        onChange={onChange}
        optionLabels={optionLabels}
        options={selectOptions.filter(o => o !== '')}
        placeholder={c.notSet}
        value={String(value ?? '')}
      />
    )
  }

  if (selectOptions) {
    return row(
      <Select
        onValueChange={next => onChange(next === EMPTY_SELECT_VALUE ? '' : next)}
        value={String(value ?? '') || EMPTY_SELECT_VALUE}
      >
        <SelectTrigger className={CONTROL_TEXT}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {selectOptions.map(option => (
            <SelectItem key={option || EMPTY_SELECT_VALUE} value={option || EMPTY_SELECT_VALUE}>
              {option
                ? (optionLabels?.[option] ?? prettyName(option))
                : schemaKey === 'display.personality'
                  ? c.none
                  : schemaKey === 'memory.provider'
                    ? c.builtinOnly
                    : c.noneParen}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    )
  }

  if (schema.type === 'number') {
    const fieldConfig = NUMERIC_FIELD_CONFIG[schemaKey]
    const min = schema.min ?? fieldConfig?.min
    const max = schema.max ?? fieldConfig?.max

    const step =
      schema.step ??
      fieldConfig?.step ??
      ((min !== undefined && min % 1 !== 0) || (max !== undefined && max % 1 !== 0) ? 0.05 : 1)

    return row(
      <NumberConfigInput
        className={CONTROL_TEXT}
        max={max}
        min={min}
        onChange={onChange}
        placeholder={c.notSet}
        step={step}
        value={value}
      />
    )
  }

  if (schema.type === 'list') {
    return row(
      <Input
        className={CONTROL_TEXT}
        onChange={e =>
          onChange(
            e.target.value
              .split(',')
              .map(s => s.trim())
              .filter(Boolean)
          )
        }
        placeholder={c.commaSeparated}
        value={Array.isArray(value) ? value.join(', ') : String(value ?? '')}
      />
    )
  }

  if (typeof value === 'object' && value !== null) {
    return row(
      <Textarea
        className={cn('min-h-28 resize-y bg-background font-mono', CONTROL_TEXT)}
        onChange={e => {
          try {
            onChange(JSON.parse(e.target.value))
          } catch {
            /* keep last valid */
          }
        }}
        placeholder={c.notSet}
        spellCheck={false}
        value={JSON.stringify(value, null, 2)}
      />,
      true
    )
  }

  const isLong = schema.type === 'text' || String(value ?? '').length > 100

  return row(
    isLong ? (
      <Textarea
        className={cn('min-h-24 resize-y bg-background', CONTROL_TEXT)}
        onChange={e => onChange(e.target.value)}
        placeholder={c.notSet}
        value={String(value ?? '')}
      />
    ) : (
      <Input
        className={CONTROL_TEXT}
        onChange={e => onChange(e.target.value)}
        placeholder={c.notSet}
        value={String(value ?? '')}
      />
    ),
    isLong
  )
}
