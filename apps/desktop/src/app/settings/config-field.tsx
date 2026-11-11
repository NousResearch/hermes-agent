import { type ReactNode, useEffect, useRef, useState } from 'react'

import { Input } from '@/components/ui/input'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { prettyName } from '@/lib/text'
import { cn } from '@/lib/utils'
import type { ConfigFieldSchema } from '@/types/hermes'

import { ComboboxInput } from './combobox-input'
import { CONTROL_TEXT, EMPTY_SELECT_VALUE, FIELD_DESCRIPTIONS, FIELD_LABELS, FREE_INPUT_KEYS } from './constants'
import { FallbackModelsField } from './fallback-models-field'
import { fieldCopyForSchemaKey } from './field-copy'
import { ListRow, ToggleRow } from './primitives'
import { SearchableSelect } from './searchable-select'

interface ConfigFieldProps {
  schemaKey: string
  schema: ConfigFieldSchema
  value: unknown
  enumOptions?: string[]
  optionLabels?: Record<string, string>
  onChange: (value: unknown) => void
  descriptionExtra?: ReactNode
  descriptionOverride?: string
  segmented?: boolean
}

export function parseListFieldDraft(raw: string): string[] {
  return raw
    .split(',')
    .map(item => item.trim())
    .filter(Boolean)
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
  descriptionExtra,
  descriptionOverride,
  segmented = false
}: ConfigFieldProps) {
  const { t } = useI18n()
  const c = t.settings.config

  const label =
    fieldCopyForSchemaKey(t.settings.fieldLabels, schemaKey) ??
    fieldCopyForSchemaKey(FIELD_LABELS, schemaKey) ??
    prettyName(schemaKey.split('.').pop() ?? schemaKey)

  const normalize = (v: string) =>
    v
      .toLowerCase()
      .normalize('NFC')
      .replace(/[^\p{L}\p{M}\p{N}]+/gu, '')

  const rawDescription = (
    descriptionOverride ??
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
  const dataTour = `field-${schemaKey}`

  const row = (action: ReactNode) => (
    <ListRow action={action} data-tour={dataTour} description={descriptionNode} title={label} />
  )

  // Editors too big for the control column (textareas, structured lists) take
  // the full width under the description.
  const wideRow = (editor: ReactNode) => (
    <ListRow
      below={<div className="mt-3">{editor}</div>}
      data-tour={dataTour}
      description={descriptionNode}
      title={label}
      wide
    />
  )

  // `fallback_providers` is a list of {provider, model} objects; the generic
  // `list` branch below would stringify them to "[object Object]". Render the
  // dedicated structured editor instead.
  if (schemaKey === 'fallback_providers') {
    return wideRow(<FallbackModelsField onChange={onChange} value={value} />)
  }

  if (schema.type === 'boolean') {
    return (
      <ToggleRow
        checked={Boolean(value)}
        data-tour={dataTour}
        description={descriptionNode}
        label={label}
        onChange={onChange}
      />
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
      <EnumControl
        onChange={onChange}
        optionLabels={optionLabels}
        options={selectOptions}
        schemaKey={schemaKey}
        segmented={segmented}
        value={value}
      />
    )
  }

  if (schema.type === 'number') {
    return row(
      <Input
        className={CONTROL_TEXT}
        onChange={e => {
          const raw = e.target.value
          const n = raw === '' ? 0 : Number(raw)

          if (!Number.isNaN(n)) {
            onChange(n)
          }
        }}
        placeholder={c.notSet}
        type="number"
        value={value === undefined || value === null ? '' : String(value)}
      />
    )
  }

  if (schema.type === 'list') {
    return row(<ListField onChange={onChange} placeholder={c.commaSeparated} value={value} />)
  }

  if (typeof value === 'object' && value !== null) {
    return wideRow(
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
      />
    )
  }

  const isLong = schema.type === 'text' || String(value ?? '').length > 100

  return isLong
    ? wideRow(
        <Textarea
          className={cn('min-h-24 resize-y bg-background', CONTROL_TEXT)}
          onChange={e => onChange(e.target.value)}
          placeholder={c.notSet}
          value={String(value ?? '')}
        />
      )
    : row(
        <Input
          className={CONTROL_TEXT}
          onChange={e => onChange(e.target.value)}
          placeholder={c.notSet}
          value={String(value ?? '')}
        />
      )
}

function ListField({
  value,
  onChange,
  placeholder
}: {
  value: unknown
  onChange: (value: unknown) => void
  placeholder: string
}) {
  const normalizedValue = Array.isArray(value) ? value.join(', ') : String(value ?? '')
  const [draft, setDraft] = useState(normalizedValue)
  const focusedRef = useRef(false)

  useEffect(() => {
    if (!focusedRef.current) {
      setDraft(normalizedValue)
    }
  }, [normalizedValue])

  const commitDraft = () => {
    const parsed = parseListFieldDraft(draft)
    setDraft(parsed.join(', '))
    onChange(parsed)
  }

  return (
    <Input
      className={CONTROL_TEXT}
      onBlur={() => {
        focusedRef.current = false
        commitDraft()
      }}
      onChange={e => setDraft(e.target.value)}
      onFocus={() => {
        focusedRef.current = true
      }}
      onKeyDown={e => {
        if (e.key === 'Enter') {
          e.currentTarget.blur()
        }
      }}
      placeholder={placeholder}
      value={draft}
    />
  )
}

interface EnumControlProps {
  schemaKey: string
  value: unknown
  options: string[]
  optionLabels?: Record<string, string>
  onChange: (value: unknown) => void
  segmented: boolean
}

function EnumControl({ schemaKey, value, options, optionLabels, onChange, segmented }: EnumControlProps) {
  const { t } = useI18n()
  const c = t.settings.config
  const selected = String(value ?? '') || EMPTY_SELECT_VALUE
  const change = (next: string) => onChange(next === EMPTY_SELECT_VALUE ? '' : next)
  const label = (option: string) =>
    option
      ? (optionLabels?.[option] ?? prettyName(option))
      : schemaKey === 'display.personality'
        ? c.none
        : schemaKey === 'memory.provider'
          ? c.builtinOnly
          : c.noneParen

  if (segmented) {
    return (
      <SegmentedControl
        onChange={change}
        options={options.map(option => ({ id: option || EMPTY_SELECT_VALUE, label: label(option) }))}
        sizing="content"
        value={selected}
      />
    )
  }

  return (
    <Select onValueChange={change} value={selected}>
      <SelectTrigger className={CONTROL_TEXT}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map(option => (
          <SelectItem key={option || EMPTY_SELECT_VALUE} value={option || EMPTY_SELECT_VALUE}>
            {label(option)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
