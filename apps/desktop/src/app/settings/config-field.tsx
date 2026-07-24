import type { ReactNode } from 'react'

import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { prettyName } from '@/lib/text'
import { cn } from '@/lib/utils'
import type { ConfigFieldSchema } from '@/types/hermes'

import { CONTROL_TEXT, EMPTY_SELECT_VALUE, FIELD_DESCRIPTIONS, FIELD_LABELS, FREE_INPUT_KEYS } from './constants'
import { FallbackModelsField } from './fallback-models-field'
import { fieldCopyForSchemaKey } from './field-copy'
import { ListRow } from './primitives'

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

  const row = (action: ReactNode, wide = false) => (
    <ListRow action={action} description={descriptionNode} title={label} wide={wide} />
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

  // Voice/model name fields are open-world (custom voice IDs, cloned voices,
  // brand-new model names) — render a free-input combobox where the known
  // options are datalist suggestions instead of a closed Select gate.
  if (selectOptions && FREE_INPUT_KEYS.has(schemaKey)) {
    const datalistId = `config-field-options-${schemaKey.replace(/\./g, '-')}`

    return row(
      <>
        <Input
          className={CONTROL_TEXT}
          list={datalistId}
          onChange={e => onChange(e.target.value)}
          placeholder={c.notSet}
          value={String(value ?? '')}
        />
        <datalist id={datalistId}>
          {selectOptions
            .filter(option => option !== '')
            .map(option => (
              <option key={option} label={optionLabels?.[option]} value={option} />
            ))}
        </datalist>
      </>
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
