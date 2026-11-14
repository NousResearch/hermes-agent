import { type ReactNode, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Field, FieldHint } from '@/components/ui/field'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import type { PluginSettingField, PluginSettingFieldType } from '@/store/agent-plugins'

// A plugin manifest's `config_schema` rendered as a form. Every non-secret
// value is edited as TEXT (the draft) and coerced to its wire type on save, so a
// half-typed number never fights the input; secrets are drafted separately and
// only ever sent to the `.env` credential route by the caller.

export type PluginSettingsDraft = Record<string, string>

export interface PluginSettingsSave {
  values: Record<string, unknown>
  secrets: Record<string, string>
}

interface ControlProps {
  field: PluginSettingField
  id: string
  raw: string
  disabled: boolean
  onChange: (raw: string) => void
  /** Localised hint for a secret that already has a stored value. */
  secretSetHint: string
}

/** What the input shows before the user touches it. */
const INITIAL_TEXT: Record<PluginSettingFieldType, (field: PluginSettingField) => string> = {
  boolean: field => (field.value === true ? 'true' : 'false'),
  enum: field => String(field.value ?? field.choices?.[0] ?? ''),
  json: field => (field.value === undefined || field.value === null ? '' : JSON.stringify(field.value, null, 2)),
  number: field => (field.value === undefined || field.value === null ? '' : String(field.value)),
  secret: () => '',
  string: field => String(field.value ?? '')
}

/** Text → wire value; a thrown Error is the field's validation message. */
const COERCE: Record<Exclude<PluginSettingFieldType, 'secret'>, (raw: string, field: PluginSettingField) => unknown> = {
  boolean: raw => raw === 'true',
  enum: (raw, field) => {
    if (!field.choices?.includes(raw)) {
      throw new Error(`${field.label}: not one of ${field.choices?.join(', ') ?? ''}`)
    }

    return raw
  },
  json: (raw, field) => {
    const parsed: unknown = JSON.parse(raw || 'null')

    if (parsed === null || typeof parsed !== 'object') {
      throw new Error(`${field.label}: expected a JSON list or object`)
    }

    return parsed
  },
  number: (raw, field) => {
    const n = Number(raw)

    if (raw.trim() === '' || Number.isNaN(n)) {
      throw new Error(`${field.label}: expected a number`)
    }

    return n
  },
  string: raw => raw
}

const TEXT_CLASS = 'h-7 text-xs'

function StringControl({ disabled, id, onChange, raw }: ControlProps) {
  return (
    <Input
      className={TEXT_CLASS}
      disabled={disabled}
      id={id}
      onChange={e => onChange(e.currentTarget.value)}
      value={raw}
    />
  )
}

function NumberControl({ disabled, id, onChange, raw }: ControlProps) {
  return (
    <Input
      className={TEXT_CLASS}
      disabled={disabled}
      id={id}
      inputMode="decimal"
      onChange={e => onChange(e.currentTarget.value)}
      type="number"
      value={raw}
    />
  )
}

function BooleanControl({ disabled, field, id, onChange, raw }: ControlProps) {
  return (
    <Switch
      aria-label={field.label}
      checked={raw === 'true'}
      disabled={disabled}
      id={id}
      onCheckedChange={on => onChange(on ? 'true' : 'false')}
    />
  )
}

function EnumControl({ disabled, field, id, onChange, raw }: ControlProps) {
  return (
    <Select disabled={disabled} onValueChange={onChange} value={raw}>
      <SelectTrigger aria-label={field.label} className={TEXT_CLASS} id={id}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {(field.choices ?? []).map(choice => (
          <SelectItem key={choice} value={choice}>
            {choice}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

function SecretControl({ disabled, field, id, onChange, raw, secretSetHint }: ControlProps) {
  return (
    <Input
      autoComplete="off"
      className={TEXT_CLASS}
      data-env={field.env}
      disabled={disabled}
      id={id}
      onChange={e => onChange(e.currentTarget.value)}
      placeholder={field.has_value ? secretSetHint : ''}
      type="password"
      value={raw}
    />
  )
}

function JsonControl({ disabled, id, onChange, raw }: ControlProps) {
  return (
    <Textarea
      className="min-h-16 font-mono text-xs"
      disabled={disabled}
      id={id}
      onChange={e => onChange(e.currentTarget.value)}
      spellCheck={false}
      value={raw}
    />
  )
}

/** Field type → control. Adding a manifest type means one row here, one in
 *  INITIAL_TEXT and one in COERCE — never a branch in the renderer. */
export const FIELD_CONTROLS: Record<PluginSettingFieldType, (props: ControlProps) => ReactNode> = {
  boolean: BooleanControl,
  enum: EnumControl,
  json: JsonControl,
  number: NumberControl,
  secret: SecretControl,
  string: StringControl
}

export function initialDraft(fields: PluginSettingField[]): PluginSettingsDraft {
  return Object.fromEntries(fields.map(field => [field.key, INITIAL_TEXT[field.type](field)]))
}

/** Split a draft into the two payloads: coerced non-secret values that
 *  CHANGED, and non-blank secrets keyed by their `.env` name. Throws the first
 *  field's validation error. */
export function collectChanges(fields: PluginSettingField[], draft: PluginSettingsDraft): PluginSettingsSave {
  const initial = initialDraft(fields)
  const values: Record<string, unknown> = {}
  const secrets: Record<string, string> = {}

  for (const field of fields) {
    const raw = draft[field.key] ?? ''

    if (field.type === 'secret') {
      if (raw && field.env) {
        secrets[field.env] = raw
      }

      continue
    }

    if (raw !== initial[field.key]) {
      values[field.key] = COERCE[field.type](raw, field)
    }
  }

  return { values, secrets }
}

export function PluginSettingsForm({
  fields,
  idPrefix,
  disabled,
  onSave
}: {
  fields: PluginSettingField[]
  idPrefix: string
  disabled: boolean
  /** Resolves true when everything landed; the form then re-seeds from `fields`. */
  onSave: (changes: PluginSettingsSave) => Promise<boolean>
}) {
  const { t } = useI18n()
  const s = t.skills.plugins.settingsForm
  const seed = useMemo(() => initialDraft(fields), [fields])
  const [draft, setDraft] = useState<PluginSettingsDraft>(seed)
  const [seedRef, setSeedRef] = useState(seed)
  const [error, setError] = useState<null | string>(null)

  // The row's refreshed copy re-seeds the form (a saved value becomes the new baseline).
  if (seedRef !== seed) {
    setSeedRef(seed)
    setDraft(seed)
  }

  const dirty = fields.some(field => (draft[field.key] ?? '') !== seed[field.key])

  const save = async () => {
    let changes: PluginSettingsSave

    try {
      changes = collectChanges(fields, draft)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))

      return
    }

    setError(null)

    if (await onSave(changes)) {
      // Secrets are never echoed back; clear them so the placeholder shows "set".
      setDraft(current =>
        Object.fromEntries(fields.map(field => [field.key, field.type === 'secret' ? '' : (current[field.key] ?? '')]))
      )
    }
  }

  return (
    <form
      className="grid gap-3"
      data-testid={`${idPrefix}-settings-form`}
      onSubmit={event => {
        event.preventDefault()
        void save()
      }}
    >
      <div className="grid items-start gap-3 sm:grid-cols-2">
        {fields.map(field => {
          const id = `${idPrefix}-${field.key}`
          const Control = FIELD_CONTROLS[field.type]

          return (
            <Field
              htmlFor={id}
              key={field.key}
              label={field.label}
              optional={!field.required}
              optionalLabel={s.optional}
            >
              <Control
                disabled={disabled}
                field={field}
                id={id}
                onChange={raw => setDraft(current => ({ ...current, [field.key]: raw }))}
                raw={draft[field.key] ?? ''}
                secretSetHint={s.secretSet}
              />
              {(field.description || field.type === 'secret') && (
                <FieldHint>
                  {field.description}
                  {field.type === 'secret' && field.env ? ` ${s.secretStoredAs(field.env)}` : ''}
                </FieldHint>
              )}
            </Field>
          )
        })}
      </div>
      {error && <FieldHint error>{error}</FieldHint>}
      <div className="flex items-center justify-end gap-2">
        <Button disabled={disabled || !dirty} size="xs" type="submit" variant="outline">
          {s.save}
        </Button>
      </div>
    </form>
  )
}
