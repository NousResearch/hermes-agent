import { type ReactNode, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { Loader2, Save, SlidersHorizontal } from '@/lib/icons'
import { cn } from '@/lib/utils'
import type { PluginSettingField, PluginSettingFieldType } from '@/store/agent-plugins'

import { CONTROL_TEXT } from './constants'
import { ListRow, Pill, SectionHeading } from './primitives'

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

// Words a humanized key keeps upper-case ("maps_api_key" → "Maps API key").
const KEY_ACRONYMS = new Set([
  'ai',
  'api',
  'cpu',
  'css',
  'dns',
  'gpu',
  'html',
  'http',
  'https',
  'id',
  'ip',
  'json',
  'llm',
  'mcp',
  'oauth',
  'sql',
  'ssh',
  'ssl',
  'tls',
  'tts',
  'ui',
  'uri',
  'url',
  'xml'
])

const ACRONYM_SPELLING: Record<string, string> = { ids: 'IDs', oauth: 'OAuth', urls: 'URLs' }

/** A config key as a sentence-case label: `daily_budget` → "Daily budget",
 *  `maps_api_key` → "Maps API key", `webhookURL` → "Webhook URL". */
export function humanizeSettingKey(key: string): string {
  const words = key
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2')
    .split(/[\s_.\-/]+/)
    .filter(Boolean)

  return words
    .map((word, index) => {
      const lower = word.toLowerCase()

      if (ACRONYM_SPELLING[lower]) {
        return ACRONYM_SPELLING[lower]
      }

      if (KEY_ACRONYMS.has(lower)) {
        return lower.toUpperCase()
      }

      return index === 0 ? lower.charAt(0).toUpperCase() + lower.slice(1) : lower
    })
    .join(' ')
}

/** The row label: the manifest's `label`/`title` when it gave one (the backend
 *  falls back to the bare key), else the key humanized. */
export function fieldLabel(field: Pick<PluginSettingField, 'key' | 'label'>): string {
  const label = field.label.trim()

  return label && label !== field.key ? label : humanizeSettingKey(field.key)
}

/** Helper copy from independent sentences (a plugin's description, the secret
 *  storage note): each part ends its sentence before the next begins. */
export function joinSentences(...parts: (null | string | undefined)[]): string {
  const kept = parts.map(part => part?.trim() ?? '').filter(Boolean)

  return kept
    .map((part, index) => (index < kept.length - 1 && !/[.!?。！？…:;]$/.test(part) ? `${part}.` : part))
    .join(' ')
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
      throw new Error(`${fieldLabel(field)}: not one of ${field.choices?.join(', ') ?? ''}`)
    }

    return raw
  },
  json: (raw, field) => {
    const parsed: unknown = JSON.parse(raw || 'null')

    if (parsed === null || typeof parsed !== 'object') {
      throw new Error(`${fieldLabel(field)}: expected a JSON list or object`)
    }

    return parsed
  },
  number: (raw, field) => {
    const n = Number(raw)

    if (raw.trim() === '' || Number.isNaN(n)) {
      throw new Error(`${fieldLabel(field)}: expected a number`)
    }

    return n
  },
  string: raw => raw
}

const TEXT_CLASS = cn('w-full', CONTROL_TEXT)

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
      aria-label={fieldLabel(field)}
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
      <SelectTrigger aria-label={fieldLabel(field)} className={TEXT_CLASS} id={id}>
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
      className={cn('min-h-28 resize-y bg-background font-mono', CONTROL_TEXT)}
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
  intro,
  onSave,
  title
}: {
  fields: PluginSettingField[]
  idPrefix: string
  disabled: boolean
  /** Lead copy under the page heading (the plugin's description). */
  intro?: ReactNode
  /** Resolves true when everything landed; the form then re-seeds from `fields`. */
  onSave: (changes: PluginSettingsSave) => Promise<boolean>
  /** Page title; the Settings breadcrumb owns it, embedded callers show it. */
  title: string
}) {
  const { t } = useI18n()
  const s = t.skills.plugins.settingsForm
  const seed = useMemo(() => initialDraft(fields), [fields])
  const [draft, setDraft] = useState<PluginSettingsDraft>(seed)
  const [seedRef, setSeedRef] = useState(seed)
  const [error, setError] = useState<null | string>(null)
  const [saving, setSaving] = useState(false)

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
    setSaving(true)

    try {
      if (await onSave(changes)) {
        // Secrets are never echoed back; clear them so the placeholder shows "set".
        setDraft(current =>
          Object.fromEntries(
            fields.map(field => [field.key, field.type === 'secret' ? '' : (current[field.key] ?? '')])
          )
        )
      }
    } finally {
      setSaving(false)
    }
  }

  return (
    <form
      data-testid={`${idPrefix}-settings-form`}
      onSubmit={event => {
        event.preventDefault()
        void save()
      }}
    >
      {/* The page's one action rides the heading row, like every Settings page
          with a primary action (Passwords & Logins' Add): never mid-canvas. */}
      <SectionHeading
        aside={
          <Button className="gap-1.5" disabled={disabled || saving || !dirty} size="sm" type="submit">
            {saving ? <Loader2 className="size-3.5 animate-spin" /> : <Save className="size-3.5" />}
            {s.save}
          </Button>
        }
        icon={SlidersHorizontal}
        page
        title={title}
      />
      {intro}
      {error && (
        <p className="mb-1 text-[length:var(--conversation-caption-font-size)] text-destructive" role="alert">
          {error}
        </p>
      )}
      <div>
        {fields.map(field => {
          const id = `${idPrefix}-${field.key}`
          const Control = FIELD_CONTROLS[field.type]
          const label = fieldLabel(field)

          const control = (
            <Control
              disabled={disabled || saving}
              field={field}
              id={id}
              onChange={raw => setDraft(current => ({ ...current, [field.key]: raw }))}
              raw={draft[field.key] ?? ''}
              secretSetHint={s.secretSet}
            />
          )

          const description =
            joinSentences(field.description, field.type === 'secret' && field.env ? s.secretStoredAs(field.env) : '') ||
            undefined

          const rowTitle = (
            <span className="inline-flex flex-wrap items-center gap-2">
              <label htmlFor={id}>{label}</label>
              {field.required && <Pill>{s.required}</Pill>}
            </span>
          )

          // Editors too big for the control column take the full width under
          // the description, as native config rows do.
          return field.type === 'json' ? (
            <ListRow
              below={<div className="mt-3">{control}</div>}
              data-tour={`plugin-field-${field.key}`}
              description={description}
              key={field.key}
              title={rowTitle}
              wide
            />
          ) : (
            <ListRow
              action={control}
              data-tour={`plugin-field-${field.key}`}
              description={description}
              key={field.key}
              title={rowTitle}
            />
          )
        })}
      </div>
    </form>
  )
}
