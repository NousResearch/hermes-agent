import type { ChangeEvent, ReactNode } from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import {
  getElevenLabsVoices,
  getHermesConfigDefaults,
  getHermesConfigRecord,
  getHermesConfigSchema,
  saveHermesConfig
} from '@/hermes'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import type { ConfigFieldSchema, HermesConfigRecord } from '@/types/hermes'

import { CONTROL_TEXT, EMPTY_SELECT_VALUE, FIELD_DESCRIPTIONS, FIELD_LABELS, SECTIONS } from './constants'
import { fieldCopyForSchemaKey } from './field-copy'
import { enumOptionsFor, getNested, prettyName, setNested } from './helpers'
import { ModelSettings } from './model-settings'
import { EmptyState, ListRow, LoadingState, SettingsContent } from './primitives'

function ConfigField({
  schemaKey,
  schema,
  value,
  enumOptions,
  optionLabels,
  onChange
}: {
  schemaKey: string
  schema: ConfigFieldSchema
  value: unknown
  enumOptions?: string[]
  optionLabels?: Record<string, string>
  onChange: (value: unknown) => void
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

  const row = (action: ReactNode, wide = false) => (
    <ListRow action={action} description={description} title={label} wide={wide} />
  )

  if (schema.type === 'boolean') {
    return row(
      <div className="flex items-center justify-end">
        <Switch checked={Boolean(value)} onCheckedChange={onChange} />
      </div>
    )
  }

  const selectOptions = enumOptions ?? (schema.type === 'select' ? (schema.options ?? []).map(String) : undefined)

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

export function ConfigSettings({
  activeSectionId,
  onConfigSaved,
  onMainModelChanged,
  importInputRef
}: {
  activeSectionId: string
  onConfigSaved?: () => void
  onMainModelChanged?: (provider: string, model: string) => void
  importInputRef: React.RefObject<HTMLInputElement | null>
}) {
  const { t } = useI18n()
  const c = t.settings.config
  const [config, setConfig] = useState<HermesConfigRecord | null>(null)
  const [_defaults, setDefaults] = useState<HermesConfigRecord | null>(null)
  const [schema, setSchema] = useState<Record<string, ConfigFieldSchema> | null>(null)
  const [elevenLabsVoiceOptions, setElevenLabsVoiceOptions] = useState<string[] | null>(null)
  const [elevenLabsVoiceLabels, setElevenLabsVoiceLabels] = useState<Record<string, string>>({})
  const saveVersionRef = useRef(0)
  const [saveVersion, setSaveVersion] = useState(0)

  useEffect(() => {
    let cancelled = false
    Promise.all([getHermesConfigRecord(), getHermesConfigDefaults(), getHermesConfigSchema()])
      .then(([c, d, s]) => {
        if (cancelled) {
          return
        }

        setConfig(c)
        setDefaults(d)
        setSchema(s.fields)
      })
      .catch(err => notifyError(err, c.failedLoad))

    return () => void (cancelled = true)
  }, [])

  useEffect(() => {
    let cancelled = false

    getElevenLabsVoices()
      .then(result => {
        if (cancelled || !result.available) {
          return
        }

        setElevenLabsVoiceOptions(result.voices.map(voice => voice.voice_id))
        setElevenLabsVoiceLabels(Object.fromEntries(result.voices.map(voice => [voice.voice_id, voice.label])))
      })
      .catch(() => {
        if (!cancelled) {
          setElevenLabsVoiceOptions(null)
          setElevenLabsVoiceLabels({})
        }
      })

    return () => void (cancelled = true)
  }, [])

  useEffect(() => {
    if (!config || saveVersion === 0) {
      return
    }

    const v = saveVersion

    const t = window.setTimeout(() => {
      void (async () => {
        try {
          await saveHermesConfig(config)

          if (saveVersionRef.current === v) {
            onConfigSaved?.()
          }
        } catch (err) {
          if (saveVersionRef.current === v) {
            notifyError(err, c.autosaveFailed)
          }
        }
      })()
    }, 550)

    return () => window.clearTimeout(t)
  }, [config, onConfigSaved, saveVersion])

  const updateConfig = (next: HermesConfigRecord) => {
    saveVersionRef.current += 1
    setConfig(next)
    setSaveVersion(saveVersionRef.current)
  }

  const sectionFields = useMemo(() => {
    if (!schema) {
      return new Map<string, [string, ConfigFieldSchema][]>()
    }

    return new Map(
      SECTIONS.map(s => [s.id, s.keys.flatMap(k => (schema[k] ? [[k, schema[k]] as [string, ConfigFieldSchema]] : []))])
    )
  }, [schema])

  const fields = sectionFields.get(activeSectionId) ?? []

  // Deep-link target from the command palette (?field=<key>): scroll the row
  // into view and flash it, then drop the param so it doesn't re-fire.
  const [searchParams, setSearchParams] = useSearchParams()
  const targetField = searchParams.get('field')

  useEffect(() => {
    if (!targetField || !config || !schema) {
      return
    }

    const element = document.getElementById(`setting-field-${targetField}`)

    if (!element) {
      return
    }

    element.scrollIntoView({ behavior: 'smooth', block: 'center' })
    element.classList.add('setting-field-highlight')

    const timeout = window.setTimeout(() => element.classList.remove('setting-field-highlight'), 1600)

    setSearchParams(
      previous => {
        const next = new URLSearchParams(previous)
        next.delete('field')

        return next
      },
      { replace: true }
    )

    return () => window.clearTimeout(timeout)
  }, [config, schema, setSearchParams, targetField])

  function handleImport(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]

    if (!file) {
      return
    }

    const reader = new FileReader()

    reader.onload = () => {
      try {
        updateConfig(JSON.parse(String(reader.result)))
        notify({ kind: 'success', title: c.imported, message: t.common.saving })
      } catch (err) {
        notifyError(err, c.invalidJson)
      }
    }

    reader.readAsText(file)
    e.target.value = ''
  }

  if (!config || !schema) {
    return <LoadingState label={c.loading} />
  }

  return (
    <SettingsContent>
      {activeSectionId === 'model' && (
        <div className="mb-6">
          <ModelSettings onMainModelChanged={onMainModelChanged} />
        </div>
      )}
      {fields.length === 0 ? (
        <EmptyState description={c.emptyDesc} title={c.emptyTitle} />
      ) : (
        <div className="grid gap-1">
          {fields.map(([key, field]) => (
            <div className="scroll-mt-6 rounded-lg" id={`setting-field-${key}`} key={key}>
              <ConfigField
                enumOptions={
                  key === 'tts.elevenlabs.voice_id'
                    ? enumOptionsFor(key, getNested(config, key), config, elevenLabsVoiceOptions ?? undefined)
                    : enumOptionsFor(key, getNested(config, key), config)
                }
                onChange={value => updateConfig(setNested(config, key, value))}
                optionLabels={key === 'tts.elevenlabs.voice_id' ? elevenLabsVoiceLabels : undefined}
                schema={field}
                schemaKey={key}
                value={getNested(config, key)}
              />
            </div>
          ))}
        </div>
      )}
      <input
        accept=".json,application/json"
        className="hidden"
        onChange={handleImport}
        ref={importInputRef}
        type="file"
      />
    </SettingsContent>
  )
}
