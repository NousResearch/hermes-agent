import type { ChangeEvent, ReactNode } from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'

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
import { fieldDescription, fieldLabel, optionLabel } from '@/lib/i18n'
import { cn } from '@/lib/utils'
import { $desktopLanguage, type DesktopLanguage } from '@/store/language'
import { notify, notifyError } from '@/store/notifications'
import { useStore } from '@nanostores/react'
import type { ConfigFieldSchema, HermesConfigRecord } from '@/types/hermes'

import { CONTROL_TEXT, EMPTY_SELECT_VALUE, FIELD_DESCRIPTIONS, FIELD_LABELS, SECTIONS } from './constants'
import { enumOptionsFor, getNested, includesQuery, prettyName, setNested } from './helpers'
import { EmptyState, ListRow, LoadingState, SettingsContent } from './primitives'
import type { SearchProps } from './types'

function ConfigField({
  schemaKey,
  schema,
  value,
  enumOptions,
  optionLabels,
  language,
  onChange
}: {
  schemaKey: string
  schema: ConfigFieldSchema
  value: unknown
  enumOptions?: string[]
  optionLabels?: Record<string, string>
  language: DesktopLanguage
  onChange: (value: unknown) => void
}) {
  const baseLabel = FIELD_LABELS[schemaKey] ?? prettyName(schemaKey.split('.').pop() ?? schemaKey)
  const label = fieldLabel(language, schemaKey, baseLabel)
  const normalize = (v: string) => v.toLowerCase().replace(/[^a-z0-9]+/g, '')
  const rawDescription = fieldDescription(language, schemaKey, FIELD_DESCRIPTIONS[schemaKey] ?? schema.description ?? '')?.trim() ?? ''
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
      <div className="flex items-center justify-end gap-3">
        <span className="text-xs text-muted-foreground">{value ? optionLabel(language, 'on', 'On') : optionLabel(language, 'off', 'Off')}</span>
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
                ? optionLabel(language, option, optionLabels?.[option] ?? prettyName(option))
                : schemaKey === 'display.personality'
                  ? optionLabel(language, 'none', 'None')
                  : language === 'zh'
                    ? '（无）'
                    : '(none)'}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    )
  }

  if (schema.type === 'number') {
    return row(
      <Input
        className={cn('h-8', CONTROL_TEXT)}
        onChange={e => {
          const raw = e.target.value
          const n = raw === '' ? 0 : Number(raw)

          if (!Number.isNaN(n)) {
            onChange(n)
          }
        }}
        placeholder={language === 'zh' ? '未设置' : 'Not set'}
        type="number"
        value={value === undefined || value === null ? '' : String(value)}
      />
    )
  }

  if (schema.type === 'list') {
    return row(
      <Input
        className={cn('h-8', CONTROL_TEXT)}
        onChange={e =>
          onChange(
            e.target.value
              .split(',')
              .map(s => s.trim())
              .filter(Boolean)
          )
        }
        placeholder={language === 'zh' ? '逗号分隔的值' : 'comma-separated values'}
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
        placeholder={language === 'zh' ? '未设置' : 'Not set'}
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
        placeholder={language === 'zh' ? '未设置' : 'Not set'}
        value={String(value ?? '')}
      />
    ) : (
      <Input
        className={cn('h-8', CONTROL_TEXT)}
        onChange={e => onChange(e.target.value)}
        placeholder={language === 'zh' ? '未设置' : 'Not set'}
        value={String(value ?? '')}
      />
    ),
    isLong
  )
}

export function ConfigSettings({
  query,
  activeSectionId,
  onConfigSaved,
  importInputRef
}: SearchProps & {
  activeSectionId: string
  onConfigSaved?: () => void
  importInputRef: React.RefObject<HTMLInputElement | null>
}) {
  const language = useStore($desktopLanguage)
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
      .catch(err => notifyError(err, language === 'zh' ? '设置加载失败' : 'Settings failed to load'))

    return () => void (cancelled = true)
  }, [language])

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
            notifyError(err, language === 'zh' ? '自动保存失败' : 'Autosave failed')
          }
        }
      })()
    }, 550)

    return () => window.clearTimeout(t)
  }, [config, language, onConfigSaved, saveVersion])

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

  const matched = useMemo(() => {
    const q = query.trim().toLowerCase()

    if (!schema || !q) {
      return []
    }

    const seen = new Set<string>()

    return SECTIONS.flatMap(s =>
      s.keys.flatMap(k => {
        if (seen.has(k) || !schema[k]) {
          return []
        }

        seen.add(k)
        const label = prettyName(k.split('.').pop() ?? k)
        const item = schema[k]

        const hit =
          k.toLowerCase().includes(q) ||
          label.toLowerCase().includes(q) ||
          includesQuery(item.category, q) ||
          includesQuery(item.description, q)

        return hit ? [[k, item] as [string, ConfigFieldSchema]] : []
      })
    )
  }, [schema, query])

  const fields = query.trim() ? matched : (sectionFields.get(activeSectionId) ?? [])

  function handleImport(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]

    if (!file) {
      return
    }

    const reader = new FileReader()

    reader.onload = () => {
      try {
        updateConfig(JSON.parse(String(reader.result)))
        notify({
          kind: 'success',
          title: language === 'zh' ? '配置已导入' : 'Config imported',
          message: language === 'zh' ? '正在保存…' : 'Saving…'
        })
      } catch (err) {
        notifyError(err, language === 'zh' ? '配置 JSON 无效' : 'Invalid config JSON')
      }
    }

    reader.readAsText(file)
    e.target.value = ''
  }

  if (!config || !schema) {
    return <LoadingState label={language === 'zh' ? '正在加载配置...' : 'Loading Hermes configuration...'} />
  }

  return (
    <SettingsContent>
      {query.trim() && (
        <div className="mb-4 text-xs text-muted-foreground">
          {language === 'zh' ? `${fields.length} 个结果` : `${fields.length} result${fields.length === 1 ? '' : 's'}`}
        </div>
      )}
      {fields.length === 0 ? (
        <EmptyState
          description={language === 'zh' ? '请尝试其他搜索词，或选择另一个设置分类。' : 'Try a different search term or choose another section.'}
          title={language === 'zh' ? '没有匹配的设置' : 'No matching settings'}
        />
      ) : (
        <div className="divide-y divide-border/40">
          {fields.map(([key, field]) => (
            <ConfigField
              enumOptions={
                key === 'tts.elevenlabs.voice_id'
                  ? enumOptionsFor(key, getNested(config, key), config, elevenLabsVoiceOptions ?? undefined)
                  : enumOptionsFor(key, getNested(config, key), config)
              }
              key={key}
              language={language}
              onChange={value => updateConfig(setNested(config, key, value))}
              optionLabels={key === 'tts.elevenlabs.voice_id' ? elevenLabsVoiceLabels : undefined}
              schema={field}
              schemaKey={key}
              value={getNested(config, key)}
            />
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
