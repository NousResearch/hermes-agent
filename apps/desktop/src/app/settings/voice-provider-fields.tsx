import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'

import { getElevenLabsVoices, getHermesConfigSchema, saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import type { HermesConfigRecord } from '@/types/hermes'

import { setHermesConfigCache, useHermesConfigRecord } from '../hooks/use-config-record'

import { ConfigField } from './config-field'
import { SECTIONS } from './constants'
import { enumOptionsFor, getNested, inferFieldSchema, setNested } from './helpers'

// The curated voice keys (Settings → Voice) are the single source of which
// per-provider fields exist; both the Voice settings page and the
// Capabilities TTS panel derive from it so the two surfaces never drift.
const VOICE_KEYS = SECTIONS.find(s => s.id === 'voice')?.keys ?? []

export function voiceProviderKeys(
  section: 'tts' | 'stt',
  providerKey: string,
  schema: Record<string, unknown> = {},
  config: HermesConfigRecord | null = null
): string[] {
  const prefix = `${section}.${providerKey}.`
  const keys = new Set(VOICE_KEYS.filter(key => key.startsWith(prefix)))

  Object.keys(schema)
    .filter(key => key.startsWith(prefix))
    .forEach(key => keys.add(key))

  const providerConfig = config ? getNested(config, `${section}.${providerKey}`) : undefined
  if (providerConfig && typeof providerConfig === 'object' && !Array.isArray(providerConfig)) {
    Object.keys(providerConfig).forEach(key => keys.add(`${prefix}${key}`))
  }

  return [...keys]
}

/**
 * Config-presence-only keys: fields present in the live config for a provider
 * but with NO schema entry (neither the curated Voice section nor a backend
 * config_fields declaration). These render through inferFieldSchema() as
 * generic inputs, so callers should mark them "(detected)" rather than
 * presenting them as a fully-supported schema field.
 */
export function detectedConfigKeys(
  section: 'tts' | 'stt',
  providerKey: string,
  schema: Record<string, unknown> = {},
  config: HermesConfigRecord | null = null
): string[] {
  const prefix = `${section}.${providerKey}.`
  const declared = new Set(VOICE_KEYS.filter(key => key.startsWith(prefix)))
  Object.keys(schema)
    .filter(key => key.startsWith(prefix))
    .forEach(key => declared.add(key))

  const providerConfig = config ? getNested(config, `${section}.${providerKey}`) : undefined
  if (!providerConfig || typeof providerConfig !== 'object' || Array.isArray(providerConfig)) {
    return []
  }
  return Object.keys(providerConfig)
    .map(key => `${prefix}${key}`)
    .filter(key => !declared.has(key))
}

/**
 * Inline voice/model settings for one TTS (or STT) provider, rendered inside
 * the Capabilities → toolset config panel underneath the provider's API-key
 * fields. Reads and writes the same `tts.<provider>.*` config keys as
 * Settings → Voice (shared ConfigField renderer + enum/free-input rules), with
 * the same debounced autosave through the shared config cache.
 */
export function VoiceProviderFields({ section, providerKey }: { section: 'tts' | 'stt'; providerKey: string }) {
  const { t } = useI18n()
  const { data: loadedConfig } = useHermesConfigRecord()

  const { data: schemaResponse } = useQuery({
    queryKey: ['hermes-config-schema'],
    queryFn: () => getHermesConfigSchema(),
    staleTime: 5 * 60 * 1000
  })

  const keys = useMemo(
    () => voiceProviderKeys(section, providerKey, schemaResponse?.fields ?? {}, loadedConfig ?? null),
    [section, providerKey, schemaResponse?.fields, loadedConfig]
  )
  const detectedKeys = useMemo(
    () => detectedConfigKeys(section, providerKey, schemaResponse?.fields ?? {}, loadedConfig ?? null),
    [section, providerKey, schemaResponse?.fields, loadedConfig]
  )

  // Local editable draft, seeded once from the shared cache (background
  // refetches must not clobber in-progress edits) — the same shape as
  // config-settings.tsx's autosave loop.
  const [config, setConfig] = useState<HermesConfigRecord | null>(null)
  const seeded = useRef(false)

  // eslint-disable-next-line no-restricted-syntax -- one-shot config seed flag, not an atom mirror
  useEffect(() => {
    if (loadedConfig && !seeded.current) {
      seeded.current = true
      setConfig(loadedConfig)
    }
  }, [loadedConfig])

  const saveVersionRef = useRef(0)
  const [saveVersion, setSaveVersion] = useState(0)

  useEffect(() => {
    if (!config || saveVersion === 0) {
      return
    }

    const timeout = window.setTimeout(() => {
      void saveHermesConfig(config)
        .then(() => setHermesConfigCache(config))
        .catch(err => notifyError(err, t.settings.config.autosaveFailed))
    }, 550)

    return () => window.clearTimeout(timeout)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- copy is stable; avoid re-scheduling autosave on locale change
  }, [config, saveVersion])

  // ElevenLabs cloned/library voices from the live account, when available —
  // mirrors the Settings → Voice dynamic voice list.
  const [elVoices, setElVoices] = useState<string[] | null>(null)
  const [elVoiceLabels, setElVoiceLabels] = useState<Record<string, string>>({})
  const wantsElevenLabs = keys.includes('tts.elevenlabs.voice_id')

  useEffect(() => {
    if (!wantsElevenLabs) {
      return
    }

    let cancelled = false

    getElevenLabsVoices()
      .then(result => {
        if (cancelled || !result.available) {
          return
        }

        setElVoices(result.voices.map(voice => voice.voice_id))
        setElVoiceLabels(Object.fromEntries(result.voices.map(voice => [voice.voice_id, voice.label])))
      })
      .catch(() => {
        if (!cancelled) {
          setElVoices(null)
          setElVoiceLabels({})
        }
      })

    return () => void (cancelled = true)
  }, [wantsElevenLabs])

  if (keys.length === 0 || !config) {
    return null
  }

  const schema = schemaResponse?.fields ?? {}

  const updateConfig = (next: HermesConfigRecord) => {
    saveVersionRef.current += 1
    setConfig(next)
    setSaveVersion(saveVersionRef.current)
  }

  return (
    <div className="grid gap-0.5 rounded-lg bg-background/55 px-2.5">
      {keys.map(key => {
        const value = getNested(config, key)
        const field = schema[key] ?? inferFieldSchema(value)
        const isElVoice = key === 'tts.elevenlabs.voice_id'
        const isDetected = !schema[key] && detectedKeys.includes(key)

        return (
          <ConfigField
            descriptionExtra={
              isDetected ? (
                <span className="text-muted-foreground text-xs italic">{t.settings.config.detected}</span>
              ) : undefined
            }
            enumOptions={enumOptionsFor(key, value, config, isElVoice ? (elVoices ?? undefined) : undefined)}
            key={key}
            onChange={next => updateConfig(setNested(config, key, next))}
            optionLabels={isElVoice ? elVoiceLabels : undefined}
            schema={field}
            schemaKey={key}
            value={value}
          />
        )
      })}
    </div>
  )
}
