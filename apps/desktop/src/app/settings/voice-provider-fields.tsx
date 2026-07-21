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

export function voiceProviderKeys(section: 'tts' | 'stt', providerKey: string): string[] {
  const prefix = `${section}.${providerKey}.`

  return VOICE_KEYS.filter(key => key.startsWith(prefix))
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
  const keys = useMemo(() => voiceProviderKeys(section, providerKey), [section, providerKey])
  const { data: loadedConfig } = useHermesConfigRecord()

  const { data: schemaResponse } = useQuery({
    queryKey: ['hermes-config-schema'],
    queryFn: getHermesConfigSchema,
    staleTime: 5 * 60 * 1000
  })

  // Local editable draft, seeded once from the shared cache (background
  // refetches must not clobber in-progress edits) — the same shape as
  // config-settings.tsx's autosave loop.
  const [config, setConfig] = useState<HermesConfigRecord | null>(null)
  const seeded = useRef(false)

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

        return (
          <ConfigField
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
