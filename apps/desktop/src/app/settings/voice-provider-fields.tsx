import { useQuery } from '@tanstack/react-query'
import { useEffect, useMemo, useRef, useState } from 'react'

import {
  getElevenLabsVoices,
  getHermesConfigSchema,
  type ProfileScope,
  profileScopeKey,
  saveHermesConfigRecord
} from '@/hermes'
import { useI18n } from '@/i18n'
import { notifyError } from '@/store/notifications'
import type { HermesConfigRecord } from '@/types/hermes'

import { hermesConfigCacheWriter, useHermesConfigRecord } from '../hooks/use-config-record'

import { ConfigField } from './config-field'
import { SECTIONS } from './constants'
import { diffConfig, enumOptionsFor, getNested, inferFieldSchema, setNested } from './helpers'

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
 * fields. Reads and writes the same `<section>.<provider>.*` config keys as
 * Settings → Voice (shared ConfigField renderer + enum/free-input rules), with
 * the same debounced autosave through the shared config cache.
 */
interface VoiceProviderFieldsProps {
  section: 'tts' | 'stt'
  providerKey: string
  profile?: ProfileScope
}

export function VoiceProviderFields({ section, providerKey, profile }: VoiceProviderFieldsProps) {
  const { t } = useI18n()
  const keys = useMemo(() => voiceProviderKeys(section, providerKey), [section, providerKey])
  const { data: loadedConfig } = useHermesConfigRecord(profile)

  const { data: schemaResponse } = useQuery({
    queryKey: profile == null ? ['hermes-config-schema'] : ['hermes-config-schema', profileScopeKey(profile)],
    queryFn: () => getHermesConfigSchema(profile),
    staleTime: 5 * 60 * 1000
  })

  // Local editable draft, seeded once from the shared cache (background
  // refetches must not clobber in-progress edits) — the same shape as
  // config-settings.tsx's autosave loop.
  const [config, setConfig] = useState<HermesConfigRecord | null>(null)
  const seeded = useRef(false)
  const configBaselineRef = useRef<HermesConfigRecord | null>(null)
  const saveQueueRef = useRef<Promise<void>>(Promise.resolve())

  // eslint-disable-next-line no-restricted-syntax -- one-shot config seed flag, not an atom mirror
  useEffect(() => {
    if (loadedConfig && !seeded.current) {
      seeded.current = true
      configBaselineRef.current = loadedConfig
      setConfig(loadedConfig)
    }
  }, [loadedConfig])

  const saveVersionRef = useRef(0)
  const [saveVersion, setSaveVersion] = useState(0)

  // eslint-disable-next-line no-restricted-syntax -- autosave bookkeeping refs, not an atom mirror
  useEffect(() => {
    if (!config || saveVersion === 0) {
      return
    }

    const snapshot = config

    const timeout = window.setTimeout(() => {
      saveQueueRef.current = saveQueueRef.current.then(async () => {
        try {
          const patch = diffConfig(configBaselineRef.current ?? {}, snapshot)
          const result = await saveHermesConfigRecord(patch, profile)

          if (!result.ok) {
            throw new Error(t.settings.config.autosaveFailed)
          }

          configBaselineRef.current = snapshot
          hermesConfigCacheWriter(profile)(snapshot)
        } catch (err) {
          notifyError(err, t.settings.config.autosaveFailed)
        }
      })
    }, 550)

    return () => window.clearTimeout(timeout)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- copy is stable; avoid re-scheduling autosave on locale change
  }, [config, saveVersion, profile])

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

    getElevenLabsVoices(profile)
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
  }, [profile, wantsElevenLabs])

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
