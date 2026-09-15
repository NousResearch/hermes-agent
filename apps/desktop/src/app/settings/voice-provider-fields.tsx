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
 * fields. Reads and writes the same `tts.<provider>.*` config keys as
 * Settings → Voice (shared ConfigField renderer + enum/free-input rules), with
 * the same debounced autosave through the shared config cache.
 */
export function VoiceProviderFields({
  section,
  providerKey,
  profile
}: {
  section: 'tts' | 'stt'
  providerKey: string
  /** Profile whose config these fields read AND write. The Capabilities panel
   *  is profile-scoped (its scope selector can target another profile, even on
   *  another gateway); rendering these fields unscoped read and autosaved the
   *  ACTIVE profile's whole config record while the UI claimed to configure
   *  profile B — a silent cross-profile clobber. Omitted = active profile,
   *  which keeps the Settings → Voice page's behavior unchanged. */
  profile?: ProfileScope
}) {
  const { t } = useI18n()
  const keys = useMemo(() => voiceProviderKeys(section, providerKey), [section, providerKey])
  const { data: loadedConfig } = useHermesConfigRecord(profile)
  // Parents pass `profile` as a fresh object literal each render; keying the
  // writer and the autosave effect on its identity would re-arm the 550ms
  // timer on every unrelated re-render. Key on the scope string instead
  // (null when unscoped, which maps to the bare cache row).
  const scopeKey = profile == null ? null : profileScopeKey(profile)
  // eslint-disable-next-line react-hooks/exhaustive-deps -- scopeKey is the identity of `profile`
  const writeConfigCache = useMemo(() => hermesConfigCacheWriter(profile), [scopeKey])

  const { data: schemaResponse } = useQuery({
    queryKey: ['hermes-config-schema', ...(scopeKey ? [scopeKey] : [])],
    queryFn: () => (profile === undefined ? getHermesConfigSchema() : getHermesConfigSchema(profile)),
    staleTime: 5 * 60 * 1000
  })

  // Local editable draft, seeded once from the shared cache (background
  // refetches must not clobber in-progress edits) — the same shape as
  // config-settings.tsx's autosave loop.
  const [config, setConfig] = useState<HermesConfigRecord | null>(null)
  // Autosave sends only what changed against this baseline (config-settings.tsx
  // pattern): the seeded record is a default-expanded snapshot, and echoing it
  // whole would overwrite keys other surfaces changed since it loaded. The
  // baseline advances to each successfully saved draft.
  const [baseline, setBaseline] = useState<HermesConfigRecord | null>(null)
  const seeded = useRef(false)

  // eslint-disable-next-line no-restricted-syntax -- one-shot config seed flag, not an atom mirror
  useEffect(() => {
    if (loadedConfig && !seeded.current) {
      seeded.current = true
      setBaseline(loadedConfig)
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
      void saveHermesConfigRecord(diffConfig(baseline ?? {}, config), profile)
        .then(() => {
          setBaseline(config)
          writeConfigCache(config)
        })
        .catch(err => notifyError(err, t.settings.config.autosaveFailed))
    }, 550)

    return () => window.clearTimeout(timeout)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- copy is stable; `profile` is keyed by scopeKey; avoid re-scheduling autosave on locale change
  }, [config, scopeKey, saveVersion, writeConfigCache])

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

    const voices = profile === undefined ? getElevenLabsVoices() : getElevenLabsVoices(profile)

    voices
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
    // eslint-disable-next-line react-hooks/exhaustive-deps -- scopeKey is the normalized owner identity
  }, [scopeKey, wantsElevenLabs])

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
            scope={profile}
            value={value}
          />
        )
      })}
    </div>
  )
}
