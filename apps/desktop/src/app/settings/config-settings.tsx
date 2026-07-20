import { useQuery } from '@tanstack/react-query'
import type { ChangeEvent } from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import { Button } from '@/components/ui/button'
import { getElevenLabsVoices, getHermesConfigSchema, saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'
import type { ConfigFieldSchema, HermesConfigRecord } from '@/types/hermes'

import { setHermesConfigCache, useHermesConfigRecord } from '../hooks/use-config-record'
import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'
import { PanelEmpty } from '../overlays/panel'

import { ConfigField } from './config-field'
import {
  enumOptionsFor,
  getNested,
  isExternalMemoryProvider,
  sectionFieldEntries,
  setNested
} from './helpers'
import { MemoryConnect } from './memory/connect'
import { ProviderConfigPanel } from './memory/provider-config-panel'
import { ModelSettings, ModelSettingsSkeleton } from './model-settings'
import { EmptyState, LoadingState, SettingsContent } from './primitives'

// On the Voice page, only surface the sub-fields of the *selected* TTS/STT
// provider — otherwise every provider's options render at once (the "totally
// crazy" wall of ~30 fields). Top-level keys (tts.provider, stt.enabled,
// voice.*) always show; STT provider fields hide entirely when STT is off.
export function voiceFieldVisible(key: string, config: HermesConfigRecord): boolean {
  const match = /^(tts|stt)\.([^.]+)\./.exec(key)

  if (!match) {
    return true
  }

  const [, domain, provider] = match

  if (domain === 'stt' && !getNested(config, 'stt.enabled')) {
    return false
  }

  return provider === String(getNested(config, `${domain}.provider`) ?? '')
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
  // The editable draft is local (debounced autosave watches it), but it's seeded
  // from — and saved back through — the shared config cache, so edits are visible
  // in the MCP/model surfaces and reopening the page doesn't reload-flash.
  const [config, setConfig] = useState<HermesConfigRecord | null>(null)
  const { data: loadedConfig, isError: configLoadFailed, refetch: refetchConfig } = useHermesConfigRecord()

  const {
    data: schemaResponse,
    isError: schemaFailed,
    refetch: refetchSchema
  } = useQuery({
    queryKey: ['hermes-config-schema'],
    queryFn: getHermesConfigSchema,
    staleTime: 5 * 60 * 1000
  })

  const schema = schemaResponse?.fields ?? null
  const [elevenLabsVoiceOptions, setElevenLabsVoiceOptions] = useState<string[] | null>(null)
  const [elevenLabsVoiceLabels, setElevenLabsVoiceLabels] = useState<Record<string, string>>({})
  const saveVersionRef = useRef(0)
  const [saveVersion, setSaveVersion] = useState(0)

  // Seed the local draft once, the first time the shared record lands.
  // Background refetches thereafter must not clobber in-progress edits.
  const configSeeded = useRef(false)

  useEffect(() => {
    if (loadedConfig && !configSeeded.current) {
      configSeeded.current = true
      setConfig(loadedConfig)
    }
  }, [loadedConfig])

  // A profile switch invalidates (but doesn't clear) the shared config query, so
  // the local draft would otherwise keep profile A's data and autosave it into
  // B. Drop the seed + draft (re-seeds from B's refetch) and zero saveVersion so
  // the pending debounced autosave is cancelled by its effect cleanup.
  useOnProfileSwitch(() => {
    configSeeded.current = false
    setConfig(null)
    saveVersionRef.current = 0
    setSaveVersion(0)
  })

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
          // Mirror the saved record into the shared cache so MCP/model surfaces
          // reflect the edit without their own refetch.
          setHermesConfigCache(config)

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
    // eslint-disable-next-line react-hooks/exhaustive-deps -- copy is stable; avoid re-scheduling autosave on locale change
  }, [config, onConfigSaved, saveVersion])

  const updateConfig = (next: HermesConfigRecord) => {
    saveVersionRef.current += 1
    setConfig(next)
    setSaveVersion(saveVersionRef.current)
  }

  const sectionFields = useMemo(() => {
    if (!schema || !config) {
      return new Map<string, [string, ConfigFieldSchema][]>()
    }

    return sectionFieldEntries(schema, config)
  }, [schema, config])

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
    // A failed config/schema fetch must surface a retry, not spin forever.
    if ((configLoadFailed && !config) || (schemaFailed && !schema)) {
      return (
        <div className="flex h-full min-h-0 flex-1">
          <PanelEmpty
            action={
              <Button
                onClick={() => {
                  void refetchConfig()
                  void refetchSchema()
                }}
                size="sm"
              >
                {t.skills.refresh}
              </Button>
            }
            icon="error"
            title={c.failedLoad}
          />
        </div>
      )
    }

    // Model keeps its shape via a skeleton (its catalog fetch is the slow part);
    // other sections are quick config/schema reads, so a light loader is fine.
    if (activeSectionId === 'model') {
      return (
        <SettingsContent>
          <div className="mb-6">
            <ModelSettingsSkeleton />
          </div>
        </SettingsContent>
      )
    }

    return <LoadingState label={c.loading} />
  }

  const visibleFields = activeSectionId === 'voice' ? fields.filter(([key]) => voiceFieldVisible(key, config)) : fields

  return (
    <SettingsContent>
      {activeSectionId === 'model' && (
        <div className="mb-6">
          <ModelSettings onMainModelChanged={onMainModelChanged} />
        </div>
      )}
      {visibleFields.length === 0 ? (
        <EmptyState description={c.emptyDesc} title={c.emptyTitle} />
      ) : (
        <div className="grid gap-1">
          {visibleFields.map(([key, field]) => (
            <div className="scroll-mt-6 rounded-lg" id={`setting-field-${key}`} key={key}>
              <ConfigField
                descriptionExtra={
                  key === 'memory.provider' && isExternalMemoryProvider(getNested(config, key)) ? (
                    <MemoryConnect provider={String(getNested(config, key))} />
                  ) : undefined
                }
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
              {key === 'memory.provider' && isExternalMemoryProvider(getNested(config, key)) ? (
                <ProviderConfigPanel key={String(getNested(config, key))} provider={String(getNested(config, key))} />
              ) : null}
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
