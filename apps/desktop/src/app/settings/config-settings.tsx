import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import type { ChangeEvent } from 'react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { getElevenLabsVoices, getHermesConfigSchema, saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { confirm } from '@/store/confirm'
import {
  $dataUrlReadMaxMb,
  clampDataUrlReadMaxMb,
  DATA_URL_READ_DEFAULT_MAX_MB,
  DATA_URL_READ_MAX_MAX_MB,
  DATA_URL_READ_MIN_MAX_MB,
  refreshDataUrlReadMaxMb,
  setDataUrlReadMaxMb
} from '@/store/data-url-read-max'
import { $disableF12, setDisableF12 } from '@/store/disable-f12'
import { $keepAwake, setKeepAwake } from '@/store/keep-awake'
import { notify, notifyError } from '@/store/notifications'
import { normalizeProfileKey } from '@/store/profile'
import { repoDiscoveryPolicyFromConfig, repoDiscoveryPolicySignature, scanAndRecordRepos } from '@/store/projects'
import { $settingsRequestProfile } from '@/store/settings-scope'
import type { ConfigFieldSchema, HermesConfigRecord } from '@/types/hermes'

import { hermesConfigCacheWriter, useHermesConfigRecord } from '../hooks/use-config-record'
import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'
import { PanelEmpty } from '../overlays/panel'

import { ConfigField } from './config-field'
import {
  clearsEnabledToolsets,
  diffConfig,
  enumOptionsFor,
  getNested,
  isExternalMemoryProvider,
  sectionFieldEntries,
  setNested,
  voiceFieldVisible
} from './helpers'
import { MemoryConnect } from './memory/connect'
import { ProviderConfigPanel } from './memory/provider-config-panel'
import { ModelSettings, ModelSettingsSkeleton } from './model-settings'
import { EmptyState, ListRow, SettingsContent, SettingsSkeleton, ToggleRow } from './primitives'
import { SettingsProfileScope } from './profile-scope'
import { QuickEntrySettings } from './quick-entry-settings'

export function ConfigSettings({
  activeSectionId,
  onConfigSaved,
  onMainModelChanged,
  importInputRef
}: ConfigSettingsProps) {
  // Shared "Applies to" scope (null → the app's active profile). Remount the
  // inner page per scope so every draft/seed/autosave ref resets wholesale
  // when the target profile changes — the same guarantee useOnProfileSwitch
  // provides for app-wide switches, without hand-clearing each piece.
  const scopeProfile = useStore($settingsRequestProfile)

  return (
    <ConfigSettingsInner
      activeSectionId={activeSectionId}
      importInputRef={importInputRef}
      key={scopeProfile ?? '__active__'}
      onConfigSaved={onConfigSaved}
      onMainModelChanged={onMainModelChanged}
      scopeProfile={scopeProfile}
    />
  )
}

interface ConfigSettingsProps {
  activeSectionId: string
  onConfigSaved?: () => void
  onMainModelChanged?: (provider: string, model: string) => void
  importInputRef: React.RefObject<HTMLInputElement | null>
}

function ConfigSettingsInner({
  activeSectionId,
  onConfigSaved,
  onMainModelChanged,
  importInputRef,
  scopeProfile
}: ConfigSettingsProps & { scopeProfile: string | undefined }) {
  const { t } = useI18n()
  const c = t.settings.config
  const keepAwake = useStore($keepAwake)
  const disableF12 = useStore($disableF12)
  // The editable draft is local (debounced autosave watches it), but it's seeded
  // from — and saved back through — the shared config cache, so edits are visible
  // in the MCP/model surfaces and reopening the page doesn't reload-flash.
  const [config, setConfig] = useState<HermesConfigRecord | null>(null)
  const { data: loadedConfig, isError: configLoadFailed, refetch: refetchConfig } = useHermesConfigRecord(scopeProfile)
  // Writes land on the same cache key the query above reads (base key when
  // following the active profile, suffixed when a scope override is set).
  const writeConfigCache = useMemo(() => hermesConfigCacheWriter(scopeProfile), [scopeProfile])

  const {
    data: schemaResponse,
    isError: schemaFailed,
    refetch: refetchSchema
  } = useQuery({
    // Base key when following the active profile (matches every pre-existing
    // consumer); suffixed only for an explicit scope override.
    queryKey:
      scopeProfile == null ? ['hermes-config-schema'] : ['hermes-config-schema', normalizeProfileKey(scopeProfile)],
    queryFn: () => getHermesConfigSchema(scopeProfile),
    staleTime: 5 * 60 * 1000
  })

  const schema = schemaResponse?.fields ?? null
  const [elevenLabsVoiceOptions, setElevenLabsVoiceOptions] = useState<string[] | null>(null)
  const [elevenLabsVoiceLabels, setElevenLabsVoiceLabels] = useState<Record<string, string>>({})
  const saveVersionRef = useRef(0)
  const savedDiscoverySignatureRef = useRef<string | undefined>(undefined)
  const [saveVersion, setSaveVersion] = useState(0)

  // Seed the local draft once, the first time the shared record lands.
  // Background refetches thereafter must not clobber in-progress edits.
  const configSeeded = useRef(false)
  // Snapshot of the record as it was when the draft was seeded. Autosave
  // diffs the draft against this (not against disk) so a field the user
  // never touched — possibly changed out-of-band by `hermes config set`
  // while this page sat open — is never resent with its stale value.
  const configBaselineRef = useRef<HermesConfigRecord | null>(null)
  // Serializes autosave requests so an older save that's still in flight can't
  // resolve after a newer one and re-advance the baseline / cache with stale
  // data — each save's diff+request only starts once the previous one lands.
  const saveQueueRef = useRef<Promise<void>>(Promise.resolve())

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    if (loadedConfig && !configSeeded.current) {
      configSeeded.current = true
      configBaselineRef.current = loadedConfig
      savedDiscoverySignatureRef.current = repoDiscoveryPolicySignature(repoDiscoveryPolicyFromConfig(loadedConfig))
      setConfig(loadedConfig)
    }
  }, [loadedConfig])

  // A profile switch invalidates (but doesn't clear) the shared config query, so
  // the local draft would otherwise keep profile A's data and autosave it into
  // B. Drop the seed + draft (re-seeds from B's refetch) and zero saveVersion so
  // the pending debounced autosave is cancelled by its effect cleanup.
  useOnProfileSwitch(() => {
    configSeeded.current = false
    configBaselineRef.current = null
    savedDiscoverySignatureRef.current = undefined
    setConfig(null)
    saveVersionRef.current = 0
    setSaveVersion(0)
    saveQueueRef.current = Promise.resolve()
  })

  useEffect(() => {
    let cancelled = false

    getElevenLabsVoices(scopeProfile)
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
    // scopeProfile is constant per mount (the inner component is keyed on it).
  }, [scopeProfile])

  // eslint-disable-next-line no-restricted-syntax -- autosave bookkeeping refs, not an atom mirror
  useEffect(() => {
    if (!config || saveVersion === 0) {
      return
    }

    const v = saveVersion
    const snapshot = config

    const t = window.setTimeout(() => {
      // Chained onto the queue (not fired directly) so an older save that's
      // still awaiting its response can't land after this one and undo its
      // baseline advance — each save's diff is computed once its predecessor
      // has fully resolved.
      saveQueueRef.current = saveQueueRef.current.then(async () => {
        try {
          const patch = diffConfig(configBaselineRef.current ?? {}, snapshot)
          const result = await saveHermesConfig(patch, scopeProfile)

          if (!result.ok) {
            throw new Error(c.autosaveFailed)
          }

          // The saved snapshot becomes the new baseline, so the next autosave
          // diffs against what's actually on disk instead of the page-load
          // (or last-baseline) copy — otherwise reverting a field to its
          // pre-save value diffs to nothing and the revert never reaches disk.
          configBaselineRef.current = snapshot

          // Mirror the saved record into the shared cache so MCP/model surfaces
          // reflect the edit without their own refetch.
          writeConfigCache(snapshot)

          if (saveVersionRef.current === v) {
            // The repo-discovery scan reads the ACTIVE profile's workspace
            // policy; skip it when this page is editing another profile.
            if (scopeProfile == null) {
              const discoverySignature = repoDiscoveryPolicySignature(repoDiscoveryPolicyFromConfig(snapshot))

              if (savedDiscoverySignatureRef.current !== discoverySignature) {
                savedDiscoverySignatureRef.current = discoverySignature
                await scanAndRecordRepos(true)
              }
            }

            onConfigSaved?.()
          }
        } catch (err) {
          if (saveVersionRef.current === v) {
            notifyError(err, c.autosaveFailed)
          }
        }
      })
    }, 550)

    return () => window.clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- copy is stable; avoid re-scheduling autosave on locale change
  }, [config, onConfigSaved, saveVersion])

  const applyConfig = (next: HermesConfigRecord) => {
    saveVersionRef.current += 1
    setConfig(next)
    setSaveVersion(saveVersionRef.current)
  }

  const updateConfig = (next: HermesConfigRecord) => {
    // Guard the single most destructive config edit: clearing the entire
    // "Enabled Toolsets" list silently disables memory, terminal, web search,
    // delegation, and most tools, and a stray select-all + Backspace can do it.
    // Auto-save is debounced with no undo, so confirm a non-empty → empty
    // transition before applying it. Every other edit passes through untouched.
    if (config && clearsEnabledToolsets(config, next)) {
      void confirm({ destructive: true, title: c.toolsetsWipeConfirm }).then(ok => {
        if (ok) {
          applyConfig(next)
        }
      })

      return
    }

    applyConfig(next)
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

    if (!element.hasAttribute('tabindex')) {
      element.tabIndex = -1
    }

    element.focus({ preventScroll: true })
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

    // Every section keeps its shape via a skeleton; model gets its bespoke one
    // (its catalog fetch is the slow part), the rest the shared field rhythm.
    if (activeSectionId === 'model') {
      return (
        <SettingsContent>
          <SettingsProfileScope className="mb-5" />
          <div className="mb-6">
            <ModelSettingsSkeleton />
          </div>
        </SettingsContent>
      )
    }

    return <SettingsSkeleton sections={[{ rows: 6 }]} />
  }

  const visibleFields = activeSectionId === 'voice' ? fields.filter(([key]) => voiceFieldVisible(key, config)) : fields

  return (
    <SettingsContent>
      {/* Which profile's config.yaml this page edits — shared across every
          config-backed settings page (and hidden for single-profile users). */}
      <SettingsProfileScope className="mb-5" />
      {activeSectionId === 'model' && (
        <div className="mb-6">
          <ModelSettings onMainModelChanged={onMainModelChanged} scopeProfile={scopeProfile} />
        </div>
      )}
      {/* Device-local desktop prefs (not config.yaml) — they live here since
          keeping the machine awake and the global Quick Entry chord are both
          power-user, this-computer-only knobs. */}
      {activeSectionId === 'advanced' && (
        <>
          <ToggleRow
            checked={keepAwake}
            description={c.keepAwakeDesc}
            label={c.keepAwakeTitle}
            onChange={setKeepAwake}
          />
          <ToggleRow
            checked={disableF12}
            description={c.disableF12Desc}
            label={c.disableF12Title}
            onChange={setDisableF12}
          />
          <QuickEntrySettings />
        </>
      )}
      {/* Device-local attach/preview byte cap (main-process IPC guard). Chat is
          where image-attachment behavior already lives, so this sits above the
          schema fields for that section. */}
      {activeSectionId === 'chat' ? <AttachmentSizeSetting /> : null}
      {visibleFields.length === 0 && activeSectionId !== 'chat' ? (
        <EmptyState description={c.emptyDesc} title={c.emptyTitle} />
      ) : visibleFields.length === 0 ? null : (
        <div className="grid gap-1">
          {visibleFields.map(([key, field]) => (
            <div className="scroll-mt-6 rounded-lg" id={`setting-field-${key}`} key={key}>
              <ConfigField
                descriptionExtra={
                  key === 'memory.provider' && isExternalMemoryProvider(getNested(config, key)) ? (
                    <MemoryConnect profile={scopeProfile} provider={String(getNested(config, key))} />
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
                <ProviderConfigPanel
                  key={String(getNested(config, key))}
                  profile={scopeProfile}
                  provider={String(getNested(config, key))}
                />
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

/** Free-form MB cap for Desktop's data-URL attach/preview path (main-process). */
function AttachmentSizeSetting() {
  const { t } = useI18n()
  const c = t.settings.config
  const stored = useStore($dataUrlReadMaxMb)
  const [draft, setDraft] = useState(String(stored))

  useEffect(() => {
    void refreshDataUrlReadMaxMb()
  }, [])

  useEffect(() => {
    setDraft(String(stored))
  }, [stored])

  const commit = () => {
    // An empty draft means "reset to the default", not the 1 MB floor
    // (Number('') === 0 would otherwise clamp down to the floor).
    const applied = draft.trim() === '' ? DATA_URL_READ_DEFAULT_MAX_MB : clampDataUrlReadMaxMb(draft)

    // Unchanged: snap the draft back to the stored value and skip the
    // pointless IPC write + haptic.
    if (applied === stored) {
      setDraft(String(stored))

      return
    }

    void setDataUrlReadMaxMb(applied).then(next => {
      setDraft(String(next))

      // On a bridge write failure the store keeps the old value; only
      // celebrate when the new cap actually landed.
      if (next === applied) {
        triggerHaptic('selection')
      }
    })
  }

  return (
    <ListRow
      action={
        <div className="flex items-center gap-2">
          <Input
            aria-label={c.attachmentSizeLabel}
            className="w-20"
            inputMode="numeric"
            max={DATA_URL_READ_MAX_MAX_MB}
            min={DATA_URL_READ_MIN_MAX_MB}
            onBlur={commit}
            onChange={event => setDraft(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter') {
                event.currentTarget.blur()
              }
            }}
            type="number"
            value={draft}
          />
          <span className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
            {c.attachmentSizeUnit}
          </span>
        </div>
      }
      description={c.attachmentSizeDesc}
      title={c.attachmentSizeTitle}
    />
  )
}
