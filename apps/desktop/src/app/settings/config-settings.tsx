import { useStore } from '@nanostores/react'
import { useQuery } from '@tanstack/react-query'
import type { ChangeEvent, ReactNode } from 'react'
import { useContext, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useSearchParams } from 'react-router'

import { usePortalRedesign } from '@/app/redesign/use-portal-redesign'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { getElevenLabsVoices, getHermesConfigSchema, saveHermesConfig } from '@/hermes'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { isSubmitEnter } from '@/lib/ime'
import { asText } from '@/lib/text'
import { confirm } from '@/store/confirm'
import { $activeConnectionId } from '@/store/connections'
import {
  $dataUrlReadMaxMb,
  clampDataUrlReadMaxMb,
  DATA_URL_READ_DEFAULT_MAX_MB,
  DATA_URL_READ_MAX_MAX_MB,
  DATA_URL_READ_MIN_MAX_MB,
  refreshDataUrlReadMaxMb,
  setDataUrlReadMaxMb
} from '@/store/data-url-read-max'
import { recordSettingsSaved } from '@/store/desktop-metrics'
import { $disableF12, setDisableF12 } from '@/store/disable-f12'
import { $alwaysExternalLinks, setAlwaysExternalLinks } from '@/store/external-links'
import { $keepAwakeMode, type KeepAwakeMode, setKeepAwakeMode } from '@/store/keep-awake'
import { refreshMemoryStatus } from '@/store/memory-status'
import { notify, notifyError } from '@/store/notifications'
import { normalizeProfileKey } from '@/store/profile'
import { repoDiscoveryPolicyFromConfig, repoDiscoveryPolicySignature, scanAndRecordRepos } from '@/store/projects'
import { $settingsRequestProfile } from '@/store/settings-scope'
import type { ConfigFieldSchema, HermesConfigRecord } from '@/types/hermes'

import { useAgentSettingsEditorCopy } from '../agent-settings/settings-copy'
import { hermesConfigCacheWriter, useHermesConfigRecord } from '../hooks/use-config-record'
import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'
import { PanelEmpty } from '../overlays/panel'

import { AgentConfigFields } from './agent-config-fields'
import { type ConfigSettingsDraft, ConfigSettingsDraftContext } from './config-draft'
import { ConfigField } from './config-field'
import { configSubpageForField } from './config-subpages'
import { DELEGATION_ROUTE_COMPANION_KEY, DelegationRouteRow } from './delegation-route-row'
import { DelegationRoutingSlot } from './delegation/routing-section'
import { SubagentModelGate } from './delegation/subagent-model-gate'
import {
  clearsEnabledToolsets,
  configFieldValue,
  diffConfig,
  enumOptionsFor,
  getNested,
  isExternalMemoryProvider,
  sectionFieldEntries,
  setConfigField,
  voiceFieldVisible
} from './helpers'
import { MemoryConnect } from './memory/connect'
import { MemoryLayersSlot } from './memory/memory-layers'
import { ProviderConfigPanel } from './memory/provider-config-panel'
import { ModelSettings, ModelSettingsSkeleton } from './model-settings'
import { OnboardingSettings } from './onboarding-settings'
import { PortalSafetySettings } from './portal-safety-fields'
import { EmptyState, ListRow, SettingsContent, SettingsSkeleton, ToggleRow } from './primitives'
import { SettingsProfileScope, useSettingsOwner } from './profile-scope'
import { QuickEntrySettings } from './quick-entry-settings'
import { SETTING_IDS, settingElementId } from './settings-manifest'
import { SharedMetricsSettings } from './shared-metrics-settings'
import { useSettingDeepLink } from './use-setting-deep-link'

const VOICE_MODE_LABELS = { chained: 'Chained', 'gpt-live': 'GPT-Live' }

export function ConfigSettings({
  activeSectionId,
  subpage,
  agentModel,
  agentLayout,
  onConfigSaved,
  onMainModelChanged,
  importInputRef
}: ConfigSettingsProps) {
  // Shared "Applies to" scope (null → the app's active profile). Remount the
  // inner page per scope so every draft/seed/autosave ref resets wholesale
  // when the target profile changes — the same guarantee useOnProfileSwitch
  // provides for app-wide switches, without hand-clearing each piece.
  const selectedProfile = useStore($settingsRequestProfile)
  const activeConnection = useStore($activeConnectionId)
  const owner = useSettingsOwner()
  const scopeProfile = owner?.profile ?? selectedProfile
  const connectionId = owner ? owner.connectionId : activeConnection

  return (
    <ConfigSettingsInner
      activeSectionId={activeSectionId}
      agentLayout={agentLayout}
      agentModel={agentModel}
      importInputRef={importInputRef}
      key={JSON.stringify([connectionId, scopeProfile ?? '__active__'])}
      onConfigSaved={onConfigSaved}
      onMainModelChanged={onMainModelChanged}
      scopeProfile={scopeProfile}
      subpage={subpage}
    />
  )
}

interface ConfigSettingsProps {
  activeSectionId: string
  /** Undefined preserves the full section for existing embedded consumers. */
  agentModel?: boolean
  agentLayout?: boolean
  subpage?: string
  onConfigSaved?: () => void
  onMainModelChanged?: (provider: string, model: string) => void
  importInputRef: React.RefObject<HTMLInputElement | null>
}

function useConfigSectionVisibility({
  activeSectionId,
  agentModel,
  subpage,
  hasOwner
}: Pick<ConfigSettingsProps, 'activeSectionId' | 'agentModel' | 'subpage'> & { hasOwner: boolean }) {
  const includesSubpage = (...pages: string[]) => agentModel || subpage === undefined || pages.includes(subpage)
  const portalSafety = usePortalRedesign() && activeSectionId === 'safety'

  return {
    showModelSettings: activeSectionId === 'model' && includesSubpage('main', 'auxiliary', 'moa'),
    showDesktopSettings: !hasOwner && activeSectionId === 'advanced' && includesSubpage('desktop'),
    showAttachments: !hasOwner && activeSectionId === 'chat' && includesSubpage('attachments'),
    portalSafety,
    showSharedMetrics: !hasOwner && activeSectionId === 'safety' && subpage === 'privacy'
  }
}

interface ConfigLoadingProps {
  configLoaded: boolean
  schemaLoaded: boolean
  configFailed: boolean
  schemaFailed: boolean
  model: boolean
  subpage?: string
  onRetry: () => void
}

function ConfigLoading({
  configLoaded,
  schemaLoaded,
  configFailed,
  schemaFailed,
  model,
  subpage,
  onRetry
}: ConfigLoadingProps) {
  const { t } = useI18n()

  if ((configFailed && !configLoaded) || (schemaFailed && !schemaLoaded)) {
    return (
      <div className="flex h-full min-h-0 flex-1">
        <PanelEmpty
          action={
            <Button onClick={onRetry} size="sm">
              {t.common.retry}
            </Button>
          }
          icon="error"
          title={t.settings.config.failedLoad}
        />
      </div>
    )
  }

  if (model) {
    return (
      <SettingsContent>
        <SettingsProfileScope className="mb-5" />
        <div className="mb-6">
          <ModelSettingsSkeleton subpage={subpage} />
        </div>
      </SettingsContent>
    )
  }

  return <SettingsSkeleton sections={[{ rows: 6 }]} />
}

function ConfigSettingsInner({
  activeSectionId,
  subpage,
  agentModel,
  agentLayout,
  onConfigSaved,
  onMainModelChanged,
  importInputRef,
  scopeProfile
}: ConfigSettingsProps & { scopeProfile: string | undefined }) {
  const { t } = useI18n()
  const owner = useSettingsOwner()
  const compactAgent = usePortalRedesign() && agentLayout
  const agentCopy = useAgentSettingsEditorCopy()
  const requestScope = owner ?? scopeProfile
  const c = t.settings.config
  const keepAwakeMode = useStore($keepAwakeMode)

  const keepAwakeOptions = [
    { id: 'off', label: c.keepAwakeOff },
    { id: 'while-working', label: c.keepAwakeWhileWorking },
    { id: 'always', label: c.keepAwakeAlways }
  ] as const satisfies readonly { id: KeepAwakeMode; label: string }[]

  const disableF12 = useStore($disableF12)
  const alwaysExternalLinks = useStore($alwaysExternalLinks)
  const retainedDraft = useContext(ConfigSettingsDraftContext)
  const localDraft = useRef<ConfigSettingsDraft | null>(null)
  const draft = retainedDraft ?? localDraft
  const [config, setConfig] = useState<HermesConfigRecord | null>(() => draft.current?.config ?? null)

  const {
    data: loadedConfig,
    isError: configLoadFailed,
    refetch: refetchConfig,
    writeScope
  } = useHermesConfigRecord(requestScope)

  const {
    data: schemaResponse,
    isError: schemaFailed,
    refetch: refetchSchema
  } = useQuery({
    // Base key when following the active profile (matches every pre-existing
    // consumer); suffixed only for an explicit scope override.
    queryKey: owner
      ? ['hermes-config-schema', owner.connectionId ?? '', owner.profile]
      : scopeProfile == null
        ? ['hermes-config-schema']
        : ['hermes-config-schema', normalizeProfileKey(scopeProfile)],
    queryFn: () => getHermesConfigSchema(requestScope),
    staleTime: 5 * 60 * 1000
  })

  const schema = schemaResponse?.fields ?? null
  const [elevenLabsVoiceOptions, setElevenLabsVoiceOptions] = useState<string[] | null>(null)
  const [elevenLabsVoiceLabels, setElevenLabsVoiceLabels] = useState<Record<string, string>>({})
  const savedDiscoverySignatureRef = useRef<string | undefined>(undefined)
  const [saveVersion, setSaveVersion] = useState(() =>
    draft.current && draft.current.version > draft.current.savedVersion ? draft.current.version : 0
  )
  const mounted = useRef(true)
  useLayoutEffect(() => {
    mounted.current = true

    return () => {
      mounted.current = false
    }
  }, [])

  // Keep the seed, baseline and write owner together across presentation changes.
  // Refetches must not replace unsaved fields or redirect their eventual write.
  // eslint-disable-next-line no-restricted-syntax -- scoped editor draft, not an atom mirror
  useEffect(() => {
    if (loadedConfig && !draft.current) {
      draft.current = {
        config: loadedConfig,
        baseline: loadedConfig,
        writeScope: owner ?? writeScope ?? scopeProfile,
        version: 0,
        savedVersion: 0,
        queue: Promise.resolve()
      }
      savedDiscoverySignatureRef.current = repoDiscoveryPolicySignature(repoDiscoveryPolicyFromConfig(loadedConfig))
      setConfig(loadedConfig)
    }
  }, [loadedConfig, draft, owner, writeScope, scopeProfile])

  // A profile switch invalidates (but doesn't clear) the shared config query, so
  // the local draft would otherwise keep profile A's data and autosave it into
  // B. Drop the seed + draft (re-seeds from B's refetch) and zero saveVersion so
  // the pending debounced autosave is cancelled by its effect cleanup.
  useOnProfileSwitch(() => {
    if (owner) {
      return
    }

    draft.current = null
    savedDiscoverySignatureRef.current = undefined
    setConfig(null)
    setSaveVersion(0)
  })

  useEffect(() => {
    let cancelled = false

    getElevenLabsVoices(requestScope)
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
  }, [requestScope])

  // eslint-disable-next-line no-restricted-syntax -- autosave bookkeeping refs, not an atom mirror
  useEffect(() => {
    const pending = draft.current

    if (!config || !pending || saveVersion === 0) {
      return
    }

    const v = saveVersion
    const snapshot = config

    const t = window.setTimeout(() => {
      // Chained onto the queue (not fired directly) so an older save that's
      // still awaiting its response can't land after this one and undo its
      // baseline advance — each save's diff is computed once its predecessor
      // has fully resolved.
      pending.queue = pending.queue.then(async () => {
        if (!mounted.current || pending.savedVersion >= v) {
          return
        }

        try {
          const patch = diffConfig(pending.baseline, snapshot)
          const result = await saveHermesConfig(patch, pending.writeScope)

          if (!result.ok) {
            throw new Error(c.autosaveFailed)
          }

          // The saved snapshot becomes the new baseline, so the next autosave
          // diffs against what's actually on disk instead of the page-load
          // (or last-baseline) copy — otherwise reverting a field to its
          // pre-save value diffs to nothing and the revert never reaches disk.
          pending.baseline = snapshot
          pending.savedVersion = v

          // Mirror the saved record into the shared cache so MCP/model surfaces
          // reflect the edit without their own refetch.
          hermesConfigCacheWriter(pending.writeScope)(snapshot)
          const savedScope = pending.writeScope

          recordSettingsSaved(patch, schema ?? {}, typeof savedScope === 'string' ? savedScope : savedScope?.profile)

          // The Memory summary's "active" provider is the saved one; reread it.
          if ('memory' in patch) {
            refreshMemoryStatus()
          }

          if (mounted.current && pending.version === v) {
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
          if (mounted.current && pending.version === v) {
            notifyError(err, c.autosaveFailed)
          }
        }
      })
    }, 550)

    return () => window.clearTimeout(t)
    // eslint-disable-next-line react-hooks/exhaustive-deps -- copy is stable; avoid re-scheduling autosave on locale change
  }, [config, draft, onConfigSaved, saveVersion, scopeProfile])

  const applyConfig = (next: HermesConfigRecord) => {
    if (!draft.current) {
      return
    }

    draft.current.config = next
    draft.current.version += 1
    setConfig(next)
    setSaveVersion(draft.current.version)
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

  const fields = (sectionFields.get(activeSectionId) ?? []).filter(
    ([key]) =>
      (compactAgent && activeSectionId !== 'advanced') ||
      agentModel ||
      subpage === undefined ||
      configSubpageForField(activeSectionId, key) === subpage
  )

  const { showModelSettings, showDesktopSettings, showAttachments, portalSafety, showSharedMetrics } =
    useConfigSectionVisibility({ activeSectionId, agentModel, subpage, hasOwner: Boolean(owner) })

  // Deep-link target from the command palette (?field=<key>): scroll the row
  // into view and flash it, then drop the param so it doesn't re-fire.
  const [searchParams, setSearchParams] = useSearchParams()
  const targetField = searchParams.get('field')

  useSettingDeepLink(
    `config:${activeSectionId}`,
    page => (compactAgent && activeSectionId !== 'advanced') || subpage === undefined || page === subpage
  )

  useEffect(() => {
    if (!targetField || !config || !schema) {
      return
    }

    const element = document.getElementById(
      `setting-field-${targetField === DELEGATION_ROUTE_COMPANION_KEY ? 'delegation.model' : targetField}`
    )

    if (!element) {
      return
    }

    let disclosure = element.parentElement?.closest('details')

    while (disclosure) {
      disclosure.open = true
      disclosure = disclosure.parentElement?.closest('details')
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
  }, [activeSectionId, config, schema, setSearchParams, subpage, targetField])

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

  // Keep the model controller and pending MoA saves at a stable position when
  // selecting siblings or returning to the first page through the parent.
  const renderPage = (children: ReactNode) => (
    <SettingsContent>
      <SettingsProfileScope className="mb-5" />
      {activeSectionId === 'model' && (
        <div className={showModelSettings ? 'mb-6' : undefined}>
          <ModelSettings
            advancedChildren={agentModel ? children : undefined}
            advancedOpen={agentModel && Boolean(subpage && !['main', 'fallbacks'].includes(subpage))}
            agentLayout={agentModel}
            mainFooter={
              agentModel && config
                ? fields
                    .filter(([key]) => key === 'fallback_providers')
                    .map(([key, field]) => (
                      <div id={`setting-field-${key}`} key={key}>
                        <ConfigField
                          onChange={value => updateConfig(setConfigField(config, key, value))}
                          schema={field}
                          schemaKey={key}
                          value={configFieldValue(config, key)}
                        />
                      </div>
                    ))
                : undefined
            }
            onMainModelChanged={onMainModelChanged}
            scopeProfile={scopeProfile}
            subpage={agentModel ? undefined : subpage}
          />
        </div>
      )}
      {!agentModel && children}
      <input
        accept=".json,application/json"
        className="hidden"
        onChange={handleImport}
        ref={importInputRef}
        type="file"
      />
    </SettingsContent>
  )

  if (!config || !schema) {
    return (
      <ConfigLoading
        configFailed={configLoadFailed}
        configLoaded={Boolean(config)}
        model={showModelSettings}
        onRetry={() => {
          void refetchConfig()
          void refetchSchema()
        }}
        schemaFailed={schemaFailed}
        schemaLoaded={Boolean(schema)}
        subpage={subpage}
      />
    )
  }

  const visibleFields = (
    activeSectionId === 'voice' ? fields.filter(([key]) => voiceFieldVisible(key, config)) : fields
  ).filter(([key]) => key !== DELEGATION_ROUTE_COMPANION_KEY && (!agentModel || key !== 'fallback_providers'))

  if (portalSafety) {
    return renderPage(
      <PortalSafetySettings
        config={config}
        fields={sectionFields.get('safety')}
        onChange={updateConfig}
        subpage={subpage}
      />
    )
  }

  const showEmptyState =
    visibleFields.length === 0 &&
    (subpage === undefined
      ? activeSectionId !== 'chat'
      : !showModelSettings && !showDesktopSettings && !showAttachments && !showSharedMetrics)

  const renderField = ([key, field]: [string, ConfigFieldSchema]) => (
    <div className="scroll-mt-6 rounded-lg" id={`setting-field-${key}`} key={key}>
      {key === 'delegation.model' ? (
        <SubagentModelGate scope={writeScope ?? scopeProfile}>
          <DelegationRouteRow config={config} onConfigChange={updateConfig} scope={writeScope ?? scopeProfile} />
        </SubagentModelGate>
      ) : (
        <ConfigField
          descriptionExtra={
            key === 'memory.provider' && isExternalMemoryProvider(getNested(config, key)) ? (
              <MemoryConnect profile={scopeProfile} provider={String(getNested(config, key))} />
            ) : undefined
          }
          descriptionOverride={
            compactAgent && key === 'voice.voice_chat_mode' ? agentCopy.voiceModeDescription : undefined
          }
          enumOptions={
            key === 'tts.elevenlabs.voice_id'
              ? enumOptionsFor(key, getNested(config, key), config, elevenLabsVoiceOptions ?? undefined)
              : enumOptionsFor(key, getNested(config, key), config)
          }
          onChange={value => updateConfig(setConfigField(config, key, value))}
          optionLabels={
            key === 'tts.elevenlabs.voice_id'
              ? elevenLabsVoiceLabels
              : compactAgent && key === 'voice.voice_chat_mode'
                ? VOICE_MODE_LABELS
                : undefined
          }
          schema={field}
          schemaKey={key}
          segmented={compactAgent && key === 'voice.voice_chat_mode'}
          value={configFieldValue(config, key)}
        />
      )}
      {key === 'memory.provider' && isExternalMemoryProvider(getNested(config, key)) ? (
        <ProviderConfigPanel
          key={String(getNested(config, key))}
          profile={scopeProfile}
          provider={String(getNested(config, key))}
        />
      ) : null}
    </div>
  )

  return renderPage(
    <>
      {/* Device-local desktop prefs (not config.yaml) — they live here since
          keeping the machine awake and the global Quick Entry chord are both
          power-user, this-computer-only knobs. */}
      {showDesktopSettings && (
        <>
          <ListRow
            action={
              <SegmentedControl
                onChange={mode => {
                  triggerHaptic('selection')
                  setKeepAwakeMode(mode)
                }}
                options={keepAwakeOptions}
                value={keepAwakeMode}
              />
            }
            description={c.keepAwakeDesc}
            id={settingElementId(SETTING_IDS.advanced.keepAwake)}
            title={c.keepAwakeTitle}
          />
          <ToggleRow
            checked={disableF12}
            description={c.disableF12Desc}
            id={settingElementId(SETTING_IDS.advanced.disableF12)}
            label={c.disableF12Title}
            onChange={setDisableF12}
          />
          <ToggleRow
            checked={alwaysExternalLinks}
            description={c.alwaysExternalLinksDesc}
            label={c.alwaysExternalLinksTitle}
            onChange={setAlwaysExternalLinks}
          />
          <QuickEntrySettings />
          <OnboardingSettings />
        </>
      )}
      {/* Device-local attach/preview byte cap (main-process IPC guard). Chat is
          where image-attachment behavior already lives, so this sits above the
          schema fields for that section. */}
      {showAttachments ? <AttachmentSizeSetting /> : null}
      {/* Shared metrics are two coupled opt-ins with a consent side effect, so they
          go through their own RPC rather than the generic field autosave. */}
      {showSharedMetrics ? <SharedMetricsSettings /> : null}
      <MemoryLayersSlot
        builtin={{
          memory: getNested(config, 'memory.memory_enabled') !== false,
          user: getNested(config, 'memory.user_profile_enabled') !== false
        }}
        profile={scopeProfile}
        provider={asText(getNested(config, 'memory.provider'))}
        sectionId={activeSectionId}
        subpage={subpage}
      />
      {/* Routing picks the strategy and owns its routes; the subagent model row below is the
          Default strategy's route and goes inert while another strategy decides. */}
      <DelegationRoutingSlot
        readScope={scopeProfile}
        sectionId={activeSectionId}
        subpage={agentModel ? undefined : subpage}
        writeScope={writeScope}
      />
      {activeSectionId === 'voice' && !compactAgent ? (
        <ListRow description={c.voiceShortcutHintDesc} title={c.voiceShortcutHintTitle} />
      ) : null}
      {showEmptyState ? (
        <EmptyState description={c.emptyDesc} title={c.emptyTitle} />
      ) : visibleFields.length === 0 ? null : compactAgent ? (
        <AgentConfigFields fields={visibleFields} renderField={renderField} sectionId={activeSectionId} />
      ) : (
        <div className="grid gap-1">{visibleFields.map(renderField)}</div>
      )}
    </>
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
              if (isSubmitEnter(event)) {
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
      id={settingElementId(SETTING_IDS.chat.attachmentSize)}
      title={c.attachmentSizeTitle}
    />
  )
}
