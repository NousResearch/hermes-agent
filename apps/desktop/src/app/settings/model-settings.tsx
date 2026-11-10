import type { ModelOptionProvider } from '@hermes/shared'
import { DEFAULT_REASONING_EFFORT, isReasoningEffort, REASONING_EFFORT_VALUES } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { Switch } from '@/components/ui/switch'
import {
  getAuxiliaryModels,
  getGlobalModelInfo,
  getGlobalModelOptions,
  getMoaModels,
  getRecommendedDefaultModel,
  saveHermesConfig,
  saveMoaModels,
  setEnvVar,
  setModelAssignment
} from '@/hermes'
import type {
  AuxiliaryModelsResponse,
  AuxiliaryTaskAssignment,
  MoaConfigResponse,
  MoaModelSlot,
  StaleAuxAssignment
} from '@/hermes'
import { useI18n } from '@/i18n'
import { isCodeSkewRestartRequired } from '@/lib/code-skew-error'
import './model-settings.css'

import { AlertTriangle, Cpu, Loader2, Plus } from '@/lib/icons'
import { isSubmitEnter } from '@/lib/ime'
import { findCatalogProvider } from '@/lib/model-options'
import { cn } from '@/lib/utils'
import { $customModels, withCustomModels } from '@/store/custom-models'
import { setMainModelAssignment } from '@/store/model-assignment'
import { notify, notifyError, readableError } from '@/store/notifications'
import { startManualLocalEndpoint, startManualOnboarding, startManualProviderOAuth } from '@/store/onboarding'

import { hermesConfigCacheWriter, invalidateHermesConfig, useHermesConfigRecord } from '../hooks/use-config-record'
import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'
import { PanelEmpty } from '../overlays/panel'

import { CONTROL_TEXT } from './constants'
import { getNested, setNested } from './helpers'
import { ModelSelect, withActive } from './model-select'
import { SectionHeading } from './primitives'
import { dismissStaleAux, readStaleAuxDismissal, staleAuxFingerprint } from './stale-aux-dismissal'
import { useDeepLinkHighlight } from './use-deep-link-highlight'

// Skeleton mirror of the Model settings DOM so the page keeps its shape while
// the provider/model catalog loads, instead of collapsing to a centered
// spinner. Same containers/rhythm as the real render below.
export function ModelSettingsSkeleton({ subpage }: Pick<ModelSettingsProps, 'subpage'> = {}) {
  return (
    <div className="grid gap-6" data-slot="model-settings-skeleton">
      {(subpage === undefined || subpage === 'main') && (
        <section className="grid gap-3.5">
          <div className="space-y-2">
            <Skeleton className="h-9 w-48" />
            <Skeleton className="h-4 w-full max-w-xl" />
          </div>
          <div className="rounded-[20px] border border-[#e2e8f0] bg-white p-5 shadow-[0_4px_16px_rgba(38,64,140,0.06)]">
            <Skeleton className="mb-4 h-3 w-14" />
            <Skeleton className="mb-4 h-12 w-full rounded-xl" />
            <Skeleton className="mb-4 h-4 w-28" />
            <div className="flex gap-3">
              <Skeleton className="h-8 w-20 rounded-full" />
              <Skeleton className="h-8 w-24 rounded-full" />
              <Skeleton className="h-8 w-28 rounded-full" />
            </div>
          </div>
          <div className="rounded-[20px] border border-[#e2e8f0] bg-white p-5 shadow-[0_4px_16px_rgba(38,64,140,0.06)]">
            <div className="flex items-center justify-between gap-4">
              <div className="min-w-0 flex-1 space-y-2">
                <Skeleton className="h-4 w-64" />
                <Skeleton className="h-8 w-full max-w-lg" />
              </div>
              <Skeleton className="h-10 w-[88px] rounded-[10px]" />
            </div>
          </div>
        </section>
      )}

      {(subpage === undefined || subpage === 'auxiliary') && (
        <section className="grid gap-3.5">
          <div className="space-y-2">
            <Skeleton className="h-9 w-64" />
            <Skeleton className="h-4 w-full max-w-2xl" />
          </div>
          <div className="overflow-hidden rounded-[20px] border border-[#e2e8f0] bg-white shadow-[0_4px_16px_rgba(38,64,140,0.06)]">
            {[0, 1, 2, 3, 4].map(row => (
              <div className="flex items-center justify-between gap-4 border-t border-[#e2e8f0] px-5 py-4 first:border-t-0" key={row}>
                <div className="min-w-0 flex-1 space-y-2">
                  <Skeleton className="h-4 w-48" />
                  <Skeleton className="h-3 w-36" />
                </div>
                <div className="flex gap-2">
                  <Skeleton className="h-8 w-24 rounded-[10px]" />
                  <Skeleton className="h-8 w-20 rounded-[10px]" />
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {subpage === 'moa' && (
        <section className="grid gap-3.5">
          <div className="space-y-2">
            <Skeleton className="h-9 w-72" />
            <Skeleton className="h-4 w-full max-w-3xl" />
            <Skeleton className="h-4 w-2/3 max-w-xl" />
          </div>
          <div className="space-y-4 rounded-[20px] border border-[#e2e8f0] bg-white p-5 shadow-[0_4px_16px_rgba(38,64,140,0.06)]">
            <Skeleton className="h-3 w-14" />
            <Skeleton className="h-10 w-full rounded-xl" />
            <div className="flex flex-wrap gap-2">
              <Skeleton className="h-8 w-28 rounded-[10px]" />
              <Skeleton className="h-8 w-24 rounded-[10px]" />
              <Skeleton className="h-8 w-20 rounded-[10px]" />
              <Skeleton className="h-8 flex-1 rounded-xl" />
              <Skeleton className="h-9 w-32 rounded-[10px]" />
            </div>
          </div>
          <div className="space-y-3 rounded-[20px] border border-[#e2e8f0] bg-white p-5 shadow-[0_4px_16px_rgba(38,64,140,0.06)]">
            <Skeleton className="h-4 w-36" />
            <Skeleton className="h-36 w-full rounded-2xl" />
            <Skeleton className="h-36 w-full rounded-2xl" />
            <Skeleton className="h-10 w-full rounded-xl" />
          </div>
          <div className="space-y-3 rounded-[20px] border border-[#e2e8f0] bg-white p-5 shadow-[0_4px_16px_rgba(38,64,140,0.06)]">
            <Skeleton className="h-4 w-28" />
            <Skeleton className="h-10 w-full rounded-xl" />
          </div>
        </section>
      )}
    </div>
  )
}

// agent.service_tier stores "fast"/"priority"/"on" for fast and "ultrafast" for OpenAI
// Ultrafast; anything else is normal (mirrors agent.fast_mode.parse_service_tier).
const isFastTier = (tier: unknown): boolean =>
  ['fast', 'priority', 'on', 'ultrafast'].includes(
    String(tier ?? '')
      .trim()
      .toLowerCase()
  )

// A provider row is "ready" to pick a model from when it reports models. The
// backend now surfaces the full `hermes model` universe (every canonical
// provider), so unconfigured providers come back with `authenticated:false`
// and an empty `models` list — those need a setup step before a model exists.
function isProviderReady(p?: ModelOptionProvider): boolean {
  return !!p && (p.authenticated !== false || (p.models?.length ?? 0) > 0)
}

// Mirrors `_AUX_TASK_SLOTS` in hermes_cli/web_server.py. Friendly labels and
// hints make the assignments readable; raw task keys (vision, mcp, …) are
// opaque to most users.
interface AuxTaskMeta {
  key: string
}

const AUX_TASKS: readonly AuxTaskMeta[] = [
  { key: 'vision' },
  { key: 'compression' },
  { key: 'skills_hub' },
  { key: 'approval' },
  { key: 'mcp' },
  { key: 'title_generation' },
  { key: 'review' },
  // Same three canonical slots the backend serves but the list below used to
  // omit (#97297): triage_specifier, kanban_decomposer, profile_describer.
  { key: 'triage_specifier' },
  { key: 'kanban_decomposer' },
  { key: 'profile_describer' },
  { key: 'curator' }
]

const NO_PROVIDERS: readonly ModelOptionProvider[] = [{ name: '—', slug: '', models: [] }]

// A slot is complete when both halves are chosen. Changing a slot's provider
// intentionally clears its model (see updateMoaSlot), so every provider change
// passes through an incomplete state while the user picks the new model.
export const moaSlotComplete = (slot: MoaModelSlot): boolean => !!(slot.provider.trim() && slot.model.trim())

// True when every slot in every preset is fully specified — the only state
// that is safe to persist. The backend rejects configs with half-filled slots
// (HTTP 422) instead of silently swapping the preset for hardcoded defaults
// (#64156), so the autosave must simply wait for the edit to finish rather
// than trying to "repair" the payload.
export const moaConfigComplete = (config: MoaConfigResponse): boolean =>
  Object.values(config.presets).every(
    preset =>
      preset.reference_models.length > 0 &&
      preset.reference_models.every(moaSlotComplete) &&
      moaSlotComplete(preset.aggregator)
  )

// Persistent mismatch: any aux slot pinned to a provider different from the
// current main, regardless of whether the user just switched. Catches the
// "I pinned aux months ago and forgot, now it bills a dead provider" case.
// A pin on a private/LAN endpoint (per-task base_url, e.g. a home Ollama box)
// never bills a provider, so the backend's `local_endpoint` verdict exempts it.
export function staleAuxAssignments(
  tasks: readonly AuxiliaryTaskAssignment[],
  mainProvider: string
): StaleAuxAssignment[] {
  const main = mainProvider.toLowerCase()

  if (!main) {
    return []
  }

  return (
    tasks
      .filter(entry => {
        const p = (entry.provider ?? '').toLowerCase()

        // 'main' is a backend alias meaning "follow the current main provider"
        // (auxiliary_client._normalize_aux_provider), so it can never be a stale pin.
        return p && p !== 'auto' && p !== 'main' && p !== main && !entry.local_endpoint
      })
      // base_url rides along for the dismissal fingerprint (see
      // stale-aux-dismissal.ts): repointing a pin at a different endpoint changes
      // the billing surface and must re-arm an acknowledged banner.
      .map(entry => ({ base_url: entry.base_url, task: entry.task, provider: entry.provider, model: entry.model }))
  )
}

interface StaleAuxWarningProps {
  applying: boolean
  /** Offered only on the persistent variant — the post-switch notice announces
   *  a change that just happened and must not be silenced. */
  onDismiss?: () => void
  onReset: () => void
  slots: readonly StaleAuxAssignment[]
  taskLabel: (key: string) => string
}

// Shared notice: auxiliary tasks still pinned to a provider that isn't the
// current main. Surfaces the silent credit-burn path (e.g. aux pinned to a
// $0-balance provider after switching main away from it) and offers the
// existing one-click reset rather than auto-clearing legitimate pins.
// Sized to be read at a glance (#66740) with the theme-aware amber text the
// app's warn badges use, so light mode keeps its contrast.
function StaleAuxWarning({ applying, onDismiss, onReset, slots, taskLabel }: StaleAuxWarningProps) {
  const { t } = useI18n()
  const m = t.settings.model

  if (!slots.length) {
    return null
  }

  const provider = slots[0].provider
  const allSameProvider = slots.every(slot => slot.provider === provider)
  const names = slots.map(slot => taskLabel(slot.task)).join(', ')

  return (
    <div className="flex flex-wrap items-center gap-2 rounded-md border border-amber-400/60 bg-amber-500/15 px-3 py-2.5 text-sm text-amber-600 dark:text-amber-300">
      <AlertTriangle className="size-4 shrink-0" />
      <span className="grow font-medium">
        {m.staleAuxBefore(slots.length, names)}
        <span className="font-mono">{allSameProvider ? provider : m.staleAuxOtherProviders}</span>
        {m.staleAuxAfter}
      </span>
      <Button disabled={applying} onClick={onReset} size="sm" variant="textStrong">
        {m.resetAllToMain}
      </Button>
      {onDismiss && (
        <Button disabled={applying} onClick={onDismiss} size="sm" variant="textStrong">
          {m.staleAuxDismiss}
        </Button>
      )}
    </div>
  )
}

interface ModelSettingsProps {
  /** Visibility only: changing pages must not reset drafts or cancel autosave. */
  subpage?: string
  /** Notified after the main model is applied, so live UI stores can sync. */
  onMainModelChanged?: (provider: string, model: string) => void
  /** Shared settings "Applies to" scope: a concrete profile to edit instead of
   *  the app's active one, or undefined to follow the active profile (default).
   *  Request-shaped on purpose — the API helpers treat `null` as "deliberately
   *  target the primary/default backend", so this prop never carries null. */
  scopeProfile?: string
}

export function ModelSettings({ onMainModelChanged, scopeProfile, subpage }: ModelSettingsProps) {
  const { t } = useI18n()
  const m = t.settings.model
  const showMain = subpage === undefined || subpage === 'main'
  const showAuxiliary = subpage === undefined || subpage === 'auxiliary'
  const showMoa = subpage === undefined || subpage === 'moa'
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [skewRestart, setSkewRestart] = useState(false)
  const [restartingBackend, setRestartingBackend] = useState(false)
  const [mainModel, setMainModel] = useState<{ model: string; provider: string } | null>(null)
  const [catalogProviders, setCatalogProviders] = useState<ModelOptionProvider[]>([])
  // Slugs typed into any picker ride along as rows of their provider, so a
  // model added from the composer is selectable here too.
  const customModels = useStore($customModels)
  const providers = useMemo(() => withCustomModels(catalogProviders, customModels), [catalogProviders, customModels])
  const [selectedProvider, setSelectedProvider] = useState('')
  const [selectedModel, setSelectedModel] = useState('')
  const [auxiliary, setAuxiliary] = useState<AuxiliaryModelsResponse | null>(null)
  const [moa, setMoa] = useState<MoaConfigResponse | null>(null)
  const [selectedMoaPreset, setSelectedMoaPreset] = useState('')
  const [newMoaPresetName, setNewMoaPresetName] = useState('')
  // agent.* defaults round-trip through the shared config cache (read → write
  // back the whole record), so a save here shows in the MCP/model surfaces.
  const { data: config, writeScope } = useHermesConfigRecord(scopeProfile)
  const setConfig = useMemo(() => hermesConfigCacheWriter(scopeProfile), [scopeProfile])
  const [applying, setApplying] = useState(false)
  const [editingAuxTask, setEditingAuxTask] = useState<null | string>(null)

  const [auxDraft, setAuxDraft] = useState<{ model: string; provider: string; reasoningEffort: string }>({
    model: '',
    provider: '',
    reasoningEffort: '__inherit__'
  })

  // Aux slots reported stale by the backend immediately after a main-model
  // switch (provider differs from the new main). Cleared on next switch/reset.
  const [switchStaleAux, setSwitchStaleAux] = useState<StaleAuxAssignment[]>([])
  // Inline API-key entry for picking an unconfigured `api_key` provider in
  // place — mirrors the onboarding ApiKeyForm but scoped to the model picker.
  const [apiKeyDraft, setApiKeyDraft] = useState('')
  const [activating, setActivating] = useState(false)

  // Deep link from the vision Capabilities detail (?tab=config:model&aux=vision):
  // scroll the auxiliary task row into view and flash it once the list loads.
  useDeepLinkHighlight({
    elementId: task => `aux-task-${task}`,
    param: 'aux',
    ready: task => showAuxiliary && !loading && AUX_TASKS.some(meta => meta.key === task)
  })

  // Every profile-scoped async here captures this and bails before writing back,
  // so a request in flight when the user switches profiles can't paint profile
  // A's models/providers into profile B (or fire onMainModelChanged for A).
  const profileEpoch = useRef(0)

  const setCaughtError = useCallback(
    (err: unknown, fallback: string) => {
      const skew = isCodeSkewRestartRequired(err)
      setSkewRestart(skew)
      setError(skew ? m.restartRequired : readableError(err, fallback).message)
    },
    [m.restartRequired]
  )

  const refresh = useCallback(
    async ({ replaceSelection = false }: { replaceSelection?: boolean } = {}) => {
      const epoch = profileEpoch.current
      setLoading(true)
      setError('')
      setSkewRestart(false)

      try {
        // Degrade per call: a hung /api/model/info (its context-length probe
        // hits the configured provider, which can be unreachable) must not
        // block the config-backed sections — auxiliary and MOA are fast
        // config-file reads — behind a single all-or-nothing Promise.all.
        const [modelInfoResult, modelOptionsResult, auxiliaryModelsResult, moaModelsResult] = await Promise.allSettled([
          getGlobalModelInfo(scopeProfile),
          getGlobalModelOptions(undefined, scopeProfile),
          getAuxiliaryModels(scopeProfile),
          getMoaModels(scopeProfile)
        ])

        if (profileEpoch.current !== epoch) {
          return
        }

        const failures: string[] = []

        const settledValue = <T,>(result: PromiseSettledResult<T>): T | null => {
          if (result.status === 'fulfilled') {
            return result.value
          }

          failures.push(result.reason instanceof Error ? result.reason.message : String(result.reason))

          return null
        }

        const modelInfo = settledValue(modelInfoResult)
        const modelOptions = settledValue(modelOptionsResult)
        const auxiliaryModels = settledValue(auxiliaryModelsResult)
        // MOA has always been optional-on-failure; keep it out of the banner.
        const moaModels = moaModelsResult.status === 'fulfilled' ? moaModelsResult.value : null
        // The main assignment also lives in the auxiliary config read, so the
        // page still knows the current model when only the live probe failed.
        const resolvedMain = modelInfo ?? auxiliaryModels?.main ?? null

        if (resolvedMain) {
          setMainModel({ model: resolvedMain.model, provider: resolvedMain.provider })

          if (replaceSelection) {
            setSelectedProvider(resolvedMain.provider)
            setSelectedModel(resolvedMain.model)
          } else {
            setSelectedProvider(prev => prev || resolvedMain.provider)
            setSelectedModel(prev => prev || resolvedMain.model)
          }
        }

        if (modelOptions) {
          setCatalogProviders(modelOptions.providers || [])
        }

        setAuxiliary(auxiliaryModels)
        setMoa(moaModels)

        if (moaModels) {
          setSelectedMoaPreset(prev => (prev && moaModels.presets[prev] ? prev : moaModels.default_preset))
        }

        if (failures.length > 0) {
          setCaughtError(new Error(failures.join('; ')), m.loadFailed)
        }

        // The config record loads via its own shared query; a model switch can
        // change it server-side (aux slots), so nudge that cache to refetch.
        void invalidateHermesConfig(scopeProfile)
      } catch (err) {
        if (profileEpoch.current === epoch) {
          setCaughtError(err, m.loadFailed)
        }
      } finally {
        if (profileEpoch.current === epoch) {
          setLoading(false)
        }
      }
    },
    [m.loadFailed, scopeProfile, setCaughtError]
  )

  useEffect(() => {
    void refresh()
  }, [refresh])

  // A profile switch swaps the backend under the mounted panel — reload for the
  // new profile (bumping the epoch first so any in-flight A request is discarded).
  useOnProfileSwitch(() => {
    profileEpoch.current += 1
    // The panel stays mounted across profile switches, so clear the previous
    // profile's draft selection before loading the new profile's source of
    // truth. Ordinary same-profile refreshes still preserve in-progress edits.
    setSelectedProvider('')
    setSelectedModel('')
    setApiKeyDraft('')
    void refresh({ replaceSelection: true })
  })

  const providerOptions = providers.length ? providers : NO_PROVIDERS

  // Radix renders a blank trigger when the controlled value has no matching
  // item. Keep a missing saved provider visible in the main selector while
  // leaving it out of the real inventory used for readiness/setup metadata.
  const mainProviderOptions = useMemo(
    () =>
      selectedProvider && !findCatalogProvider(providers, selectedProvider)
        ? [{ name: selectedProvider, slug: selectedProvider, models: [] }, ...providers]
        : providerOptions,
    [providerOptions, providers, selectedProvider]
  )

  // MoA reference/aggregator slots must never be the moa virtual provider —
  // that would create a recursive MoA tree (the backend rejects it on save).
  // Hide it from the slot selectors so it isn't offered as a dead choice.
  const moaSlotProviderOptions = providerOptions.filter(provider => (provider.slug || '').toLowerCase() !== 'moa')

  const selectedProviderRow = useMemo(
    () => findCatalogProvider(providers, selectedProvider),
    [providers, selectedProvider]
  )

  const selectedProviderModels = selectedProviderRow?.models ?? []

  // An unconfigured provider was picked: no credentials yet, so there are no
  // models to choose. `api_key` providers can be activated inline (paste key);
  // OAuth / external flows hand off to the onboarding sign-in.
  const needsSetup = !!selectedProvider && !isProviderReady(selectedProviderRow)
  const setupIsApiKey = needsSetup && selectedProviderRow?.auth_type === 'api_key' && !!selectedProviderRow?.key_env

  // Clear any half-typed key when switching provider so it can't leak across.
  useEffect(() => {
    setApiKeyDraft('')
  }, [selectedProvider])

  const auxDraftProviderModels = useMemo(
    () => findCatalogProvider(providers, auxDraft.provider)?.models ?? [],
    [auxDraft.provider, providers]
  )

  const modelsForProvider = useCallback(
    (provider: string) => findCatalogProvider(providers, provider)?.models ?? [],
    [providers]
  )

  const currentMoaPreset = useMemo(() => {
    if (!moa) {
      return null
    }

    return moa.presets[selectedMoaPreset] || moa.presets[moa.default_preset] || Object.values(moa.presets)[0] || null
  }, [moa, selectedMoaPreset])

  // Mirror of `moa` so inline edits compute the next state purely (outside the
  // setState updater) and hand it straight to the debounced autosave.
  const moaRef = useRef<MoaConfigResponse | null>(null)

  // eslint-disable-next-line no-restricted-syntax -- legitimate non-atom ref write (see eslint rule comment)
  useEffect(() => {
    moaRef.current = moa
  }, [moa])

  const moaSaveTimer = useRef<number | null>(null)

  useEffect(
    () => () => {
      if (moaSaveTimer.current) {
        window.clearTimeout(moaSaveTimer.current)
      }
    },
    []
  )

  // Guard against stale save responses overwriting newer state.
  const moaSaveGeneration = useRef(0)

  // Quiet debounced persist for inline MoA edits — mirrors the config page's
  // autosave so slot/aggregator tweaks save themselves, matching the
  // preset-level ops (set default / add / delete) that already persist on
  // click. No `applying` spinner, so selecting stays responsive.
  //
  // While any slot is half-filled (provider picked, model pending) the save is
  // HELD, not sent: the previous complete config stays on disk and the next
  // edit that completes the slot flushes the whole preset. Every edit bumps
  // the generation so an in-flight response from an older save can never
  // repaint over the user's mid-edit state.
  const scheduleMoaSave = useCallback(
    (next: MoaConfigResponse) => {
      if (moaSaveTimer.current) {
        window.clearTimeout(moaSaveTimer.current)
        moaSaveTimer.current = null
      }

      const generation = moaSaveGeneration.current + 1
      moaSaveGeneration.current = generation

      if (!moaConfigComplete(next)) {
        return
      }

      moaSaveTimer.current = window.setTimeout(() => {
        void saveMoaModels(next, scopeProfile)
          .then(saved => {
            if (moaSaveGeneration.current === generation) {
              setMoa(saved)
            }
          })
          .catch(err => {
            if (moaSaveGeneration.current === generation) {
              setCaughtError(err, m.loadFailed)
            }
          })
      }, 600)
    },
    [m.loadFailed, scopeProfile, setCaughtError]
  )

  const updateMoaPreset = useCallback(
    (updater: (preset: NonNullable<typeof currentMoaPreset>) => NonNullable<typeof currentMoaPreset>) => {
      const prev = moaRef.current

      if (!prev || !selectedMoaPreset || !prev.presets[selectedMoaPreset]) {
        return
      }

      const next: MoaConfigResponse = {
        ...prev,
        presets: {
          ...prev.presets,
          [selectedMoaPreset]: updater(prev.presets[selectedMoaPreset])
        }
      }

      moaRef.current = next
      setMoa(next)
      scheduleMoaSave(next)
    },
    [scheduleMoaSave, selectedMoaPreset]
  )

  const updateMoaSlot = useCallback((slot: MoaModelSlot, patch: Partial<MoaModelSlot>): MoaModelSlot => {
    const next = { ...slot, ...patch }

    // Picking a new provider invalidates the model choice (models are
    // per-provider). A same-provider update must not wipe the model — Radix
    // filters same-value changes, but programmatic callers may not.
    if (patch.provider && patch.provider !== slot.provider) {
      next.model = ''
    }

    return next
  }, [])

  const saveMoa = useCallback(
    async (next: MoaConfigResponse) => {
      const epoch = profileEpoch.current

      // Explicit preset ops (set default / add / delete) supersede any pending
      // debounced slot autosave — cancel it and invalidate in-flight responses
      // so the two writers can't race each other's state.
      if (moaSaveTimer.current) {
        window.clearTimeout(moaSaveTimer.current)
        moaSaveTimer.current = null
      }

      moaSaveGeneration.current += 1
      setApplying(true)
      setError('')

      try {
        const saved = await saveMoaModels(next, scopeProfile)

        if (profileEpoch.current !== epoch) {
          return
        }

        setMoa(saved)
      } catch (err) {
        setCaughtError(err, m.loadFailed)
      } finally {
        setApplying(false)
      }
    },
    [m.loadFailed, scopeProfile, setCaughtError]
  )

  const auxiliaryTaskLabel = useCallback((key: string) => m.tasks[key]?.label ?? key, [m.tasks])

  const persistentStaleAux = useMemo<StaleAuxAssignment[]>(
    () => staleAuxAssignments(auxiliary?.tasks ?? [], mainModel?.provider ?? ''),
    [auxiliary, mainModel]
  )

  // Acknowledgement of the persistent stale-aux banner (#66740): a dismissal
  // is bound to the exact pin configuration it acknowledged, so any slot edit,
  // main switch, or endpoint repoint produces a different fingerprint and
  // re-arms the warning. Seeded lazily at first render (before the data can
  // paint, so an acknowledged banner never flashes); the panel stays mounted
  // across profile switches, so re-read when the scope changes.
  const [dismissedStaleAux, setDismissedStaleAux] = useState<null | string>(() => readStaleAuxDismissal(scopeProfile))

  useEffect(() => {
    setDismissedStaleAux(readStaleAuxDismissal(scopeProfile))
  }, [scopeProfile])

  const staleAuxDismissed =
    persistentStaleAux.length > 0 &&
    dismissedStaleAux === staleAuxFingerprint(mainModel?.provider ?? '', persistentStaleAux)

  // Capabilities of the APPLIED main model — gates the profile-default
  // reasoning/speed controls the same way the composer picker gates per-model
  // edits (reasoning defaults on, fast defaults off when unreported).
  const mainCaps = useMemo(() => {
    const row = mainModel ? findCatalogProvider(providers, mainModel.provider) : undefined

    return mainModel ? row?.capabilities?.[mainModel.model] : undefined
  }, [providers, mainModel])

  const reasoningSupported = mainCaps?.reasoning ?? true
  const fastSupported = mainCaps?.fast ?? false

  // Hand-written `reasoning_effort: false`/`off` reaches us as boolean false
  // ("false" once stringified) — show it as Off, not an empty select.
  const rawEffort = String(getNested(config ?? {}, 'agent.reasoning_effort') ?? '')
    .trim()
    .toLowerCase()

  const effortValue = rawEffort === 'false' || rawEffort === 'disabled' ? 'none' : rawEffort || DEFAULT_REASONING_EFFORT

  const fastOn = isFastTier(getNested(config ?? {}, 'agent.service_tier'))

  const contextLength = Number(getNested(config ?? {}, 'model_context_length') ?? 0) || 0
  const [contextDraft, setContextDraft] = useState(String(contextLength))

  useEffect(() => {
    setContextDraft(String(contextLength))
  }, [contextLength])

  // Persist a single agent.* default as a sparse patch (PUT /api/config
  // deep-merges onto disk). Never send the whole cached record: it is a
  // default-expanded snapshot, and echoing it back rewrites every key another
  // surface changed meanwhile — a CLI-pinned auxiliary slot came back as
  // provider "auto" / model "" (#95460). Optimistic, with rollback on failure.
  const writeAgentDefault = useCallback(
    async (key: string, value: string | number) => {
      if (!config) {
        return
      }

      const prev = config
      const next = setNested(config, key, value)
      setConfig(next)

      try {
        await saveHermesConfig(setNested({}, key, value), writeScope ?? scopeProfile)
      } catch (err) {
        setConfig(prev)
        notifyError(err, m.defaultsFailed)
      }
    },
    [config, m.defaultsFailed, scopeProfile, setConfig, writeScope]
  )

  const commitContextLength = useCallback(() => {
    const parsed = contextDraft.trim() === '' ? 0 : Number(contextDraft)
    const next = Number.isFinite(parsed) && parsed >= 0 ? Math.floor(parsed) : contextLength

    setContextDraft(String(next))

    if (next !== contextLength) {
      void writeAgentDefault('model_context_length', next)
    }
  }, [contextDraft, contextLength, writeAgentDefault])

  // Paste an API key for the selected `api_key` provider, persist it, then
  // refresh so the now-authenticated provider's models populate. Auto-selects
  // the recommended default model so the user can Apply in one more click.
  const activateApiKeyProvider = useCallback(async () => {
    const keyEnv = selectedProviderRow?.key_env
    const slug = selectedProviderRow?.slug

    if (!keyEnv || !slug || !apiKeyDraft.trim()) {
      return
    }

    const epoch = profileEpoch.current
    setActivating(true)
    setError('')

    try {
      await setEnvVar(keyEnv, apiKeyDraft.trim(), scopeProfile, { providerSetup: true })
      setApiKeyDraft('')

      // Pick a sensible default for the freshly-activated provider (mirrors
      // `hermes model` curation). Best-effort — fall through to the refreshed
      // model list if it fails.
      let nextModel = ''

      try {
        const rec = await getRecommendedDefaultModel(slug, scopeProfile)
        nextModel = rec.model || ''
      } catch {
        nextModel = ''
      }

      const options = await getGlobalModelOptions(undefined, scopeProfile)

      if (profileEpoch.current !== epoch) {
        return
      }

      setCatalogProviders(options.providers || [])
      const refreshedRow = options.providers?.find(p => p.slug === slug)
      const fallbackModel = refreshedRow?.models?.[0] ?? ''
      setSelectedModel(nextModel || fallbackModel)
    } catch (err) {
      setCaughtError(err, m.loadFailed)
    } finally {
      setActivating(false)
    }
  }, [apiKeyDraft, m.loadFailed, scopeProfile, selectedProviderRow, setCaughtError])

  // OAuth / external providers can't be activated with a pasted key — hand off
  // to the shared onboarding flow scoped to this provider's real sign-in. The
  // custom / local endpoint is NOT an OAuth provider, so it gets the dedicated
  // local-endpoint form (URL + optional API key) instead of being dead-ended
  // on the OAuth picker (the original "booted back to the first screen" loop).
  const startProviderSetup = useCallback(() => {
    const rowSlug = selectedProviderRow?.slug.trim() ?? ''
    const slug = rowSlug || selectedProvider.trim()

    if (!slug) {
      return
    }

    const lower = slug.toLowerCase()

    if (lower === 'custom' || lower === 'local' || lower.startsWith('custom:')) {
      startManualLocalEndpoint(null, scopeProfile)
    } else if (rowSlug) {
      startManualProviderOAuth(rowSlug, scopeProfile)
    } else {
      // An absent row has no trustworthy auth metadata. Open the generic
      // provider picker instead of deep-linking an unknown or stale slug.
      startManualOnboarding(undefined, scopeProfile)
    }
  }, [scopeProfile, selectedProvider, selectedProviderRow])

  const applyMainModel = useCallback(async () => {
    if (!selectedProvider || !selectedModel) {
      return
    }

    const epoch = profileEpoch.current
    setApplying(true)
    setError('')

    try {
      const result = await setMainModelAssignment(
        {
          model: selectedModel,
          provider: selectedProvider,
          ...(selectedProviderRow?.api_url ? { base_url: selectedProviderRow.api_url } : {})
        },
        scopeProfile
      )

      if (profileEpoch.current !== epoch) {
        return
      }

      const provider = result.provider || selectedProvider
      const model = result.model || selectedModel
      setMainModel({ provider, model })
      setSwitchStaleAux(result.stale_aux ?? [])
      notify({ kind: 'success', title: m.mainAppliedTitle, message: m.mainAppliedMessage(model) })

      // Live UI stores mirror the ACTIVE profile's model; a scoped apply
      // changed a different profile and must not repaint them.
      if (scopeProfile == null) {
        onMainModelChanged?.(provider, model)
      }

      await refresh()
    } catch (err) {
      setCaughtError(err, m.loadFailed)
    } finally {
      setApplying(false)
    }
  }, [
    m,
    onMainModelChanged,
    refresh,
    scopeProfile,
    selectedModel,
    selectedProvider,
    selectedProviderRow,
    setCaughtError
  ])

  // Sibling of the applyMainModel endpoint passthrough (#65254): auxiliary
  // assignments targeting a user-defined provider must carry that provider's
  // endpoint too, or the backend pins the slot without a base_url and the
  // aux resolver falls back to the (possibly different, possibly cleared)
  // main endpoint.
  const endpointForProvider = useCallback(
    (provider: string) => {
      const row = findCatalogProvider(providers, provider)

      return row?.api_url ? { base_url: row.api_url } : {}
    },
    [providers]
  )

  const setAuxiliaryToMain = useCallback(
    async (task: string) => {
      if (!mainModel) {
        return
      }

      setApplying(true)
      setError('')

      try {
        await setModelAssignment(
          {
            model: mainModel.model,
            provider: mainModel.provider,
            scope: 'auxiliary',
            task,
            ...endpointForProvider(mainModel.provider)
          },
          scopeProfile
        )
        await refresh()
      } catch (err) {
        setCaughtError(err, m.loadFailed)
      } finally {
        setApplying(false)
      }
    },
    [endpointForProvider, m.loadFailed, mainModel, refresh, scopeProfile, setCaughtError]
  )

  const applyAuxiliaryDraft = useCallback(
    async (task: string) => {
      if (!auxDraft.provider || !auxDraft.model) {
        return
      }

      setApplying(true)
      setError('')

      try {
        await setModelAssignment(
          {
            model: auxDraft.model,
            provider: auxDraft.provider,
            reasoning_effort: auxDraft.reasoningEffort === '__inherit__' ? null : auxDraft.reasoningEffort,
            scope: 'auxiliary',
            task,
            ...endpointForProvider(auxDraft.provider)
          },
          scopeProfile
        )
        setEditingAuxTask(null)
        await refresh()
      } catch (err) {
        setCaughtError(err, m.loadFailed)
      } finally {
        setApplying(false)
      }
    },
    [auxDraft, endpointForProvider, m.loadFailed, refresh, scopeProfile, setCaughtError]
  )

  const beginAuxiliaryEdit = useCallback(
    (task: string) => {
      const current = auxiliary?.tasks.find(entry => entry.task === task)

      const initialProvider =
        current?.provider && current.provider !== 'auto' ? current.provider : (mainModel?.provider ?? '')

      const initialModel = current?.model || mainModel?.model || ''
      const initialReasoningEffort = current?.reasoning_effort ?? '__inherit__'
      setAuxDraft({ provider: initialProvider, model: initialModel, reasoningEffort: initialReasoningEffort })
      setEditingAuxTask(task)
    },
    [auxiliary, mainModel]
  )

  const resetAuxiliaryModels = useCallback(async () => {
    if (!mainModel) {
      return
    }

    setApplying(true)
    setError('')

    try {
      await setModelAssignment(
        {
          model: mainModel.model,
          provider: mainModel.provider,
          scope: 'auxiliary',
          task: '__reset__'
        },
        scopeProfile
      )
      setSwitchStaleAux([])
      await refresh()
    } catch (err) {
      setCaughtError(err, m.loadFailed)
    } finally {
      setApplying(false)
    }
  }, [m.loadFailed, mainModel, refresh, scopeProfile, setCaughtError])

  const recycleStaleBackend = useCallback(async () => {
    setRestartingBackend(true)
    setError('')
    setSkewRestart(false)

    try {
      await window.hermesDesktop?.recycleBackend?.(scopeProfile)
      await refresh({ replaceSelection: true })
    } catch (err) {
      setCaughtError(err, m.restartFailed)
    } finally {
      setRestartingBackend(false)
    }
  }, [m.restartFailed, refresh, scopeProfile, setCaughtError])

  // Fallbacks are schema-backed controls in ConfigSettings. Keep this
  // controller alive there without rendering controls from another child page.
  if (!showMain && !showAuxiliary && !showMoa) {
    return null
  }

  if (loading && !mainModel) {
    return <ModelSettingsSkeleton subpage={subpage} />
  }

  const errorNotice = error && (
    <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-destructive">
      <span>{error}</span>
      {skewRestart && (
        <Button disabled={restartingBackend} onClick={() => void recycleStaleBackend()} size="sm" variant="textStrong">
          {restartingBackend && <Loader2 className="size-3.5 animate-spin" />}
          {restartingBackend ? m.restartingBackend : m.restartBackend}
        </Button>
      )}
    </div>
  )

  return (
    <div className="grid gap-6">
      {!showMain && errorNotice}
      {showMain && (
        <section className="grid gap-3.5" data-model-main>
          <header className="space-y-1.5">
            <h1 className="text-[32px] font-extrabold leading-none tracking-tight text-[#0f172a] dark:text-foreground">
              {t.settings.subpages.modelMain}
            </h1>
            <p className="text-[14px] text-[#475569]">{m.appliesDesc}</p>
          </header>

          <div className="flex flex-col gap-4 rounded-[20px] border border-[#e2e8f0] bg-white p-5 shadow-[0_4px_16px_rgba(38,64,140,0.06)] dark:border-(--ui-stroke-tertiary) dark:bg-(--ui-chat-bubble-background) dark:shadow-none">
            <p className="text-[13px] font-semibold text-[#64748b]">{m.model}</p>

            <Select onValueChange={setSelectedProvider} value={selectedProviderRow?.slug ?? selectedProvider}>
              <SelectTrigger
                className={cn(
                  'h-12 w-full rounded-xl border-[#e2e8f0] bg-[#f7f9fc] px-4 text-[15px] font-semibold text-[#0f172a] shadow-none dark:bg-(--ui-bg-quinary)',
                  CONTROL_TEXT
                )}
              >
                <SelectValue placeholder={m.provider} />
              </SelectTrigger>
              <SelectContent>
                {mainProviderOptions.map(provider => (
                  <SelectItem key={provider.slug || 'none'} value={provider.slug || 'none'}>
                    {provider.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            {needsSetup ? (
              setupIsApiKey ? (
                <div className="flex flex-wrap items-center gap-2">
                  <Input
                    autoComplete="off"
                    className={cn('min-w-60 flex-1', CONTROL_TEXT)}
                    onChange={event => setApiKeyDraft(event.target.value)}
                    onKeyDown={event => {
                      if (isSubmitEnter(event)) {
                        void activateApiKeyProvider()
                      }
                    }}
                    placeholder={`Paste ${selectedProviderRow?.key_env ?? 'API key'}`}
                    type="password"
                    value={apiKeyDraft}
                  />
                  <Button
                    disabled={!apiKeyDraft.trim() || activating}
                    onClick={() => void activateApiKeyProvider()}
                    size="sm"
                  >
                    {activating && <Loader2 className="size-3.5 animate-spin" />}
                    {activating ? 'Activating...' : 'Activate'}
                  </Button>
                </div>
              ) : (
                <Button
                  className="h-auto justify-start p-0 text-[14px] font-semibold text-[#2f5bea] hover:bg-transparent hover:text-[#1d4ed8]"
                  onClick={startProviderSetup}
                  size="sm"
                  variant="ghost"
                >
                  {m.setUpProvider(selectedProviderRow?.name ?? m.setupProviderFallback)}
                </Button>
              )
            ) : (
              <div className="flex flex-wrap items-center gap-2" data-model-select-row>
                <ModelSelect
                  className="min-w-0 flex-1"
                  models={selectedProviderModels}
                  onValueChange={setSelectedModel}
                  provider={selectedProviderRow}
                  providerSlug={selectedProvider}
                  value={selectedModel}
                />
                <Button
                  disabled={!selectedProvider || !selectedModel || applying}
                  onClick={() => void applyMainModel()}
                  size="sm"
                >
                  {applying && <Loader2 className="size-3.5 animate-spin" />}
                  {applying ? m.applying : t.common.apply}
                </Button>
              </div>
            )}

            {needsSetup && !setupIsApiKey && selectedProviderRow ? (
              <p className="text-[13px] text-[#64748b]">
                {selectedProviderRow?.auth_type === 'api_key'
                  ? `${selectedProviderRow?.name} needs an API key — set it up to choose a model.`
                  : `${selectedProviderRow?.name} signs in through your browser — Hermes runs the flow for you.`}
              </p>
            ) : null}

            {config && mainModel && (reasoningSupported || fastSupported) ? (
              <div className="flex flex-wrap items-center gap-3">
                <span className="rounded-full border border-[#9db7f0] bg-[#e3ecfd] px-3.5 py-2 text-[13px] font-semibold text-[#2f5bea]">
                  {m.defaultsLabel}
                </span>
                {reasoningSupported ? (
                  <div className="flex items-center">
                    <span className="rounded-full bg-white px-3.5 py-2 text-[13px] font-semibold text-[#0f172a] dark:bg-transparent dark:text-foreground">
                      {m.reasoning}
                    </span>
                    <Select
                      onValueChange={value => void writeAgentDefault('agent.reasoning_effort', value)}
                      value={effortValue}
                    >
                      <SelectTrigger
                        className={cn(
                          'h-auto min-w-[6.5rem] rounded-full border-[#e2e8f0] bg-white py-2 pl-3.5 pr-3 text-[13px] font-semibold text-[#0f172a] shadow-none dark:bg-(--ui-bg-quinary)',
                          CONTROL_TEXT
                        )}
                      >
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {REASONING_EFFORT_VALUES.map(value => (
                          <SelectItem key={value} value={value}>
                            {value === 'none' ? m.reasoningOff : t.shell.modelOptions[value]}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                ) : null}
                {fastSupported ? (
                  <label className="flex items-center gap-2 rounded-full border border-[#e2e8f0] bg-white px-3.5 py-2 text-[13px] font-semibold text-[#0f172a] dark:bg-(--ui-bg-quinary) dark:text-foreground">
                    {t.shell.modelOptions.fast}
                    <Switch
                      checked={fastOn}
                      className="data-[state=checked]:border-transparent data-[state=checked]:bg-[#7c3aed]"
                      onCheckedChange={checked =>
                        void writeAgentDefault('agent.service_tier', checked ? 'fast' : 'normal')
                      }
                      size="xs"
                    />
                  </label>
                ) : null}
              </div>
            ) : null}

            {errorNotice}
            {switchStaleAux.length > 0 ? (
              <StaleAuxWarning
                applying={applying}
                onReset={() => void resetAuxiliaryModels()}
                slots={switchStaleAux}
                taskLabel={auxiliaryTaskLabel}
              />
            ) : null}
          </div>

          <div className="flex items-center justify-between gap-4 rounded-[20px] border border-[#e2e8f0] bg-white p-5 shadow-[0_4px_16px_rgba(38,64,140,0.06)] dark:border-(--ui-stroke-tertiary) dark:bg-(--ui-chat-bubble-background) dark:shadow-none">
            <div className="min-w-0 flex-1 space-y-1.5">
              <h2 className="text-[16px] font-semibold text-[#0f172a] dark:text-foreground">
                {t.settings.fieldLabels.modelContextLength}
              </h2>
              <p className="max-w-3xl text-[13px] leading-snug text-[#475569]">
                {t.settings.fieldDescriptions.modelContextLength}
              </p>
            </div>
            <Input
              aria-label={t.settings.fieldLabels.modelContextLength}
              className="h-10 w-[88px] rounded-[10px] border-[#e2e8f0] bg-[#f7f9fc] text-center text-[15px] font-semibold text-[#0f172a] shadow-none dark:bg-(--ui-bg-quinary)"
              inputMode="numeric"
              min={0}
              onBlur={commitContextLength}
              onChange={event => setContextDraft(event.target.value)}
              onKeyDown={event => {
                if (isSubmitEnter(event)) {
                  event.currentTarget.blur()
                }
              }}
              type="number"
              value={contextDraft}
            />
          </div>
        </section>
      )}

      {showAuxiliary && (
        <section className="grid gap-3.5" data-model-auxiliary>
          <header className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0 flex-1 space-y-1.5">
              {subpage === undefined ? (
                <SectionHeading icon={Cpu} title={m.auxiliaryTitle} />
              ) : (
                <h1 className="text-[32px] font-extrabold leading-none tracking-tight text-[#0f172a] dark:text-foreground">
                  {m.auxiliaryTitle}
                </h1>
              )}
              <p className="max-w-3xl text-[14px] text-[#475569]">{m.auxiliaryDesc}</p>
            </div>
            <Button
              className="shrink-0 text-[13px] font-semibold text-[#2f5bea] hover:bg-transparent hover:text-[#1d4ed8]"
              disabled={!mainModel || applying}
              onClick={() => void resetAuxiliaryModels()}
              size="sm"
              variant="ghost"
            >
              {m.resetAllToMain}
            </Button>
          </header>

          {(switchStaleAux.length === 0 || !showMain) && persistentStaleAux.length > 0 && !staleAuxDismissed && (
            <StaleAuxWarning
              applying={applying}
              onDismiss={() => {
                const mainProvider = mainModel?.provider ?? ''

                dismissStaleAux(scopeProfile, mainProvider, persistentStaleAux)
                setDismissedStaleAux(staleAuxFingerprint(mainProvider, persistentStaleAux))
              }}
              onReset={() => void resetAuxiliaryModels()}
              slots={persistentStaleAux}
              taskLabel={auxiliaryTaskLabel}
            />
          )}

          <div data-model-auxiliary-card="">
            {AUX_TASKS.map(meta => {
              const copy = m.tasks[meta.key] ?? { label: meta.key, hint: meta.key }
              const current = auxiliary?.tasks.find(entry => entry.task === meta.key)
              const isAuto = !current || !current.provider || current.provider === 'auto'
              const isEditing = editingAuxTask === meta.key

              return (
                <div
                  className="scroll-mt-6 px-5 py-4"
                  data-model-auxiliary-row=""
                  id={`aux-task-${meta.key}`}
                  key={meta.key}
                >
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div className="min-w-0 flex-1 space-y-1">
                      <div className="flex min-w-0 flex-wrap items-baseline gap-2">
                        <span className="text-[15px] font-semibold text-[#0f172a] dark:text-foreground">{copy.label}</span>
                        <span className="text-[13px] font-medium text-[#94a3b8]">{copy.hint}</span>
                      </div>
                      <p className="text-[13px] text-[#64748b]">
                        {isAuto ? m.autoUseMain : `${current.provider} · ${current.model || m.providerDefault}`}
                        {!isAuto && current.base_url ? ` · ${current.base_url}` : null}
                        {current?.reasoning_effort
                          ? ` · ${
                              current.reasoning_effort === 'none'
                                ? `${m.reasoning} ${m.reasoningOff}`
                                : isReasoningEffort(current.reasoning_effort)
                                  ? t.shell.modelOptions[current.reasoning_effort]
                                  : current.reasoning_effort
                            }`
                          : null}
                      </p>
                    </div>

                    {!isEditing ? (
                      <div className="flex shrink-0 items-center gap-2">
                        <Button
                          className="h-8 rounded-[10px] border-0 bg-[#f1f5f9] px-3 text-[13px] font-semibold text-[#475569] shadow-none hover:bg-[#e2e8f0] dark:bg-(--ui-bg-quinary) dark:text-(--ui-text-secondary)"
                          disabled={!mainModel || applying}
                          onClick={() => void setAuxiliaryToMain(meta.key)}
                          size="sm"
                          variant="secondary"
                        >
                          {m.setToMain}
                        </Button>
                        <Button
                          className="h-8 rounded-[10px] border-0 bg-[#e3ecfd] px-3.5 text-[13px] font-semibold text-[#2f5bea] shadow-none hover:bg-[#d4e2fb]"
                          disabled={!providers.length || applying}
                          onClick={() => beginAuxiliaryEdit(meta.key)}
                          size="sm"
                          variant="secondary"
                        >
                          {m.change}
                        </Button>
                      </div>
                    ) : null}
                  </div>

                  {isEditing ? (
                    <div className="mt-3 grid gap-2.5 rounded-xl border border-[#e2e8f0] bg-[#f7f9fc] p-3.5 dark:border-(--ui-stroke-tertiary) dark:bg-(--ui-bg-quinary)">
                      <div className="flex flex-wrap items-center gap-2">
                        <Select
                          onValueChange={value => setAuxDraft(prev => ({ ...prev, provider: value, model: '' }))}
                          value={findCatalogProvider(providers, auxDraft.provider)?.slug ?? auxDraft.provider}
                        >
                          <SelectTrigger
                            aria-label={`${copy.label} provider`}
                            className={cn(
                              'h-10 min-w-32 rounded-xl border-[#e2e8f0] bg-white shadow-none dark:bg-(--ui-chat-bubble-background)',
                              CONTROL_TEXT
                            )}
                          >
                            <SelectValue placeholder={m.provider} />
                          </SelectTrigger>
                          <SelectContent>
                            {providerOptions.map(provider => (
                              <SelectItem key={provider.slug || 'none'} value={provider.slug || 'none'}>
                                {provider.name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                        <ModelSelect
                          aria-label={`${copy.label} model`}
                          className="min-w-48"
                          models={auxDraftProviderModels}
                          onValueChange={value => setAuxDraft(prev => ({ ...prev, model: value }))}
                          provider={findCatalogProvider(providers, auxDraft.provider)}
                          providerSlug={auxDraft.provider}
                          value={auxDraft.model}
                        />
                      </div>
                      <div className="flex flex-wrap items-center gap-2 text-[13px]">
                        <span className="font-medium text-[#64748b]">{m.reasoning}</span>
                        <Select
                          onValueChange={value => setAuxDraft(prev => ({ ...prev, reasoningEffort: value }))}
                          value={auxDraft.reasoningEffort}
                        >
                          <SelectTrigger
                            aria-label={`${copy.label} reasoning effort`}
                            className={cn(
                              'h-10 min-w-32 rounded-xl border-[#e2e8f0] bg-white shadow-none dark:bg-(--ui-chat-bubble-background)',
                              CONTROL_TEXT
                            )}
                          >
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="__inherit__">{m.inheritMainEffort}</SelectItem>
                            {REASONING_EFFORT_VALUES.map(value => (
                              <SelectItem key={value} value={value}>
                                {value === 'none' ? m.reasoningOff : t.shell.modelOptions[value]}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        <Button
                          className="h-8 rounded-[10px] bg-[#2f5bea] px-3.5 text-[13px] font-semibold text-white hover:bg-[#2548c7]"
                          disabled={!auxDraft.provider || !auxDraft.model || applying}
                          onClick={() => void applyAuxiliaryDraft(meta.key)}
                          size="sm"
                        >
                          {applying ? m.applying : t.common.apply}
                        </Button>
                        <Button
                          className="h-8 rounded-[10px] px-3 text-[13px] font-semibold text-[#475569]"
                          onClick={() => setEditingAuxTask(null)}
                          size="sm"
                          variant="ghost"
                        >
                          {t.common.cancel}
                        </Button>
                      </div>
                    </div>
                  ) : null}
                </div>
              )
            })}
          </div>
        </section>
      )}
      {showMoa && moa && currentMoaPreset && (
        <section className="grid gap-3.5" data-model-moa>
          <header className="space-y-1.5">
            {subpage === undefined ? (
              <SectionHeading icon={Cpu} title={m.moaTitle} />
            ) : (
              <h1 className="text-[32px] font-extrabold leading-none tracking-tight text-[#0f172a] dark:text-foreground">
                {m.moaTitle}
              </h1>
            )}
            <p className="max-w-4xl text-[14px] leading-snug text-[#475569]">{m.moaDescription}</p>
          </header>

          <div className="grid gap-4 p-5" data-model-moa-card="preset">
            <p className="text-[13px] font-semibold text-[#64748b]">{m.moaPreset}</p>

            <Select onValueChange={setSelectedMoaPreset} value={selectedMoaPreset || moa.default_preset}>
              <SelectTrigger className={cn('w-full', CONTROL_TEXT)}>
                <SelectValue placeholder={m.moaPreset} />
              </SelectTrigger>
              <SelectContent>
                {Object.keys(moa.presets).map(name => (
                  <SelectItem key={name} value={name}>
                    {name === moa.default_preset ? `${name} (default)` : name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <div className="flex flex-wrap items-center gap-2">
              <label className="flex items-center gap-2.5 pr-1 text-[13px] font-semibold text-[#0f172a] dark:text-foreground">
                <Switch
                  checked={currentMoaPreset.enabled !== false}
                  className="h-[22px] w-10 data-[state=checked]:border-transparent data-[state=checked]:bg-[#7c3aed] [&_[data-slot=switch-thumb]]:size-[18px] data-[state=checked]:[&_[data-slot=switch-thumb]]:translate-x-[18px]"
                  disabled={applying}
                  onCheckedChange={checked => updateMoaPreset(prev => ({ ...prev, enabled: checked }))}
                  size="default"
                />
                {m.moaEnabled}
              </label>

              <Button
                className="h-8 rounded-[10px] border-0 bg-[#f1f5f9] px-3 text-[13px] font-semibold text-[#475569] shadow-none hover:bg-[#e2e8f0] dark:bg-(--ui-bg-quinary) dark:text-(--ui-text-secondary)"
                disabled={applying}
                onClick={() => {
                  const next: MoaConfigResponse = {
                    ...moa,
                    default_preset: selectedMoaPreset || moa.default_preset
                  }

                  void saveMoa(next)
                }}
                size="sm"
                variant="secondary"
              >
                {m.moaSetDefault}
              </Button>

              <Button
                className="h-8 rounded-[10px] border-0 bg-[#fef2f2] px-3 text-[13px] font-semibold text-[#dc2626] shadow-none hover:bg-[#fee2e2]"
                disabled={Object.keys(moa.presets).length <= 1 || applying}
                onClick={() => {
                  if (Object.keys(moa.presets).length <= 1) {
                    return
                  }

                  const presets = { ...moa.presets }
                  delete presets[selectedMoaPreset]
                  const fallback = Object.keys(presets)[0]

                  const next: MoaConfigResponse = {
                    ...moa,
                    presets,
                    default_preset: moa.default_preset === selectedMoaPreset ? fallback : moa.default_preset,
                    active_preset: moa.active_preset === selectedMoaPreset ? '' : moa.active_preset
                  }

                  setSelectedMoaPreset(Object.keys(moa.presets).find(name => name !== selectedMoaPreset) || '')
                  void saveMoa(next)
                }}
                size="sm"
                variant="secondary"
              >
                {t.common.delete}
              </Button>

              <Input
                className={cn('min-w-[10rem] flex-1', CONTROL_TEXT)}
                onChange={event => setNewMoaPresetName(event.target.value)}
                onKeyDown={event => {
                  if (!isSubmitEnter(event)) {
                    return
                  }

                  const name = newMoaPresetName.trim()

                  if (!name || moa.presets[name] || applying) {
                    return
                  }

                  const next: MoaConfigResponse = {
                    ...moa,
                    presets: {
                      ...moa.presets,
                      [name]: { ...currentMoaPreset, reference_models: [...currentMoaPreset.reference_models] }
                    }
                  }

                  setSelectedMoaPreset(name)
                  setNewMoaPresetName('')
                  void saveMoa(next)
                }}
                placeholder={m.moaNewPresetPlaceholder}
                value={newMoaPresetName}
              />

              <Button
                className="h-9 rounded-[10px] border-0 bg-[#7c3aed] px-3.5 text-[13px] font-semibold text-white shadow-none hover:bg-[#6d28d9]"
                disabled={!newMoaPresetName.trim() || !!moa.presets[newMoaPresetName.trim()] || applying}
                onClick={() => {
                  const name = newMoaPresetName.trim()

                  const next: MoaConfigResponse = {
                    ...moa,
                    presets: {
                      ...moa.presets,
                      [name]: { ...currentMoaPreset, reference_models: [...currentMoaPreset.reference_models] }
                    }
                  }

                  setSelectedMoaPreset(name)
                  setNewMoaPresetName('')
                  void saveMoa(next)
                }}
                size="sm"
              >
                <Plus className="size-3.5" />
                {m.moaAddPreset}
              </Button>
            </div>
          </div>

          <div className="grid gap-3 p-5" data-model-moa-card="references">
            <p className="text-[13px] font-semibold text-[#64748b]">{m.moaReferenceModels}</p>

            {currentMoaPreset.reference_models.map((slot, index) => (
              <div
                className={cn('grid gap-3 p-4', slot.enabled === false && 'opacity-60')}
                data-model-moa-slot=""
                key={`${selectedMoaPreset}-${index}`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0 space-y-0.5">
                    <p className="text-[15px] font-semibold text-[#0f172a] dark:text-foreground">
                      {m.moaReferenceTitle(index + 1)}
                    </p>
                    <p className="text-[12px] text-[#64748b]">{m.moaReferenceHint}</p>
                  </div>
                  <Switch
                    aria-label={m.moaReferenceToggle(slot.enabled !== false, index + 1)}
                    checked={slot.enabled !== false}
                    className="mt-0.5 h-[22px] w-10 shrink-0 data-[state=checked]:border-transparent data-[state=checked]:bg-[#7c3aed] [&_[data-slot=switch-thumb]]:size-[18px] data-[state=checked]:[&_[data-slot=switch-thumb]]:translate-x-[18px]"
                    disabled={applying}
                    onCheckedChange={checked =>
                      updateMoaPreset(prev => ({
                        ...prev,
                        reference_models: prev.reference_models.map((s, i) =>
                          i === index ? { ...s, enabled: checked === true } : s
                        )
                      }))
                    }
                    size="default"
                  />
                </div>

                <div className="grid gap-2 sm:grid-cols-2">
                  <Select
                    onValueChange={value =>
                      updateMoaPreset(prev => ({
                        ...prev,
                        reference_models: prev.reference_models.map((s, i) =>
                          i === index ? updateMoaSlot(s, { provider: value }) : s
                        )
                      }))
                    }
                    value={slot.provider}
                  >
                    <SelectTrigger className={cn('w-full', CONTROL_TEXT)}>
                      <SelectValue placeholder={m.provider} />
                    </SelectTrigger>
                    <SelectContent>
                      {withActive(
                        moaSlotProviderOptions.map(p => p.slug || 'none'),
                        slot.provider
                      ).map(slug => {
                        const provider = moaSlotProviderOptions.find(p => (p.slug || 'none') === slug)

                        return (
                          <SelectItem key={slug} value={slug}>
                            {provider?.name || slug}
                          </SelectItem>
                        )
                      })}
                    </SelectContent>
                  </Select>
                  <ModelSelect
                    className="min-w-0 w-full"
                    models={modelsForProvider(slot.provider)}
                    onValueChange={value =>
                      updateMoaPreset(prev => ({
                        ...prev,
                        reference_models: prev.reference_models.map((s, i) =>
                          i === index ? updateMoaSlot(s, { model: value }) : s
                        )
                      }))
                    }
                    provider={findCatalogProvider(providers, slot.provider)}
                    providerSlug={slot.provider}
                    value={slot.model}
                  />
                </div>

                <div className="flex justify-end">
                  <Button
                    className="h-8 rounded-[10px] border-0 bg-[#fef2f2] px-3 text-[13px] font-semibold text-[#dc2626] shadow-none hover:bg-[#fee2e2]"
                    disabled={currentMoaPreset.reference_models.length <= 1 || applying}
                    onClick={() =>
                      updateMoaPreset(prev => ({
                        ...prev,
                        reference_models: prev.reference_models.filter((_, i) => i !== index)
                      }))
                    }
                    size="sm"
                    variant="secondary"
                  >
                    {t.common.remove}
                  </Button>
                </div>
              </div>
            ))}

            <Button
              className="h-10 w-full rounded-xl border-0 bg-[#e3ecfd] text-[13px] font-semibold text-[#2f5bea] shadow-none hover:bg-[#d4e2fb]"
              disabled={applying}
              onClick={() =>
                updateMoaPreset(prev => ({
                  ...prev,
                  reference_models: [...prev.reference_models, { ...prev.aggregator, enabled: true }]
                }))
              }
              size="sm"
              variant="secondary"
            >
              <Plus className="size-3.5" />
              {m.moaAddReference}
            </Button>
          </div>

          <div className="grid gap-3 p-5" data-model-moa-card="aggregator">
            <div className="space-y-0.5">
              <p className="text-[13px] font-semibold text-[#64748b]">{m.moaAggregator}</p>
              <p className="text-[12px] text-[#64748b]">{m.moaAggregatorBilled}</p>
            </div>

            <div className="grid gap-2 sm:grid-cols-2">
              <Select
                onValueChange={value =>
                  updateMoaPreset(prev => ({
                    ...prev,
                    aggregator: updateMoaSlot(prev.aggregator, { provider: value })
                  }))
                }
                value={currentMoaPreset.aggregator.provider}
              >
                <SelectTrigger className={cn('w-full', CONTROL_TEXT)}>
                  <SelectValue placeholder={m.provider} />
                </SelectTrigger>
                <SelectContent>
                  {withActive(
                    moaSlotProviderOptions.map(p => p.slug || 'none'),
                    currentMoaPreset.aggregator.provider
                  ).map(slug => {
                    const provider = moaSlotProviderOptions.find(p => (p.slug || 'none') === slug)

                    return (
                      <SelectItem key={slug} value={slug}>
                        {provider?.name || slug}
                      </SelectItem>
                    )
                  })}
                </SelectContent>
              </Select>
              <ModelSelect
                className="min-w-0 w-full"
                models={modelsForProvider(currentMoaPreset.aggregator.provider)}
                onValueChange={value =>
                  updateMoaPreset(prev => ({
                    ...prev,
                    aggregator: updateMoaSlot(prev.aggregator, { model: value })
                  }))
                }
                provider={findCatalogProvider(providers, currentMoaPreset.aggregator.provider)}
                providerSlug={currentMoaPreset.aggregator.provider}
                value={currentMoaPreset.aggregator.model}
              />
            </div>
          </div>
        </section>
      )}
      {subpage === 'moa' && !loading && (!moa || !currentMoaPreset) && (
        <PanelEmpty
          action={
            <Button onClick={() => void refresh()} size="sm">
              {t.skills.refresh}
            </Button>
          }
          icon="error"
          title={m.loadFailed}
        />
      )}
    </div>
  )
}
