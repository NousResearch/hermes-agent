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
  MoaConfigResponse,
  MoaModelSlot,
  ModelOptionProvider,
  StaleAuxAssignment
} from '@/hermes'
import { useI18n } from '@/i18n'
import { AlertTriangle, Cpu, Loader2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import { startManualLocalEndpoint, startManualProviderOAuth } from '@/store/onboarding'

import { invalidateHermesConfig, setHermesConfigCache, useHermesConfigRecord } from '../hooks/use-config-record'
import { useOnProfileSwitch } from '../hooks/use-on-profile-switch'

import { CONTROL_TEXT } from './constants'
import { getNested, setNested } from './helpers'
import { ListRow, Pill, SectionHeading } from './primitives'

// Skeleton mirror of the Model settings DOM so the page keeps its shape while
// the provider/model catalog loads, instead of collapsing to a centered
// spinner. Same containers/rhythm as the real render below.
export function ModelSettingsSkeleton() {
  return (
    <div className="grid gap-6" data-slot="model-settings-skeleton">
      <section>
        <Skeleton className="mb-3 h-3 w-72 max-w-full" />
        <div className="flex flex-wrap items-center gap-2">
          <Skeleton className="h-8 w-40" />
          <Skeleton className="h-8 w-60 max-w-full" />
          <Skeleton className="h-8 w-16" />
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-3">
          <Skeleton className="h-3 w-16" />
          <Skeleton className="h-8 w-28" />
          <Skeleton className="h-6 w-20" />
        </div>
      </section>

      <section>
        <div className="mb-2.5 flex items-center gap-2 pt-2">
          <Skeleton className="size-4" />
          <Skeleton className="h-4 w-36" />
        </div>
        <div className="grid gap-1">
          {[0, 1, 2, 3].map(row => (
            <div
              className="grid gap-3 py-3 @2xl:grid-cols-[minmax(0,1fr)_minmax(15rem,22rem)] @2xl:items-center"
              key={row}
            >
              <div className="min-w-0 space-y-1.5">
                <Skeleton className="h-3.5 w-32" />
                <Skeleton className="h-3 w-52 max-w-full" />
              </div>
              <Skeleton className="h-8 w-full @2xl:justify-self-end @2xl:w-56" />
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

// Hermes' reasoning levels (VALID_REASONING_EFFORTS); `none` = thinking off.
// Empty config = Hermes default (medium), shown as Medium.
const EFFORT_VALUES = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'] as const

// agent.service_tier stores "fast"/"priority"/"on" for fast; anything else is
// normal (mirrors tui_gateway _load_service_tier).
const isFastTier = (tier: unknown): boolean =>
  ['fast', 'priority', 'on'].includes(
    String(tier ?? '')
      .trim()
      .toLowerCase()
  )

// Reuse the composer's effort labels.
const effortLabelKey = (v: string) => v as 'high' | 'low' | 'max' | 'medium' | 'minimal' | 'ultra' | 'xhigh'

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
  { key: 'web_extract' },
  { key: 'compression' },
  { key: 'skills_hub' },
  { key: 'approval' },
  { key: 'mcp' },
  { key: 'title_generation' },
  { key: 'curator' }
]

const NO_PROVIDERS: readonly ModelOptionProvider[] = [{ name: '—', slug: '', models: [] }]

// Radix <Select> renders a blank trigger when `value` matches no <SelectItem>.
// A custom model (e.g. one added via config that isn't in the provider's
// curated list) would vanish — surface the active value so it stays selectable.
export const withActive = (models: readonly string[], active: string): readonly string[] =>
  active && !models.includes(active) ? [active, ...models] : models

interface StaleAuxWarningProps {
  applying: boolean
  onReset: () => void
  slots: readonly StaleAuxAssignment[]
  taskLabel: (key: string) => string
}

// Shared notice: auxiliary tasks still pinned to a provider that isn't the
// current main. Surfaces the silent credit-burn path (e.g. aux pinned to a
// $0-balance provider after switching main away from it) and offers the
// existing one-click reset rather than auto-clearing legitimate pins.
function StaleAuxWarning({ applying, onReset, slots, taskLabel }: StaleAuxWarningProps) {
  if (!slots.length) {
    return null
  }

  const provider = slots[0].provider
  const allSameProvider = slots.every(slot => slot.provider === provider)
  const names = slots.map(slot => taskLabel(slot.task)).join(', ')

  return (
    <div className="flex flex-wrap items-center gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
      <AlertTriangle className="size-3.5 shrink-0" />
      <span className="grow">
        {slots.length} auxiliary task{slots.length === 1 ? '' : 's'} ({names}) still run on{' '}
        <span className="font-mono">{allSameProvider ? provider : 'other providers'}</span>, not your main model.
      </span>
      <Button disabled={applying} onClick={onReset} size="sm" variant="textStrong">
        Reset all to main
      </Button>
    </div>
  )
}

interface ModelSettingsProps {
  /** Notified after the main model is applied, so live UI stores can sync. */
  onMainModelChanged?: (provider: string, model: string) => void
}

export function ModelSettings({ onMainModelChanged }: ModelSettingsProps) {
  const { t } = useI18n()
  const m = t.settings.model
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [mainModel, setMainModel] = useState<{ model: string; provider: string } | null>(null)
  const [providers, setProviders] = useState<ModelOptionProvider[]>([])
  const [selectedProvider, setSelectedProvider] = useState('')
  const [selectedModel, setSelectedModel] = useState('')
  const [auxiliary, setAuxiliary] = useState<AuxiliaryModelsResponse | null>(null)
  const [moa, setMoa] = useState<MoaConfigResponse | null>(null)
  const [selectedMoaPreset, setSelectedMoaPreset] = useState('')
  const [newMoaPresetName, setNewMoaPresetName] = useState('')
  // agent.* defaults round-trip through the shared config cache (read → write
  // back the whole record), so a save here shows in the MCP/config surfaces.
  const { data: config } = useHermesConfigRecord()
  const setConfig = setHermesConfigCache
  const [applying, setApplying] = useState(false)
  const [editingAuxTask, setEditingAuxTask] = useState<null | string>(null)
  const [auxDraft, setAuxDraft] = useState<{ model: string; provider: string }>({ model: '', provider: '' })
  // Aux slots reported stale by the backend immediately after a main-model
  // switch (provider differs from the new main). Cleared on next switch/reset.
  const [switchStaleAux, setSwitchStaleAux] = useState<StaleAuxAssignment[]>([])
  // Inline API-key entry for picking an unconfigured `api_key` provider in
  // place — mirrors the onboarding ApiKeyForm but scoped to the model picker.
  const [apiKeyDraft, setApiKeyDraft] = useState('')
  const [activating, setActivating] = useState(false)

  // Every profile-scoped async here captures this and bails before writing back,
  // so a request in flight when the user switches profiles can't paint profile
  // A's models/providers into profile B (or fire onMainModelChanged for A).
  const profileEpoch = useRef(0)

  const refresh = useCallback(async () => {
    const epoch = profileEpoch.current
    setLoading(true)
    setError('')

    try {
      const [modelInfo, modelOptions, auxiliaryModels, moaModels] = await Promise.all([
        getGlobalModelInfo(),
        getGlobalModelOptions(),
        getAuxiliaryModels(),
        getMoaModels().catch(() => null)
      ])

      if (profileEpoch.current !== epoch) {
        return
      }

      setMainModel({ model: modelInfo.model, provider: modelInfo.provider })
      setProviders(modelOptions.providers || [])
      setSelectedProvider(prev => prev || modelInfo.provider)
      setSelectedModel(prev => prev || modelInfo.model)
      setAuxiliary(auxiliaryModels)
      setMoa(moaModels)

      if (moaModels) {
        setSelectedMoaPreset(prev => (prev && moaModels.presets[prev] ? prev : moaModels.default_preset))
      }

      // The config record loads via its own shared query; a model switch can
      // change it server-side (aux slots), so nudge that cache to refetch.
      void invalidateHermesConfig()
    } catch (err) {
      if (profileEpoch.current === epoch) {
        setError(err instanceof Error ? err.message : String(err))
      }
    } finally {
      if (profileEpoch.current === epoch) {
        setLoading(false)
      }
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  // A profile switch swaps the backend under the mounted panel — reload for the
  // new profile (bumping the epoch first so any in-flight A request is discarded).
  useOnProfileSwitch(() => {
    profileEpoch.current += 1
    void refresh()
  })

  const providerOptions = providers.length ? providers : NO_PROVIDERS

  // MoA reference/aggregator slots must never be the moa virtual provider —
  // that would create a recursive MoA tree (the backend rejects it on save).
  // Hide it from the slot selectors so it isn't offered as a dead choice.
  const moaSlotProviderOptions = providerOptions.filter(provider => (provider.slug || '').toLowerCase() !== 'moa')

  const selectedProviderRow = useMemo(
    () => providers.find(provider => provider.slug === selectedProvider),
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
    () => providers.find(provider => provider.slug === auxDraft.provider)?.models ?? [],
    [auxDraft.provider, providers]
  )

  const modelsForProvider = useCallback(
    (provider: string) => providers.find(row => row.slug === provider)?.models ?? [],
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

  // Quiet debounced persist for inline MoA edits — mirrors the config page's
  // autosave so slot/aggregator tweaks save themselves, matching the
  // preset-level ops (set default / add / delete) that already persist on
  // click. No `applying` spinner, so selecting stays responsive.
  const scheduleMoaSave = useCallback((next: MoaConfigResponse) => {
    if (moaSaveTimer.current) {
      window.clearTimeout(moaSaveTimer.current)
    }

    moaSaveTimer.current = window.setTimeout(() => {
      void saveMoaModels(next)
        .then(setMoa)
        .catch(err => setError(err instanceof Error ? err.message : String(err)))
    }, 600)
  }, [])

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

    if (patch.provider) {
      next.model = ''
    }

    return next
  }, [])

  const saveMoa = useCallback(async (next: MoaConfigResponse) => {
    const epoch = profileEpoch.current
    setApplying(true)
    setError('')

    try {
      const saved = await saveMoaModels(next)

      if (profileEpoch.current !== epoch) {
        return
      }

      setMoa(saved)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setApplying(false)
    }
  }, [])

  const auxiliaryTaskLabel = useCallback((key: string) => m.tasks[key]?.label ?? key, [m.tasks])

  // Persistent mismatch: any aux slot pinned to a provider different from the
  // current main, regardless of whether the user just switched. Catches the
  // "I pinned aux months ago and forgot, now it bills a dead provider" case.
  const persistentStaleAux = useMemo<StaleAuxAssignment[]>(() => {
    const mainProvider = (mainModel?.provider ?? '').toLowerCase()

    if (!mainProvider || !auxiliary) {
      return []
    }

    return auxiliary.tasks
      .filter(entry => {
        const p = (entry.provider ?? '').toLowerCase()

        return p && p !== 'auto' && p !== mainProvider
      })
      .map(entry => ({ task: entry.task, provider: entry.provider, model: entry.model }))
  }, [auxiliary, mainModel])

  // Capabilities of the APPLIED main model — gates the profile-default
  // reasoning/speed controls the same way the composer picker gates per-model
  // edits (reasoning defaults on, fast defaults off when unreported).
  const mainCaps = useMemo(() => {
    const row = providers.find(provider => provider.slug === mainModel?.provider)

    return mainModel ? row?.capabilities?.[mainModel.model] : undefined
  }, [providers, mainModel])

  const reasoningSupported = mainCaps?.reasoning ?? true
  const fastSupported = mainCaps?.fast ?? false

  // Hand-written `reasoning_effort: false`/`off` reaches us as boolean false
  // ("false" once stringified) — show it as Off, not an empty select.
  const rawEffort = String(getNested(config ?? {}, 'agent.reasoning_effort') ?? '')
    .trim()
    .toLowerCase()

  const effortValue = rawEffort === 'false' || rawEffort === 'disabled' ? 'none' : rawEffort || 'medium'

  const fastOn = isFastTier(getNested(config ?? {}, 'agent.service_tier'))

  // Persist a single agent.* default by round-tripping the whole config record
  // (PUT /api/config replaces it) — optimistic, with rollback on failure.
  const writeAgentDefault = useCallback(
    async (key: string, value: string) => {
      if (!config) {
        return
      }

      const prev = config
      const next = setNested(config, key, value)
      setConfig(next)

      try {
        await saveHermesConfig(next)
      } catch (err) {
        setConfig(prev)
        notifyError(err, m.defaultsFailed)
      }
    },
    [config, m.defaultsFailed]
  )

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
      await setEnvVar(keyEnv, apiKeyDraft.trim())
      setApiKeyDraft('')

      // Pick a sensible default for the freshly-activated provider (mirrors
      // `hermes model` curation). Best-effort — fall through to the refreshed
      // model list if it fails.
      let nextModel = ''

      try {
        const rec = await getRecommendedDefaultModel(slug)
        nextModel = rec.model || ''
      } catch {
        nextModel = ''
      }

      const options = await getGlobalModelOptions()

      if (profileEpoch.current !== epoch) {
        return
      }

      setProviders(options.providers || [])
      const refreshedRow = options.providers?.find(p => p.slug === slug)
      const fallbackModel = refreshedRow?.models?.[0] ?? ''
      setSelectedModel(nextModel || fallbackModel)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setActivating(false)
    }
  }, [apiKeyDraft, selectedProviderRow])

  // OAuth / external providers can't be activated with a pasted key — hand off
  // to the shared onboarding flow scoped to this provider's real sign-in. The
  // custom / local endpoint is NOT an OAuth provider, so it gets the dedicated
  // local-endpoint form (URL + optional API key) instead of being dead-ended
  // on the OAuth picker (the original "booted back to the first screen" loop).
  const startProviderSetup = useCallback(() => {
    const slug = selectedProviderRow?.slug

    if (!slug) {
      return
    }

    const lower = slug.toLowerCase()

    if (lower === 'custom' || lower === 'local' || lower.startsWith('custom:')) {
      startManualLocalEndpoint()
    } else {
      startManualProviderOAuth(slug)
    }
  }, [selectedProviderRow])

  const applyMainModel = useCallback(async () => {
    if (!selectedProvider || !selectedModel) {
      return
    }

    const epoch = profileEpoch.current
    setApplying(true)
    setError('')

    try {
      const result = await setModelAssignment({ model: selectedModel, provider: selectedProvider, scope: 'main' })

      if (profileEpoch.current !== epoch) {
        return
      }

      const provider = result.provider || selectedProvider
      const model = result.model || selectedModel
      setMainModel({ provider, model })
      setSwitchStaleAux(result.stale_aux ?? [])
      onMainModelChanged?.(provider, model)
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setApplying(false)
    }
  }, [onMainModelChanged, refresh, selectedModel, selectedProvider])

  const setAuxiliaryToMain = useCallback(
    async (task: string) => {
      if (!mainModel) {
        return
      }

      setApplying(true)
      setError('')

      try {
        await setModelAssignment({ model: mainModel.model, provider: mainModel.provider, scope: 'auxiliary', task })
        await refresh()
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      } finally {
        setApplying(false)
      }
    },
    [mainModel, refresh]
  )

  const applyAuxiliaryDraft = useCallback(
    async (task: string) => {
      if (!auxDraft.provider || !auxDraft.model) {
        return
      }

      setApplying(true)
      setError('')

      try {
        await setModelAssignment({ model: auxDraft.model, provider: auxDraft.provider, scope: 'auxiliary', task })
        setEditingAuxTask(null)
        await refresh()
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      } finally {
        setApplying(false)
      }
    },
    [auxDraft, refresh]
  )

  const beginAuxiliaryEdit = useCallback(
    (task: string) => {
      const current = auxiliary?.tasks.find(entry => entry.task === task)

      const initialProvider =
        current?.provider && current.provider !== 'auto' ? current.provider : (mainModel?.provider ?? '')

      const initialModel = current?.model || mainModel?.model || ''
      setAuxDraft({ provider: initialProvider, model: initialModel })
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
      await setModelAssignment({
        model: mainModel.model,
        provider: mainModel.provider,
        scope: 'auxiliary',
        task: '__reset__'
      })
      setSwitchStaleAux([])
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setApplying(false)
    }
  }, [mainModel, refresh])

  if (loading && !mainModel) {
    return <ModelSettingsSkeleton />
  }

  return (
    <div className="grid gap-6">
      <section>
        <p className="mb-3 text-xs text-muted-foreground">{m.appliesDesc}</p>
        <div className="flex flex-wrap items-center gap-2">
          <Select onValueChange={setSelectedProvider} value={selectedProvider}>
            <SelectTrigger className={cn('min-w-40', CONTROL_TEXT)}>
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
          {needsSetup ? (
            setupIsApiKey ? (
              <>
                <Input
                  autoComplete="off"
                  className={cn('min-w-60 flex-1', CONTROL_TEXT)}
                  onChange={event => setApiKeyDraft(event.target.value)}
                  onKeyDown={event => {
                    if (event.key === 'Enter') {
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
              </>
            ) : (
              <Button onClick={startProviderSetup} size="sm" variant="textStrong">
                Set up {selectedProviderRow?.name ?? 'provider'}
              </Button>
            )
          ) : (
            <>
              <Select onValueChange={setSelectedModel} value={selectedModel}>
                <SelectTrigger className={cn('min-w-60', CONTROL_TEXT)}>
                  <SelectValue placeholder={m.model} />
                </SelectTrigger>
                <SelectContent>
                  {withActive(selectedProviderModels, selectedModel).map(model => (
                    <SelectItem key={model} value={model}>
                      {model}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                disabled={!selectedProvider || !selectedModel || applying}
                onClick={() => void applyMainModel()}
                size="sm"
              >
                {applying && <Loader2 className="size-3.5 animate-spin" />}
                {applying ? m.applying : t.common.apply}
              </Button>
            </>
          )}
        </div>
        {needsSetup && !setupIsApiKey && (
          <p className="mt-2 text-xs text-muted-foreground">
            {selectedProviderRow?.auth_type === 'api_key'
              ? `${selectedProviderRow?.name} needs an API key — set it up to choose a model.`
              : `${selectedProviderRow?.name} signs in through your browser — Hermes runs the flow for you.`}
          </p>
        )}
        {config && mainModel && (reasoningSupported || fastSupported) && (
          <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-3">
            <span className="text-xs text-muted-foreground">{m.defaultsLabel}</span>
            {reasoningSupported && (
              <div className="flex items-center gap-2 text-xs">
                {m.reasoning}
                <Select
                  onValueChange={value => void writeAgentDefault('agent.reasoning_effort', value)}
                  value={effortValue}
                >
                  <SelectTrigger className={cn('min-w-28', CONTROL_TEXT)}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {EFFORT_VALUES.map(value => (
                      <SelectItem key={value} value={value}>
                        {value === 'none' ? m.reasoningOff : t.shell.modelOptions[effortLabelKey(value)]}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            )}
            {fastSupported && (
              <label className="flex items-center gap-2 text-xs">
                {t.shell.modelOptions.fast}
                <Switch
                  checked={fastOn}
                  onCheckedChange={checked => void writeAgentDefault('agent.service_tier', checked ? 'fast' : 'normal')}
                  size="xs"
                />
              </label>
            )}
          </div>
        )}
        {error && <div className="mt-2 text-xs text-destructive">{error}</div>}
        {switchStaleAux.length > 0 && (
          <div className="mt-2">
            <StaleAuxWarning
              applying={applying}
              onReset={() => void resetAuxiliaryModels()}
              slots={switchStaleAux}
              taskLabel={auxiliaryTaskLabel}
            />
          </div>
        )}
      </section>

      <section>
        <div className="mb-2.5 flex items-center justify-between">
          <SectionHeading icon={Cpu} title={m.auxiliaryTitle} />
          <Button
            disabled={!mainModel || applying}
            onClick={() => void resetAuxiliaryModels()}
            size="sm"
            variant="textStrong"
          >
            {m.resetAllToMain}
          </Button>
        </div>
        <p className="mb-2 text-xs text-muted-foreground">{m.auxiliaryDesc}</p>
        {switchStaleAux.length === 0 && persistentStaleAux.length > 0 && (
          <div className="mb-2.5">
            <StaleAuxWarning
              applying={applying}
              onReset={() => void resetAuxiliaryModels()}
              slots={persistentStaleAux}
              taskLabel={auxiliaryTaskLabel}
            />
          </div>
        )}
        <div className="grid gap-1">
          {AUX_TASKS.map(meta => {
            const copy = m.tasks[meta.key] ?? { label: meta.key, hint: meta.key }
            const current = auxiliary?.tasks.find(entry => entry.task === meta.key)
            const isAuto = !current || !current.provider || current.provider === 'auto'
            const isEditing = editingAuxTask === meta.key

            return (
              <ListRow
                action={
                  !isEditing && (
                    <div className="flex shrink-0 items-center gap-1.5">
                      <Button
                        disabled={!mainModel || applying}
                        onClick={() => void setAuxiliaryToMain(meta.key)}
                        size="sm"
                        variant="text"
                      >
                        {m.setToMain}
                      </Button>
                      <Button
                        disabled={!providers.length || applying}
                        onClick={() => beginAuxiliaryEdit(meta.key)}
                        size="sm"
                        variant="textStrong"
                      >
                        {m.change}
                      </Button>
                    </div>
                  )
                }
                below={
                  isEditing && (
                    <div className="mt-2 flex flex-wrap items-center gap-2 pt-1">
                      <Select
                        onValueChange={value => setAuxDraft(prev => ({ ...prev, provider: value, model: '' }))}
                        value={auxDraft.provider}
                      >
                        <SelectTrigger className={cn('min-w-32', CONTROL_TEXT)}>
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
                      <Select
                        onValueChange={value => setAuxDraft(prev => ({ ...prev, model: value }))}
                        value={auxDraft.model}
                      >
                        <SelectTrigger className={cn('min-w-48', CONTROL_TEXT)}>
                          <SelectValue placeholder={m.model} />
                        </SelectTrigger>
                        <SelectContent>
                          {withActive(auxDraftProviderModels, auxDraft.model).map(model => (
                            <SelectItem key={model} value={model}>
                              {model}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Button
                        disabled={!auxDraft.provider || !auxDraft.model || applying}
                        onClick={() => void applyAuxiliaryDraft(meta.key)}
                        size="sm"
                      >
                        {applying ? m.applying : t.common.apply}
                      </Button>
                      <Button onClick={() => setEditingAuxTask(null)} size="sm" variant="ghost">
                        {t.common.cancel}
                      </Button>
                    </div>
                  )
                }
                description={
                  <span className="font-mono text-[0.68rem]">
                    {isAuto ? m.autoUseMain : `${current.provider} · ${current.model || m.providerDefault}`}
                  </span>
                }
                key={meta.key}
                title={
                  <span className="flex items-baseline gap-2">
                    {copy.label}
                    <Pill>{copy.hint}</Pill>
                  </span>
                }
              />
            )
          })}
        </div>
      </section>
      {moa && currentMoaPreset && (
        <section>
          <SectionHeading icon={Cpu} title="Mixture of Agents" />
          <p className="mb-2 text-xs text-muted-foreground">
            Configure named presets that appear as models under the Mixture of Agents provider. The aggregator is the
            acting model.
          </p>
          <div className="mb-2 flex flex-wrap items-center gap-2">
            <Select onValueChange={setSelectedMoaPreset} value={selectedMoaPreset || moa.default_preset}>
              <SelectTrigger className={cn('min-w-40', CONTROL_TEXT)}>
                <SelectValue placeholder="Preset" />
              </SelectTrigger>
              <SelectContent>
                {Object.keys(moa.presets).map(name => (
                  <SelectItem key={name} value={name}>
                    {name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              disabled={applying}
              onClick={() => {
                const next: MoaConfigResponse = {
                  ...moa,
                  default_preset: selectedMoaPreset || moa.default_preset
                }

                void saveMoa(next)
              }}
              size="sm"
              variant="text"
            >
              Set default
            </Button>
            <Button
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
              variant="ghost"
            >
              Delete
            </Button>
            <Input
              className={cn('w-40', CONTROL_TEXT)}
              onChange={event => setNewMoaPresetName(event.target.value)}
              placeholder="new preset"
              value={newMoaPresetName}
            />
            <Button
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
              variant="textStrong"
            >
              Add preset
            </Button>
          </div>
          <div className="mb-2 text-xs text-muted-foreground">
            Default: <span className="font-mono">{moa.default_preset}</span>
          </div>
          <div className="grid gap-1">
            {currentMoaPreset.reference_models.map((slot, index) => (
              <ListRow
                below={
                  <div className="mt-2 flex flex-wrap items-center gap-2 pt-1">
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
                      <SelectTrigger className={cn('min-w-32', CONTROL_TEXT)}>
                        <SelectValue placeholder={m.provider} />
                      </SelectTrigger>
                      <SelectContent>
                        {moaSlotProviderOptions.map(provider => (
                          <SelectItem key={provider.slug || 'none'} value={provider.slug || 'none'}>
                            {provider.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Select
                      onValueChange={value =>
                        updateMoaPreset(prev => ({
                          ...prev,
                          reference_models: prev.reference_models.map((s, i) =>
                            i === index ? updateMoaSlot(s, { model: value }) : s
                          )
                        }))
                      }
                      value={slot.model}
                    >
                      <SelectTrigger className={cn('min-w-48', CONTROL_TEXT)}>
                        <SelectValue placeholder={m.model} />
                      </SelectTrigger>
                      <SelectContent>
                        {withActive(modelsForProvider(slot.provider), slot.model).map(model => (
                          <SelectItem key={model} value={model}>
                            {model}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <Button
                      disabled={currentMoaPreset.reference_models.length <= 1 || applying}
                      onClick={() =>
                        updateMoaPreset(prev => ({
                          ...prev,
                          reference_models: prev.reference_models.filter((_, i) => i !== index)
                        }))
                      }
                      size="sm"
                      variant="ghost"
                    >
                      Remove
                    </Button>
                  </div>
                }
                description={
                  <span className="font-mono text-[0.68rem]">
                    {slot.provider} · {slot.model}
                  </span>
                }
                key={`${selectedMoaPreset}-${slot.provider}-${slot.model}-${index}`}
                title={`Reference ${index + 1}`}
              />
            ))}
            <Button
              disabled={applying}
              onClick={() =>
                updateMoaPreset(prev => ({ ...prev, reference_models: [...prev.reference_models, prev.aggregator] }))
              }
              size="sm"
              variant="textStrong"
            >
              Add reference model
            </Button>
            <ListRow
              below={
                <div className="mt-2 flex flex-wrap items-center gap-2 pt-1">
                  <Select
                    onValueChange={value =>
                      updateMoaPreset(prev => ({
                        ...prev,
                        aggregator: updateMoaSlot(prev.aggregator, { provider: value })
                      }))
                    }
                    value={currentMoaPreset.aggregator.provider}
                  >
                    <SelectTrigger className={cn('min-w-32', CONTROL_TEXT)}>
                      <SelectValue placeholder={m.provider} />
                    </SelectTrigger>
                    <SelectContent>
                      {moaSlotProviderOptions.map(provider => (
                        <SelectItem key={provider.slug || 'none'} value={provider.slug || 'none'}>
                          {provider.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select
                    onValueChange={value =>
                      updateMoaPreset(prev => ({
                        ...prev,
                        aggregator: updateMoaSlot(prev.aggregator, { model: value })
                      }))
                    }
                    value={currentMoaPreset.aggregator.model}
                  >
                    <SelectTrigger className={cn('min-w-48', CONTROL_TEXT)}>
                      <SelectValue placeholder={m.model} />
                    </SelectTrigger>
                    <SelectContent>
                      {withActive(
                        modelsForProvider(currentMoaPreset.aggregator.provider),
                        currentMoaPreset.aggregator.model
                      ).map(model => (
                        <SelectItem key={model} value={model}>
                          {model}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              }
              description={
                <span className="font-mono text-[0.68rem]">
                  {currentMoaPreset.aggregator.provider} · {currentMoaPreset.aggregator.model}
                </span>
              }
              title="Aggregator"
            />
          </div>
        </section>
      )}
    </div>
  )
}
