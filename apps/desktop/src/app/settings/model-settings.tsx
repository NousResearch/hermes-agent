import { useCallback, useEffect, useMemo, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  applyModelProfile,
  createModelProfile,
  deleteModelProfile,
  getAuxiliaryModels,
  getGlobalModelInfo,
  getGlobalModelOptions,
  getModelProfiles,
  getRecommendedDefaultModel,
  setEnvVar,
  setModelAssignment,
  updateModelProfile
} from '@/hermes'
import type {
  AuxiliaryModelsResponse,
  ModelOptionProvider,
  ModelProfilesResponse,
  ModelRoutingPayload,
  StaleAuxAssignment
} from '@/hermes'
import { useI18n } from '@/i18n'
import { AlertTriangle, Cpu, Layers3, Loader2, Save, Sparkles, Trash2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { startManualProviderOAuth } from '@/store/onboarding'

import { CONTROL_TEXT } from './constants'
import { ListRow, LoadingState, Pill, SectionHeading } from './primitives'

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
const NO_PROFILE_VALUE = '__none__'

type PendingAction =
  | 'aux-reset'
  | 'main-apply'
  | 'profile-apply'
  | 'profile-delete'
  | 'profile-save'
  | 'profile-update'
  | `aux-${string}`

type AuxiliaryTaskAssignment = AuxiliaryModelsResponse['tasks'][number]

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

function normalizeAuxiliaryTasks(tasks: readonly Partial<AuxiliaryTaskAssignment>[] | undefined) {
  return AUX_TASKS.map(meta => {
    const entry = tasks?.find(task => task.task === meta.key)

    return {
      base_url: entry?.base_url || '',
      model: entry?.model || '',
      provider: entry?.provider || 'auto',
      task: meta.key
    }
  })
}

function routingSignature(routing: ModelRoutingPayload) {
  return JSON.stringify({
    auxiliary: normalizeAuxiliaryTasks(routing.auxiliary).map(task => ({
      base_url: task.base_url || '',
      model: task.model || '',
      provider: task.provider || 'auto',
      task: task.task
    })),
    main: {
      model: routing.main.model || '',
      provider: routing.main.provider || ''
    }
  })
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
  const [currentAuxiliary, setCurrentAuxiliary] = useState<AuxiliaryModelsResponse | null>(null)
  const [modelProfiles, setModelProfiles] = useState<ModelProfilesResponse>({ active: '', profiles: [] })
  const [selectedProfileId, setSelectedProfileId] = useState(NO_PROFILE_VALUE)
  const [newProfileName, setNewProfileName] = useState('')
  const [pendingAction, setPendingAction] = useState<null | PendingAction>(null)
  const [editingAuxTask, setEditingAuxTask] = useState<null | string>(null)
  const [auxDraft, setAuxDraft] = useState<{ model: string; provider: string }>({ model: '', provider: '' })
  // Aux slots reported stale by the backend immediately after a main-model
  // switch (provider differs from the new main). Cleared on next switch/reset.
  const [switchStaleAux, setSwitchStaleAux] = useState<StaleAuxAssignment[]>([])
  // Inline API-key entry for picking an unconfigured `api_key` provider in
  // place — mirrors the onboarding ApiKeyForm but scoped to the model picker.
  const [apiKeyDraft, setApiKeyDraft] = useState('')
  const [activating, setActivating] = useState(false)

  const refresh = useCallback(async () => {
    setLoading(true)
    setError('')

    try {
      const [modelInfo, modelOptions, auxiliaryModels] = await Promise.all([
        getGlobalModelInfo(),
        getGlobalModelOptions(),
        getAuxiliaryModels()
      ])

      const profiles = await getModelProfiles()

      setMainModel({ model: modelInfo.model, provider: modelInfo.provider })
      setProviders(modelOptions.providers || [])
      setSelectedProvider(prev => prev || modelInfo.provider)
      setSelectedModel(prev => prev || modelInfo.model)
      setCurrentAuxiliary(auxiliaryModels)
      setAuxiliary(auxiliaryModels)
      setModelProfiles(profiles)
      setSelectedProfileId(prev => {
        if (profiles.active && profiles.profiles.some(profile => profile.id === profiles.active)) {
          return profiles.active
        }

        if (prev !== NO_PROFILE_VALUE && profiles.profiles.some(profile => profile.id === prev)) {
          return prev
        }

        return NO_PROFILE_VALUE
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const providerOptions = providers.length ? providers : NO_PROVIDERS
  const selectedProfile = modelProfiles.profiles.find(profile => profile.id === selectedProfileId)
  const activeProfile = modelProfiles.profiles.find(profile => profile.id === modelProfiles.active)
  const busy = pendingAction !== null

  const duplicateProfileName = useMemo(() => {
    const normalized = newProfileName.trim().toLowerCase()

    return Boolean(
      normalized &&
        modelProfiles.profiles.some(profile => profile.name.trim().toLowerCase() === normalized)
    )
  }, [modelProfiles.profiles, newProfileName])

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

  // Paste an API key for the selected `api_key` provider, persist it, then
  // refresh so the now-authenticated provider's models populate. Auto-selects
  // the recommended default model so the user can Apply in one more click.
  const activateApiKeyProvider = useCallback(async () => {
    const keyEnv = selectedProviderRow?.key_env
    const slug = selectedProviderRow?.slug

    if (!keyEnv || !slug || !apiKeyDraft.trim()) {
      return
    }

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
  // to the shared onboarding flow scoped to this provider's real sign-in.
  const startProviderSetup = useCallback(() => {
    if (selectedProviderRow?.slug) {
      startManualProviderOAuth(selectedProviderRow.slug)
    }
  }, [selectedProviderRow])

  useEffect(() => {
    if (!selectedProfile) {
      return
    }

    setSelectedProvider(selectedProfile.main.provider)
    setSelectedModel(selectedProfile.main.model)
    setAuxiliary({
      main: selectedProfile.main,
      tasks: normalizeAuxiliaryTasks(selectedProfile.auxiliary)
    })
  }, [selectedProfile])

  const draftRouting = useMemo<ModelRoutingPayload>(
    () => ({
      auxiliary: normalizeAuxiliaryTasks(auxiliary?.tasks),
      main: { model: selectedModel, provider: selectedProvider }
    }),
    [auxiliary, selectedModel, selectedProvider]
  )

  const selectedProfileRouting = useMemo<ModelRoutingPayload | null>(() => {
    if (!selectedProfile) {
      return null
    }

    return {
      auxiliary: normalizeAuxiliaryTasks(selectedProfile.auxiliary),
      main: selectedProfile.main
    }
  }, [selectedProfile])

  const selectedProfileDirty = selectedProfileRouting
    ? routingSignature(draftRouting) !== routingSignature(selectedProfileRouting)
    : false

  const selectProfile = useCallback(
    (profileId: string) => {
      setSelectedProfileId(profileId)

      if (profileId === NO_PROFILE_VALUE && currentAuxiliary) {
        setSelectedProvider(currentAuxiliary.main.provider)
        setSelectedModel(currentAuxiliary.main.model)
        setAuxiliary(currentAuxiliary)
      }
    },
    [currentAuxiliary]
  )

  const applyDraftRouting = useCallback(
    async (pending: PendingAction, options: { preferSavedProfile?: boolean } = {}) => {
      const routing = draftRouting

      if (options.preferSavedProfile && selectedProfile && !selectedProfileDirty) {
        setPendingAction(pending)
        setError('')

        try {
          const result = await applyModelProfile(selectedProfile.id)

          if (result.profile?.main.provider && result.profile.main.model) {
            setSelectedProvider(result.profile.main.provider)
            setSelectedModel(result.profile.main.model)
            onMainModelChanged?.(result.profile.main.provider, result.profile.main.model)
          }

          setSwitchStaleAux([])
          await refresh()
        } catch (err) {
          setError(err instanceof Error ? err.message : String(err))
        } finally {
          setPendingAction(null)
        }

        return
      }

      if (!routing.main.provider || !routing.main.model) {
        setError('Provider and model required')

        return
      }

      setPendingAction(pending)
      setError('')

      try {
        const result = await setModelAssignment({
          model: routing.main.model,
          provider: routing.main.provider,
          scope: 'main'
        })

        await setModelAssignment({
          model: routing.main.model,
          provider: routing.main.provider,
          scope: 'auxiliary',
          task: '__reset__'
        })

        for (const task of routing.auxiliary) {
          if (!task.provider || task.provider === 'auto') {
            continue
          }

          await setModelAssignment({
            model: task.model || '',
            provider: task.provider,
            scope: 'auxiliary',
            task: task.task
          })
        }

        const provider = result.provider || routing.main.provider
        const model = result.model || routing.main.model
        const appliedAuxiliary = { main: { provider, model }, tasks: routing.auxiliary }
        setMainModel({ provider, model })
        setSelectedProvider(provider)
        setSelectedModel(model)
        setAuxiliary(appliedAuxiliary)
        setCurrentAuxiliary(appliedAuxiliary)
        setSelectedProfileId(NO_PROFILE_VALUE)
        setSwitchStaleAux([])
        onMainModelChanged?.(provider, model)
        await refresh()
        setSelectedProfileId(NO_PROFILE_VALUE)
        setSelectedProvider(provider)
        setSelectedModel(model)
        setAuxiliary(appliedAuxiliary)
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
      } finally {
        setPendingAction(null)
      }
    },
    [draftRouting, onMainModelChanged, refresh, selectedProfile, selectedProfileDirty]
  )

  const applyRouting = useCallback(async () => {
    await applyDraftRouting('main-apply')
  }, [applyDraftRouting])

  const saveCurrentAsProfile = useCallback(async () => {
    const name = newProfileName.trim()

    if (!name) {
      setError('Profile name required')

      return
    }

    if (duplicateProfileName) {
      setError('Profile name already exists')

      return
    }

    setPendingAction('profile-save')
    setError('')

    try {
      const result = await createModelProfile(name, draftRouting)
      setNewProfileName('')
      setSelectedProfileId(result.profile?.id || result.active || NO_PROFILE_VALUE)
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setPendingAction(null)
    }
  }, [draftRouting, duplicateProfileName, newProfileName, refresh])

  const applySelectedProfile = useCallback(async () => {
    if (selectedProfileId === NO_PROFILE_VALUE) {
      return
    }

    await applyDraftRouting('profile-apply', { preferSavedProfile: true })
  }, [applyDraftRouting, selectedProfileId])

  const updateSelectedProfile = useCallback(async () => {
    if (selectedProfileId === NO_PROFILE_VALUE) {
      return
    }

    setPendingAction('profile-update')
    setError('')

    try {
      await updateModelProfile(selectedProfileId, { routing: draftRouting })
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setPendingAction(null)
    }
  }, [draftRouting, refresh, selectedProfileId])

  const deleteSelectedProfile = useCallback(async () => {
    if (selectedProfileId === NO_PROFILE_VALUE) {
      return
    }

    setPendingAction('profile-delete')
    setError('')

    try {
      await deleteModelProfile(selectedProfileId)
      setSelectedProfileId(NO_PROFILE_VALUE)
      await refresh()
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setPendingAction(null)
    }
  }, [refresh, selectedProfileId])

  const setAuxiliaryToMain = useCallback((task: string) => {
    setAuxiliary(prev => ({
      main: { model: selectedModel, provider: selectedProvider },
      tasks: normalizeAuxiliaryTasks(prev?.tasks).map(entry =>
        entry.task === task ? { ...entry, base_url: '', model: '', provider: 'auto' } : entry
      )
    }))
  }, [selectedModel, selectedProvider])

  const applyAuxiliaryDraft = useCallback(
    (task: string) => {
      if (!auxDraft.provider || !auxDraft.model) {
        return
      }

      setAuxiliary(prev => ({
        main: { model: selectedModel, provider: selectedProvider },
        tasks: normalizeAuxiliaryTasks(prev?.tasks).map(entry =>
          entry.task === task
            ? { ...entry, base_url: '', model: auxDraft.model, provider: auxDraft.provider }
            : entry
        )
      }))
      setEditingAuxTask(null)
    },
    [auxDraft, selectedModel, selectedProvider]
  )

  const beginAuxiliaryEdit = useCallback(
    (task: string) => {
      const current = auxiliary?.tasks.find(entry => entry.task === task)

      const initialProvider =
        current?.provider && current.provider !== 'auto' ? current.provider : selectedProvider

      const initialModel = current?.model || selectedModel
      setAuxDraft({ provider: initialProvider, model: initialModel })
      setEditingAuxTask(task)
    },
    [auxiliary, selectedModel, selectedProvider]
  )

  const resetAuxiliaryModels = useCallback(() => {
    setAuxiliary({
      main: { model: selectedModel, provider: selectedProvider },
      tasks: normalizeAuxiliaryTasks(undefined)
    })
    setSwitchStaleAux([])
  }, [selectedModel, selectedProvider])

  if (loading && !mainModel) {
    return <LoadingState label={m.loading} />
  }

  return (
    <div className="grid gap-6">
      <section>
        <SectionHeading
          icon={Layers3}
          meta={activeProfile ? `active · ${activeProfile.name}` : undefined}
          title="Model profiles"
        />
        <p className="mb-3 text-xs text-muted-foreground">
          Save and switch named combinations of the main model plus auxiliary model assignments.
        </p>
        <div className="flex flex-wrap items-center gap-2">
          <Select onValueChange={selectProfile} value={selectedProfileId}>
            <SelectTrigger className={cn('min-w-64', CONTROL_TEXT)}>
              <SelectValue placeholder="Model profile" />
            </SelectTrigger>
            <SelectContent>
              {modelProfiles.profiles.length ? (
                <>
                  <SelectItem value={NO_PROFILE_VALUE}>No profile selected</SelectItem>
                  {modelProfiles.profiles.map(profile => (
                    <SelectItem key={profile.id} value={profile.id}>
                      {profile.name}
                    </SelectItem>
                  ))}
                </>
              ) : (
                <SelectItem disabled value={NO_PROFILE_VALUE}>
                  No model profiles
                </SelectItem>
              )}
            </SelectContent>
          </Select>
          <Button
            disabled={selectedProfileId === NO_PROFILE_VALUE || busy}
            onClick={() => void applySelectedProfile()}
            size="sm"
          >
            {pendingAction === 'profile-apply' ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Sparkles className="size-3.5" />
            )}
            {pendingAction === 'profile-apply' ? 'Applying...' : 'Apply profile'}
          </Button>
          <Button
            disabled={selectedProfileId === NO_PROFILE_VALUE || busy}
            onClick={() => void updateSelectedProfile()}
            size="sm"
            variant="outline"
          >
            {pendingAction === 'profile-update' ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Save className="size-3.5" />
            )}
            {pendingAction === 'profile-update' ? 'Updating...' : 'Update profile'}
          </Button>
          <Button
            disabled={selectedProfileId === NO_PROFILE_VALUE || busy}
            onClick={() => void deleteSelectedProfile()}
            size="sm"
            variant="ghost"
          >
            {pendingAction === 'profile-delete' ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Trash2 className="size-3.5" />
            )}
            {pendingAction === 'profile-delete' ? 'Deleting...' : 'Delete'}
          </Button>
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <Input
            className={cn('min-w-64', CONTROL_TEXT)}
            onChange={event => setNewProfileName(event.target.value)}
            placeholder="New profile name"
            value={newProfileName}
          />
          <Button
            disabled={!newProfileName.trim() || duplicateProfileName || busy}
            onClick={() => void saveCurrentAsProfile()}
            size="sm"
          >
            {pendingAction === 'profile-save' ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <Save className="size-3.5" />
            )}
            {pendingAction === 'profile-save' ? 'Saving...' : 'Save as profile'}
          </Button>
          {duplicateProfileName && <span className="text-xs text-destructive">Profile name already exists.</span>}
        </div>
        {selectedProfile && (
          <div className="mt-2 font-mono text-[0.68rem] text-muted-foreground">
            {selectedProfile.main.provider || '(no provider)'} · {selectedProfile.main.model || '(no model)'} ·{' '}
            {selectedProfile.auxiliary_overrides} auxiliary override
            {selectedProfile.auxiliary_overrides === 1 ? '' : 's'}
          </div>
        )}
      </section>

      <section>
        <p className="mb-3 text-xs text-muted-foreground">
          {m.appliesDesc}
        </p>
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
                  {(selectedProviderModels.length ? selectedProviderModels : []).map(model => (
                    <SelectItem key={model} value={model}>
                      {model}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                disabled={!selectedProvider || !selectedModel || busy}
                onClick={() => void applyRouting()}
                size="sm"
              >
                {pendingAction === 'main-apply' && <Loader2 className="size-3.5 animate-spin" />}
                {pendingAction === 'main-apply' ? m.applying : t.common.apply}
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
        {error && <div className="mt-2 text-xs text-destructive">{error}</div>}
        {switchStaleAux.length > 0 && (
          <div className="mt-2">
            <StaleAuxWarning
              applying={busy}
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
            disabled={busy}
            onClick={() => resetAuxiliaryModels()}
            size="sm"
            variant="textStrong"
          >
            {m.resetAllToMain}
          </Button>
        </div>
        <p className="mb-2 text-xs text-muted-foreground">
          {m.auxiliaryDesc}
        </p>
        {switchStaleAux.length === 0 && persistentStaleAux.length > 0 && (
          <div className="mb-2.5">
            <StaleAuxWarning
              applying={busy}
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
                        disabled={busy}
                        onClick={() => setAuxiliaryToMain(meta.key)}
                        size="sm"
                        variant="text"
                      >
                        {m.setToMain}
                      </Button>
                      <Button
                        disabled={!providers.length || busy}
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
                          {(auxDraftProviderModels.length ? auxDraftProviderModels : []).map(model => (
                            <SelectItem key={model} value={model}>
                              {model}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <Button
                        disabled={!auxDraft.provider || !auxDraft.model || busy}
                        onClick={() => applyAuxiliaryDraft(meta.key)}
                        size="sm"
                      >
                        {t.common.apply}
                      </Button>
                      <Button onClick={() => setEditingAuxTask(null)} size="sm" variant="ghost">
                        {t.common.cancel}
                      </Button>
                    </div>
                  )
                }
                description={
                  <span className="font-mono text-[0.68rem]">
                    {isAuto
                      ? m.autoUseMain
                      : `${current.provider} · ${current.model || m.providerDefault}`}
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
    </div>
  )
}
