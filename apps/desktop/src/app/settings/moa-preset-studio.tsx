import type { ModelOptionProvider } from '@hermes/shared'
import { useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Input } from '@/components/ui/input'
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { getMoaModels, saveMoaModels } from '@/hermes'
import { useI18n } from '@/i18n'
import { Cpu } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $activeConnectionId } from '@/store/connections'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $activeSessionId, getComposerSelectionGeneration } from '@/store/session'
import type { MoaConfigResponse, MoaModelSlot, MoaPresetConfig } from '@/types/hermes'

import { CONTROL_TEXT } from './constants'
import {
  addMoaReference,
  buildMoaFanout,
  createMoaPreset,
  deleteMoaPreset,
  duplicateMoaPreset,
  getOwnMoaPreset,
  moaConfigComplete,
  moaFanoutEveryN,
  moaFanoutMode,
  moveMoaReference,
  prepareMoaConfigForSave,
  renameMoaPreset,
  updateMoaSlot,
  validateMoaPresetName
} from './moa-preset-helpers'
import { ListRow, Pill, SectionHeading } from './primitives'

export interface MoaPresetStudioProps {
  config: MoaConfigResponse
  onUseMoaPreset?: (name: string) => boolean | Promise<boolean> | Promise<void> | void
  scopeProfile?: string
  providers: readonly ModelOptionProvider[]
}

type SaveStatus = 'dirty' | 'error' | 'idle' | 'incomplete' | 'saved' | 'saving'

const EFFORT_VALUES = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultra'] as const
const PROVIDER_DEFAULT_EFFORT = '__provider_default__'

const withActive = (values: readonly string[], active: string): readonly string[] =>
  active && !values.includes(active) ? [active, ...values] : values

const firstPresetName = (config: MoaConfigResponse): string =>
  (getOwnMoaPreset(config, config.default_preset) && config.default_preset) || Object.keys(config.presets)[0] || ''

const ownEnabledMoaPreset = (config: MoaConfigResponse, name: string): boolean =>
  getOwnMoaPreset(config, name)?.enabled === true

const errorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error))

const moaSnapshotsMatch = (left: MoaConfigResponse, right: MoaConfigResponse): boolean =>
  JSON.stringify(prepareMoaConfigForSave(left)) === JSON.stringify(prepareMoaConfigForSave(right))

const optionalFiniteInput = (value: string): number | null => {
  if (!value.trim()) {
    return null
  }

  const parsed = Number(value)

  return Number.isFinite(parsed) ? parsed : Number.NaN
}

const numberInputValue = (value: null | number | undefined): number | string =>
  typeof value === 'number' && Number.isFinite(value) ? value : ''

const setSlotEffort = (slot: MoaModelSlot, effort: string): MoaModelSlot => {
  if (effort === PROVIDER_DEFAULT_EFFORT) {
    const { reasoning_effort: _reasoningEffort, ...rest } = slot

    return rest as MoaModelSlot
  }

  return { ...slot, reasoning_effort: effort }
}

const BLOCKED_SAVE_STATUSES = new Set<SaveStatus>(['dirty', 'error', 'incomplete', 'saving'])

const activationOwner = () =>
  JSON.stringify([
    $activeConnectionId.get(),
    normalizeProfileKey($activeGatewayProfile.get()),
    $activeSessionId.get(),
    getComposerSelectionGeneration()
  ])

const editingActiveProfile = (profile?: string) =>
  profile === undefined || normalizeProfileKey(profile) === normalizeProfileKey($activeGatewayProfile.get())

const activationReady = (draft: MoaConfigResponse, saved: MoaConfigResponse, name: string) =>
  [draft, saved].every(config => moaConfigComplete(config) && ownEnabledMoaPreset(config, name))

function useMoaPresetStudio({ config, onUseMoaPreset, providers, scopeProfile }: MoaPresetStudioProps) {
  const { t } = useI18n()
  const copy = t.settings.moa
  const modelCopy = t.settings.model
  const queryClient = useQueryClient()
  const initialPreset = firstPresetName(config)
  const [moa, setMoa] = useState(config)
  const [selectedPreset, setSelectedPresetState] = useState(initialPreset)
  const [presetName, setPresetName] = useState(initialPreset)
  const [newPresetName, setNewPresetName] = useState('')
  const [blankDraftPreset, setBlankDraftPreset] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null)
  const [applying, setApplying] = useState(false)
  const [dirty, setDirtyState] = useState(false)
  const [saveStatus, setSaveStatusState] = useState<SaveStatus>('idle')
  const [saveError, setSaveError] = useState('')

  const moaRef = useRef(config)
  const persistedRef = useRef(config)
  const selectedPresetRef = useRef(initialPreset)
  const dirtyRef = useRef(false)
  const saveStatusRef = useRef<SaveStatus>('idle')
  const sourceGeneration = useRef(0)
  const lifecycleOperation = useRef(0)
  const applyingRef = useRef(false)

  const setSaveStatus = useCallback((status: SaveStatus) => {
    saveStatusRef.current = status
    setSaveStatusState(status)
  }, [])

  const setDirty = useCallback((next: boolean) => {
    dirtyRef.current = next
    setDirtyState(next)
  }, [])

  const selectPreset = useCallback((name: string) => {
    selectedPresetRef.current = name
    setSelectedPresetState(name)
    setPresetName(name)
  }, [])

  const replaceDraft = useCallback((next: MoaConfigResponse) => {
    moaRef.current = next
    setMoa(next)
  }, [])

  const acceptSaved = useCallback(
    (saved: MoaConfigResponse) => {
      persistedRef.current = saved
      replaceDraft(saved)

      const nextSelected = getOwnMoaPreset(saved, selectedPresetRef.current)
        ? selectedPresetRef.current
        : firstPresetName(saved)

      selectPreset(nextSelected)
      setBlankDraftPreset(null)
      setDirty(false)
      setSaveStatus('saved')
      setSaveError('')
      void queryClient.invalidateQueries({ queryKey: ['model-options'] })
    },
    [queryClient, replaceDraft, selectPreset, setDirty, setSaveStatus]
  )

  const applyDraft = useCallback(
    (next: MoaConfigResponse, nextSelected?: string): boolean => {
      if (applyingRef.current) {
        return false
      }

      replaceDraft(next)

      if (nextSelected !== undefined) {
        selectPreset(nextSelected)
      }

      const nextDirty = !moaSnapshotsMatch(next, persistedRef.current)
      setDirty(nextDirty)
      setSaveError('')

      if (!moaConfigComplete(prepareMoaConfigForSave(next))) {
        setSaveStatus('incomplete')
      } else {
        setSaveStatus(nextDirty ? 'dirty' : 'idle')
      }

      if (!nextDirty) {
        setBlankDraftPreset(null)
      }

      return true
    },
    [replaceDraft, selectPreset, setDirty, setSaveStatus]
  )

  // A parent config replacement is a new authoritative source. Unsaved local
  // edits are intentionally discarded; explicit Save is the only write path.
  // eslint-disable-next-line no-restricted-syntax -- prop replacement invalidates a draft transaction, not an atom mirror
  useEffect(() => {
    sourceGeneration.current += 1
    lifecycleOperation.current += 1
    applyingRef.current = false
    moaRef.current = config
    persistedRef.current = config
    setMoa(config)

    const nextSelected = getOwnMoaPreset(config, selectedPresetRef.current)
      ? selectedPresetRef.current
      : firstPresetName(config)

    selectPreset(nextSelected)
    setBlankDraftPreset(null)
    setDirty(false)
    setDeleteTarget(null)
    setApplying(false)
    setSaveStatus('idle')
    setSaveError('')
  }, [config, selectPreset, setDirty, setSaveStatus])

  useEffect(
    () => () => {
      sourceGeneration.current += 1
      lifecycleOperation.current += 1
      applyingRef.current = false
    },
    []
  )

  const updatePreset = useCallback(
    (updater: (preset: MoaPresetConfig) => MoaPresetConfig) => {
      if (applyingRef.current) {
        return
      }

      const previous = moaRef.current
      const name = selectedPresetRef.current
      const preset = getOwnMoaPreset(previous, name)

      if (!preset) {
        return
      }

      const next: MoaConfigResponse = {
        ...previous,
        presets: { ...previous.presets, [name]: updater(preset) }
      }

      applyDraft(next)
    },
    [applyDraft]
  )

  const saveDraft = useCallback(async (): Promise<boolean> => {
    if (applyingRef.current || !dirtyRef.current) {
      return false
    }

    const source = sourceGeneration.current
    const prepared = prepareMoaConfigForSave(moaRef.current)

    if (!moaConfigComplete(prepared)) {
      setSaveStatus('incomplete')

      return false
    }

    const operation = lifecycleOperation.current + 1
    lifecycleOperation.current = operation
    applyingRef.current = true
    setApplying(true)
    setSaveStatus('saving')
    setSaveError('')

    try {
      const saved = await saveMoaModels(prepared, scopeProfile)

      if (lifecycleOperation.current !== operation || sourceGeneration.current !== source) {
        return false
      }

      acceptSaved(saved)

      return true
    } catch (error) {
      if (lifecycleOperation.current === operation && sourceGeneration.current === source) {
        setSaveStatus('error')
        setSaveError(errorMessage(error))
      }

      return false
    } finally {
      if (lifecycleOperation.current === operation) {
        applyingRef.current = false
        setApplying(false)
      }
    }
  }, [acceptSaved, setSaveStatus, scopeProfile])

  const handleUsePreset = useCallback(async (): Promise<boolean> => {
    const source = sourceGeneration.current
    const operation = lifecycleOperation.current + 1
    const owner = activationOwner()
    const name = selectedPresetRef.current
    const current = moaRef.current
    const persisted = persistedRef.current

    const blockedSaveStatus = dirtyRef.current || BLOCKED_SAVE_STATUSES.has(saveStatusRef.current)

    if (
      applyingRef.current ||
      !onUseMoaPreset ||
      !editingActiveProfile(scopeProfile) ||
      blockedSaveStatus ||
      !activationReady(current, persisted, name)
    ) {
      return false
    }

    lifecycleOperation.current = operation
    applyingRef.current = true
    setApplying(true)

    try {
      // getMoaModels captures the active REST profile synchronously. Treat this
      // fresh response, not the editable draft, as activation authority.
      const fresh = await getMoaModels(scopeProfile)

      if (
        sourceGeneration.current !== source ||
        lifecycleOperation.current !== operation ||
        selectedPresetRef.current !== name ||
        activationOwner() !== owner
      ) {
        return false
      }

      // Top-level enabled is the default preset's compatibility view, not a
      // global switch. Only the selected preset authorizes this activation.
      if (!moaConfigComplete(fresh) || !ownEnabledMoaPreset(fresh, name)) {
        return false
      }

      return (await onUseMoaPreset(name)) !== false
    } catch {
      // A transient authority read or selection failure must fail closed without
      // turning Save into a permanent error state; releasing applying allows a
      // deliberate retry.
      return false
    } finally {
      if (lifecycleOperation.current === operation) {
        applyingRef.current = false
        setApplying(false)
      }
    }
  }, [onUseMoaPreset, scopeProfile])

  const slotProviders = useMemo(
    () => providers.filter(provider => provider.slug.trim().toLowerCase() !== 'moa'),
    [providers]
  )

  const providerSlugs = useMemo(() => slotProviders.map(provider => provider.slug).filter(Boolean), [slotProviders])

  const providerName = useCallback(
    (slug: string) => slotProviders.find(provider => provider.slug === slug)?.name ?? slug,
    [slotProviders]
  )

  const modelsForProvider = useCallback(
    (provider: string) => slotProviders.find(row => row.slug === provider)?.models ?? [],
    [slotProviders]
  )

  const preset = getOwnMoaPreset(moa, selectedPreset) ?? getOwnMoaPreset(moa, firstPresetName(moa))
  const renameIssue = validateMoaPresetName(moa, presetName, selectedPreset)
  const duplicateIssue = validateMoaPresetName(moa, newPresetName)
  const persisted = persistedRef.current

  const addPresetDisabled =
    applying ||
    blankDraftPreset !== null ||
    !!duplicateIssue ||
    !moaConfigComplete(moa) ||
    !moaConfigComplete(persisted)

  const usePresetDisabled =
    applying ||
    dirty ||
    !onUseMoaPreset ||
    !editingActiveProfile(scopeProfile) ||
    BLOCKED_SAVE_STATUSES.has(saveStatus) ||
    !activationReady(moa, persisted, selectedPreset)

  const saveDisabled = applying || !dirty || !moaConfigComplete(prepareMoaConfigForSave(moa))

  const effortLabel = useCallback(
    (effort: (typeof EFFORT_VALUES)[number]) =>
      effort === 'none'
        ? modelCopy.reasoningOff
        : t.shell.modelOptions[effort as Exclude<(typeof EFFORT_VALUES)[number], 'none'>],
    [modelCopy.reasoningOff, t.shell.modelOptions]
  )

  return {
    t,
    copy,
    modelCopy,
    moa,
    selectedPreset,
    presetName,
    setPresetName,
    newPresetName,
    setNewPresetName,
    blankDraftPreset,
    setBlankDraftPreset,
    deleteTarget,
    setDeleteTarget,
    applying,
    dirty,
    saveStatus,
    saveError,
    moaRef,
    selectPreset,
    applyDraft,
    updatePreset,
    saveDraft,
    handleUsePreset,
    providerSlugs,
    providerName,
    modelsForProvider,
    preset,
    renameIssue,
    duplicateIssue,
    addPresetDisabled,
    usePresetDisabled,
    saveDisabled,
    effortLabel
  }
}

function PresetActions({ state }: { state: ReturnType<typeof useMoaPresetStudio> }) {
  const {
    copy,
    moa,
    selectedPreset,
    presetName,
    setPresetName,
    newPresetName,
    setNewPresetName,
    blankDraftPreset,
    setBlankDraftPreset,
    setDeleteTarget,
    applying,
    moaRef,
    selectPreset,
    applyDraft,
    handleUsePreset,
    preset,
    renameIssue,
    duplicateIssue,
    addPresetDisabled,
    usePresetDisabled
  } = state

  return (
    <>
      <div className="flex flex-wrap items-center gap-2">
        <Select disabled={blankDraftPreset !== null} onValueChange={selectPreset} value={selectedPreset}>
          <SelectTrigger aria-label={copy.preset} className={cn('min-w-40', CONTROL_TEXT)}>
            <SelectValue placeholder={copy.preset} />
          </SelectTrigger>
          <SelectContent>
            {Object.keys(moa.presets).map(name => (
              <SelectItem key={name} value={name}>
                {name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {selectedPreset === moa.default_preset ? <Pill tone="primary">{copy.defaultBadge}</Pill> : null}
        {selectedPreset === moa.active_preset ? <Pill>{copy.activeBadge}</Pill> : null}
        <Button disabled={usePresetDisabled} onClick={() => void handleUsePreset()} size="sm">
          {copy.useInThisChat}
        </Button>
        <Button
          disabled={applying || blankDraftPreset !== null || selectedPreset === moa.default_preset}
          onClick={() => applyDraft({ ...moaRef.current, default_preset: selectedPreset })}
          size="sm"
          variant="textStrong"
        >
          {copy.setDefault}
        </Button>
        <Button
          disabled={applying || Object.keys(moa.presets).length <= 1}
          onClick={() => setDeleteTarget(selectedPreset)}
          size="sm"
          variant="destructive"
        >
          {copy.deletePreset}
        </Button>
      </div>

      <div className="mt-3 grid gap-2 @2xl:grid-cols-2">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <Input
            aria-label={copy.presetName}
            className={cn('min-w-40 flex-1', CONTROL_TEXT)}
            onChange={event => setPresetName(event.target.value)}
            value={presetName}
          />
          <Button
            disabled={applying || blankDraftPreset !== null || !!renameIssue || presetName.trim() === selectedPreset}
            onClick={() => {
              const name = presetName.trim()
              const next = renameMoaPreset(moaRef.current, selectedPreset, name)
              applyDraft(next, name)
            }}
            size="sm"
            variant="textStrong"
          >
            {copy.renamePreset}
          </Button>
        </div>
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <Input
            aria-label={copy.newPresetName}
            className={cn('min-w-40 flex-1', CONTROL_TEXT)}
            disabled={blankDraftPreset !== null}
            onChange={event => setNewPresetName(event.target.value)}
            value={newPresetName}
          />
          <Button
            disabled={addPresetDisabled}
            onClick={() => {
              const name = newPresetName.trim()
              const next = createMoaPreset(moaRef.current, name)

              applyDraft(next, name)
              setNewPresetName('')
              setBlankDraftPreset(name)
            }}
            size="sm"
          >
            {copy.addPreset}
          </Button>
          <Button
            disabled={applying || blankDraftPreset !== null || !!duplicateIssue}
            onClick={() => {
              const name = newPresetName.trim()
              const next = duplicateMoaPreset(moaRef.current, selectedPreset, name)
              setNewPresetName('')
              applyDraft(next, name)
            }}
            size="sm"
            variant="textStrong"
          >
            {copy.duplicatePreset}
          </Button>
        </div>
      </div>
      {presetName !== selectedPreset && renameIssue ? (
        <p className="mt-1 text-xs text-destructive">{renameIssue === 'blank' ? copy.nameBlank : copy.nameDuplicate}</p>
      ) : null}
      {newPresetName && duplicateIssue ? (
        <p className="mt-1 text-xs text-destructive">
          {duplicateIssue === 'blank' ? copy.nameBlank : copy.nameDuplicate}
        </p>
      ) : null}
    </>
  )
}

export function MoaPresetStudio(props: MoaPresetStudioProps) {
  const state = useMoaPresetStudio(props)

  const {
    t,
    copy,
    modelCopy,
    moa,
    selectedPreset,
    presetName,
    setPresetName,
    newPresetName,
    setNewPresetName,
    blankDraftPreset,
    setBlankDraftPreset,
    deleteTarget,
    setDeleteTarget,
    applying,
    dirty,
    saveStatus,
    saveError,
    moaRef,
    selectPreset,
    applyDraft,
    updatePreset,
    saveDraft,
    handleUsePreset,
    providerSlugs,
    providerName,
    modelsForProvider,
    preset,
    renameIssue,
    duplicateIssue,
    addPresetDisabled,
    usePresetDisabled,
    saveDisabled,
    effortLabel
  } = state

  if (!preset) {
    return (
      <section>
        <SectionHeading icon={Cpu} title={copy.title} />
        <div className="text-xs text-destructive" role="alert">
          {copy.unavailable}
        </div>
      </section>
    )
  }

  const renderProviderSelect = (slot: MoaModelSlot, label: string, onChange: (value: string) => void) => (
    <Select onValueChange={onChange} value={slot.provider}>
      <SelectTrigger aria-label={label} className={cn('min-w-32', CONTROL_TEXT)}>
        <SelectValue placeholder={modelCopy.provider} />
      </SelectTrigger>
      <SelectContent>
        {withActive(providerSlugs, slot.provider).map(slug => (
          <SelectItem key={slug} value={slug}>
            {providerName(slug)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )

  const renderModelSelect = (slot: MoaModelSlot, label: string, onChange: (value: string) => void) => (
    <Select onValueChange={onChange} value={slot.model}>
      <SelectTrigger aria-label={label} className={cn('min-w-48', CONTROL_TEXT)}>
        <SelectValue placeholder={modelCopy.model} />
      </SelectTrigger>
      <SelectContent>
        {withActive(modelsForProvider(slot.provider), slot.model).map(model => (
          <SelectItem key={model} value={model}>
            {model}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )

  const renderEffortSelect = (slot: MoaModelSlot, label: string, onChange: (value: string) => void) => (
    <Select onValueChange={onChange} value={slot.reasoning_effort || PROVIDER_DEFAULT_EFFORT}>
      <SelectTrigger aria-label={label} className={cn('min-w-36', CONTROL_TEXT)}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={PROVIDER_DEFAULT_EFFORT}>{copy.providerDefault}</SelectItem>
        {EFFORT_VALUES.map(value => (
          <SelectItem key={value} value={value}>
            {effortLabel(value)}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )

  return (
    <section>
      <SectionHeading icon={Cpu} title={copy.title} />
      <p className="mb-3 text-xs text-(--ui-text-tertiary)">{copy.description}</p>

      <fieldset className="m-0 min-w-0 border-0 p-0" disabled={applying}>
        <PresetActions state={state} />
        <ListRow
          action={
            <Switch
              aria-label={copy.presetEnabled}
              checked={preset.enabled}
              onCheckedChange={enabled => updatePreset(current => ({ ...current, enabled }))}
              size="xs"
            />
          }
          description={copy.enabledDescription}
          title={copy.presetEnabled}
        />

        <SectionHeading icon={Cpu} title={copy.referencesTitle} />
        <div className="grid gap-1">
          {preset.reference_models.map((slot, index) => {
            const ordinal = index + 1

            return (
              <ListRow
                below={
                  <div className="mt-2 flex flex-wrap items-center gap-2 pt-1">
                    {renderProviderSelect(slot, copy.referenceProvider(ordinal), provider =>
                      updatePreset(current => ({
                        ...current,
                        reference_models: current.reference_models.map((candidate, candidateIndex) =>
                          candidateIndex === index ? updateMoaSlot(candidate, { provider }) : candidate
                        )
                      }))
                    )}
                    {renderModelSelect(slot, copy.referenceModel(ordinal), model =>
                      updatePreset(current => ({
                        ...current,
                        reference_models: current.reference_models.map((candidate, candidateIndex) =>
                          candidateIndex === index ? updateMoaSlot(candidate, { model }) : candidate
                        )
                      }))
                    )}
                    {renderEffortSelect(slot, copy.referenceEffort(ordinal), effort =>
                      updatePreset(current => ({
                        ...current,
                        reference_models: current.reference_models.map((candidate, candidateIndex) =>
                          candidateIndex === index ? setSlotEffort(candidate, effort) : candidate
                        )
                      }))
                    )}
                    <label className="flex items-center gap-2 text-xs">
                      <span className="sr-only">{copy.referenceEnabled(ordinal)}</span>
                      <Switch
                        aria-label={copy.referenceEnabled(ordinal)}
                        checked={slot.enabled !== false}
                        onCheckedChange={enabled =>
                          updatePreset(current => ({
                            ...current,
                            reference_models: current.reference_models.map((candidate, candidateIndex) =>
                              candidateIndex === index ? { ...candidate, enabled } : candidate
                            )
                          }))
                        }
                        size="xs"
                      />
                    </label>
                    <Button
                      aria-label={copy.moveReferenceUp(ordinal)}
                      disabled={applying || index === 0}
                      onClick={() => updatePreset(current => moveMoaReference(current, index, index - 1))}
                      size="sm"
                      variant="text"
                    >
                      {copy.moveUp}
                    </Button>
                    <Button
                      aria-label={copy.moveReferenceDown(ordinal)}
                      disabled={applying || index === preset.reference_models.length - 1}
                      onClick={() => updatePreset(current => moveMoaReference(current, index, index + 1))}
                      size="sm"
                      variant="text"
                    >
                      {copy.moveDown}
                    </Button>
                    <Button
                      aria-label={copy.removeReference(ordinal)}
                      disabled={applying || preset.reference_models.length <= 1}
                      onClick={() =>
                        updatePreset(current => ({
                          ...current,
                          reference_models: current.reference_models.filter(
                            (_, candidateIndex) => candidateIndex !== index
                          )
                        }))
                      }
                      size="sm"
                      variant="text"
                    >
                      {copy.remove}
                    </Button>
                  </div>
                }
                className={cn(slot.enabled === false && 'opacity-60')}
                description={
                  <span className="font-mono text-[0.68rem]">
                    {slot.provider} · {slot.model || modelCopy.model}
                  </span>
                }
                key={slot.continuity_id || `${selectedPreset}-${index}`}
                title={copy.reference(ordinal)}
              />
            )
          })}
          <Button
            disabled={applying}
            onClick={() => updatePreset(current => addMoaReference(current))}
            size="sm"
            variant="textStrong"
          >
            {copy.addReference}
          </Button>
          <ListRow
            below={
              <div className="mt-2 flex flex-wrap items-center gap-2 pt-1">
                {renderProviderSelect(preset.aggregator, copy.aggregatorProvider, provider =>
                  updatePreset(current => ({
                    ...current,
                    aggregator: updateMoaSlot(current.aggregator, { provider })
                  }))
                )}
                {renderModelSelect(preset.aggregator, copy.aggregatorModel, model =>
                  updatePreset(current => ({
                    ...current,
                    aggregator: updateMoaSlot(current.aggregator, { model })
                  }))
                )}
                {renderEffortSelect(preset.aggregator, copy.aggregatorEffort, effort =>
                  updatePreset(current => ({
                    ...current,
                    aggregator: setSlotEffort(current.aggregator, effort)
                  }))
                )}
              </div>
            }
            description={
              <span className="font-mono text-[0.68rem]">
                {preset.aggregator.provider} · {preset.aggregator.model || modelCopy.model}
              </span>
            }
            title={copy.aggregator}
          />
        </div>

        <SectionHeading icon={Cpu} title={copy.executionTitle} />
        <ListRow
          action={
            <div className="flex flex-wrap items-center justify-end gap-2">
              <SegmentedControl
                onChange={mode =>
                  updatePreset(current => ({
                    ...current,
                    fanout: buildMoaFanout(mode, moaFanoutEveryN(current.fanout, 3))
                  }))
                }
                options={[
                  { id: 'user_turn', label: copy.oncePerUserTurn },
                  { id: 'per_iteration', label: copy.everyToolIteration },
                  { id: 'every_n', label: copy.everyNIterations }
                ]}
                value={moaFanoutMode(preset.fanout)}
              />
              {moaFanoutMode(preset.fanout) === 'every_n' ? (
                <Input
                  aria-label={copy.everyNCount}
                  className={cn('w-20', CONTROL_TEXT)}
                  min={2}
                  onChange={event => {
                    const raw = Number(event.target.value)

                    if (!Number.isInteger(raw)) {
                      return
                    }

                    updatePreset(current => ({
                      ...current,
                      fanout: buildMoaFanout('every_n', raw >= 2 ? raw : 2)
                    }))
                  }}
                  step={1}
                  type="number"
                  value={moaFanoutEveryN(preset.fanout, 3)}
                />
              ) : null}
            </div>
          }
          description={copy.cadenceDescription}
          title={copy.cadence}
        />
        <ListRow
          action={
            <Input
              aria-label={copy.advisorTemperature}
              className={cn('w-28', CONTROL_TEXT)}
              onChange={event =>
                updatePreset(current => ({
                  ...current,
                  reference_temperature: optionalFiniteInput(event.target.value)
                }))
              }
              step="any"
              type="number"
              value={numberInputValue(preset.reference_temperature)}
            />
          }
          description={copy.advisorTemperatureDescription}
          title={copy.advisorTemperature}
        />
        <ListRow
          action={
            <Input
              aria-label={copy.aggregatorTemperature}
              className={cn('w-28', CONTROL_TEXT)}
              onChange={event =>
                updatePreset(current => ({
                  ...current,
                  aggregator_temperature: optionalFiniteInput(event.target.value)
                }))
              }
              step="any"
              type="number"
              value={numberInputValue(preset.aggregator_temperature)}
            />
          }
          description={copy.aggregatorTemperatureDescription}
          title={copy.aggregatorTemperature}
        />

        <div className="mt-4 flex items-center justify-between gap-3 border-t border-border/60 pt-4">
          <div aria-live="polite" className="min-h-4 text-xs text-(--ui-text-tertiary)" role="status">
            {saveStatus === 'saving'
              ? copy.saving
              : saveStatus === 'saved'
                ? copy.saved
                : saveStatus === 'incomplete'
                  ? copy.incomplete
                  : saveStatus === 'dirty' || dirty
                    ? copy.unsaved
                    : null}
          </div>
          <Button disabled={saveDisabled} onClick={() => void saveDraft()} size="sm">
            {copy.saveChanges}
          </Button>
        </div>
        {saveStatus === 'error' ? (
          <div className="mt-1 grid gap-0.5 text-xs text-destructive" role="alert">
            <span>{copy.saveFailedRetained}</span>
            <span>{saveError}</span>
          </div>
        ) : null}
      </fieldset>
      <ConfirmDialog
        busyLabel={copy.saving}
        confirmLabel={copy.deletePreset}
        description={copy.deleteConfirm(deleteTarget ?? selectedPreset)}
        destructive
        doneLabel={copy.unsaved}
        onClose={() => setDeleteTarget(null)}
        onConfirm={() => {
          if (!deleteTarget || Object.keys(moaRef.current.presets).length <= 1) {
            return
          }

          const next = deleteMoaPreset(moaRef.current, deleteTarget)
          applyDraft(next, firstPresetName(next))

          if (deleteTarget === blankDraftPreset) {
            setBlankDraftPreset(null)
          }
        }}
        open={deleteTarget !== null}
        title={copy.deletePreset}
      />
    </section>
  )
}
