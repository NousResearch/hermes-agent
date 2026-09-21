import { type ComponentProps, type KeyboardEvent, type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent } from '@/components/ui/dialog'
import {
  getGlobalModelOptions,
  getHermesConfigRecord,
  type ProfileScope,
  saveHermesConfigRecord,
  setModelAssignment
} from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import { Check, ChevronLeft, ChevronRight, KeyRound, Loader2, RefreshCw, ShieldLock, Volume2 } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import { startManualOnboarding } from '@/store/onboarding'
import type { HermesConfigRecord, ModelAssignmentResponse, ModelOptionsResponse } from '@/types/hermes'

import { getNested, setNested } from '../settings/helpers'

import {
  approvalConfigMode,
  initialJarvisOnboardingState,
  JARVIS_ONBOARDING_STEPS,
  JARVIS_ONBOARDING_VERSION,
  type JarvisApprovalProductMode,
  jarvisOnboardingComplete,
  type JarvisOnboardingScope,
  jarvisOnboardingScopeKey,
  type JarvisOnboardingSelections,
  type JarvisOnboardingState,
  type JarvisOnboardingStep,
  type JarvisVoiceMode,
  normalizeJarvisOnboardingScope,
  readJarvisOnboardingState,
  writeJarvisOnboardingState
} from './onboarding-state'

type ConfigurationCheckResult = { ok: true; message?: string } | { ok: false; message: string }
type JarvisOnboardingCopy = Translations['jarvisOnboarding']

class StaleOnboardingTransactionError extends Error {
  constructor() {
    super('Jarvis onboarding transaction is no longer current')
    this.name = 'StaleOnboardingTransactionError'
  }
}

interface ProviderOption {
  authenticated?: boolean
  models: string[]
  name: string
  slug: string
}

function stepIndex(step: JarvisOnboardingStep): number {
  return Math.max(0, JARVIS_ONBOARDING_STEPS.indexOf(step))
}

function normalizeProviders(options: ModelOptionsResponse | null): ProviderOption[] {
  return (options?.providers ?? [])
    .map(provider => ({
      authenticated: provider.authenticated,
      models: provider.models ?? [],
      name: provider.name,
      slug: provider.slug
    }))
    .filter(provider => provider.slug && provider.name)
}

function firstModel(provider: ProviderOption | undefined, fallback?: string): string {
  if (!provider) {
    return fallback ?? ''
  }

  if (fallback && provider.models.includes(fallback)) {
    return fallback
  }

  return provider.models[0] ?? fallback ?? ''
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error ? error.message : String(error || fallback)
}

function updatedState(
  state: JarvisOnboardingState,
  patch: Partial<JarvisOnboardingState> & { selections?: JarvisOnboardingSelections }
): JarvisOnboardingState {
  return {
    ...state,
    ...patch,
    version: JARVIS_ONBOARDING_VERSION,
    selections: {
      ...(state.selections ?? {}),
      ...(patch.selections ?? {})
    }
  }
}

export interface JarvisOnboardingProps {
  initialStep?: JarvisOnboardingStep
  loadConfig?: (scope: JarvisOnboardingScope) => Promise<HermesConfigRecord>
  loadModelOptions?: (scope: JarvisOnboardingScope) => Promise<ModelOptionsResponse>
  providerConfigurationCheck?: (input: {
    model: string
    provider: string
    scope: JarvisOnboardingScope
  }) => Promise<ConfigurationCheckResult>
  requestGateway?: (method: string, params?: Record<string, unknown>) => Promise<unknown>
  saveConfig?: (config: HermesConfigRecord, scope: JarvisOnboardingScope) => Promise<{ ok: boolean }>
  saveModel?: (
    body: { model: string; provider: string },
    scope: JarvisOnboardingScope
  ) => Promise<ModelAssignmentResponse | { ok: boolean }>
  isScopeCurrent?: (scope: JarvisOnboardingScope) => boolean
  onComplete?: () => void
  scope?: JarvisOnboardingScope
}

const defaultLoadConfig = (scope: JarvisOnboardingScope) => getHermesConfigRecord(scope as ProfileScope)

const defaultLoadModelOptions = (scope: JarvisOnboardingScope) =>
  getGlobalModelOptions({ refresh: true, includeUnconfigured: true, explicitOnly: false }, scope as ProfileScope)

const defaultSaveConfig = (config: HermesConfigRecord, scope: JarvisOnboardingScope) =>
  saveHermesConfigRecord(config, scope as ProfileScope)

const defaultSaveModel = (body: { model: string; provider: string }, scope: JarvisOnboardingScope) =>
  setModelAssignment({ scope: 'main', ...body }, scope as ProfileScope)

export function JarvisOnboarding({
  initialStep,
  loadConfig = defaultLoadConfig,
  loadModelOptions = defaultLoadModelOptions,
  providerConfigurationCheck,
  requestGateway,
  saveConfig = defaultSaveConfig,
  saveModel = defaultSaveModel,
  isScopeCurrent,
  onComplete,
  scope: rawScope
}: JarvisOnboardingProps) {
  const { t } = useI18n()
  const copy = t.jarvisOnboarding
  const scope = useMemo(() => normalizeJarvisOnboardingScope(rawScope), [rawScope])
  const scopeKey = useMemo(() => jarvisOnboardingScopeKey(scope), [scope])
  const loadedState = useMemo(() => readJarvisOnboardingState(undefined, scope), [scope])
  const [state, setState] = useState<JarvisOnboardingState>(() => loadedState ?? initialJarvisOnboardingState(initialStep))
  const [config, setConfig] = useState<HermesConfigRecord | null>(null)
  const [providers, setProviders] = useState<ProviderOption[]>([])
  const [loading, setLoading] = useState(true)
  const [completed, setCompleted] = useState(() => jarvisOnboardingComplete(loadedState))
  const [configurationStatus, setConfigurationStatus] = useState<'idle' | 'checking' | 'failed' | 'passed'>('idle')
  const [configurationMessage, setConfigurationMessage] = useState('')
  const [accessStatus, setAccessStatus] = useState<'idle' | 'checking' | 'failed' | 'passed'>('idle')
  const [accessMessage, setAccessMessage] = useState('')
  const [saveError, setSaveError] = useState('')
  const mounted = useRef(true)
  const loadToken = useRef(0)
  const requestToken = useRef(0)
  const stateRef = useRef(state)
  const selectedModelRef = useRef({ model: '', provider: '' })
  const scopeKeyRef = useRef(scopeKey)

  const currentIndex = stepIndex(state.currentStep)
  const currentStep = state.currentStep
  const selectedProvider = String(state.selections?.engine ?? providers[0]?.slug ?? '')
  const provider = providers.find(p => p.slug === selectedProvider) ?? providers[0]
  const selectedModel = String(state.selections?.model ?? firstModel(provider))
  const voiceMode = String(state.selections?.voiceMode ?? 'quiet') as JarvisVoiceMode
  const approvalsMode = String(state.selections?.approvalsMode ?? 'balanced') as JarvisApprovalProductMode

  stateRef.current = state
  selectedModelRef.current = { provider: selectedProvider, model: selectedModel }

  if (scopeKeyRef.current !== scopeKey) {
    scopeKeyRef.current = scopeKey
    loadToken.current += 1
    requestToken.current += 1
  }

  const requestStillCurrent = (token: number, expectedScopeKey: string, expectedScope: JarvisOnboardingScope = scope) =>
    mounted.current &&
    requestToken.current === token &&
    scopeKeyRef.current === expectedScopeKey &&
    (isScopeCurrent?.(expectedScope) ?? true)

  const selectionStillCurrent = (token: number, expectedScopeKey: string, providerName: string, modelName: string) => {
    const selections = selectedModelRef.current

    return (
      requestStillCurrent(token, expectedScopeKey) &&
      selections.provider === providerName &&
      selections.model === modelName
    )
  }

  useEffect(
    () => () => {
      mounted.current = false
      loadToken.current += 1
      requestToken.current += 1
    },
    []
  )

  const loadInitialState = useCallback(
    (token: number, loadScope: JarvisOnboardingScope, loadScopeKey: string) => {
      const loadStillCurrent = () => mounted.current && loadToken.current === token && scopeKeyRef.current === loadScopeKey

      setLoading(true)
      void Promise.all([
        Promise.resolve().then(() => loadConfig(loadScope)),
        Promise.resolve().then(() => loadModelOptions(loadScope))
      ])
        .then(([cfg, options]) => {
          if (!loadStillCurrent()) {
            return
          }

          const nextProviders = normalizeProviders(options)
          const activeProvider = String(options.provider ?? nextProviders[0]?.slug ?? '')
          const active = nextProviders.find(p => p.slug === activeProvider) ?? nextProviders[0]
          const activeModel = String(options.model ?? firstModel(active))
          const autoTts = Boolean(getNested(cfg, 'voice.auto_tts'))

          const approvalMode: JarvisApprovalProductMode =
            getNested(cfg, 'approvals.mode') === 'manual' ? 'strict' : 'balanced'

          setConfig(cfg)
          setProviders(nextProviders)
          setState(prev => {
            const next = updatedState(prev, {
              selections: {
                profile: 'active',
                engine: String(prev.selections?.engine ?? activeProvider),
                model: String(prev.selections?.model ?? activeModel),
                voiceMode: prev.selections?.voiceMode ?? (autoTts ? 'spoken' : 'quiet'),
                approvalsMode: prev.selections?.approvalsMode ?? approvalMode
              }
            })

            stateRef.current = next

            return next
          })
        })
        .catch(error => {
          if (loadStillCurrent()) {
            setSaveError(errorMessage(error, copy.errors.config))
          }
        })
        .finally(() => {
          if (loadStillCurrent()) {
            setLoading(false)
          }
        })
    },
    [copy.errors.config, loadConfig, loadModelOptions]
  )

  useEffect(() => {
    if (completed) {
      return
    }

    loadInitialState(++loadToken.current, scope, scopeKey)
  }, [completed, loadInitialState, scope, scopeKey])

  useEffect(() => {
    if (!completed) {
      const ok = writeJarvisOnboardingState(state, globalThis.localStorage, scope)

      if (!ok && mounted.current) {
        setSaveError(copy.errors.save)
      }
    }
  }, [completed, copy.errors.save, scope, state])

  if (completed) {
    return null
  }

  const persistState = (next: JarvisOnboardingState) => {
    if (!writeJarvisOnboardingState(next, globalThis.localStorage, scope)) {
      setSaveError(copy.errors.save)

      return false
    }

    setSaveError('')
    stateRef.current = next
    setState(next)

    return true
  }

  const selectStep = (step: JarvisOnboardingStep) => {
    persistState(updatedState(state, { currentStep: step }))
  }

  const markCurrentDone = (step: JarvisOnboardingStep = currentStep) => {
    const completedSteps = Array.from(new Set([...state.completedSteps, step]))

    return updatedState(state, { completedSteps })
  }

  const invalidateModelDependentState = (next: JarvisOnboardingState): JarvisOnboardingState =>
    updatedState(
      {
        ...next,
        completedSteps: next.completedSteps.filter(step => !['model', 'access', 'approvals'].includes(step))
      },
      {
        selections: {
          validatedAccessModel: undefined,
          validatedAccessProvider: undefined,
          validatedModel: undefined,
          validatedProvider: undefined
        }
      }
    )

  const goNext = () => {
    const nextStep = JARVIS_ONBOARDING_STEPS[Math.min(JARVIS_ONBOARDING_STEPS.length - 1, currentIndex + 1)]
    persistState({ ...markCurrentDone(), currentStep: nextStep })
  }

  const goBack = () => {
    const previous = JARVIS_ONBOARDING_STEPS[Math.max(0, currentIndex - 1)]
    selectStep(previous)
  }

  const chooseProvider = (slug: string) => {
    const nextProvider = providers.find(p => p.slug === slug)
    const nextModel = firstModel(nextProvider)

    requestToken.current += 1
    selectedModelRef.current = { provider: slug, model: nextModel }
    setConfigurationStatus('idle')
    setConfigurationMessage('')
    setAccessStatus('idle')
    setAccessMessage('')
    persistState(
      invalidateModelDependentState(
        updatedState(state, {
          selections: {
            engine: slug,
            model: nextModel
          }
        })
      )
    )
  }

  const chooseModel = (model: string) => {
    requestToken.current += 1
    selectedModelRef.current = { provider: selectedProvider, model }
    setConfigurationStatus('idle')
    setConfigurationMessage('')
    setAccessStatus('idle')
    setAccessMessage('')
    persistState(
      invalidateModelDependentState(
        updatedState(state, {
          selections: { model }
        })
      )
    )
  }

  const runConfigurationCheck = async () => {
    const providerAtRequest = selectedProvider
    const modelAtRequest = selectedModel
    const token = ++requestToken.current
    const requestScope = scope
    const requestScopeKey = scopeKey

    setConfigurationStatus('checking')
    setConfigurationMessage('')

    try {
      const result = providerConfigurationCheck
        ? await providerConfigurationCheck({ provider: providerAtRequest, model: modelAtRequest, scope: requestScope })
        : await defaultProviderConfigurationCheck({
            copy,
            loadModelOptions,
            model: modelAtRequest,
            provider: providerAtRequest,
            scope: requestScope
          })

      if (!selectionStillCurrent(token, requestScopeKey, providerAtRequest, modelAtRequest)) {
        return
      }

      if (!result.ok) {
        setConfigurationStatus('failed')
        setConfigurationMessage(result.message)

        return
      }

      const base = stateRef.current
      const completedSteps = Array.from(new Set([...base.completedSteps, 'model' as JarvisOnboardingStep]))

      if (
        persistState(
          updatedState(base, {
            completedSteps,
            selections: {
              validatedModel: modelAtRequest,
              validatedProvider: providerAtRequest
            }
          })
        )
      ) {
        setConfigurationStatus('passed')
      } else {
        setConfigurationStatus('failed')
      }
    } catch (error) {
      if (selectionStillCurrent(token, requestScopeKey, providerAtRequest, modelAtRequest)) {
        setConfigurationStatus('failed')
        setConfigurationMessage(errorMessage(error, copy.errors.model))
      }
    }
  }

  const openSecureProviderSetup = () => {
    startManualOnboarding(null, scope)
    persistState(updatedState(state, { selections: { accessOpened: true } }))
  }

  const validateAccess = async () => {
    const providerAtRequest = selectedProvider
    const modelAtRequest = selectedModel
    const token = ++requestToken.current
    const requestScope = scope
    const requestScopeKey = scopeKey

    setAccessStatus('checking')
    setAccessMessage('')

    try {
      const options = await loadModelOptions(requestScope)
      const nextProviders = normalizeProviders(options)
      const match = nextProviders.find(item => item.slug === providerAtRequest)

      if (!selectionStillCurrent(token, requestScopeKey, providerAtRequest, modelAtRequest)) {
        return
      }

      setProviders(nextProviders)

      if (!match || match.authenticated === false || (match.models.length > 0 && !match.models.includes(modelAtRequest))) {
        setAccessStatus('failed')
        setAccessMessage(match?.name ? copy.errors.providerUnavailable : copy.errors.model)

        return
      }

      const base = stateRef.current
      const completedSteps = Array.from(new Set([...base.completedSteps, 'access' as JarvisOnboardingStep]))

      if (
        persistState(
          updatedState(base, {
            completedSteps,
            selections: {
              validatedAccessModel: modelAtRequest,
              validatedAccessProvider: providerAtRequest
            }
          })
        )
      ) {
        setAccessStatus('passed')
      } else {
        setAccessStatus('failed')
      }
    } catch (error) {
      if (selectionStillCurrent(token, requestScopeKey, providerAtRequest, modelAtRequest)) {
        setAccessStatus('failed')
        setAccessMessage(errorMessage(error, copy.errors.providerUnavailable))
      }
    }
  }

  const ensureFinalFingerprint = (providerName: string, modelName: string): boolean => {
    const selections = stateRef.current.selections ?? {}

    return (
      selections.validatedProvider === providerName &&
      selections.validatedModel === modelName &&
      selections.validatedAccessProvider === providerName &&
      selections.validatedAccessModel === modelName
    )
  }

  const validateRefreshedModel = (options: ModelOptionsResponse, providerName: string, modelName: string) => {
    const nextProviders = normalizeProviders(options)
    const match = nextProviders.find(item => item.slug === providerName)

    setProviders(nextProviders)

    return Boolean(match && match.authenticated !== false && (match.models.length === 0 || match.models.includes(modelName)))
  }

  const assertModelAssignmentResult = (result: ModelAssignmentResponse | { ok: boolean }) => {
    if ('confirm_required' in result && result.confirm_required) {
      throw new Error(result.confirm_message || copy.errors.providerUnavailable)
    }

    if ('ok' in result && result.ok === false) {
      throw new Error(copy.errors.save)
    }
  }

  const rollbackModelAssignment = async (
    snapshot: { model?: null | string; provider?: null | string },
    rollbackScope: JarvisOnboardingScope
  ) => {
    const previousProvider = String(snapshot.provider ?? '').trim()
    const previousModel = String(snapshot.model ?? '').trim()

    if (!previousProvider || !previousModel) {
      return
    }

    assertModelAssignmentResult(await saveModel({ provider: previousProvider, model: previousModel }, rollbackScope))
  }

  const hasRollbackModelSnapshot = (snapshot: { model?: null | string; provider?: null | string }) =>
    Boolean(String(snapshot.provider ?? '').trim() && String(snapshot.model ?? '').trim())

  const rollbackConfig = async (snapshot: HermesConfigRecord, rollbackScope: JarvisOnboardingScope) => {
    const result = await saveConfig(snapshot, rollbackScope)

    if (!result.ok) {
      throw new Error(copy.errors.save)
    }
  }

  const setRecoveryError = (operationError: unknown, recoveryError: unknown) => {
    if (!mounted.current) {
      notifyError(recoveryError, copy.errors.recovery)

      return
    }

    setSaveError(
      `${errorMessage(operationError, copy.errors.save)} ${copy.errors.recovery}: ${errorMessage(
        recoveryError,
        copy.errors.save
      )}`
    )
  }

  const reportStaleRecoveryFailure = (error: unknown) => {
    notifyError(error, copy.errors.recovery)
  }

  const finish = async () => {
    const providerAtRequest = selectedProvider
    const modelAtRequest = selectedModel
    const token = ++requestToken.current
    const requestScope = scope
    const requestScopeKey = scopeKey
    let modelWritten = false
    let configWritten = false
    let reloadAttempted = false

    const stillCurrent = () => selectionStillCurrent(token, requestScopeKey, providerAtRequest, modelAtRequest)

    const assertStillCurrent = () => {
      if (!stillCurrent()) {
        throw new StaleOnboardingTransactionError()
      }
    }

    const reloadRuntime = async () => {
      reloadAttempted = true
      await requestGateway?.('reload.env')
    }

    const compensateWrittenState = async ({ resyncRuntime = false }: { resyncRuntime?: boolean } = {}) => {
      if (!snapshotConfig || !snapshotOptions) {
        return stillCurrent()
      }

      let currentAfterRollback = stillCurrent()

      if (configWritten) {
        await rollbackConfig(snapshotConfig, requestScope)
        currentAfterRollback = stillCurrent()
      }

      if (modelWritten) {
        await rollbackModelAssignment({ provider: snapshotOptions.provider, model: snapshotOptions.model }, requestScope)
        currentAfterRollback = currentAfterRollback && stillCurrent()
      }

      if (resyncRuntime && reloadAttempted) {
        await requestGateway?.('reload.env')
      }

      return currentAfterRollback && stillCurrent()
    }

    let snapshotConfig: HermesConfigRecord | null = null
    let snapshotOptions: ModelOptionsResponse | null = null

    setSaveError('')

    try {
      if (!ensureFinalFingerprint(providerAtRequest, modelAtRequest)) {
        throw new Error(copy.errors.model)
      }

      const [loadedConfig, loadedOptions] = await Promise.all([
        loadConfig(requestScope),
        loadModelOptions(requestScope)
      ])

      if (!stillCurrent()) {
        return
      }

      snapshotConfig = loadedConfig
      snapshotOptions = loadedOptions

      if (!hasRollbackModelSnapshot(snapshotOptions)) {
        throw new Error(copy.errors.rollbackSnapshot)
      }

      if (!validateRefreshedModel(snapshotOptions, providerAtRequest, modelAtRequest)) {
        throw new Error(copy.errors.providerUnavailable)
      }

      let nextConfig = setNested(snapshotConfig, 'voice.auto_tts', voiceMode === 'spoken')
      nextConfig = setNested(nextConfig, 'approvals.mode', approvalConfigMode(approvalsMode))

      assertModelAssignmentResult(await saveModel({ provider: providerAtRequest, model: modelAtRequest }, requestScope))
      modelWritten = true
      assertStillCurrent()

      try {
        const result = await saveConfig(nextConfig, requestScope)

        if (!result.ok) {
          throw new Error(copy.errors.save)
        }

        configWritten = true
        assertStillCurrent()
      } catch (error) {
        try {
          if (!(await compensateWrittenState())) {
            return
          }
        } catch (rollbackError) {
          if (error instanceof StaleOnboardingTransactionError || !mounted.current) {
            reportStaleRecoveryFailure(rollbackError)

            return
          }

          setRecoveryError(error, rollbackError)

          return
        }

        if (error instanceof StaleOnboardingTransactionError) {
          return
        }

        throw error
      }

      assertStillCurrent()

      try {
        await reloadRuntime()
        assertStillCurrent()
      } catch (error) {
        try {
          if (!(await compensateWrittenState({ resyncRuntime: true }))) {
            return
          }
        } catch (rollbackError) {
          if (error instanceof StaleOnboardingTransactionError || !mounted.current) {
            reportStaleRecoveryFailure(rollbackError)

            return
          }

          setRecoveryError(error, rollbackError)

          return
        }

        if (error instanceof StaleOnboardingTransactionError) {
          return
        }

        throw error
      }

      const finalState = {
        ...updatedState(stateRef.current, { completedSteps: [...JARVIS_ONBOARDING_STEPS] }),
        completedSteps: [...JARVIS_ONBOARDING_STEPS]
      }

      if (!writeJarvisOnboardingState(finalState, globalThis.localStorage, requestScope)) {
        const error = new Error(copy.errors.save)

        try {
          if (!(await compensateWrittenState({ resyncRuntime: true }))) {
            return
          }
        } catch (rollbackError) {
          setRecoveryError(error, rollbackError)

          return
        }

        throw error
      }

      if (!requestStillCurrent(token, requestScopeKey)) {
        return
      }

      setConfig(nextConfig)
      setState(finalState)
      onComplete?.()
      setCompleted(true)
    } catch (error) {
      if (error instanceof StaleOnboardingTransactionError) {
        if (modelWritten || configWritten) {
          try {
            await compensateWrittenState({ resyncRuntime: reloadAttempted })
          } catch (rollbackError) {
            reportStaleRecoveryFailure(rollbackError)
          }
        }

        return
      }

      if (requestStillCurrent(token, requestScopeKey)) {
        setSaveError(errorMessage(error, copy.errors.save))
      }
    }
  }

  const nextDisabled =
    loading ||
    (currentStep === 'engine' && !selectedProvider) ||
    (currentStep === 'model' && configurationStatus !== 'passed') ||
    (currentStep === 'access' && !state.completedSteps.includes('access')) ||
    (currentStep === 'approvals' && !approvalsMode)

  return (
    <Dialog modal onOpenChange={() => undefined} open>
      <DialogContent
        aria-labelledby="jarvis-onboarding-title"
        bodyClassName="grid max-h-[calc(100vh-2rem)] gap-4 overflow-y-auto p-4 sm:max-h-[calc(100vh-3rem)] sm:p-6 lg:grid-cols-[17rem_minmax(0,1fr)]"
        className="z-(--z-onboarding) w-[calc(100vw-2rem)] max-w-5xl overflow-hidden border-white/12 bg-[#0B0D10] text-[#F5F7FA] sm:w-[calc(100vw-3rem)]"
        data-testid="jarvis-onboarding"
        showCloseButton={false}
      >
        <aside className="grid content-start gap-4">
          <div>
            <p className="text-xs font-semibold uppercase text-[#00B7FF]">{copy.productName}</p>
            <h1 className="mt-2 text-2xl font-semibold tracking-normal" id="jarvis-onboarding-title">
              {copy.intro.title}
            </h1>
            <p className="mt-2 text-sm leading-6 text-[#9299A5]">{copy.intro.subtitle}</p>
          </div>
          <ol aria-label={copy.stepsLabel} className="grid gap-2">
            {JARVIS_ONBOARDING_STEPS.map((step, index) => {
              const active = step === currentStep
              const done = state.completedSteps.includes(step)

              return (
                <li key={step}>
                  <button
                    aria-current={active ? 'step' : undefined}
                    className={cn(
                      'flex min-h-11 w-full items-center gap-3 rounded-md border px-3 text-left text-sm transition focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[#00B7FF]/50',
                      active
                        ? 'border-[#00B7FF]/70 bg-[#00B7FF]/12 text-white'
                        : 'border-white/10 bg-[#101318] text-[#9299A5] hover:border-white/20 hover:text-white'
                    )}
                    onClick={() => selectStep(step)}
                    type="button"
                  >
                    <span className="flex size-6 shrink-0 items-center justify-center rounded-full bg-white/10 text-xs">
                      {done ? <Check className="size-3.5" /> : index + 1}
                    </span>
                    <span>{copy.steps[step]}</span>
                  </button>
                </li>
              )
            })}
          </ol>
        </aside>

        <section className="grid min-h-[31rem] min-w-0 grid-rows-[auto_minmax(0,1fr)_auto] gap-4 rounded-md border border-white/10 bg-[#101318] p-4 sm:p-5">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-4">
            <div>
              <p className="text-xs text-[#9299A5]">
                {copy.progress(currentIndex + 1, JARVIS_ONBOARDING_STEPS.length)}
              </p>
              <h2 className="mt-1 text-xl font-semibold tracking-normal">{copy.steps[currentStep]}</h2>
            </div>
            {loading ? <Loader2 className="size-5 animate-spin text-[#00B7FF]" /> : null}
          </div>

          <div className="min-h-0 overflow-y-auto pr-1">
            {currentStep === 'profile' ? (
              <ProfileStep activeLabel={copy.profile.active} body={copy.profile.body} title={copy.profile.title} />
            ) : null}
            {currentStep === 'engine' ? (
              <EngineStep
                body={copy.engine.body}
                copy={copy.engine}
                onSelect={chooseProvider}
                providers={providers}
                selected={selectedProvider}
                title={copy.engine.title}
              />
            ) : null}
            {currentStep === 'model' ? (
              <ModelStep
                body={copy.model.body}
                checking={configurationStatus === 'checking'}
                checkLabel={copy.actions.checkConfiguration}
                message={configurationMessage}
                models={provider?.models ?? []}
                onCheck={() => void runConfigurationCheck()}
                onSelect={chooseModel}
                selected={selectedModel}
                status={configurationStatus}
                successLabel={copy.model.success}
                title={copy.model.title}
              />
            ) : null}
            {currentStep === 'voice' ? (
              <VoiceStep
                copy={copy.voice}
                mode={voiceMode}
                onSelect={mode => persistState(updatedState(state, { selections: { voiceMode: mode } }))}
              />
            ) : null}
            {currentStep === 'access' ? (
              <AccessStep
                checking={accessStatus === 'checking'}
                copy={copy.access}
                message={accessMessage}
                onOpen={openSecureProviderSetup}
                onValidate={() => void validateAccess()}
                opened={state.selections?.accessOpened === true}
                status={accessStatus}
              />
            ) : null}
            {currentStep === 'approvals' ? (
              <ApprovalsStep
                copy={copy.approvals}
                mode={approvalsMode}
                onSelect={mode => persistState(updatedState(state, { selections: { approvalsMode: mode } }))}
              />
            ) : null}
          </div>

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-white/10 pt-4">
            <div className="min-h-5 text-sm text-red-300" role="alert">
              {saveError}
            </div>
            <div className="flex gap-2">
              <Button className="min-h-11" disabled={currentIndex === 0} onClick={goBack} type="button" variant="outline">
                <ChevronLeft className="size-4" />
                {copy.actions.back}
              </Button>
              {currentStep === 'approvals' ? (
                <Button className="min-h-11" disabled={loading} onClick={() => void finish()} type="button">
                  {copy.actions.finish}
                </Button>
              ) : (
                <Button className="min-h-11" disabled={nextDisabled} onClick={goNext} type="button">
                  {copy.actions.next}
                  <ChevronRight className="size-4" />
                </Button>
              )}
            </div>
          </div>
        </section>
      </DialogContent>
    </Dialog>
  )
}

async function defaultProviderConfigurationCheck({
  copy,
  loadModelOptions,
  model,
  provider,
  scope
}: {
  copy: JarvisOnboardingCopy
  loadModelOptions: (scope: JarvisOnboardingScope) => Promise<ModelOptionsResponse>
  model: string
  provider: string
  scope: JarvisOnboardingScope
}): Promise<ConfigurationCheckResult> {
  if (!provider || !model) {
    return { ok: false, message: copy.errors.model }
  }

  const options = await loadModelOptions(scope)
  const match = normalizeProviders(options).find(item => item.slug === provider)

  if (!match || match.authenticated === false || (match.models.length > 0 && !match.models.includes(model))) {
    return { ok: false, message: match?.name ? copy.errors.providerUnavailable : copy.errors.model }
  }

  return { ok: true }
}

function ProfileStep({ activeLabel, body, title }: { activeLabel: string; body: string; title: string }) {
  return (
    <div className="grid gap-4">
      <p className="text-lg font-semibold">{title}</p>
      <p className="max-w-2xl text-sm leading-6 text-[#C7CBD1]">{body}</p>
      <div className="rounded-md border border-white/10 bg-black/20 p-4 text-sm text-[#9299A5]">{activeLabel}</div>
    </div>
  )
}

function EngineStep({
  body,
  copy,
  onSelect,
  providers,
  selected,
  title
}: {
  body: string
  copy: JarvisOnboardingCopy['engine']
  onSelect: (slug: string) => void
  providers: ProviderOption[]
  selected: string
  title: string
}) {
  return (
    <div className="grid gap-4">
      <p className="text-lg font-semibold">{title}</p>
      <p className="max-w-2xl text-sm leading-6 text-[#C7CBD1]">{body}</p>
      <div className="grid gap-2 sm:grid-cols-2">
        {providers.length === 0 ? <p className="text-sm text-[#C7CBD1]">{copy.noProviders}</p> : null}
        {providers.map(provider => (
          <button
            className={cn(
              'min-h-16 rounded-md border p-3 text-left focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[#00B7FF]/50',
              selected === provider.slug ? 'border-[#00B7FF] bg-[#00B7FF]/12' : 'border-white/10 bg-black/20'
            )}
            key={provider.slug}
            onClick={() => onSelect(provider.slug)}
            type="button"
          >
            <span className="block text-sm font-medium">{provider.name}</span>
            <span className="mt-1 block text-xs text-[#9299A5]">{copy.modelCount(provider.models.length)}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

function ModelStep({
  body,
  checking,
  checkLabel,
  message,
  models,
  onCheck,
  onSelect,
  selected,
  status,
  successLabel,
  title
}: {
  body: string
  checking: boolean
  checkLabel: string
  message: string
  models: string[]
  onCheck: () => void
  onSelect: (model: string) => void
  selected: string
  status: 'idle' | 'checking' | 'failed' | 'passed'
  successLabel: string
  title: string
}) {
  return (
    <div className="grid gap-4">
      <p className="text-lg font-semibold">{title}</p>
      <p className="max-w-2xl text-sm leading-6 text-[#C7CBD1]">{body}</p>
      <select
        className="min-h-11 rounded-md border border-white/10 bg-black/30 px-3 text-sm text-white focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[#00B7FF]/50"
        onChange={event => onSelect(event.target.value)}
        value={selected}
      >
        {models.map(model => (
          <option key={model} value={model}>
            {model}
          </option>
        ))}
      </select>
      <div className="flex flex-wrap items-center gap-3">
        <Button className="min-h-11" disabled={checking || !selected} onClick={onCheck} type="button" variant="secondary">
          {checking ? <Loader2 className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
          {checkLabel}
        </Button>
        {status === 'passed' ? <span className="text-sm text-[#29E68C]">{successLabel}</span> : null}
      </div>
      {message ? <p className="text-sm text-red-300">{message}</p> : null}
    </div>
  )
}

function VoiceStep({
  copy,
  mode,
  onSelect
}: {
  copy: JarvisOnboardingCopy['voice']
  mode: 'quiet' | 'spoken'
  onSelect: (mode: 'quiet' | 'spoken') => void
}) {
  return (
    <div className="grid gap-4">
      <p className="text-lg font-semibold">{copy.title}</p>
      <div className="grid gap-2 sm:grid-cols-2">
        <ChoiceCard
          active={mode === 'quiet'}
          description={copy.quietHint}
          icon={<Volume2 className="size-4" />}
          label={copy.quiet}
          onClick={() => onSelect('quiet')}
        />
        <ChoiceCard
          active={mode === 'spoken'}
          description={copy.spokenHint}
          icon={<Volume2 className="size-4" />}
          label={copy.spoken}
          onClick={() => onSelect('spoken')}
        />
      </div>
    </div>
  )
}

function AccessStep({
  checking,
  copy,
  message,
  onOpen,
  onValidate,
  opened,
  status
}: {
  checking: boolean
  copy: JarvisOnboardingCopy['access']
  message: string
  onOpen: () => void
  onValidate: () => void
  opened: boolean
  status: 'idle' | 'checking' | 'failed' | 'passed'
}) {
  return (
    <div className="grid gap-4">
      <p className="text-lg font-semibold">{copy.title}</p>
      <p className="max-w-2xl text-sm leading-6 text-[#C7CBD1]">{copy.body}</p>
      <div className="flex flex-wrap items-center gap-3">
        <Button className="min-h-11 w-fit" onClick={onOpen} type="button" variant="secondary">
          <KeyRound className="size-4" />
          {copy.secureAction}
        </Button>
        <Button className="min-h-11 w-fit" disabled={checking} onClick={onValidate} type="button" variant="outline">
          {checking ? <Loader2 className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
          {copy.validateAction}
        </Button>
      </div>
      {opened ? <p className="text-sm text-[#C7CBD1]">{copy.opened}</p> : null}
      {status === 'passed' ? <p className="text-sm text-[#29E68C]">{copy.validated}</p> : null}
      {message ? <p className="text-sm text-red-300">{message}</p> : null}
    </div>
  )
}

function ApprovalsStep({
  copy,
  mode,
  onSelect
}: {
  copy: JarvisOnboardingCopy['approvals']
  mode: JarvisApprovalProductMode
  onSelect: (mode: JarvisApprovalProductMode) => void
}) {
  const modes: JarvisApprovalProductMode[] = ['balanced', 'strict']

  const moveSelection = (nextMode: JarvisApprovalProductMode) => {
    onSelect(nextMode)
    requestAnimationFrame(() => {
      document.querySelector<HTMLElement>(`[data-approval-mode="${nextMode}"]`)?.focus()
    })
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const index = Math.max(0, modes.indexOf(mode))
    let nextIndex = index

    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      nextIndex = (index + 1) % modes.length
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      nextIndex = (index - 1 + modes.length) % modes.length
    } else if (event.key === 'Home') {
      nextIndex = 0
    } else if (event.key === 'End') {
      nextIndex = modes.length - 1
    } else {
      return
    }

    event.preventDefault()
    moveSelection(modes[nextIndex])
  }

  return (
    <fieldset className="grid gap-4">
      <legend className="text-lg font-semibold">{copy.title}</legend>
      <div aria-label={copy.title} className="grid gap-2 sm:grid-cols-2" onKeyDown={handleKeyDown} role="radiogroup">
        <ChoiceCard
          active={mode === 'balanced'}
          data-approval-mode="balanced"
          description={copy.balancedHint}
          icon={<ShieldLock className="size-4" />}
          label={copy.balanced}
          onClick={() => onSelect('balanced')}
          role="radio"
          tabIndex={mode === 'balanced' ? 0 : -1}
        />
        <ChoiceCard
          active={mode === 'strict'}
          data-approval-mode="strict"
          description={copy.strictHint}
          icon={<ShieldLock className="size-4" />}
          label={copy.strict}
          onClick={() => onSelect('strict')}
          role="radio"
          tabIndex={mode === 'strict' ? 0 : -1}
        />
      </div>
    </fieldset>
  )
}

function ChoiceCard({
  active,
  description,
  icon,
  label,
  onClick,
  role,
  tabIndex,
  ...props
}: {
  active: boolean
  description: string
  icon: ReactNode
  label: string
  onClick: () => void
  role?: 'radio'
  tabIndex?: number
} & Omit<ComponentProps<'button'>, 'aria-checked' | 'aria-label' | 'className' | 'onClick' | 'role' | 'type'>) {
  return (
    <button
      aria-checked={role === 'radio' ? active : undefined}
      aria-label={label}
      className={cn(
        'min-h-24 rounded-md border p-4 text-left focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[#00B7FF]/50',
        active ? 'border-[#00B7FF] bg-[#00B7FF]/12' : 'border-white/10 bg-black/20 hover:border-white/20'
      )}
      onClick={onClick}
      role={role}
      tabIndex={tabIndex}
      type="button"
      {...props}
    >
      <span className="flex items-center gap-2 text-sm font-semibold">
        {icon}
        {label}
      </span>
      <span className="mt-2 block text-sm leading-6 text-[#C7CBD1]">{description}</span>
    </button>
  )
}
