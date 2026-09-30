import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from 'react'

import type { LiveVoiceProviderId } from '@/api/voice-realtime'
import logoUrl from '@/assets/czesiek-logo.png'
import { ModelBrandIcon } from '@/components/model-brand-icon'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent } from '@/components/ui/dialog'
import {
  getGlobalModelOptions,
  getHermesConfigRecord,
  type ProfileScope,
  saveHermesConfigRecord,
  setEnvVar,
  setModelAssignment,
  setToolsetEnabled,
  validateProviderCredential
} from '@/hermes'
import { type Translations, useI18n } from '@/i18n'
import {
  Check,
  ChevronLeft,
  ChevronRight,
  KeyRound,
  Loader2,
  RefreshCw,
  ShieldLock,
  Sparkles,
  Volume2,
  X,
  Zap
} from '@/lib/icons'
import { cn } from '@/lib/utils'
import { notify, notifyError } from '@/store/notifications'
import { startManualOnboarding } from '@/store/onboarding'
import { applyVoiceEngineFromConfig } from '@/store/voice-prefs'
import type { HermesConfigRecord, ModelAssignmentResponse, ModelOptionsResponse } from '@/types/hermes'

import { getNested, setNested } from '../settings/helpers'

import { CHATGPT_PROVIDER_SLUG, type ChatGptConnectResult, chatGptWorkModel } from './chatgpt-connect'
import { ChatGptQuickConnect } from './chatgpt-quick-connect'
import {
  applyJarvisToolsetPlan,
  JARVIS_DEFAULT_COMPUTER_MODE,
  type JarvisComputerMode,
  jarvisToolsetPlan
} from './computer-capabilities'
import { type JarvisRoleId, selectionForRole } from './connections-catalog'
import { withCoordinatorPrompt } from './coordinator-prompt'
import { ChoiceCard, choiceRadioKeyHandler } from './onboarding-choice-card'
import { ComputerStep, type ComputerStepProps } from './onboarding-computer'
import { ConnectionsStep } from './onboarding-connections'
import { PersonalityStep } from './onboarding-personality'
import {
  approvalConfigMode,
  dismissJarvisOnboarding,
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
  markJarvisOnboardingCompleted,
  normalizeJarvisOnboardingScope,
  readJarvisOnboardingState,
  writeJarvisOnboardingState
} from './onboarding-state'
import {
  OnboardingTransactionBusyError,
  runOnboardingTransaction,
  StaleOnboardingTransactionError
} from './onboarding-transaction'
import { WelcomeStep } from './onboarding-welcome'
import { OPENROUTER_ENV_KEY, type OpenRouterConnectResult } from './openrouter-connect'
import { OPENROUTER_PROVIDER_SLUG, resolveOpenRouterPresets } from './openrouter-presets'
import { OpenRouterQuickConnect } from './openrouter-quick-connect'
import { withPersonality } from './personality'
import { setupCopy } from './setup-copy'

/**
 * Product default (docs/product/AI_EVOLUTION_JARVIS_DESIGN.md §2): a first
 * setup speaks Polish. The backend's own default is English and cannot be told
 * apart from a chosen one, so the switch happens only for a wizard that was
 * never started, and only after the saved config has actually loaded — the
 * write then lands on a live backend. The navigation's PL/EN toggle undoes it.
 */
function usePolishFirstRun(scope: JarvisOnboardingProps['scope']) {
  const { configLoadError, isLoadingConfig, locale, setLocale } = useI18n()

  const fresh = useMemo(
    () => readJarvisOnboardingState(undefined, normalizeJarvisOnboardingScope(scope)) === null,
    [scope]
  )

  const sawLoad = useRef(false)
  const applied = useRef(false)

  // eslint-disable-next-line no-restricted-syntax -- one-shot latches, not atom mirrors
  useEffect(() => {
    if (isLoadingConfig) {
      sawLoad.current = true

      return
    }

    if (!fresh || applied.current || !sawLoad.current || configLoadError || locale !== 'en') {
      return
    }

    applied.current = true
    setLocale('pl').catch(() => {
      // A failed save already rolled the UI back; the next load may try again.
      applied.current = false
    })
  }, [configLoadError, fresh, isLoadingConfig, locale, setLocale])
}

type ConfigurationCheckResult = { ok: true; message?: string } | { ok: false; message: string }
type JarvisOnboardingCopy = Translations['jarvisOnboarding']

/** Which Live provider a voice mode means (`voice.realtime.provider`); classic modes have none. */
const LIVE_VOICE_PROVIDERS: Record<JarvisVoiceMode, LiveVoiceProviderId | null> = {
  gemini: 'gemini',
  live: 'openai',
  quiet: null,
  spoken: null
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
  /** Test seam for the computer step's backend probes. */
  computerStatus?: Pick<ComputerStepProps, 'grantPermissions' | 'loadStatus'>
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
  /** Applies the computer step's toolsets once setup commits. */
  setToolsetEnabled?: (name: string, enabled: boolean, scope: JarvisOnboardingScope) => Promise<unknown>
  isScopeCurrent?: (scope: JarvisOnboardingScope) => boolean
  onComplete?: () => void
  /**
   * The person closed the wizard without finishing it. Setup stays incomplete
   * on purpose: Settings is the place to add a provider key or a model later.
   */
  onDismiss?: () => void
  scope?: JarvisOnboardingScope
}

const defaultLoadConfig = (scope: JarvisOnboardingScope) => getHermesConfigRecord(scope as ProfileScope)

const defaultLoadModelOptions = (scope: JarvisOnboardingScope) =>
  getGlobalModelOptions({ refresh: true, includeUnconfigured: true, explicitOnly: false }, scope as ProfileScope)

const defaultSaveConfig = (config: HermesConfigRecord, scope: JarvisOnboardingScope) =>
  saveHermesConfigRecord(config, scope as ProfileScope)

const defaultSetToolsetEnabled = (name: string, enabled: boolean, scope: JarvisOnboardingScope) =>
  setToolsetEnabled(name, enabled, scope as ProfileScope)

const defaultSaveModel = (body: { model: string; provider: string }, scope: JarvisOnboardingScope) =>
  setModelAssignment({ scope: 'main', ...body }, scope as ProfileScope)

export function JarvisOnboarding({
  computerStatus,
  initialStep,
  loadConfig = defaultLoadConfig,
  loadModelOptions = defaultLoadModelOptions,
  providerConfigurationCheck,
  requestGateway,
  saveConfig = defaultSaveConfig,
  saveModel = defaultSaveModel,
  setToolsetEnabled: applyToolset = defaultSetToolsetEnabled,
  isScopeCurrent,
  onComplete,
  onDismiss,
  scope: rawScope
}: JarvisOnboardingProps) {
  const { t, locale } = useI18n()
  const copy = t.jarvisOnboarding
  const guide = setupCopy[locale]
  usePolishFirstRun(rawScope)
  const scope = useMemo(() => normalizeJarvisOnboardingScope(rawScope), [rawScope])
  const scopeKey = useMemo(() => jarvisOnboardingScopeKey(scope), [scope])
  const loadedState = useMemo(() => readJarvisOnboardingState(undefined, scope), [scope])
  const [showSplash, setShowSplash] = useState(() => loadedState === null && initialStep === undefined)
  const [splashLeaving, setSplashLeaving] = useState(false)
  const splashTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => () => {
    if (splashTimer.current) {
      clearTimeout(splashTimer.current)
    }
  }, [])

  const openWizard = () => {
    if (splashLeaving) {
      return
    }

    setSplashLeaving(true)

    try {
      const AudioContextClass = window.AudioContext

      if (AudioContextClass) {
        const audio = new AudioContextClass()
        const now = audio.currentTime

        for (const [index, frequency] of [392, 523.25, 659.25].entries()) {
          const oscillator = audio.createOscillator()
          const gain = audio.createGain()
          const start = now + index * 0.095
          oscillator.type = 'sine'
          oscillator.frequency.value = frequency
          gain.gain.setValueAtTime(0, start)
          gain.gain.linearRampToValueAtTime(0.035, start + 0.025)
          gain.gain.exponentialRampToValueAtTime(0.001, start + 0.35)
          oscillator.connect(gain).connect(audio.destination)
          oscillator.start(start)
          oscillator.stop(start + 0.36)
        }

        window.setTimeout(() => void audio.close(), 800)
      }
    } catch {
      // The onboarding remains usable when audio is unavailable.
    }

    const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
    splashTimer.current = setTimeout(() => setShowSplash(false), reducedMotion ? 0 : 420)
  }

  const [state, setState] = useState<JarvisOnboardingState>(
    () => loadedState ?? initialJarvisOnboardingState(initialStep)
  )

  const [config, setConfig] = useState<HermesConfigRecord | null>(null)
  // The role picked on the connections step; only used to swap its suggested tools for another role's.
  const [connectionRole, setConnectionRole] = useState<JarvisRoleId | null>(null)
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
  const computerMode = (state.selections?.computerMode ?? JARVIS_DEFAULT_COMPUTER_MODE) as JarvisComputerMode

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

  // Lifecycle tracking is not reactive state mirroring; StrictMode reruns setup.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    mounted.current = true

    return () => {
      mounted.current = false
      loadToken.current += 1
      requestToken.current += 1
    }
  }, [])

  const loadInitialState = useCallback(
    (token: number, loadScope: JarvisOnboardingScope, loadScopeKey: string) => {
      const loadStillCurrent = () =>
        mounted.current && loadToken.current === token && scopeKeyRef.current === loadScopeKey

      setLoading(true)
      let timeout: ReturnType<typeof setTimeout>

      const deadline = new Promise<never>((_, reject) => {
        timeout = setTimeout(() => reject(new Error(copy.errors.config)), 15_000)
      })

      void Promise.race([
        Promise.all([
          Promise.resolve().then(() => loadConfig(loadScope)),
          Promise.resolve().then(() => loadModelOptions(loadScope))
        ]),
        deadline
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
          const liveVoice = getNested(cfg, 'voice.engine') === 'realtime'
          const liveMode: JarvisVoiceMode = getNested(cfg, 'voice.realtime.provider') === 'gemini' ? 'gemini' : 'live'

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
                voiceMode: prev.selections?.voiceMode ?? (liveVoice ? liveMode : autoTts ? 'spoken' : 'quiet'),
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
          clearTimeout(timeout)

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

  // Closing is a decision, not an accident: the skip is persisted per connection
  // + profile so the wizard does not come back on the next launch, and Settings
  // remains the place to add a provider key or a model afterwards.
  const close = () => {
    dismissJarvisOnboarding(undefined, scope)
    onDismiss?.()
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

  // Quick connect saves the key and refreshes the catalog; the model is only
  // committed by finish(), like any other choice made here.
  const adoptOpenRouter = (result: Extract<OpenRouterConnectResult, { ok: true }>) => {
    const nextProviders = normalizeProviders(result.options)
    const openRouter = nextProviders.find(item => item.slug === OPENROUTER_PROVIDER_SLUG)
    const nextModel = result.model ?? firstModel(openRouter)

    setProviders(nextProviders)
    requestToken.current += 1
    selectedModelRef.current = { provider: OPENROUTER_PROVIDER_SLUG, model: nextModel }
    setConfigurationStatus('idle')
    setConfigurationMessage('')
    setAccessStatus('idle')
    setAccessMessage('')
    persistState({
      ...invalidateModelDependentState(
        updatedState(state, { selections: { engine: OPENROUTER_PROVIDER_SLUG, model: nextModel } })
      ),
      completedSteps: Array.from(new Set([...state.completedSteps, 'engine' as JarvisOnboardingStep])),
      currentStep: 'model'
    })
  }

  // Same shape as the OpenRouter path: the catalog is refreshed, the model is only committed by finish().
  const adoptChatGpt = (result: Extract<ChatGptConnectResult, { ok: true }>) => {
    const nextProviders = normalizeProviders(result.options)
    const chatGpt = nextProviders.find(item => item.slug === CHATGPT_PROVIDER_SLUG)
    const nextModel = result.model ?? firstModel(chatGpt)

    setProviders(nextProviders)
    requestToken.current += 1
    selectedModelRef.current = { provider: CHATGPT_PROVIDER_SLUG, model: nextModel }
    setConfigurationStatus('idle')
    setConfigurationMessage('')
    setAccessStatus('idle')
    setAccessMessage('')
    persistState({
      ...invalidateModelDependentState(
        updatedState(state, { selections: { engine: CHATGPT_PROVIDER_SLUG, model: nextModel } })
      ),
      completedSteps: Array.from(new Set([...state.completedSteps, 'engine' as JarvisOnboardingStep])),
      currentStep: 'model'
    })
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

      if (
        !match ||
        match.authenticated === false ||
        (match.models.length > 0 && !match.models.includes(modelAtRequest))
      ) {
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

    return Boolean(
      match && match.authenticated !== false && (match.models.length === 0 || match.models.includes(modelName))
    )
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

  const performFinish = async () => {
    const providerAtRequest = selectedProvider
    const modelAtRequest = selectedModel
    const computerModeAtRequest = computerMode
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
        await rollbackModelAssignment(
          { provider: snapshotOptions.provider, model: snapshotOptions.model },
          requestScope
        )
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

      let nextConfig = setNested(snapshotConfig, 'voice.auto_tts', voiceMode !== 'quiet')
      const liveProvider = LIVE_VOICE_PROVIDERS[voiceMode]

      nextConfig = setNested(nextConfig, 'voice.engine', liveProvider ? 'realtime' : 'classic')

      if (liveProvider) {
        nextConfig = setNested(nextConfig, 'voice.realtime.provider', liveProvider)
      }

      nextConfig = setNested(nextConfig, 'approvals.mode', approvalConfigMode(approvalsMode))
      nextConfig = setNested(
        nextConfig,
        'custom_prompt',
        withCoordinatorPrompt(withPersonality(snapshotConfig.custom_prompt, stateRef.current.selections?.personality))
      )

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

      // Toolsets sit OUTSIDE the model/config transaction on purpose: they are a
      // separate backend surface, they are reversible from Tools, and a refusal
      // here must not undo a setup that has already committed. So this runs
      // last, and a failure is reported rather than rolled back.
      const { failed } = await applyJarvisToolsetPlan(jarvisToolsetPlan(computerModeAtRequest), (name, enabled) =>
        applyToolset(name, enabled, requestScope)
      )

      if (failed.length > 0 && requestStillCurrent(token, requestScopeKey)) {
        notify({ kind: 'warning', message: copy.errors.toolsets(failed.join(', ')) })
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
      applyVoiceEngineFromConfig(nextConfig)
      onComplete?.()
      setCompleted(true)
      markJarvisOnboardingCompleted()
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

  const finish = async () => {
    try {
      await runOnboardingTransaction(scope, () => performFinish())
    } catch (error) {
      if (error instanceof OnboardingTransactionBusyError) {
        setSaveError(copy.errors.save)

        return
      }

      throw error
    }
  }

  const isLastStep = currentIndex === JARVIS_ONBOARDING_STEPS.length - 1
  const informational = currentStep === 'welcome' || currentStep === 'access'
  const optional = ['profile', 'voice', 'connections'].includes(currentStep)

  const stepLabel = (step: JarvisOnboardingStep) =>
    step === 'profile' ? guide.personality : step === 'engine' ? guide.ownApi : copy.steps[step]

  const nextDisabled =
    (loading && ['engine', 'model', 'approvals'].includes(currentStep)) ||
    (currentStep === 'engine' && !selectedProvider) ||
    (currentStep === 'model' && configurationStatus !== 'passed') ||
    (currentStep === 'approvals' && !approvalsMode)

  if (showSplash) {
    const splashCopy = locale === 'pl'
      ? { title: 'Poznaj Cześka', action: 'Kliknij logo, aby rozpocząć' }
      : locale === 'zh'
        ? { title: '认识 Czesiek', action: '点击标志开始' }
        : { title: 'Meet Czesiek', action: 'Click the logo to begin' }

    return (
      <Dialog
        modal
        onOpenChange={open => {
          if (!open) {
            close()
          }
        }}
        open
      >
        <DialogContent
          aria-labelledby="czesiek-splash-title"
          className="z-(--z-onboarding) w-[calc(100vw-2rem)] max-w-xl border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) text-(--ui-text-primary)"
          data-testid="czesiek-onboarding-splash"
          showCloseButton={false}
        >
          <div className="flex min-h-[26rem] flex-col items-center justify-center gap-5 text-center">
            <button
              aria-label={splashCopy.action}
              className={cn('rounded-full p-3 transition-all duration-400 hover:scale-105 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-(--ui-accent)', splashLeaving && 'scale-125 opacity-0')}
              onClick={openWizard}
              type="button"
            >
              <img alt="" className="size-44 object-contain drop-shadow-[0_0_30px_rgba(67,108,255,0.36)] sm:size-56" src={logoUrl} />
            </button>
            <h1 className="text-2xl font-semibold" id="czesiek-splash-title">{splashCopy.title}</h1>
            <p className="text-sm text-(--ui-text-secondary)">{splashCopy.action}</p>
          </div>
        </DialogContent>
      </Dialog>
    )
  }

  return (
    <Dialog
      modal
      onOpenChange={open => {
        if (!open) {
          close()
        }
      }}
      open
    >
      <DialogContent
        aria-labelledby="jarvis-onboarding-title"
        bodyClassName="grid max-h-[calc(100vh-2rem)] gap-4 overflow-y-auto bg-(--ui-bg-elevated) p-4 text-(--ui-text-primary) sm:max-h-[calc(100vh-3rem)] sm:p-6 lg:grid-cols-[17rem_minmax(0,1fr)]"
        className="z-(--z-onboarding) w-[calc(100vw-2rem)] max-w-5xl overflow-hidden border-(--ui-stroke-secondary) bg-(--ui-bg-elevated) text-(--ui-text-primary) shadow-[0_28px_90px_-24px_rgba(0,0,0,0.9)] backdrop-blur-2xl sm:w-[calc(100vw-3rem)]"
        data-testid="jarvis-onboarding"
        showCloseButton={false}
      >
        <aside className="grid content-start gap-4">
          <div>
            <p className="text-xs font-semibold uppercase text-(--ui-accent)">{copy.productName}</p>
            <h1 className="mt-2 text-2xl font-semibold tracking-normal" id="jarvis-onboarding-title">
              {copy.intro.title}
            </h1>
            <p className="mt-2 text-sm leading-6 text-(--ui-text-tertiary)">{copy.intro.subtitle}</p>
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
                      'jarvis-choice flex min-h-11 w-full items-center gap-3 rounded-xl px-3 text-left text-sm focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[#00B7FF]/50',
                      active
                        ? 'jarvis-choice-on text-(--ui-text-primary)'
                        : 'jarvis-well text-(--ui-text-tertiary) hover:text-(--ui-text-primary)'
                    )}
                    onClick={() => selectStep(step)}
                    type="button"
                  >
                    <span
                      className={cn(
                        'flex size-6 shrink-0 items-center justify-center rounded-full text-xs',
                        done ? 'bg-emerald-500/20 text-emerald-500' : active ? 'jarvis-segment-on' : 'bg-(--ui-bg-tertiary)'
                      )}
                    >
                      {done ? <Check className="size-3.5" /> : index + 1}
                    </span>
                    <span>{stepLabel(step)}</span>
                  </button>
                </li>
              )
            })}
          </ol>
        </aside>

        <section className="jarvis-panel grid min-h-[31rem] min-w-0 grid-rows-[auto_minmax(0,1fr)_auto] gap-4 rounded-2xl bg-(--ui-bg-quaternary) p-4 backdrop-blur-xl sm:p-5">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-(--ui-stroke-tertiary) pb-4">
            <div>
              <p className="text-xs text-(--ui-text-tertiary)">
                {copy.progress(currentIndex + 1, JARVIS_ONBOARDING_STEPS.length)}
              </p>
              <div
                aria-hidden="true"
                className="jarvis-progress mt-2 h-1.5 w-40 overflow-hidden rounded-full"
              >
                <span
                  className="jarvis-progress-fill block h-full rounded-full"
                  style={{ width: `${((currentIndex + 1) / JARVIS_ONBOARDING_STEPS.length) * 100}%` }}
                />
              </div>
              <h2 className="mt-1 text-xl font-semibold tracking-normal">{stepLabel(currentStep)}</h2>
              <p className="mt-2 text-sm text-(--ui-accent)">
                {informational ? guide.information : optional ? guide.optional : guide.required}
              </p>
            </div>
            <div className="flex items-center gap-2">
              {loading ? <Loader2 className="size-5 animate-spin text-(--ui-accent)" /> : null}
              <button
                aria-label={copy.actions.close}
                className="inline-flex size-10 items-center justify-center rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) text-(--ui-text-secondary) transition hover:border-white/25 hover:bg-(--chrome-action-hover) hover:text-(--ui-text-primary) focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[#00B7FF]/50"
                onClick={close}
                type="button"
              >
                <X className="size-4" />
              </button>
            </div>
          </div>

          <div className="jarvis-rise min-h-0 overflow-y-auto pr-1" key={currentStep}>
            {currentStep === 'welcome' ? <WelcomeStep copy={copy.welcome} /> : null}
            {currentStep === 'profile' ? (
              <PersonalityStep
                onChange={personality => persistState(updatedState(state, { selections: { personality } }))}
                value={state.selections?.personality}
              />
            ) : null}
            {currentStep === 'engine' ? (
              <EngineStep
                body={copy.engine.body}
                copy={copy.engine}
                onOtherProviders={openSecureProviderSetup}
                onSelect={chooseProvider}
                otherProvidersLabel={guide.alternatives}
                providers={providers}
                quickConnect={
                  <>
                    {providers.some(item => item.slug === CHATGPT_PROVIDER_SLUG && item.authenticated !== false) ? null : (
                      <div className="grid gap-2 rounded-md border border-[#10A37F]/40 bg-[#10A37F]/8 p-4">
                        <ChatGptQuickConnect
                          loadOptions={() => loadModelOptions(scope)}
                          onConnected={adoptChatGpt}
                          scope={scope}
                          tone="dark"
                        />
                      </div>
                    )}
                    {providers.some(item => item.slug === OPENROUTER_PROVIDER_SLUG && item.authenticated !== false) ? null : (
                      <div className="grid gap-2 rounded-md border border-[#00B7FF]/40 bg-[#00B7FF]/8 p-4">
                        <p className="text-sm font-semibold">{copy.engine.quickStartTitle}</p>
                        <OpenRouterQuickConnect
                          deps={{
                            loadOptions: () => loadModelOptions(scope),
                            saveKey: key => setEnvVar(OPENROUTER_ENV_KEY, key, scope),
                            validate: key => validateProviderCredential(OPENROUTER_ENV_KEY, key, undefined, scope)
                          }}
                          onConnected={adoptOpenRouter}
                          scope={scope}
                          tone="dark"
                        />
                      </div>
                    )}
                  </>
                }
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
                providerSlug={selectedProvider}
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
                saveKey={(envKey, key) => setEnvVar(envKey, key, scope)}
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
            {currentStep === 'computer' ? (
              <ComputerStep
                {...computerStatus}
                copy={copy.computer}
                mode={computerMode}
                onSelect={mode => persistState(updatedState(state, { selections: { computerMode: mode } }))}
              />
            ) : null}
            {currentStep === 'connections' ? (
              <ConnectionsStep
                catalog={t.jarvisConnections}
                copy={copy.connections}
                onSelectRole={next => {
                  persistState(
                    updatedState(state, {
                      selections: { connections: selectionForRole(state.selections?.connections ?? [], next, connectionRole) }
                    })
                  )
                  setConnectionRole(next)
                }}
                onToggle={id => {
                  const chosen = state.selections?.connections ?? []

                  persistState(
                    updatedState(state, {
                      selections: {
                        connections: chosen.includes(id) ? chosen.filter(item => item !== id) : [...chosen, id]
                      }
                    })
                  )
                }}
                role={connectionRole}
                selected={state.selections?.connections ?? []}
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

          <div className="flex flex-wrap items-center justify-between gap-3 border-t border-(--ui-stroke-tertiary) pt-4">
            {loading ? <p className="w-full text-sm text-(--ui-text-secondary)">{guide.loading}</p> : null}
            {nextDisabled && !loading ? (
              <p className="w-full text-sm text-(--ui-text-secondary)">
                {currentStep === 'model' ? guide.checkModel : guide.chooseProvider}
              </p>
            ) : null}
            {saveError && !loading ? (
              <Button onClick={() => loadInitialState(++loadToken.current, scope, scopeKey)} variant="secondary">
                {guide.retry}
              </Button>
            ) : null}
            <div className="min-h-5 text-sm text-red-300" role="alert">
              {saveError}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {optional ? (
                <Button onClick={goNext} variant="text">
                  {guide.skip}
                </Button>
              ) : null}
              <button
                className="min-h-11 rounded-md px-2 text-sm text-(--ui-text-tertiary) underline-offset-4 transition hover:text-(--ui-text-primary) hover:underline focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[#00B7FF]/50"
                onClick={close}
                type="button"
              >
                {copy.actions.finishLater}
              </button>
              <Button
                className="min-h-11"
                disabled={currentIndex === 0}
                onClick={goBack}
                type="button"
                variant="outline"
              >
                <ChevronLeft className="size-4" />
                {copy.actions.back}
              </Button>
              {isLastStep ? (
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

function EngineStep({
  body,
  copy,
  onSelect,
  onOtherProviders,
  otherProvidersLabel,
  providers,
  quickConnect,
  selected,
  title
}: {
  body: string
  copy: JarvisOnboardingCopy['engine']
  onSelect: (slug: string) => void
  onOtherProviders: () => void
  otherProvidersLabel: string
  providers: ProviderOption[]
  /** The OpenRouter fast path, shown until OpenRouter is connected. */
  quickConnect?: ReactNode
  selected: string
  title: string
}) {
  return (
    <div className="grid gap-4">
      <p className="text-lg font-semibold">{title}</p>
      <p className="max-w-2xl text-sm leading-6 text-(--ui-text-secondary)">{body}</p>
      {quickConnect ? <div className="grid gap-3">{quickConnect}</div> : null}
      <Button onClick={onOtherProviders} variant="secondary">
        {otherProvidersLabel}
      </Button>
      <div className="grid gap-2 sm:grid-cols-2">
        {providers.length === 0 ? <p className="text-sm text-(--ui-text-secondary)">{copy.noProviders}</p> : null}
        {providers.map(provider => (
          <button
            className={cn(
              'min-h-16 rounded-md border p-3 text-left focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[#00B7FF]/50',
              selected === provider.slug
                ? 'border-[#00B7FF] bg-[#00B7FF]/12'
                : 'border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary)'
            )}
            key={provider.slug}
            onClick={() => onSelect(provider.slug)}
            type="button"
          >
            <span className="flex items-center gap-3">
              <ModelBrandIcon hints={[provider.slug, provider.name]} />
              <span className="min-w-0">
                <span className="block text-sm font-medium">{provider.name}</span>
                <span className="mt-0.5 block text-xs text-(--ui-text-tertiary)">
                  {copy.modelCount(provider.models.length)}
                </span>
              </span>
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}

/** The few models worth a card: the presets the provider serves, else its first handful. */
function quickModels(providerSlug: string, models: string[]): string[] {
  const listed = [{ authenticated: true, models, name: providerSlug, slug: providerSlug }]

  if (providerSlug === OPENROUTER_PROVIDER_SLUG) {
    return resolveOpenRouterPresets(listed).presets.map(preset => preset.model)
  }

  if (providerSlug === CHATGPT_PROVIDER_SLUG) {
    const best = chatGptWorkModel(listed)

    return [best, ...models.filter(model => model !== best)].filter((model): model is string => Boolean(model)).slice(0, 4)
  }

  return models.slice(0, 4)
}

function shortModelName(model: string): string {
  return model.includes('/') ? model.slice(model.indexOf('/') + 1) : model
}

function ModelStep({
  body,
  checking,
  checkLabel,
  message,
  models,
  onCheck,
  onSelect,
  providerSlug,
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
  providerSlug: string
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
      <p className="max-w-2xl text-sm leading-6 text-(--ui-text-secondary)">{body}</p>
      <div className="grid gap-2 sm:grid-cols-2" data-testid="jarvis-model-cards">
        {quickModels(providerSlug, models).map(model => (
          <button
            aria-pressed={selected === model}
            className={cn(
              'jarvis-choice flex min-h-14 items-center gap-3 rounded-xl px-3 py-2 text-left focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[#00B7FF]/50',
              selected === model ? 'jarvis-choice-on' : 'jarvis-well'
            )}
            key={model}
            onClick={() => onSelect(model)}
            type="button"
          >
            <ModelBrandIcon hints={[providerSlug]} model={model} />
            <span className="min-w-0 flex-1">
              <span className="block truncate text-sm font-medium">{shortModelName(model)}</span>
              <span className="block truncate text-xs text-(--ui-text-tertiary)">{model.includes('/') ? model.slice(0, model.indexOf('/')) : providerSlug}</span>
            </span>
            {selected === model ? <Check className="size-4 shrink-0 text-(--ui-accent)" /> : null}
          </button>
        ))}
      </div>
      <select
        className="min-h-11 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-input) px-3 text-sm text-(--ui-text-primary) focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[#00B7FF]/50"
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
        <Button
          className="min-h-11"
          disabled={checking || !selected}
          onClick={onCheck}
          type="button"
          variant="secondary"
        >
          {checking ? <Loader2 className="size-4 animate-spin" /> : <RefreshCw className="size-4" />}
          {checkLabel}
        </Button>
        {status === 'passed' ? <span className="text-sm text-(--ui-green)">{successLabel}</span> : null}
      </div>
      {message ? <p className="text-sm text-red-300">{message}</p> : null}
    </div>
  )
}

/** The key each Live provider needs, and where to get it. */
const LIVE_KEY_FIELDS: Record<'gemini' | 'live', { env: string; placeholder: string; url?: string }> = {
  gemini: { env: 'GEMINI_API_KEY', placeholder: 'AIza…', url: 'https://aistudio.google.com/apikey' },
  live: { env: 'OPENAI_API_KEY', placeholder: 'sk-…' }
}

function LiveKeyPanel({
  copy,
  mode,
  saveKey
}: {
  copy: JarvisOnboardingCopy['voice']
  mode: 'gemini' | 'live'
  saveKey: (envKey: string, key: string) => Promise<unknown>
}) {
  const field = LIVE_KEY_FIELDS[mode]
  const [key, setKey] = useState('')
  const [keyState, setKeyState] = useState<'idle' | 'saving' | 'saved' | 'failed'>('idle')
  const hint = mode === 'gemini' ? copy.geminiKeyHint : copy.liveKeyHint
  const label = mode === 'gemini' ? copy.geminiKeyLabel : copy.liveKeyLabel

  const save = async () => {
    if (!key.trim()) {
      return
    }

    setKeyState('saving')

    try {
      await saveKey(field.env, key.trim())
      setKey('')
      setKeyState('saved')
    } catch {
      setKeyState('failed')
    }
  }

  return (
    <div
      className="grid gap-2 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) p-4"
      data-live-key={mode}
    >
      <p className="text-sm font-semibold text-(--ui-text-primary)">{label}</p>
      <p className="text-sm text-(--ui-text-secondary)">{hint}</p>
      <div className="flex min-w-0 gap-2">
        <input
          aria-label={label}
          autoComplete="off"
          className="min-h-11 min-w-0 flex-1 rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-input) px-3 font-mono text-xs text-(--ui-text-primary) focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-[#00B7FF]/50"
          onChange={event => setKey(event.target.value)}
          placeholder={field.placeholder}
          type="password"
          value={key}
        />
        <Button
          className="min-h-11"
          disabled={!key.trim() || keyState === 'saving'}
          onClick={() => void save()}
          type="button"
          variant="secondary"
        >
          {keyState === 'saving' ? <Loader2 className="size-4 animate-spin" /> : <KeyRound className="size-4" />}
          {copy.liveKeySave}
        </Button>
      </div>
      {field.url ? (
        <a className="text-sm text-(--ui-accent) hover:underline" href={field.url} rel="noreferrer" target="_blank">
          {copy.geminiGetKey}
        </a>
      ) : null}
      {keyState === 'saved' ? <p className="text-sm text-(--ui-green)">{copy.liveKeySaved}</p> : null}
      {keyState === 'failed' ? <p className="text-sm text-red-300">{copy.liveKeyFailed}</p> : null}
    </div>
  )
}

function VoiceStep({
  copy,
  mode,
  onSelect,
  saveKey
}: {
  copy: JarvisOnboardingCopy['voice']
  mode: JarvisVoiceMode
  onSelect: (mode: JarvisVoiceMode) => void
  saveKey: (envKey: string, key: string) => Promise<unknown>
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
        <ChoiceCard
          active={mode === 'live'}
          description={copy.liveHint}
          icon={<Zap className="size-4" />}
          label={copy.live}
          onClick={() => onSelect('live')}
        />
        <ChoiceCard
          active={mode === 'gemini'}
          data-voice-mode="gemini"
          description={copy.geminiHint}
          icon={<Sparkles className="size-4" />}
          label={copy.gemini}
          onClick={() => onSelect('gemini')}
        />
      </div>
      <div className="grid gap-3 lg:grid-cols-2">
        <LiveKeyPanel copy={copy} mode="live" saveKey={saveKey} />
        <LiveKeyPanel copy={copy} mode="gemini" saveKey={saveKey} />
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
      <p className="max-w-2xl text-sm leading-6 text-(--ui-text-secondary)">{copy.body}</p>
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
      {opened ? <p className="text-sm text-(--ui-text-secondary)">{copy.opened}</p> : null}
      {status === 'passed' ? <p className="text-sm text-(--ui-green)">{copy.validated}</p> : null}
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
  const handleKeyDown = choiceRadioKeyHandler<JarvisApprovalProductMode>({
    attribute: 'data-approval-mode',
    current: mode,
    onSelect,
    values: ['balanced', 'strict']
  })

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
