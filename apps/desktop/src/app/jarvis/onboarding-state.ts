import { atom } from 'nanostores'

import { getApiRequestConnection, getApiRequestProfile } from '@/hermes'

import { isJarvisComputerMode, type JarvisComputerMode } from './computer-capabilities'
import { isJarvisConnectionId, type JarvisConnectionId } from './connections-catalog'
import { type CzesiekPersonality, normalizePersonality } from './personality'

export const JARVIS_ONBOARDING_VERSION = 3
export const JARVIS_ONBOARDING_STATE_KEY = 'ai-evolution-jarvis-onboarding-v1'

export const JARVIS_ONBOARDING_STEPS = [
  'welcome',
  'profile',
  'engine',
  'model',
  'voice',
  'access',
  'computer',
  'connections',
  'approvals'
] as const

/**
 * The steps each earlier version shipped with. Payloads from those versions
 * are migrated, not dropped: v1 had no computer step, v2 no welcome and no
 * connections step.
 */
const PREVIOUS_VERSION_STEPS: Readonly<Record<number, readonly string[]>> = {
  1: ['profile', 'engine', 'model', 'voice', 'access', 'approvals'],
  2: ['profile', 'engine', 'model', 'voice', 'access', 'computer', 'approvals']
}

export type JarvisOnboardingStep = (typeof JARVIS_ONBOARDING_STEPS)[number]

export type JarvisApprovalProductMode = 'balanced' | 'strict'
/**
 * `live` and `gemini` are Live voice (`voice.engine: realtime`): OpenAI
 * Realtime or Gemini Live (`voice.realtime.provider`).
 */
export type JarvisVoiceMode = 'gemini' | 'live' | 'quiet' | 'spoken'

export interface JarvisOnboardingScope {
  connectionId?: null | string
  profile?: null | string
}

export interface JarvisOnboardingSelections {
  personality?: CzesiekPersonality
  accessOpened?: boolean
  approvalsMode?: JarvisApprovalProductMode
  computerMode?: JarvisComputerMode
  /** What the person wants connected; set up afterwards on the Połączenia page. */
  connections?: JarvisConnectionId[]
  engine?: string
  model?: string
  profile?: string
  validatedAccessModel?: string
  validatedAccessProvider?: string
  validatedModel?: string
  validatedProvider?: string
  voiceMode?: JarvisVoiceMode
}

export interface JarvisOnboardingState {
  version: number
  completedSteps: JarvisOnboardingStep[]
  currentStep: JarvisOnboardingStep
  selections?: JarvisOnboardingSelections
  /**
   * The person closed the wizard on purpose ("I'll finish later").
   *
   * A skip is not a completion: nothing is claimed as chosen, the wizard simply
   * stops demanding attention on this connection + profile. Settings remains the
   * place to add a provider key or a model afterwards, so the state has to be
   * readable from both, and `completedSteps` stays honest about what was picked.
   */
  skipped?: boolean
}

const STEP_SET = new Set<string>(JARVIS_ONBOARDING_STEPS)

export function initialJarvisOnboardingState(step: JarvisOnboardingStep = 'welcome'): JarvisOnboardingState {
  return {
    version: JARVIS_ONBOARDING_VERSION,
    completedSteps: [],
    currentStep: step,
    selections: {}
  }
}

function isStep(value: unknown): value is JarvisOnboardingStep {
  return typeof value === 'string' && STEP_SET.has(value)
}

interface NormalizedJarvisOnboardingScope {
  connectionId: string
  profile: string
}

function normalizeScope(scope?: JarvisOnboardingScope): NormalizedJarvisOnboardingScope {
  return {
    connectionId: String(scope?.connectionId || 'local').trim() || 'local',
    profile: String(scope?.profile || 'default').trim() || 'default'
  }
}

export function normalizeJarvisOnboardingScope(scope?: JarvisOnboardingScope): NormalizedJarvisOnboardingScope {
  return normalizeScope(scope)
}

export function jarvisOnboardingScopeKey(scope?: JarvisOnboardingScope): string {
  const normalized = normalizeScope(scope)

  return `${encodeURIComponent(normalized.connectionId)}::${encodeURIComponent(normalized.profile)}`
}

export function currentJarvisOnboardingScope(): NormalizedJarvisOnboardingScope {
  return normalizeScope({
    connectionId: getApiRequestConnection(),
    profile: getApiRequestProfile()
  })
}

export function jarvisOnboardingStorageKey(scope: JarvisOnboardingScope = currentJarvisOnboardingScope()): string {
  const normalized = normalizeScope(scope)
  const suffix = `${encodeURIComponent(normalized.connectionId)}::${encodeURIComponent(normalized.profile)}`

  return `${JARVIS_ONBOARDING_STATE_KEY}:${suffix}`
}

function safeString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined
}

function safeBoolean(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined
}

function isApprovalMode(value: unknown): value is JarvisApprovalProductMode {
  return value === 'balanced' || value === 'strict'
}

function isVoiceMode(value: unknown): value is JarvisVoiceMode {
  return value === 'quiet' || value === 'spoken' || value === 'live' || value === 'gemini'
}

function sanitizeSelections(value: unknown): JarvisOnboardingSelections {
  if (!value || typeof value !== 'object') {
    return {}
  }

  const raw = value as Record<string, unknown>
  const selections: JarvisOnboardingSelections = {}
  if (raw.personality && typeof raw.personality === 'object') {
    selections.personality = normalizePersonality(raw.personality)
  }
  const accessOpened = safeBoolean(raw.accessOpened)
  const engine = safeString(raw.engine)
  const model = safeString(raw.model)
  const profile = safeString(raw.profile)
  const validatedAccessModel = safeString(raw.validatedAccessModel)
  const validatedAccessProvider = safeString(raw.validatedAccessProvider)
  const validatedModel = safeString(raw.validatedModel)
  const validatedProvider = safeString(raw.validatedProvider)

  if (accessOpened !== undefined) {
    selections.accessOpened = accessOpened
  }

  if (isApprovalMode(raw.approvalsMode)) {
    selections.approvalsMode = raw.approvalsMode
  }

  if (isJarvisComputerMode(raw.computerMode)) {
    selections.computerMode = raw.computerMode
  }

  if (Array.isArray(raw.connections)) {
    selections.connections = Array.from(new Set(raw.connections.filter(isJarvisConnectionId)))
  }

  if (engine) {
    selections.engine = engine
  }

  if (model) {
    selections.model = model
  }

  if (profile) {
    selections.profile = profile
  }

  if (validatedAccessModel) {
    selections.validatedAccessModel = validatedAccessModel
  }

  if (validatedAccessProvider) {
    selections.validatedAccessProvider = validatedAccessProvider
  }

  if (validatedModel) {
    selections.validatedModel = validatedModel
  }

  if (validatedProvider) {
    selections.validatedProvider = validatedProvider
  }

  if (isVoiceMode(raw.voiceMode)) {
    selections.voiceMode = raw.voiceMode
  }

  return selections
}

export function sanitizeJarvisOnboardingState(state: JarvisOnboardingState): JarvisOnboardingState {
  const currentStep = isStep(state.currentStep) ? state.currentStep : 'welcome'

  const completedSteps = Array.from(
    new Set((Array.isArray(state.completedSteps) ? state.completedSteps : []).filter(isStep))
  )

  const skipped = safeBoolean(state.skipped)

  return {
    version: JARVIS_ONBOARDING_VERSION,
    completedSteps,
    currentStep,
    ...(skipped === undefined ? {} : { skipped }),
    selections: sanitizeSelections(state.selections)
  }
}

function isSupportedVersion(version: unknown): version is number {
  return version === JARVIS_ONBOARDING_VERSION || (typeof version === 'number' && version in PREVIOUS_VERSION_STEPS)
}

/**
 * Carry an earlier version's payload forward.
 *
 * Someone who already finished setup must not be dragged back through the
 * wizard because we added a step, so a complete earlier run counts every new
 * step as done. New steps are marked complete without inventing selections: a
 * v1 run gets no `computerMode` and an older run no `connections` — nothing was
 * chosen, so nothing is claimed; the tips window and the Połączenia page
 * introduce what is new. A half-finished run keeps its progress and meets the
 * new steps on the way through.
 */
function migrateCompletedSteps(version: number, completedSteps: unknown[]): JarvisOnboardingStep[] {
  const steps = completedSteps.filter(isStep)
  const previous = PREVIOUS_VERSION_STEPS[version]

  if (!previous) {
    return steps
  }

  const wasComplete = previous.every(step => (steps as readonly string[]).includes(step))

  return wasComplete ? [...JARVIS_ONBOARDING_STEPS] : steps
}

export function serializeJarvisOnboardingState(state: JarvisOnboardingState): string {
  return JSON.stringify(sanitizeJarvisOnboardingState(state))
}

export function parseJarvisOnboardingState(raw: string | null): JarvisOnboardingState | null {
  if (!raw) {
    return null
  }

  try {
    const parsed = JSON.parse(raw) as Partial<JarvisOnboardingState>

    if (!isSupportedVersion(parsed.version) || !isStep(parsed.currentStep)) {
      return null
    }

    const completedSteps: unknown[] = Array.isArray(parsed.completedSteps) ? parsed.completedSteps : []

    return sanitizeJarvisOnboardingState({
      version: JARVIS_ONBOARDING_VERSION,
      currentStep: parsed.currentStep,
      completedSteps: migrateCompletedSteps(parsed.version, completedSteps),
      // A closed wizard must survive the reload, or it comes back on next launch.
      ...(parsed.skipped === true ? { skipped: true } : {}),
      selections: parsed.selections && typeof parsed.selections === 'object' ? parsed.selections : {}
    })
  } catch {
    return null
  }
}

export function readJarvisOnboardingState(
  storage: Storage | undefined = globalThis.localStorage,
  scope: JarvisOnboardingScope = currentJarvisOnboardingScope()
): JarvisOnboardingState | null {
  if (!storage) {
    return null
  }

  try {
    return parseJarvisOnboardingState(storage.getItem(jarvisOnboardingStorageKey(scope)))
  } catch {
    return null
  }
}

export function writeJarvisOnboardingState(
  state: JarvisOnboardingState,
  storage?: Storage,
  scope: JarvisOnboardingScope = currentJarvisOnboardingScope()
): boolean {
  const targetStorage = arguments.length >= 2 ? storage : globalThis.localStorage

  if (!targetStorage) {
    return false
  }

  try {
    targetStorage.setItem(jarvisOnboardingStorageKey(scope), serializeJarvisOnboardingState(state))

    return true
  } catch {
    return false
  }
}

/**
 * Bumped the moment setup commits.
 *
 * The wizard writes completion to storage, and storage does not notify — so
 * surfaces that ask "is setup done?" (the tips window) would keep answering
 * from the read they did when they mounted, i.e. mid-wizard. This is the one
 * seam between them, and it carries no data: subscribers re-read the scoped
 * state they already know how to read.
 */
export const $jarvisOnboardingCompletedAt = atom(0)

export function markJarvisOnboardingCompleted(): void {
  $jarvisOnboardingCompletedAt.set(Date.now())
}

export function jarvisOnboardingComplete(state: JarvisOnboardingState | null): boolean {
  return Boolean(
    state &&
      state.version === JARVIS_ONBOARDING_VERSION &&
      JARVIS_ONBOARDING_STEPS.every(step => state.completedSteps.includes(step))
  )
}

export function shouldShowJarvisOnboarding(state: JarvisOnboardingState | null): boolean {
  if (state?.skipped) {
    return false
  }

  return !jarvisOnboardingComplete(state)
}

/**
 * Close the wizard for this connection + profile without claiming completion.
 *
 * Returns false when storage refused the write — the caller must not act as if
 * the wizard were dismissed, or the next render brings it straight back.
 */
export function dismissJarvisOnboarding(
  storage: Storage | undefined = globalThis.localStorage,
  scope: JarvisOnboardingScope = currentJarvisOnboardingScope()
): boolean {
  if (!storage) {
    return false
  }

  const current = readJarvisOnboardingState(storage, scope) ?? initialJarvisOnboardingState()
  const written = writeJarvisOnboardingState({ ...current, skipped: true }, storage, scope)

  if (written) {
    // Same seam as completing: storage does not notify, so other surfaces that
    // ask "should the wizard be up?" need the nudge to re-read.
    markJarvisOnboardingCompleted()
  }

  return written
}

export function approvalConfigMode(mode: JarvisApprovalProductMode): 'manual' | 'smart' {
  return mode === 'strict' ? 'manual' : 'smart'
}
