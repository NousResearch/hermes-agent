/**
 * The questionnaire's one store: where it is, what it knows, what was answered.
 *
 * `phase` is the funnel the metrics hook reads (idle → shown → done | skipped | failed). While the
 * phase is `shown` the overlay registers the onboarding presence surface, so every "not during
 * setup" surface waits. Side effects of an answer (accent, layout) live in `apply.ts`; this file only
 * holds state and the one revert hook for a step's unconfirmed preview (D25).
 */

import { atom, computed } from 'nanostores'

import { setOnboardingSurfaceActive } from '@/store/onboarding-presence'

import { type Answers, EMPTY_ANSWERS, type Facts, FLOW, type StepId, visibleSteps } from './flow'

export type QuestionnairePhase = 'done' | 'failed' | 'idle' | 'shown' | 'skipped'

/** Start or Skip in flight; one at a time, and no answer changes until it ends. */
export type QuestionnairePending = 'skip' | 'start'

/** `steps`: a question. `review`: the prompt preview and Start. `preparing`: skipped while the account is still being made. */
export type QuestionnaireView = 'preparing' | 'review' | 'steps'

/** The chat Start created for the first message: runtime and stored session ids. */
export interface FirstChat {
  runtimeSessionId: string
  sessionId: string
}

export interface QuestionnaireState {
  phase: QuestionnairePhase
  view: QuestionnaireView
  stepId: null | StepId
  answers: Answers
  facts: Facts
  pending: null | QuestionnairePending
  /** Kept past close so a retried Start can check it before sending the first message again. */
  firstChat: FirstChat | null
}

export const LOADING_FACTS: Facts = { connectors: { status: 'loading' }, local: undefined, machine: null, plugins: undefined }

const IDLE: QuestionnaireState = {
  answers: EMPTY_ANSWERS,
  facts: LOADING_FACTS,
  firstChat: null,
  pending: null,
  phase: 'idle',
  stepId: null,
  view: 'steps'
}

export const $questionnaire = atom<QuestionnaireState>(IDLE)

export const $questionnairePhase = computed($questionnaire, state => state.phase)

/** The overlay shows the questionnaire (any view) while this is true. */
export const $questionnaireOpen = computed($questionnaire, state => state.phase === 'shown')

$questionnaireOpen.subscribe(open => setOnboardingSurfaceActive('questionnaire', open))

let revertPreview: (() => void) | null = null

function patch(next: Partial<QuestionnaireState>): void {
  $questionnaire.set({ ...$questionnaire.get(), ...next })
}

/** A step painted an answer it has not confirmed yet; `revert` puts the previous look back. */
export function holdPreview(revert: () => void): void {
  revertPreview ??= revert
}

/** The preview became the answer (Confirm): nothing to put back. */
export function keepPreview(): void {
  revertPreview = null
}

function dropPreview(): void {
  const revert = revertPreview

  revertPreview = null
  revert?.()
}

export function openQuestionnaire(): void {
  if ($questionnaire.get().phase === 'shown') {
    return
  }

  revertPreview = null
  $questionnaire.set({ ...IDLE, phase: 'shown', stepId: 'name' })
}

/** A step that stops applying while it is on screen (the connector list became unavailable) gives way to the next. */
function onVisibleStep(state: QuestionnaireState): QuestionnaireState {
  const ids = visibleIds(state)
  const current = state.stepId

  if (state.view !== 'steps' || !current || ids.includes(current)) {
    return state
  }

  const order = FLOW.map(step => step.id)
  const next = ids.find(id => order.indexOf(id) > order.indexOf(current))

  return { ...state, ...(next ? { stepId: next } : { stepId: null, view: 'review' }) }
}

export function setFacts(next: Partial<Facts>): void {
  const state = $questionnaire.get()

  $questionnaire.set(onVisibleStep({ ...state, facts: { ...state.facts, ...next } }))
}

export function setAnswers(next: Partial<Answers>): void {
  const state = $questionnaire.get()

  patch({ answers: { ...state.answers, ...next } })
}

function visibleIds(state: QuestionnaireState): StepId[] {
  return visibleSteps(state.facts, state.answers).map(step => step.id)
}

/** The step after `stepId` in flow order, or the review screen after the last one. */
function advance(state: QuestionnaireState, stepId: StepId): Partial<QuestionnaireState> {
  const ids = visibleIds(state)
  const next = ids[ids.indexOf(stepId) + 1]

  return next ? { stepId: next, view: 'steps' } : { stepId: null, view: 'review' }
}

export function confirmStep(stepId: StepId): void {
  const state = $questionnaire.get()

  keepPreview()
  const answers = { ...state.answers, skipped: state.answers.skipped.filter(id => id !== stepId) }

  patch({ answers, ...advance({ ...state, answers }, stepId) })
}

export function skipStep(stepId: StepId): void {
  const state = $questionnaire.get()

  dropPreview()
  const skipped = state.answers.skipped.includes(stepId) ? state.answers.skipped : [...state.answers.skipped, stepId]
  const answers = { ...state.answers, skipped }

  patch({ answers, ...advance({ ...state, answers }, stepId) })
}

/** Go back to an answered step from the trail, or one step back. */
export function goToStep(stepId: StepId): void {
  dropPreview()
  patch({ stepId, view: 'steps' })
}

export function goBack(): void {
  const state = $questionnaire.get()
  const ids = visibleIds(state)
  const index = state.view === 'review' ? ids.length : state.stepId ? ids.indexOf(state.stepId) : 0

  if (index > 0) {
    goToStep(ids[index - 1])
  }
}

/** Steps already passed, in flow order, for the answer trail. */
export function passedSteps(state: QuestionnaireState): StepId[] {
  const ids = visibleIds(state)

  return state.view === 'review' ? ids : ids.slice(0, state.stepId ? ids.indexOf(state.stepId) : 0)
}

export function setPending(pending: null | QuestionnairePending): void {
  patch({ pending })
}

export function noteFirstChat(firstChat: FirstChat): void {
  patch({ firstChat })
}

/** Skip setup while the free account is still being made: hold the overlay on "Starting Hermes…". */
export function showPreparing(): void {
  dropPreview()
  patch({ view: 'preparing' })
}

/** Leave the questionnaire. Drops any unconfirmed preview; confirmed answers stay applied. */
export function closeQuestionnaire(phase: Exclude<QuestionnairePhase, 'idle' | 'shown'>): void {
  if ($questionnaire.get().phase !== 'shown') {
    return
  }

  dropPreview()
  patch({ pending: null, phase })
}
