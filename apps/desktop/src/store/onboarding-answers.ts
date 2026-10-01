import { atom } from 'nanostores'

import { readJson, writeJson } from '@/lib/storage'

export interface OnboardingAnswers {
  accent: null | string
  committed: string[]
  connectors: string[]
  context: string
  plugins: string[]
  pluginOutcomes: Record<string, PluginOutcome>
  name: string
  layout: string
}

export interface PluginOutcome {
  state: 'failed' | 'installed' | 'skipped'
  detail: string
  skill: string
  tools: string[]
}

export const ANSWERS_KEY = 'hermes-onboarding-wizard-answers-v1'

export const DEFAULT_ANSWERS: OnboardingAnswers = {
  accent: null,
  committed: [],
  connectors: [],
  context: '',
  name: '',
  layout: 'basic',
  plugins: [],
  pluginOutcomes: {}
}

export function loadAnswers(): OnboardingAnswers {
  const raw = readJson<Partial<OnboardingAnswers>>(ANSWERS_KEY)

  return {
    accent: raw?.accent ?? DEFAULT_ANSWERS.accent,
    committed: raw?.committed ?? [...DEFAULT_ANSWERS.committed],
    connectors: raw?.connectors ?? [...DEFAULT_ANSWERS.connectors],
    context: raw?.context ?? DEFAULT_ANSWERS.context,
    name: raw?.name ?? DEFAULT_ANSWERS.name,
    layout: raw?.layout ?? DEFAULT_ANSWERS.layout,
    plugins: raw?.plugins ?? [],
    pluginOutcomes: raw?.pluginOutcomes ?? {}
  }
}

export const $onboardingAnswers = atom<OnboardingAnswers>(loadAnswers())

export function setOnboardingAnswers(patch: Partial<OnboardingAnswers>): void {
  const next = { ...$onboardingAnswers.get(), ...patch }

  $onboardingAnswers.set(next)
  writeJson(ANSWERS_KEY, next)
}

export function markStepCommitted(step: string): void {
  const { committed } = $onboardingAnswers.get()

  if (!committed.includes(step)) {
    setOnboardingAnswers({ committed: [...committed, step] })
  }
}
