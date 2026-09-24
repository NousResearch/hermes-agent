import { atom } from 'nanostores'

import { readJson, writeJson } from '@/lib/storage'

export interface OnboardingAnswers {
  accent: null | string
  connectors: string[]
  context: string
  name: string
  layout: string
}

// Keep existing fork users' answers when they move to upstream.
export const ANSWERS_KEY = 'hermes-onboarding-wizard-answers-v1'

export const DEFAULT_ANSWERS: OnboardingAnswers = {
  accent: null,
  connectors: [],
  context: '',
  name: '',
  layout: 'basic'
}

export function loadAnswers(): OnboardingAnswers {
  const raw = readJson<Partial<OnboardingAnswers>>(ANSWERS_KEY)

  // Project the retained fields so retired wizard preferences cannot be sent
  // to personalization or written back on the next answer.
  return {
    accent: raw?.accent ?? DEFAULT_ANSWERS.accent,
    connectors: raw?.connectors ?? [...DEFAULT_ANSWERS.connectors],
    context: raw?.context ?? DEFAULT_ANSWERS.context,
    name: raw?.name ?? DEFAULT_ANSWERS.name,
    layout: raw?.layout ?? DEFAULT_ANSWERS.layout
  }
}

export const $onboardingAnswers = atom<OnboardingAnswers>(loadAnswers())

export function setOnboardingAnswers(patch: Partial<OnboardingAnswers>): void {
  const next = { ...$onboardingAnswers.get(), ...patch }

  $onboardingAnswers.set(next)
  writeJson(ANSWERS_KEY, next)
}
