import { atom } from 'nanostores'

import { readJson, writeJson } from '@/lib/storage'

export interface OnboardingAnswers {
  accent: null | string
}

const ANSWERS_KEY = 'hermes-onboarding-wizard-answers-v1'

export const DEFAULT_ANSWERS: OnboardingAnswers = { accent: null }

function loadAnswers(): OnboardingAnswers {
  const raw = readJson<Partial<OnboardingAnswers>>(ANSWERS_KEY)

  return { accent: raw?.accent ?? DEFAULT_ANSWERS.accent }
}

export const $onboardingAnswers = atom<OnboardingAnswers>(loadAnswers())

export function setOnboardingAnswers(patch: Partial<OnboardingAnswers>): void {
  const next = { ...$onboardingAnswers.get(), ...patch }

  $onboardingAnswers.set(next)
  writeJson(ANSWERS_KEY, next)
}
