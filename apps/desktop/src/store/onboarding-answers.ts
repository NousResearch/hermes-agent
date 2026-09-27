import { atom } from 'nanostores'

import { readJson, writeJson } from '@/lib/storage'

export interface OnboardingAnswers {
  accent: null | string
  /** Cards the user has already pressed Continue on. The card's own React
   *  state dies on every transcript reconcile (the hidden submit and the
   *  turn-end hydrate both rebuild the message list), so a Done button that
   *  lived there came back live and let the step be answered twice. */
  committed: string[]
  connectors: string[]
  context: string
  /** Catalog plugin names picked on the connectors card. A pick is a wish, not an install. */
  plugins: string[]
  /** The settled install card's result per plugin, written when the guide's manage_catalog card settles. A
   *  picked plugin with no entry was not offered for install (the chosen task did not need it). */
  pluginOutcomes: Record<string, PluginOutcome>
  name: string
  layout: string
}

export interface PluginOutcome {
  state: 'failed' | 'installed' | 'skipped'
  detail: string
  /** The plugin's qualified skill name (`<plugin key>:<skill>`), the only name skill_view resolves. */
  skill: string
  tools: string[]
}

// Keep existing fork users' answers when they move to upstream.
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

  // Project the retained fields so retired wizard preferences cannot be sent
  // to personalization or written back on the next answer.
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
