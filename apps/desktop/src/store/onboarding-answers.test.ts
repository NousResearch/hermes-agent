import { beforeEach, expect, it } from 'vitest'

import { writeJson } from '@/lib/storage'

import {
  $onboardingAnswers,
  ANSWERS_KEY,
  DEFAULT_ANSWERS,
  loadAnswers,
  setOnboardingAnswers
} from './onboarding-answers'

beforeEach(() => window.localStorage.clear())

it('retains saved answers and defaults missing fields without carrying retired preferences into future writes', () => {
  const saved = {
    accent: '#abc123',
    connectors: ['github'],
    name: 'Sam',
    focus: ['coding'],
    theme: 'retired-theme',
    keepInDock: false,
    openAtLogin: true
  }

  writeJson(ANSWERS_KEY, saved)
  const answers = loadAnswers()

  expect(answers).toEqual({
    accent: saved.accent,
    connectors: saved.connectors,
    name: saved.name,
    context: DEFAULT_ANSWERS.context,
    layout: DEFAULT_ANSWERS.layout
  })
  expect(Object.keys(answers).sort()).toEqual(Object.keys(DEFAULT_ANSWERS).sort())
  $onboardingAnswers.set(answers)
  setOnboardingAnswers({ context: 'A shared project' })
  expect(loadAnswers()).toEqual({ ...answers, context: 'A shared project' })
})
