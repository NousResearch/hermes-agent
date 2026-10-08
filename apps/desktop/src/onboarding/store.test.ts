import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $onboardingSurfaces, markQuestionnaireDecided, onboardingSurfaceActive } from '@/store/onboarding-presence'

import { FIXTURES } from './fixtures.test-util'
import {
  $questionnaire,
  closeQuestionnaire,
  confirmStep,
  goBack,
  holdPreview,
  openQuestionnaire,
  passedSteps,
  setFacts,
  skipStep
} from './store'

beforeEach(() => {
  markQuestionnaireDecided()
  openQuestionnaire()
  setFacts(FIXTURES.spark)
})

afterEach(() => closeQuestionnaire('skipped'))

describe('questionnaire store', () => {
  it('registers the presence surface while open and drops it on close', () => {
    expect($onboardingSurfaces.get().has('questionnaire')).toBe(true)
    expect(onboardingSurfaceActive()).toBe(true)

    closeQuestionnaire('done')

    expect($questionnaire.get().phase).toBe('done')
    expect(onboardingSurfaceActive()).toBe(false)
  })

  it('walks the visible steps in flow order into the review screen', () => {
    for (const step of ['name', 'accent', 'layout', 'local', 'apps', 'connectors', 'task', 'tour'] as const) {
      expect($questionnaire.get().stepId).toBe(step)
      confirmStep(step)
    }

    expect($questionnaire.get().view).toBe('review')
    expect(passedSteps($questionnaire.get())).toHaveLength(8)
  })

  it('Skip reverts only the current unconfirmed preview', () => {
    const confirmedRevert = vi.fn()
    const previewRevert = vi.fn()

    holdPreview(confirmedRevert)
    confirmStep('name')
    holdPreview(previewRevert)
    skipStep('accent')

    expect(confirmedRevert).not.toHaveBeenCalled()
    expect(previewRevert).toHaveBeenCalledOnce()
    expect($questionnaire.get().answers.skipped).toEqual(['accent'])
  })

  it('keeps the first look when a step previews several times', () => {
    const original = vi.fn()
    const second = vi.fn()

    holdPreview(original)
    holdPreview(second)
    closeQuestionnaire('skipped')

    expect(original).toHaveBeenCalledOnce()
    expect(second).not.toHaveBeenCalled()
  })

  it('confirming a skipped step again clears its skip', () => {
    skipStep('name')
    goBack()
    confirmStep('name')

    expect($questionnaire.get().answers.skipped).toEqual([])
  })

  it('moves on when the connector list becomes unavailable while it is on screen', () => {
    for (const step of ['name', 'accent', 'layout', 'local', 'apps'] as const) {
      confirmStep(step)
    }

    expect($questionnaire.get().stepId).toBe('connectors')

    setFacts({ connectors: { status: 'unavailable' } })

    expect($questionnaire.get().stepId).toBe('task')
  })

  it('waits on Local and Apps while their facts are still being read', () => {
    closeQuestionnaire('skipped')
    openQuestionnaire()

    for (const step of ['name', 'accent', 'layout'] as const) {
      confirmStep(step)
    }

    expect($questionnaire.get().stepId).toBe('local')

    setFacts({ local: null, plugins: FIXTURES.spark.plugins })

    expect($questionnaire.get().stepId).toBe('apps')
    expect(passedSteps($questionnaire.get())).toEqual(['name', 'accent', 'layout'])
  })
})
