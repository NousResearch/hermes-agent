import { act, cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import type { OnboardingRequester } from '@/onboarding/due'
import { FIXTURES } from '@/onboarding/fixtures.test-util'
import { Questionnaire } from '@/onboarding/Questionnaire'
import { $questionnaire, closeQuestionnaire, openQuestionnaire, setFacts, skipStep } from '@/onboarding/store'
import { markQuestionnaireDecided } from '@/store/onboarding-presence'

import { QuestionnaireLayer } from './questionnaire-layer'

/** A backend that never answers, so the questionnaire stays on the facts it was given. */
const silent: OnboardingRequester = () => new Promise(() => {})

beforeEach(() => {
  markQuestionnaireDecided()
  openQuestionnaire()
})

afterEach(() => {
  cleanup()
  closeQuestionnaire('skipped')
})

// jsdom has no layout, so the height bound and the scroller are read from their classes.
const classes = (element: Element | null) => (element?.className ?? '').split(/\s+/)

describe('QuestionnaireLayer', () => {
  it('caps the card at the overlay height and scrolls the answers, keeping Skip setup outside the scroller', async () => {
    render(
      <>
        <Questionnaire
          enabled={false}
          openDefaultChat={async () => 'runtime-1'}
          openLandedChat={async () => false}
          requestGateway={silent}
        />
        <QuestionnaireLayer refreshReadiness={async () => {}} statusbarVisible />
      </>
    )

    act(() => {
      setFacts(FIXTURES.spark)

      for (let stepId = $questionnaire.get().stepId; stepId; stepId = $questionnaire.get().stepId) {
        skipStep(stepId)
      }
    })

    const start = await screen.findByRole('button', { name: 'Start' })
    const skip = screen.getByRole('button', { name: 'Skip setup' })
    const card = start.closest('.rounded-xl')

    expect(classes(card)).toEqual(expect.arrayContaining(['max-h-full', 'flex', 'flex-col']))

    const scroller = start.closest('.overflow-y-auto')

    expect(scroller).not.toBeNull()
    expect(card?.contains(scroller ?? null)).toBe(true)
    expect(classes(scroller)).toContain('min-h-0')
    expect(skip.closest('.overflow-y-auto')).toBeNull()
  })
})
