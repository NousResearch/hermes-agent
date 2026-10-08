import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { FIXTURES } from '../fixtures.test-util'
import { MEMORY_LINE } from '../handoff'
import { closeQuestionnaire, openQuestionnaire, setAnswers, setFacts } from '../store'

import { ReviewStep } from './review'

beforeEach(() => {
  openQuestionnaire()
  setFacts(FIXTURES.spark)
  setAnswers({ connectors: ['github', 'gmail', 'slack'], name: 'Sid' })
})

afterEach(() => {
  cleanup()
  closeQuestionnaire('skipped')
})

describe('ReviewStep', () => {
  it('shows the whole prompt, memory line included, in a box that fits a full prompt', () => {
    render(<ReviewStep onStart={() => {}} />)

    const prompt = screen.getByText(content => content.includes(MEMORY_LINE))
    const box = prompt.closest<HTMLElement>('.overflow-y-auto')

    // FadeScroll's cap (its edge fade is a mask jsdom drops; fade-scroll.test covers the mask).
    expect(box?.style.maxHeight).toBe('22rem')
  })
})
