// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { OnboardingChatDirective } from './directive'

describe('OnboardingChatDirective', () => {
  afterEach(cleanup)

  it('keeps the card mounted and inert while the turn is still streaming', () => {
    const { rerender } = render(<OnboardingChatDirective attrs={{ step: 'look' }} streaming />)

    const card = document.querySelector('[data-onboarding-card]')

    expect(card).toBeTruthy()
    expect(card?.hasAttribute('inert')).toBe(true)
    expect((screen.getByRole('button', { name: 'Continue' }) as HTMLButtonElement).disabled).toBe(true)

    rerender(<OnboardingChatDirective attrs={{ step: 'look' }} streaming={false} />)

    expect(document.querySelector('[data-onboarding-card]')).toBe(card)
    expect(card?.hasAttribute('inert')).toBe(false)
    expect((screen.getByRole('button', { name: 'Continue' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('renders nothing for an unknown step', () => {
    const { container } = render(<OnboardingChatDirective attrs={{ step: 'nope' }} streaming={false} />)

    expect(container.firstChild).toBeNull()
  })
})
