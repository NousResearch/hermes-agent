import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { requestGatewayForAgent } from '@/store/gateway'
import { $onboardingAnswers, DEFAULT_ANSWERS, loadAnswers } from '@/store/onboarding-answers'
import { $activeSessionId, $selectedStoredSessionId, setSessionOwnerHint } from '@/store/session'

import { LayoutCard } from './setup'

vi.mock('@/store/gateway', async importOriginal => ({
  ...await importOriginal(), requestGatewayForAgent: vi.fn(async () => ({ finished: true }))
}))
vi.mock('@/components/onboarding-chat/assembly', async importOriginal => ({
  ...await importOriginal(), assembleChatOnboarding: vi.fn()
}))

afterEach(() => {
  cleanup()
  $activeSessionId.set(null)
  $selectedStoredSessionId.set(null)
  $onboardingAnswers.set(DEFAULT_ANSWERS)
})

it('does not start global metering merely by selecting a layout', () => {
  setSessionOwnerHint('setup-guide', { connectionId: 'setup-remote', profile: 'hermes-setup' })
  $activeSessionId.set('setup-guide')
  $selectedStoredSessionId.set('setup-guide')
  $onboardingAnswers.set(DEFAULT_ANSWERS)
  render(<LayoutCard attrs={{}} locked={false} />)
  const choices = screen.getAllByRole('button').filter(button => button.textContent !== 'Continue')
  expect(choices.length).toBeGreaterThan(0)
  fireEvent.click(choices[0])
  expect(requestGatewayForAgent).not.toHaveBeenCalled()
  expect(loadAnswers().committed).not.toContain('layout')
})
