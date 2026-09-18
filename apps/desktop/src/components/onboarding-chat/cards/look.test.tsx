import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { requestComposerSubmit } from '@/app/chat/composer/focus'
import type { CardProps } from '@/components/onboarding-chat/cards/frame'
import { LookCard } from '@/components/onboarding-chat/cards/setup'
import { $onboardingAnswers, DEFAULT_ANSWERS, loadAnswers, setOnboardingAnswers } from '@/store/onboarding-answers'
import { $accentOverride, setAccentOverride } from '@/themes/accent-override'

vi.mock('@/themes', () => ({ useTheme: () => ({ renderedMode: 'dark' }) }))
vi.mock('@/app/chat/composer/focus', () => ({ requestComposerSubmit: vi.fn(() => true) }))

afterEach(() => {
  cleanup()
  $onboardingAnswers.set({ ...DEFAULT_ANSWERS, committed: [] })
  setAccentOverride(null)
  localStorage.clear()
  vi.clearAllMocks()
})

it('applies and persists a custom color through the same Continue path as a preset', () => {
  const custom = '#5a7d62'
  const view = render(<LookCard attrs={{}} locked={false} />)
  fireEvent.change(screen.getByLabelText('Custom color'), { target: { value: custom } })
  expect($accentOverride.get()).toBe(custom)
  expect(loadAnswers().accent).toBe(custom)

  view.unmount()
  act(() => {
    setAccentOverride(null)
    $onboardingAnswers.set(loadAnswers())
  })
  render(<LookCard attrs={{}} locked={false} />)
  expect($accentOverride.get()).toBe(custom)
  expect(screen.getByLabelText<HTMLInputElement>('Custom color').value).toBe(custom)
  fireEvent.click(screen.getByRole('button', { name: 'Continue' }))
  expect(requestComposerSubmit).toHaveBeenCalledWith(`[setup] accent color: ${custom}`, expect.any(Object))
  expect(loadAnswers().committed).toContain('look')
})

function ChatColor(props: CardProps) {
  return (
    <>
      <LookCard attrs={{}} locked={false} />
      <LookCard {...props} />
    </>
  )
}

it('applies a settled chat color once without replaying old requests over later choices', () => {
  const requested = '#654321'
  const picked = '#5a7d62'
  const view = render(<ChatColor attrs={{ value: requested }} locked messageId="color-1" />)
  expect($accentOverride.get()).toBeNull()
  expect(screen.getAllByLabelText('Custom color')).toHaveLength(1)
  view.rerender(<ChatColor attrs={{ value: requested }} locked={false} messageId="color-1" />)
  expect($accentOverride.get()).toBe(requested)
  fireEvent.change(screen.getByLabelText('Custom color'), { target: { value: picked } })
  view.unmount()

  const replay = render(<ChatColor attrs={{ value: requested }} locked={false} messageId="color-1" />)
  expect($accentOverride.get()).toBe(picked)
  replay.rerender(<ChatColor attrs={{ value: requested }} locked={false} messageId="color-2" />)
  expect($accentOverride.get()).toBe(requested)
  replay.rerender(<ChatColor attrs={{ value: 'not-a-color' }} locked={false} messageId="color-3" />)
  expect(loadAnswers().accent).toBe(requested)
  act(() => setOnboardingAnswers({ committed: [...loadAnswers().committed, 'look'] }))
  replay.rerender(<ChatColor attrs={{ value: picked }} locked={false} messageId="color-4" />)
  expect(loadAnswers().accent).toBe(requested)
})
