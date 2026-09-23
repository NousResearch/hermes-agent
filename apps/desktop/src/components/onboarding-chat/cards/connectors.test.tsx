import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { ConnectorPicks } from '@/components/onboarding-chat/cards/setup'
import { $onboardingAnswers, DEFAULT_ANSWERS } from '@/store/onboarding-answers'
import type { OnboardingPlugin } from '@/store/onboarding-plugins'

const plugins: OnboardingPlugin[] = [
  {
    app_state: 'unknown',
    description: '',
    name: 'blender',
    platforms: [],
    sentence: '',
    tier: 'official',
    title: 'Blender'
  },
  {
    app_state: 'missing_app',
    description: '',
    name: 'nvidia-app',
    platforms: ['windows'],
    sentence: 'needs NVIDIA App',
    tier: 'official',
    title: 'NVIDIA App'
  }
]

afterEach(() => {
  cleanup()
  $onboardingAnswers.set({ ...DEFAULT_ANSWERS, committed: [] })
})

it('lists plugins before connectors; a plugin whose app is missing is greyed with its reason and still picks', () => {
  const sent: string[] = []

  render(
    <ConnectorPicks
      catalog={{ rows: [{ connected: false, connector: 'gmail', enabled: true }], status: 'ready' }}
      commit={summary => sent.push(summary) > 0}
      done={false}
      locked={false}
      plugins={plugins}
    />
  )

  const labels = screen.getAllByRole('button', { pressed: false }).map(button => button.textContent ?? '')
  expect(labels.findIndex(text => text.includes('Blender'))).toBeLessThan(labels.findIndex(text => /gmail/i.test(text)))

  const nvidia = screen.getByRole('button', { name: /NVIDIA App/ })
  expect(nvidia.textContent).toContain('needs NVIDIA App')
  expect(nvidia.className).toContain('opacity-60')

  fireEvent.click(nvidia)
  fireEvent.click(screen.getByRole('button', { name: /gmail/i }))
  expect($onboardingAnswers.get().plugins).toEqual(['nvidia-app'])

  fireEvent.click(screen.getByRole('button', { name: 'Continue with 2' }))
  expect(sent).toEqual(['apps I use, not connected yet: gmail; plugins picked, not installed yet: nvidia-app'])
})
