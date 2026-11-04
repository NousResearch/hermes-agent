import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, expect, it } from 'vitest'

import { $tourActive } from '@/lib/tour/tour-active'
import { $desktopOnboarding } from '@/store/onboarding'
import { $onboardingDecided, $onboardingSurfaces } from '@/store/onboarding-presence'
import { $gatewayState } from '@/store/session'
import { $agentSidebarChoice } from '@/store/simplified-agent-mode'

import { SimplifiedAgentOffer } from './simplified-agent-offer'

const initialOnboarding = $desktopOnboarding.get()

beforeEach(() => {
  $agentSidebarChoice.set(null)
  $onboardingDecided.set(true)
  $onboardingSurfaces.set(new Set())
  $gatewayState.set('open')
  $tourActive.set(false)
  $desktopOnboarding.set({ ...initialOnboarding, manual: false, freeTierReady: false })
})

afterEach(() => {
  cleanup()
  $agentSidebarChoice.set(false)
  $gatewayState.set('idle')
  $tourActive.set(false)
  $desktopOnboarding.set(initialOnboarding)
})

it.each([
  ['Yes, try it', true],
  ['No, keep current sidebar', false]
] as const)('remembers %s and does not offer again after remounting', (label, choice) => {
  const view = render(<SimplifiedAgentOffer />)
  expect(screen.getByRole('dialog', { name: 'Try the new Hermes sidebar?' })).toBeTruthy()
  expect(screen.getByText('Settings → Appearance → Window & layout → Simplified agent mode')).toBeTruthy()
  fireEvent.click(screen.getByRole('button', { name: label }))
  expect($agentSidebarChoice.get()).toBe(choice)
  expect(localStorage.getItem('hermes.desktop.simplifiedAgentMode.v1')).toBe(String(choice))
  expect(screen.queryByRole('dialog')).toBeNull()
  view.unmount()
  render(<SimplifiedAgentOffer />)
  expect(screen.queryByRole('dialog')).toBeNull()
})

it('waits for the gateway, onboarding, welcome screen, and tours without recording an answer', () => {
  $gatewayState.set('idle')
  $onboardingDecided.set(false)
  render(<SimplifiedAgentOffer />)
  expect(screen.queryByRole('dialog')).toBeNull()
  act(() => $gatewayState.set('open'))
  expect(screen.queryByRole('dialog')).toBeNull()
  act(() => {
    $onboardingDecided.set(true)
    $onboardingSurfaces.set(new Set(['onboarding']))
  })
  expect(screen.queryByRole('dialog')).toBeNull()
  act(() => {
    $onboardingSurfaces.set(new Set())
    $desktopOnboarding.set({ ...$desktopOnboarding.get(), freeTierReady: true })
  })
  expect(screen.queryByRole('dialog')).toBeNull()
  act(() => {
    $desktopOnboarding.set({ ...$desktopOnboarding.get(), freeTierReady: false })
    $tourActive.set(true)
  })
  expect(screen.queryByRole('dialog')).toBeNull()
  expect($agentSidebarChoice.get()).toBeNull()
  act(() => $tourActive.set(false))
  expect(screen.getByRole('dialog')).toBeTruthy()
  act(() => $tourActive.set(true))
  expect($agentSidebarChoice.get()).toBeNull()
})

it('treats Escape as declining, keeping the original sidebar', () => {
  render(<SimplifiedAgentOffer />)
  fireEvent.keyDown(screen.getByRole('dialog'), { key: 'Escape' })
  expect($agentSidebarChoice.get()).toBe(false)
  expect(screen.queryByRole('dialog')).toBeNull()
})
