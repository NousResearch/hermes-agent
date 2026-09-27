import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { I18nProvider } from '@/i18n'

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('starts the skipped-film splash before the backend connects and removes it only after adoption', async () => {
  vi.resetModules()
  vi.stubGlobal('hermesDesktop', { guestOnboardingEnabled: true, skipIntro: true })
  const { IntroRevealGate } = await import('@/components/intro-reveal')
  const { OnboardingChatGate } = await import('./gate')
  const { $onboardingGate } = await import('@/store/onboarding-gate')
  const { $desktopOnboarding } = await import('@/store/onboarding')
  $onboardingGate.set({ phase: 'idle', guideQueued: false, guideKickoff: 'idle' })
  $desktopOnboarding.set({ ...$desktopOnboarding.get(), firstRunSkipped: false })

  let complete = (_ready: boolean) => {}

  const pending = new Promise<boolean>(resolve => {
    complete = resolve
  })

  const kickoff = vi.fn(() => pending)

  const request = async () => {
    throw new Error('No provider notice available')
  }

  const view = (enabled: boolean) => (
    <I18nProvider>
      <IntroRevealGate enabled={enabled} />
      <OnboardingChatGate enabled={enabled} onKickoff={kickoff} requestGateway={request} />
    </I18nProvider>
  )

  const { rerender } = render(view(false))
  expect(screen.getByRole('status').textContent).toMatch(/Starting Hermes/)
  expect(kickoff).not.toHaveBeenCalled()
  rerender(view(true))
  await waitFor(() => expect(kickoff).toHaveBeenCalledOnce())
  expect(screen.getByRole('status')).toBeTruthy()
  await act(async () => {
    complete(true)
    await pending
  })
  expect(screen.queryByRole('status')).toBeNull()
  expect($onboardingGate.get().guideKickoff).toBe('started')
})

it.each(['refused', 'rejected'])('restores the ordinary app after %s startup', async outcome => {
  vi.resetModules()
  vi.stubGlobal('hermesDesktop', { guestOnboardingEnabled: true, skipIntro: true })
  const { OnboardingChatGate } = await import('./gate')
  const { $onboardingGate } = await import('@/store/onboarding-gate')
  const { $chatOnboardingSolo } = await import('./assembly')
  $onboardingGate.set({ phase: 'cinematic', guideQueued: true, guideKickoff: 'idle' })

  const kickoff = vi.fn(async () => {
    if (outcome === 'rejected') {
      throw new Error('Backend unavailable')
    }

    return false
  })

  const request = async () => {
    throw new Error('No provider notice available')
  }

  render(
    <I18nProvider>
      <OnboardingChatGate enabled onKickoff={kickoff} requestGateway={request} />
    </I18nProvider>
  )
  await waitFor(() => expect($onboardingGate.get().phase).toBe('skipped'))
  expect(screen.queryByRole('status')).toBeNull()
  expect($chatOnboardingSolo.get()).toBe(false)
  expect(kickoff).toHaveBeenCalledOnce()
})
