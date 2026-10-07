import type { OnboardingStateResult } from '@hermes/shared'
import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// The gate's stores are module singletons; each test loads a fresh window's worth.
async function loadWindow() {
  vi.resetModules()

  const [{ OnboardingChatGate }, gate, onboarding] = await Promise.all([
    import('./gate'),
    import('@/store/onboarding-gate'),
    import('@/store/onboarding')
  ])

  return { OnboardingChatGate, gate, onboarding }
}

const UNSEEN_STATE: OnboardingStateResult = { eligible: true, intro: 'unseen', failed_starts: 0, profile: 'setup' }

const unseenFirstRun = async <T,>(method: string): Promise<T> => {
  const reply = method === 'onboarding.state' ? UNSEEN_STATE : {}

  // SAFETY: the gate reads `onboarding.state` as OnboardingStateResult; any other call only awaits the reply.
  return reply as T
}

const neverKicksOff = () => new Promise<never>(() => {})

beforeEach(() => {
  Object.assign(window, { hermesDesktop: { guestOnboardingEnabled: true } })
})

afterEach(() => {
  cleanup()
  Reflect.deleteProperty(window, 'hermesDesktop')
})

describe('OnboardingChatGate', () => {
  it('lets a window that does not run the intro open the provider picker once the state is read', async () => {
    const { OnboardingChatGate, gate, onboarding } = await loadWindow()

    onboarding.requestDesktopOnboarding('No provider configured')
    render(
      <OnboardingChatGate enabled onKickoff={neverKicksOff} requestGateway={unseenFirstRun} runsIntro={false} />
    )

    await waitFor(() => expect(onboarding.$desktopOnboarding.get().requested).toBe(true))
    expect(gate.$onboardingGate.get().phase).toBe('idle')
    expect(gate.$setupProfileName.get()).toBe('setup')
  })

  it('queues the guided first run in the window that runs the intro', async () => {
    const { OnboardingChatGate, gate, onboarding } = await loadWindow()

    onboarding.requestDesktopOnboarding('No provider configured')
    render(<OnboardingChatGate enabled onKickoff={neverKicksOff} requestGateway={unseenFirstRun} runsIntro />)

    await waitFor(() => expect(gate.$onboardingStateRead.get()).toBe(true))
    expect(gate.$onboardingGate.get().phase).toBe('pending')
    expect(onboarding.$desktopOnboarding.get().requested).toBe(false)
  })
})
