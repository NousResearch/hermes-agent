import { act, cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

import { IntroRevealGate } from '@/components/intro-reveal'
import { $freeTierStatus, freeTierStripPending } from '@/store/free-tier'
import { $introReveal, finishIntroReveal } from '@/store/intro-reveal'
import { $desktopOnboarding } from '@/store/onboarding'
import { $onboardingGate, devResetOnboardingFlow } from '@/store/onboarding-gate'
import { resetOnboardingPresenceForTests } from '@/store/onboarding-presence'

import { OnboardingChatGate } from './gate'

beforeEach(() => {
  localStorage.clear()
  devResetOnboardingFlow()
  $introReveal.set({ phase: 'hidden' })
  resetOnboardingPresenceForTests()
  $desktopOnboarding.set({ ...$desktopOnboarding.get(), firstRunSkipped: false, freeTierReady: true })
})

afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

it('acks as the cinematic starts and consumes its close edge once across remounts', async () => {
  vi.stubGlobal('hermesDesktop', {
    guestOnboardingEnabled: true,
    introReveal: {
      open: vi.fn().mockResolvedValue({ ok: true }),
      close: vi.fn().mockResolvedValue({ ok: true }),
      onSkip: () => () => undefined,
      onClosed: () => () => undefined
    }
  })
  const status = {
    label: 'Nous',
    model: 'nous/welcome',
    has_guest: true,
    notice_pending: true,
    available: true,
    enabled: true
  }

  const requestGateway = vi.fn().mockImplementation(async method => {
    if (method === 'free_tier.ack_notice') {
      status.notice_pending = false

      return { acked: true }
    }

    return status
  })

  $freeTierStatus.set(status)
  let finish: (started: boolean) => void = () => undefined
  const pending = new Promise<boolean>(resolve => {
    finish = resolve
  })
  const onKickoff = vi.fn(() => pending)

  const gates = () => (
    <>
      <IntroRevealGate enabled />
      <OnboardingChatGate enabled onKickoff={onKickoff} requestGateway={requestGateway} />
    </>
  )

  const first = render(gates())
  expect($introReveal.get().phase).toBe('playing')
  expect($desktopOnboarding.get().freeTierReady).toBe(false)
  expect(freeTierStripPending($freeTierStatus.get(), false)).toBe(false)
  expect(requestGateway).toHaveBeenCalledWith('free_tier.ack_notice')
  expect(onKickoff).not.toHaveBeenCalled()
  act(finishIntroReveal)
  await waitFor(() => expect(onKickoff).toHaveBeenCalledTimes(1))
  expect($onboardingGate.get().phase).toBe('cinematic')
  first.unmount()
  const second = render(gates())
  await act(async () => {
    finish(true)
    await pending
  })
  expect($onboardingGate.get()).toEqual({ phase: 'guided', guideQueued: false })
  second.unmount()
  render(gates())
  expect(onKickoff).toHaveBeenCalledTimes(1)
  expect(requestGateway.mock.calls.filter(([method]) => method === 'free_tier.ack_notice')).toHaveLength(1)
})

it('leaves the classic ready state alone and adds no calls with the flag absent', async () => {
  const open = vi.fn()
  vi.stubGlobal('hermesDesktop', { introReveal: { open } })
  const requestGateway = vi.fn()
  const onKickoff = vi.fn().mockResolvedValue(true)
  render(
    <>
      <IntroRevealGate enabled />
      <OnboardingChatGate enabled onKickoff={onKickoff} requestGateway={requestGateway} />
    </>
  )
  await act(async () => undefined)
  expect($desktopOnboarding.get().freeTierReady).toBe(true)
  expect($introReveal.get().phase).toBe('hidden')
  expect(requestGateway).not.toHaveBeenCalled()
  expect(onKickoff).not.toHaveBeenCalled()
  expect(open).not.toHaveBeenCalled()
})
