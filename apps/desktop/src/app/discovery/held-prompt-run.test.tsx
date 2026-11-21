import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $heldPrompt, holdPrompt, takeHeldPrompt } from '@/store/held-prompt'
import { $onboardingPhase, $setupGuidePhase } from '@/store/onboarding-phase'
import { $gatewayState } from '@/store/session'
import { $shellChoice } from '@/store/shell-variant'
import { $visitorAccess } from '@/store/visitor-access'

import { useHeldPromptRun } from './held-prompt-run'

beforeEach(() => {
  window.localStorage.clear()
  takeHeldPrompt()
  $shellChoice.set(true)
  $visitorAccess.set({ kind: 'open' })
  $gatewayState.set('open')
  $onboardingPhase.set('idle')
  $setupGuidePhase.set('idle')
})

afterEach(cleanup)

describe('a message held until sign up', () => {
  it('runs first, exactly once, as a new chat once the visitor is signed in and connected', async () => {
    holdPrompt('Send me a morning briefing')
    const submit = vi.fn(async () => ({}))

    const { rerender } = renderHook(() => useHeldPromptRun(submit))
    rerender()
    act(() => $gatewayState.set('open'))

    expect(submit).toHaveBeenCalledTimes(1)
    expect(submit).toHaveBeenCalledWith('Send me a morning briefing')
    expect($heldPrompt.get()).toBeNull()

    // A second window (or this one after a reload) finds nothing left to run.
    renderHook(() => useHeldPromptRun(submit))
    expect(submit).toHaveBeenCalledTimes(1)
  })

  it('waits while sign up is still required, while disconnected and while setup is on screen', () => {
    holdPrompt('Clean up my inbox')
    const submit = vi.fn(async () => ({}))
    $visitorAccess.set({ kind: 'sign_up_required', signUpUrl: 'https://portal.example/sign-up' })
    $gatewayState.set('connecting')
    $setupGuidePhase.set('shown')

    renderHook(() => useHeldPromptRun(submit))
    act(() => $visitorAccess.set({ kind: 'open' }))
    act(() => $gatewayState.set('open'))
    expect(submit).not.toHaveBeenCalled()
    expect($heldPrompt.get()).toBe('Clean up my inbox')

    act(() => $setupGuidePhase.set('done'))
    expect(submit).toHaveBeenCalledWith('Clean up my inbox')
  })

  it('keeps a message that could not be sent in the composer rather than losing it', async () => {
    holdPrompt('Summarize a long document')
    const inserted = vi.fn()
    window.addEventListener('hermes:composer-insert', event => inserted((event as CustomEvent).detail))
    const submit = vi.fn(async () => Promise.reject(new Error('refused')))

    renderHook(() => useHeldPromptRun(submit))
    await vi.waitFor(() => expect(inserted).toHaveBeenCalled())

    expect(inserted.mock.calls[0]?.[0]).toMatchObject({ target: 'main', text: 'Summarize a long document' })
  })
})
