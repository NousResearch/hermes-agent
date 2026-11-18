import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { holdPrompt, takeHeldPrompt } from '@/store/held-prompt'
import { $shellChoice } from '@/store/shell-variant'
import { $visitorAccess } from '@/store/visitor-access'

import { useIntroUnlessHeld } from '../discovery/held-exchange'

import { useChatLanding, useComposerLanding } from './landing'

afterEach(() => {
  cleanup()
  $shellChoice.set(null)
})

it('centers normal-shell drafts, preserving classic and onboarding', () => {
  $shellChoice.set(false)

  const { result, rerender } = renderHook(
    ({ draft, onboarding, setup }) => useChatLanding(draft, onboarding, setup, 'friendly', 3),
    { initialProps: { draft: true, onboarding: false, setup: false } }
  )

  expect(result.current.intro).toEqual({ personality: 'friendly', seed: 3 })
  act(() => $shellChoice.set(true))
  expect(result.current).toEqual({ centered: true, intro: undefined })
  rerender({ draft: true, onboarding: true, setup: false })
  expect(result.current.centered).toBe(false)
  rerender({ draft: true, onboarding: false, setup: true })
  expect(result.current.centered).toBe(false)
  rerender({ draft: false, onboarding: false, setup: false })
  expect(result.current).toEqual({ centered: false, intro: undefined })
})

it('gives the composer no ideas to point at while a held message takes the new chat’s landing', () => {
  $shellChoice.set(true)
  $visitorAccess.set({ kind: 'sign_up_required', signUpUrl: 'https://portal.example/sign-up' })

  // ChatView's chain: the held message decides the intro, the intro the landing, the landing what the
  // composer carries (and so whether its placeholder may promise ideas below it).
  const { result } = renderHook(() => {
    const introShown = useIntroUnlessHeld(true, { primary: true, routed: false })
    const landing = useChatLanding(introShown, false, false)

    return useComposerLanding(landing.centered, true, 'main', true).around
  })

  expect(result.current).toBeDefined()

  act(() => holdPrompt('Compare the top 5 standing desks'))
  expect(result.current).toBeUndefined()

  takeHeldPrompt()
  $visitorAccess.set({ kind: 'open' })
})
