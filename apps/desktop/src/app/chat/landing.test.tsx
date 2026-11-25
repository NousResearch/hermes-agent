import { act, cleanup, render, renderHook } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { $guidanceLedger } from '@/store/guidance-ledger'
import { holdPrompt, takeHeldPrompt } from '@/store/held-prompt'
import { $shellChoice } from '@/store/shell-variant'
import { $visitorAccess } from '@/store/visitor-access'

import { useIntroUnlessHeld } from '../discovery/held-exchange'

import { useChatLanding, useComposerLanding } from './landing'

afterEach(() => {
  cleanup()
  $guidanceLedger.set({})
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

it('lays the newcomer’s greeting above the redesign landing, and nothing under it', () => {
  $shellChoice.set(true)
  $guidanceLedger.set({ renameSplash: 'not_applicable' })

  const { result } = renderHook(() => useComposerLanding(true, true, 'main', true))
  const around = result.current.around

  // The landing carries the greeting and nothing else: the composer's `below` slot — the three-item
  // prompt-snippet / starter-ideas strip (HADESK-65) — is gone, so nothing is laid under the composer.
  expect(Object.keys(around ?? {})).toEqual(['above'])

  const { container } = render(<>{around?.above}</>)

  expect(container.querySelector('.discovery-landing-greeting')).toBeTruthy()
  expect(container.querySelector('[data-slot="starter-ideas"], .discovery-ideas')).toBeNull()
})

it('lays nothing on the landing in the classic shell', () => {
  $shellChoice.set(false)

  const { result } = renderHook(() => {
    const { centered } = useChatLanding(true, false, false)

    return useComposerLanding(centered, true, 'main', true)
  })

  expect(result.current.around).toBeUndefined()
})
