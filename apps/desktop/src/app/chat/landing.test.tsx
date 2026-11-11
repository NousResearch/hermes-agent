import { act, cleanup, render, renderHook } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { $shellChoice } from '@/store/shell-variant'

import { ChatStageArt, useChatLanding } from './landing'

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

it('keeps one decorative stage photo from landing into the conversation, redesign only', () => {
  $shellChoice.set(true)

  const view = render(<ChatStageArt landing />)
  const art = view.container.querySelector('img')

  expect(art?.getAttribute('aria-hidden')).toBe('true')
  expect(art?.hasAttribute('data-landing')).toBe(true)
  view.rerender(<ChatStageArt landing={false} />)
  expect(view.container.querySelector('img')).toBe(art)
  expect(art?.hasAttribute('data-landing')).toBe(false)
  act(() => $shellChoice.set(false))
  expect(view.container.querySelector('img')).toBeNull()
})
