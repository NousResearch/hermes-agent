import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ModelLogo } from '@/components/ui/model-logo'
import type { SessionDotState } from '@/store/session-dot-state'

import { SessionStatusDot } from './session-status-dot'

const { states, colors } = await vi.hoisted(async () => {
  const { atom } = await import('nanostores')

  return { states: atom<Record<string, string>>({}), colors: atom({}) }
})

vi.mock('@/store/session-dot-state', () => ({ $sessionDotStateById: states }))
vi.mock('@/store/session-color', () => ({
  $sessionColorById: colors,
  sessionColorFor: () => '#123456'
}))

afterEach(cleanup)

describe('SessionStatusDot optional model glyph', () => {
  it.each<[SessionDotState, string, boolean]>([
    ['working', 'text-(--ui-accent)', false],
    ['needs-input', 'text-amber-500', false],
    ['unread', 'text-(--ui-success)', false],
    ['stalled', 'text-(--ui-accent)', true],
    ['background', 'text-(--ui-text-tertiary)', true],
    ['draft', 'text-(--ui-text-quaternary)', true],
    ['idle', 'text-(--ui-text-quaternary)', false]
  ])('preserves %s semantics and existing hover text while painting the glyph', (state, colorClass, dimmed) => {
    states.set({ s1: state })
    const dotView = render(<SessionStatusDot branchStem="└─ " storedSessionId="s1" />)

    const logoView = render(
      <SessionStatusDot branchStem="└─ " glyph={<ModelLogo brand="openai" />} storedSessionId="s1" />
    )

    const dot = dotView.container.querySelector('.rounded-full')!
    const glyph = logoView.container.querySelector('[data-session-state]')!

    expect(dot).toBeTruthy()
    expect(logoView.container.querySelector('.rounded-full')).toBeNull()
    expect(glyph.querySelectorAll('[data-model-brand]')).toHaveLength(1)
    expect(glyph.classList.contains(colorClass)).toBe(true)
    expect(glyph.classList.contains('opacity-50')).toBe(dimmed)

    for (const name of ['title', 'role', 'aria-label', 'aria-hidden']) {
      expect(glyph.getAttribute(name)).toBe(dot.getAttribute(name))
    }

    expect(logoView.container.textContent).toBe(dotView.container.textContent)
    expect(glyph.className).not.toMatch(/animate|spin|pulse/)

    if (state === 'idle') {
      expect((glyph as HTMLElement).style.color).toBe((dot as HTMLElement).style.backgroundColor)
    }
  })
})
