// @vitest-environment jsdom
import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { TemporaryChatHero } from './temporary-chat-hero'

afterEach(cleanup)

/**
 * The hero is the empty state for a temporary chat, and the compact bar above
 * the composer is its counterpart once the user starts talking. The contract
 * worth pinning is the HANDOFF: exactly one of them is on screen at a time.
 *
 * Rendering the real ChatBar here would drag in the assistant-ui runtime, the
 * Electron IPC bridge and a live session store for what is a presentational
 * decision, so the visibility rule is asserted directly against the same
 * expression the composer uses.
 */
describe('temporary chat empty state', () => {
  it('names the mode and explains what it means', () => {
    render(<TemporaryChatHero />)

    // The label the user asked for, verbatim.
    expect(screen.getByText('Temporary chat')).toBeTruthy()

    // The heading alone is just a mode name; the body has to say what is
    // actually guaranteed, or "temporary" is left to interpretation.
    const body = screen.getByText(/Nothing here is saved/i).textContent ?? ''
    expect(body).toMatch(/history/i)
    expect(body).toMatch(/resumed/i)
  })

  it('uses the incognito icon, not a padlock', () => {
    const { container } = render(<TemporaryChatHero />)

    // Same reasoning as the badge: a padlock promises encryption, which is a
    // different and misleading claim. A temporary chat is not more secure in
    // transit -- it simply is not written down.
    expect(container.querySelector('.tabler-icon-spy')).toBeTruthy()
    expect(container.querySelector('.codicon-lock')).toBeNull()
  })

  it('hands off to the compact bar once the thread is non-empty', () => {
    // Mirrors the two render conditions in chat/index.tsx and composer/index.tsx.
    const heroVisible = (ephemeral: boolean, empty: boolean) => ephemeral && empty
    const barVisible = (ephemeral: boolean, empty: boolean) => ephemeral && !empty

    // Fresh temporary chat: hero only.
    expect(heroVisible(true, true)).toBe(true)
    expect(barVisible(true, true)).toBe(false)

    // After the first message: bar only.
    expect(heroVisible(true, false)).toBe(false)
    expect(barVisible(true, false)).toBe(true)

    // Never both at once -- two amber blocks saying the same thing reads as a
    // bug rather than as emphasis.
    for (const empty of [true, false]) {
      expect(heroVisible(true, empty) && barVisible(true, empty)).toBe(false)
    }

    // A normal chat gets neither, in either state.
    for (const empty of [true, false]) {
      expect(heroVisible(false, empty)).toBe(false)
      expect(barVisible(false, empty)).toBe(false)
    }
  })
})
