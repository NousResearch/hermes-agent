/**
 * The `chat.empty` slot mounts EVERY contributor, not just the first.
 *
 * Ownership of an empty transcript is per session and is not known until each
 * plugin has loaded its own data, so a first-wins slot let whichever plugin
 * happened to register first suppress the one that actually owns the chat —
 * silently, permanently, and only for some users, since registration order
 * depends on which plugins are installed.
 */

import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib'
import { CHAT_EMPTY_AREA, type ChatEmptyContribution } from '@/lib/chat-empty'

import { ChatEmptySlot } from './chat-empty-slot'

const disposers: (() => void)[] = []

function contribute(id: string, render: ChatEmptyContribution['render']) {
  disposers.push(registry.register({ area: CHAT_EMPTY_AREA, data: { render }, id }))
}

afterEach(() => {
  for (const dispose of disposers.splice(0)) {
    dispose()
  }
})

describe('an empty transcript asks every contributor', () => {
  it('renders the owner even when an earlier contributor declined', () => {
    contribute('declines', () => null)
    contribute('owns', ({ sessionId }) => <span data-testid="owner">owner of {sessionId}</span>)

    render(<ChatEmptySlot sessionId="s-1" />)

    expect(screen.getByTestId('owner').textContent).toBe('owner of s-1')
  })

  it('renders nothing when everyone declines', () => {
    contribute('a', () => null)
    contribute('b', () => null)

    const { container } = render(<ChatEmptySlot sessionId="s-1" />)

    expect(container.textContent).toBe('')
  })

  it('renders nothing when nobody contributes at all', () => {
    const { container } = render(<ChatEmptySlot sessionId="s-1" />)

    expect(container.textContent).toBe('')
  })

  it('shows a conflict rather than hiding one of the claimants', () => {
    contribute('first', () => <span>first</span>)
    contribute('second', () => <span>second</span>)

    render(<ChatEmptySlot sessionId="s-1" />)

    expect(screen.getByText('first')).toBeTruthy()
    expect(screen.getByText('second')).toBeTruthy()
  })

  it('isolates a throwing contributor from the ones beside it', () => {
    contribute('boom', () => {
      throw new Error('contributor exploded')
    })
    contribute('owns', () => <span data-testid="owner">still here</span>)

    render(<ChatEmptySlot sessionId="s-1" />)

    expect(screen.getByTestId('owner').textContent).toBe('still here')
  })
})
