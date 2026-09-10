/**
 * The `thread.tail` slot mounts EVERY contributor and lets each decline —
 * the same contract as `chat.empty`, because ownership of a session's tail is
 * only known once each plugin has loaded its own data.
 */

import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { registry } from '@/contrib'
import { THREAD_TAIL_AREA, type ThreadTailContribution } from '@/lib/thread-tail'

import { ThreadTailSlot } from './thread-tail-slot'

const disposers: (() => void)[] = []

function contribute(id: string, render: ThreadTailContribution['render']) {
  disposers.push(registry.register({ area: THREAD_TAIL_AREA, data: { render }, id }))
}

afterEach(() => {
  for (const dispose of disposers.splice(0)) {
    dispose()
  }
})

describe('the transcript tail asks every contributor', () => {
  it('renders the owner even when an earlier contributor declined', () => {
    contribute('declines', () => null)
    contribute('owns', ({ sessionId }) => <span data-testid="owner">tail of {sessionId}</span>)

    render(<ThreadTailSlot contributions={registry.getArea(THREAD_TAIL_AREA)} sessionId="s-1" />)

    expect(screen.getByTestId('owner').textContent).toBe('tail of s-1')
    expect(screen.queryByText(/declines/)).toBeNull()
  })

  it('renders nothing for a contribution without a render payload', () => {
    disposers.push(registry.register({ area: THREAD_TAIL_AREA, data: {}, id: 'empty' }))

    const { container } = render(<ThreadTailSlot contributions={registry.getArea(THREAD_TAIL_AREA)} sessionId="s-2" />)

    expect(container.textContent).toBe('')
  })

  it('walls off a contributor that throws, without taking the slot down', () => {
    contribute('crashes', () => {
      throw new Error('boom')
    })
    contribute('fine', () => <span data-testid="fine">still here</span>)

    render(<ThreadTailSlot contributions={registry.getArea(THREAD_TAIL_AREA)} sessionId="s-3" />)

    expect(screen.getByTestId('fine')).toBeTruthy()
  })
})
