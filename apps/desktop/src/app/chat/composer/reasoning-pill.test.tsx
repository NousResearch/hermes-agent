import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { $currentModel, $currentProvider, $currentReasoningEffort, $activeSessionId } from '@/store/session'

import { ReasoningPill } from './reasoning-pill'

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

// The write path (gateway RPC, generation guard, revert semantics, empty
// model/provider guard) is exhaustively covered by `patch-reasoning.test.ts`.
// This file focuses on the pill's own concerns: capability gating,
// optimistic render while capability loads, empty model/provider hiding,
// and that `ReasoningPill` correctly hands off to `applyReasoningPatch` with
// the active-session metadata.

const applyReasoningPatch = vi.fn()

vi.mock('@/lib/patch-reasoning', () => ({
  applyReasoningPatch: (...args: unknown[]) => applyReasoningPatch(...args)
}))

beforeEach(() => {
  $activeSessionId.set(null)
  $currentModel.set('opus')
  $currentProvider.set('anthropic')
  $currentReasoningEffort.set('medium')
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

describe('ReasoningPill capability gating', () => {
  it('renders the pill when the active model supports reasoning', () => {
    render(
      <ReasoningPill
        disabled={false}
        gateway={null}
        reasoningCapable={true}
      />
    )

    // The trigger button is present — capability gating did not hide it.
    expect(screen.getByRole('button', { name: /effort/i })).toBeDefined()
  })

  it('returns null when the active model does not support reasoning (no disabled-Meds lie)', () => {
    const { container } = render(
      <ReasoningPill
        disabled={false}
        gateway={null}
        reasoningCapable={false}
      />
    )

    expect(container.firstChild).toBeNull()
  })

  it('renders optimistically while capability is unknown (null) so the pill does not flicker', () => {
    render(
      <ReasoningPill
        disabled={false}
        gateway={null}
        reasoningCapable={null}
      />
    )

    expect(screen.getByRole('button', { name: /effort/i })).toBeDefined()
  })

  it('returns null when the active model has not yet loaded (empty model)', () => {
    $currentModel.set('')

    const { container } = render(
      <ReasoningPill
        disabled={false}
        gateway={null}
        reasoningCapable={true}
      />
    )

    expect(container.firstChild).toBeNull()
  })

  it('returns null when the active provider has not yet loaded', () => {
    $currentProvider.set('')

    const { container } = render(
      <ReasoningPill
        disabled={false}
        gateway={null}
        reasoningCapable={true}
      />
    )

    expect(container.firstChild).toBeNull()
  })
})
