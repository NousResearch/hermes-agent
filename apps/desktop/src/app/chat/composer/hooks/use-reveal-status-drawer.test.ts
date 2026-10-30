import { renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { useRevealStatusDrawer } from './use-status-drawer'

describe('useRevealStatusDrawer', () => {
  it('opens a collapsed drawer once per new alert, then respects the user collapsing it again', () => {
    const toggle = vi.fn()

    const { rerender } = renderHook(
      ({ collapsed, token }: { collapsed: boolean; token: number | null }) =>
        useRevealStatusDrawer(token, collapsed, toggle),
      { initialProps: { collapsed: true, token: null as number | null } }
    )

    expect(toggle).not.toHaveBeenCalled()

    rerender({ collapsed: true, token: 1 })
    expect(toggle).toHaveBeenCalledTimes(1)

    // The user hides it again for the same wall: it stays hidden.
    rerender({ collapsed: false, token: 1 })
    rerender({ collapsed: true, token: 1 })
    expect(toggle).toHaveBeenCalledTimes(1)

    // A new wall reveals it again; an already-open drawer is left alone.
    rerender({ collapsed: true, token: 2 })
    expect(toggle).toHaveBeenCalledTimes(2)
    rerender({ collapsed: false, token: 3 })
    expect(toggle).toHaveBeenCalledTimes(2)
  })
})
