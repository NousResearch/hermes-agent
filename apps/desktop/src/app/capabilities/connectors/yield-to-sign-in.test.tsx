import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $freeTierSignIn, $openAppWindows } from '@/store/free-tier-sign-in'

import { useYieldToSignIn } from './connectors-tab'

afterEach(() => $freeTierSignIn.set({ status: 'closed' }))

describe('an open app window and the sign-in', () => {
  it('holds the offer back while open, and closes when a sign-in starts', () => {
    const close = vi.fn()
    const { unmount } = renderHook(() => useYieldToSignIn(true, close))

    expect($openAppWindows.get()).toBe(1)
    expect(close).not.toHaveBeenCalled()

    act(() => $freeTierSignIn.set({ status: 'requested' }))
    expect(close).toHaveBeenCalledWith(null)

    unmount()
    expect($openAppWindows.get()).toBe(0)
  })
})
