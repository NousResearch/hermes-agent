// @vitest-environment jsdom
import { act, renderHook } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { useProfileSwitchLatch } from './use-profile-switch-latch'

describe('useProfileSwitchLatch', () => {
  it('stays pending until the query stamp moves past the armed one', () => {
    const { rerender, result } = renderHook(useProfileSwitchLatch, { initialProps: { dataUpdatedAt: 1 } })

    expect(result.current.pending).toBe(false)

    act(() => result.current.arm())
    // Same object reference, same stamp: still the outgoing profile's record.
    rerender({ dataUpdatedAt: 1 })
    expect(result.current.pending).toBe(true)

    rerender({ dataUpdatedAt: 2 })
    expect(result.current.pending).toBe(false)
  })

  it('releases on a fresh error stamp when one is supplied', () => {
    const { rerender, result } = renderHook(useProfileSwitchLatch, {
      initialProps: { dataUpdatedAt: 1, errorUpdatedAt: 0 }
    })

    act(() => result.current.arm())
    rerender({ dataUpdatedAt: 1, errorUpdatedAt: 0 })
    expect(result.current.pending).toBe(true)

    rerender({ dataUpdatedAt: 1, errorUpdatedAt: 5 })
    expect(result.current.pending).toBe(false)
  })
})
