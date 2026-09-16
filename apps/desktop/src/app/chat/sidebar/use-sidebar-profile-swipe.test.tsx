import { renderHook } from '@testing-library/react'
import type { RefObject } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useSidebarProfileSwipe } from './use-sidebar-profile-swipe'

const { cycleProfile, triggerHaptic } = vi.hoisted(() => ({
  cycleProfile: vi.fn(),
  triggerHaptic: vi.fn()
}))

vi.mock('@/store/profile', () => ({ cycleProfile }))
vi.mock('@/lib/haptics', () => ({ triggerHaptic }))

const dispatchWheel = (target: HTMLElement, deltaX: number, deltaY = 0) => {
  target.dispatchEvent(new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaX, deltaY }))
}

afterEach(() => {
  cycleProfile.mockClear()
  triggerHaptic.mockClear()
})

describe('useSidebarProfileSwipe', () => {
  it('cycles from accumulated sidebar motion but preserves rail and vertical wheel gestures', () => {
    const sidebar = globalThis.document.createElement('div')
    const child = globalThis.document.createElement('div')
    sidebar.append(child)
    const ref = { current: sidebar } as RefObject<HTMLElement>
    const { unmount } = renderHook(() => useSidebarProfileSwipe(ref, true))

    dispatchWheel(sidebar, 20)
    dispatchWheel(sidebar, 16)
    expect(cycleProfile).toHaveBeenCalledOnce()
    expect(cycleProfile).toHaveBeenCalledWith(1)
    expect(triggerHaptic).toHaveBeenCalledWith('selection')

    child.dataset.slot = 'profile-rail'
    dispatchWheel(child, -60)
    dispatchWheel(sidebar, 0, 60)
    expect(cycleProfile).toHaveBeenCalledOnce()

    unmount()
    dispatchWheel(sidebar, -60)
    expect(cycleProfile).toHaveBeenCalledOnce()
  })
})
