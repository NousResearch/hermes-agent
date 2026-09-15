import { describe, expect, it, vi } from 'vitest'

import { isPointerInsideBounds, routeNativeSwipe } from './sidebar-native-swipe'

describe('sidebar native swipe routing', () => {
  it('uses live geometry for a stationary pointer after the sidebar renders', () => {
    const bounds = { bottom: 600, left: 0, right: 320, top: 20 }

    expect(isPointerInsideBounds({ clientX: 160, clientY: 300 }, bounds)).toBe(true)
    expect(isPointerInsideBounds({ clientX: 321, clientY: 300 }, bounds)).toBe(false)
  })

  it('gives the sidebar ownership while preserving preview navigation elsewhere', () => {
    const cycleProfile = vi.fn()
    const navigatePreview = vi.fn()

    expect(routeNativeSwipe('right', true, { cycleProfile, navigatePreview })).toBe('profile')
    expect(cycleProfile).toHaveBeenCalledWith(1)
    expect(navigatePreview).not.toHaveBeenCalled()

    expect(routeNativeSwipe('left', false, { cycleProfile, navigatePreview })).toBe('preview')
    expect(navigatePreview).toHaveBeenCalledWith('back')
  })
})
