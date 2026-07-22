import { act, cleanup, renderHook } from '@testing-library/react'
import { StrictMode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'

import { useOnProfileSwitch } from './use-on-profile-switch'

describe('useOnProfileSwitch', () => {
  beforeEach(() => {
    $activeGatewayProfile.set('default')
  })

  afterEach(() => {
    cleanup()
    $activeGatewayProfile.set('default')
  })

  it('does not fire for StrictMode effect replay when the profile did not change', () => {
    const onSwitch = vi.fn()

    const { rerender } = renderHook(() => useOnProfileSwitch(onSwitch), {
      wrapper: StrictMode
    })

    rerender()

    expect(onSwitch).not.toHaveBeenCalled()
  })

  it('fires once when the active gateway profile changes', () => {
    const onSwitch = vi.fn()

    renderHook(() => useOnProfileSwitch(onSwitch), {
      wrapper: StrictMode
    })

    act(() => {
      $activeGatewayProfile.set('coder')
    })

    expect(onSwitch).toHaveBeenCalledTimes(1)
  })
})
