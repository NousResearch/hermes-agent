import { act, renderHook } from '@testing-library/react'
import { StrictMode } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'

import { useOnProfileSwitch } from './use-on-profile-switch'

afterEach(() => {
  $activeGatewayProfile.set('default')
})

describe('useOnProfileSwitch', () => {
  it('does not fire on mount, including under StrictMode double-invoke', () => {
    const onSwitch = vi.fn()

    renderHook(() => useOnProfileSwitch(onSwitch), {
      wrapper: StrictMode
    })

    expect(onSwitch).not.toHaveBeenCalled()
  })

  it('fires when the active gateway profile actually changes', () => {
    const onSwitch = vi.fn()

    renderHook(() => useOnProfileSwitch(onSwitch), {
      wrapper: StrictMode
    })

    act(() => {
      $activeGatewayProfile.set('coder')
    })

    expect(onSwitch).toHaveBeenCalledTimes(1)
  })

  it('does not fire when the profile atom is set to the same value', () => {
    const onSwitch = vi.fn()

    renderHook(() => useOnProfileSwitch(onSwitch), {
      wrapper: StrictMode
    })

    act(() => {
      $activeGatewayProfile.set('default')
    })

    expect(onSwitch).not.toHaveBeenCalled()
  })

  it('does not fire when the raw value changes but the normalized key does not', () => {
    const onSwitch = vi.fn()

    renderHook(() => useOnProfileSwitch(onSwitch), {
      wrapper: StrictMode
    })

    // '' and ' default ' both normalize to 'default' — not a real switch.
    act(() => {
      $activeGatewayProfile.set('')
    })
    act(() => {
      $activeGatewayProfile.set(' default ')
    })

    expect(onSwitch).not.toHaveBeenCalled()
  })
})
