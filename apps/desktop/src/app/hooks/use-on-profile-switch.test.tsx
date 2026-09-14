import { StrictMode } from 'react'
import { act, renderHook } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { useOnProfileSwitch } from '@/app/hooks/use-on-profile-switch'
import { $activeGatewayProfile } from '@/store/profile'

describe('useOnProfileSwitch', () => {
  afterEach(() => {
    $activeGatewayProfile.set('default')
  })

  it('does not treat StrictMode effect replay as a profile switch', () => {
    const onSwitch = vi.fn()

    renderHook(() => useOnProfileSwitch(onSwitch), {
      wrapper: StrictMode
    })

    expect(onSwitch).not.toHaveBeenCalled()
  })

  it('fires once when the active profile actually changes', async () => {
    const onSwitch = vi.fn()

    renderHook(() => useOnProfileSwitch(onSwitch), {
      wrapper: StrictMode
    })

    await act(async () => {
      $activeGatewayProfile.set('profile-b')
    })

    expect(onSwitch).toHaveBeenCalledTimes(1)
  })
})
