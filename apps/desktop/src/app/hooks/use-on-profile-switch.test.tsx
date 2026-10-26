import { act, cleanup, renderHook } from '@testing-library/react'
import { atom } from 'nanostores'
import { StrictMode } from 'react'
import { afterEach, expect, it, vi } from 'vitest'

vi.mock('@/store/profile', () => ({ $activeGatewayProfile: atom('default') }))

import { $activeGatewayProfile } from '@/store/profile'

import { useOnProfileSwitch } from './use-on-profile-switch'

afterEach(cleanup)

it('ignores StrictMode effect replay and reacts only to real profile changes', () => {
  const onSwitch = vi.fn()
  const { rerender } = renderHook(() => useOnProfileSwitch(onSwitch), { wrapper: StrictMode })
  expect(onSwitch).not.toHaveBeenCalled()
  rerender()
  expect(onSwitch).not.toHaveBeenCalled()
  act(() => $activeGatewayProfile.set('other'))
  expect(onSwitch).toHaveBeenCalledTimes(1)
  act(() => $activeGatewayProfile.set('default'))
  expect(onSwitch).toHaveBeenCalledTimes(2)
})
