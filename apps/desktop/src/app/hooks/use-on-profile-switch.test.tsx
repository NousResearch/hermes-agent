import { act, cleanup, render } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'

const activeProfile = atom('default')

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: activeProfile
}))

beforeEach(() => {
  activeProfile.set('default')
})

afterEach(() => {
  cleanup()
})

it('invalidates profile-local state synchronously when the profile atom changes', async () => {
  const onSwitch = vi.fn()
  const { useOnProfileSwitch } = await import('./use-on-profile-switch')

  function Probe() {
    useOnProfileSwitch(onSwitch)

    return null
  }

  render(<Probe />)
  expect(onSwitch).not.toHaveBeenCalled()

  act(() => {
    activeProfile.set('worker')
    expect(onSwitch).toHaveBeenCalledTimes(1)
  })
})
