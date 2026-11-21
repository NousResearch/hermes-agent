import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'

import { en } from '@/i18n/en'
import { $freeTierStatus } from '@/store/free-tier'
import { $freeTierSignIn } from '@/store/free-tier-sign-in'
import { setInterfaceMode } from '@/store/interface-mode'
import { markOnboardingDecided } from '@/store/onboarding-presence'
import type { FreeTierStatus } from '@/types/hermes'

import { PortalFreeTierSignIn } from './portal-free-tier-sign-in'

afterEach(() => {
  cleanup()
  $freeTierStatus.set(null)
  $freeTierSignIn.set({ status: 'closed' })
  setInterfaceMode('advanced')
})

it.each(['simple', 'advanced'] as const)(
  'keeps the free tier Sign in on the rail in %s mode and opens the one sign-in dialog',
  mode => {
    setInterfaceMode(mode)
    markOnboardingDecided()
    render(<PortalFreeTierSignIn />)
    expect(screen.queryByRole('button', { name: en.freeTier.signIn })).toBeNull()

    act(() => $freeTierStatus.set({ available: true, has_guest: true } as FreeTierStatus))
    fireEvent.click(screen.getByRole('button', { name: en.freeTier.signIn }))
    expect($freeTierSignIn.get().status).toBe('requested')

    act(() => $freeTierStatus.set({ available: false, has_guest: false } as FreeTierStatus))
    expect(screen.queryByRole('button', { name: en.freeTier.signIn })).toBeNull()
  }
)
