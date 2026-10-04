import { webcrypto } from 'node:crypto'

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import { $activeGatewayProfile } from '@/store/profile'
import { $profileLocks, setProfilePasscode, $unlockedProfile } from '@/store/profile-lock'

import { ProfileLockGate } from './profile-lock-gate'

if (!globalThis.crypto?.subtle) {
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: webcrypto, writable: true })
}

function renderGate() {
  return render(
    <I18nProvider configClient={null} initialLocale="en">
      <ProfileLockGate />
    </I18nProvider>
  )
}

describe('ProfileLockGate', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $profileLocks.set({})
    $unlockedProfile.set(null)
    $activeGatewayProfile.set('default')
  })

  afterEach(() => {
    cleanup()
  })

  it('renders nothing for an unlocked profile', () => {
    renderGate()
    expect(screen.queryByLabelText('Passcode')).toBeNull()
  })

  it('covers the app with a passcode prompt for a locked profile at launch', async () => {
    await setProfilePasscode('work', '1234')
    $activeGatewayProfile.set('work')
    renderGate()
    expect(screen.getByLabelText('Passcode')).toBeTruthy()
    expect(screen.getByText('Unlock')).toBeTruthy()
  })

  it('shows an error on a wrong passcode and stays locked', async () => {
    await setProfilePasscode('work', '1234')
    $activeGatewayProfile.set('work')
    renderGate()

    fireEvent.change(screen.getByLabelText('Passcode'), { target: { value: 'nope' } })
    fireEvent.click(screen.getByText('Unlock'))

    await waitFor(() => expect(screen.getByText('Incorrect passcode. Try again.')).toBeTruthy())
    expect(screen.getByLabelText('Passcode')).toBeTruthy()
  })

  it('unlocks with the correct passcode and unmounts the gate', async () => {
    await setProfilePasscode('work', '1234')
    $activeGatewayProfile.set('work')
    renderGate()

    fireEvent.change(screen.getByLabelText('Passcode'), { target: { value: '1234' } })
    fireEvent.click(screen.getByText('Unlock'))

    await waitFor(() => expect(screen.queryByLabelText('Passcode')).toBeNull())
  })

  it('re-locks after switching to another profile and back', async () => {
    await setProfilePasscode('work', '1234')
    $activeGatewayProfile.set('work')
    renderGate()

    fireEvent.change(screen.getByLabelText('Passcode'), { target: { value: '1234' } })
    fireEvent.click(screen.getByText('Unlock'))
    await waitFor(() => expect(screen.queryByLabelText('Passcode')).toBeNull())

    // Leave the profile; the unlock must not follow.
    await act(async () => {
      $activeGatewayProfile.set('default')
    })
    expect(screen.queryByLabelText('Passcode')).toBeNull()

    // Come back — the gate re-prompts. The unlock-clear runs in the
    // noteProfileChange effect, so wait for it (React 19 + useStore).
    await act(async () => {
      $activeGatewayProfile.set('work')
    })
    await waitFor(() => expect(screen.getByLabelText('Passcode')).toBeTruthy())
  })
})
