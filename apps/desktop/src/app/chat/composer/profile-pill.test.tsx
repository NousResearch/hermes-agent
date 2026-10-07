// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  $activeGatewayProfile,
  $newChatConnectionId,
  $newChatProfile,
  $newChatRoute,
  $profileColors,
  $profileOrder,
  $profiles,
  $profilesByConnection
} from '@/store/profile'

import { ProfilePill } from './profile-pill'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      profiles: { title: 'Profile' },
      sidebar: { row: { ownedByProfile: (profile: string) => `Owned by ${profile}` } }
    }
  })
}))

const defaultProfile = {
  display_name: undefined,
  has_env: true,
  is_default: true,
  model: null,
  name: 'default',
  path: '/profiles/default',
  provider: null,
  skill_count: 0
}

const researchProfile = {
  display_name: 'Research desk',
  has_env: true,
  is_default: false,
  model: null,
  name: 'research',
  path: '/profiles/research',
  provider: null,
  skill_count: 0
}

function resetProfiles() {
  $activeGatewayProfile.set('default')
  $newChatConnectionId.set(null)
  $newChatProfile.set(null)
  $newChatRoute.set(null)
  $profileColors.set({})
  $profileOrder.set([])
  $profiles.set([defaultProfile, researchProfile])
  $profilesByConnection.set(new Map())
}

afterEach(() => {
  cleanup()
  resetProfiles()
})

describe('ProfilePill draft chooser', () => {
  it('lets an unsent draft choose a named profile without switching the active profile', async () => {
    resetProfiles()
    render(<ProfilePill mode="draft" owner={{ profile: 'default' }} />)

    const trigger = screen.getByRole('button', { name: 'Profile: default' })
    await act(async () => {
      fireEvent.pointerDown(trigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })
      await Promise.resolve()
    })

    const research = await screen.findByRole('menuitemradio', { name: /Research desk.*research/ })
    await act(async () => {
      fireEvent.click(research)
      await Promise.resolve()
    })

    expect($newChatProfile.get()).toBe('research')
    expect($activeGatewayProfile.get()).toBe('default')
    expect(screen.getByRole('button', { name: 'Profile: Research desk · research' })).toBeTruthy()

    const researchTrigger = screen.getByRole('button', { name: 'Profile: Research desk · research' })
    await act(async () => {
      fireEvent.pointerDown(researchTrigger, { button: 0, ctrlKey: false, pointerType: 'mouse' })
      await Promise.resolve()
    })

    const defaultOption = await screen.findByRole('menuitemradio', { name: 'default' })
    await act(async () => {
      fireEvent.click(defaultOption)
      await Promise.resolve()
    })

    expect($newChatProfile.get()).toBe('default')
    expect($activeGatewayProfile.get()).toBe('default')
  })

  it('supports selecting a profile by keyboard from the focused menu item', async () => {
    resetProfiles()
    render(<ProfilePill mode="draft" owner={{ profile: 'default' }} />)

    const trigger = screen.getByRole('button', { name: 'Profile: default' })
    await act(async () => {
      fireEvent.keyDown(trigger, { key: 'Enter', code: 'Enter', keyCode: 13 })
      await Promise.resolve()
    })

    const defaultOption = await screen.findByRole('menuitemradio', { name: 'default' })
    expect(globalThis.document.activeElement).toBe(defaultOption)

    await act(async () => {
      fireEvent.keyDown(defaultOption, { key: 'ArrowDown', code: 'ArrowDown', keyCode: 40 })
      await Promise.resolve()
    })

    const researchOption = screen.getByRole('menuitemradio', { name: /Research desk.*research/ })
    await waitFor(() => expect(globalThis.document.activeElement).toBe(researchOption))

    await act(async () => {
      fireEvent.keyDown(researchOption, { key: 'Enter', code: 'Enter', keyCode: 13 })
      await Promise.resolve()
    })

    expect($newChatProfile.get()).toBe('research')
    expect($activeGatewayProfile.get()).toBe('default')
  })

  it('shows the full immutable owner without a dropdown after the session starts', () => {
    resetProfiles()
    $newChatProfile.set('default')

    render(<ProfilePill mode="started" owner={{ profile: 'research' }} />)

    expect(screen.getByText('Research desk · research')).toBeTruthy()
    expect(screen.getByLabelText('Owned by Research desk · research')).toBeTruthy()
    expect(screen.queryByRole('button')).toBeNull()

    // Other draft-level state must not rewrite the selected session's owner.
    act(() => $newChatProfile.set('default'))
    expect(screen.getByText('Research desk · research')).toBeTruthy()
  })

  it('removes the selector affordance while first submission is in flight', () => {
    resetProfiles()
    $newChatProfile.set('research')

    render(<ProfilePill mode="submitting" owner={{ profile: 'default' }} />)

    expect(screen.getByText('Research desk · research')).toBeTruthy()
    expect(screen.queryByRole('button')).toBeNull()
  })
})
