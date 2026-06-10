import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

// profile.ts imports from @/hermes; stub it so the store module loads cleanly.
vi.mock('@/hermes', () => ({
  getProfileAvatar: vi.fn(),
  getProfiles: vi.fn(),
  setApiRequestProfile: vi.fn()
}))

import { $profileAvatars } from '@/store/profile'

import { ProfileAvatar } from './profile-avatar'

describe('ProfileAvatar', () => {
  beforeEach(() => {
    $profileAvatars.set({})
  })

  afterEach(() => {
    cleanup()
  })

  it('renders the cached picture when one exists', () => {
    $profileAvatars.set({ writer: 'data:image/png;base64,AAA' })
    const { container } = render(<ProfileAvatar name="writer" />)

    // alt="" makes the avatar decorative (no "img" role), so query by tag.
    const img = container.querySelector('img')
    expect(img?.getAttribute('src')).toBe('data:image/png;base64,AAA')
  })

  it('falls back to the uppercased initial when there is no picture', () => {
    const { container } = render(<ProfileAvatar name="writer" />)

    expect(container.querySelector('img')).toBeNull()
    expect(screen.getByText('W')).toBeTruthy()
  })
})
