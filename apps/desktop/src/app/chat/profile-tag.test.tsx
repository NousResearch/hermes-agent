import { cleanup, render, screen } from '@testing-library/react'
import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

// Keep store/profile's side-effecting imports inert (gateway socket layer +
// REST client) — same seam as store/profile.test.ts.
vi.mock('@/store/gateway', () => ({
  $gateway: atom<unknown>(null),
  ensureGatewayForProfile: vi.fn(async () => undefined)
}))
vi.mock('@/hermes', () => ({
  getProfiles: vi.fn(async () => ({ profiles: [] })),
  setApiRequestProfile: vi.fn()
}))
vi.mock('@/lib/query-client', () => ({ queryClient: { invalidateQueries: vi.fn() } }))
vi.mock('@/store/starmap', () => ({ resetStarmapGraph: vi.fn() }))

const { ProfileTag } = await import('./profile-tag')
const { $profiles, setProfileColor } = await import('@/store/profile')

afterEach(cleanup)

describe('ProfileTag', () => {
  it('shows the profile initial with an accessible owner label', () => {
    render(<ProfileTag profile="xavier" />)

    const tag = screen.getByRole('img', { name: 'Profile: xavier' })
    expect(tag.textContent).toBe('x')
  })

  it('uses a cached display name for a renamed profile badge', () => {
    $profiles.set([{ display_name: 'Homelab', name: 'it-homelab' }] as never)

    render(<ProfileTag profile="it-homelab" />)

    const tag = screen.getByRole('img', { name: 'Profile: Homelab (it-homelab)' })
    expect(tag.textContent).toBe('H')
  })

  it('normalizes an empty profile to default, which shows the home glyph', () => {
    render(<ProfileTag profile="" />)

    const tag = screen.getByRole('img', { name: 'Profile: default' })
    // The default profile has no initial and no identity color — it is the
    // home icon, same as the profiles panel and the rail.
    expect(tag.textContent).toBe('')
    expect(tag.querySelector('.codicon-home')).not.toBeNull()
    expect(tag.style.color).toBe('')
  })

  it('uses the profile identity color (user override wins)', () => {
    setProfileColor('xavier', 'hsl(120 68% 58%)')

    render(<ProfileTag profile="xavier" />)

    const tag = screen.getByRole('img', { name: 'Profile: xavier' })
    // jsdom normalizes hsl() to rgb(); assert the override landed, not the format.
    expect(tag.style.color).toBe('rgb(75, 221, 75)')
  })
})
