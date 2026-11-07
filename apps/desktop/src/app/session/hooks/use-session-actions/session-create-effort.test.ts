import { atom } from 'nanostores'
import { afterEach, describe, expect, it, vi } from 'vitest'

vi.mock('@/store/gateway', () => ({ requestGatewayForAgent: vi.fn() }))

vi.mock('@/store/profile', () => ({
  $activeGatewayProfile: atom('default'),
  $newChatProfile: atom<null | string>(null),
  ensureGatewayAgent: vi.fn(async () => undefined),
  ensureGatewayProfile: vi.fn(async () => undefined),
  normalizeProfileKey: (profile: string) => profile,
  resolveNewChatOwnerRoute: () => null
}))

import { setCurrentReasoningEffort, setDefaultReasoningEffort } from '@/store/session'

import { desktopSessionCreateParams } from './session-create-request'

// The composer is seeded from the profile default: shipping that seed would make every new
// chat an explicit pick (adaptive reasoning off, per-model reasoning_overrides clobbered).
describe('desktopSessionCreateParams reasoning effort', () => {
  afterEach(() => {
    setCurrentReasoningEffort('')
    setDefaultReasoningEffort('')
  })

  it('omits an effort that only mirrors the profile default', async () => {
    setDefaultReasoningEffort('high')
    setCurrentReasoningEffort('high')

    expect(await desktopSessionCreateParams('/w')).not.toHaveProperty('reasoning_effort')
  })

  it('ships a distinct pick', async () => {
    setDefaultReasoningEffort('high')
    setCurrentReasoningEffort('low')

    expect(await desktopSessionCreateParams('/w')).toMatchObject({ reasoning_effort: 'low' })
  })
})
