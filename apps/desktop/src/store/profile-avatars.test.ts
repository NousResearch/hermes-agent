import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { ProfileInfo } from '@/types/hermes'

const getProfileAvatar = vi.fn()

// profile.ts pulls these from @/hermes; only getProfileAvatar matters here.
// HermesGateway is a minimal stand-in for the routing-follow tests, where
// selectProfile() opens a secondary socket for the target profile (the real
// connect is skipped — window.hermesDesktop doesn't exist under jsdom).
vi.mock('@/hermes', () => ({
  getProfileAvatar: (name: string) => getProfileAvatar(name),
  getProfiles: vi.fn(),
  setApiRequestProfile: vi.fn(),
  HermesGateway: class {
    connectionState = 'closed'
    onEvent() {
      return () => {}
    }
    onState() {
      return () => {}
    }
    async connect() {}
    close() {}
  }
}))

import {
  $activeGatewayProfile,
  $activeProfile,
  $newChatProfile,
  $profileAvatars,
  $profileColors,
  $profileOrder,
  $profiles,
  ensureProfileAvatars,
  removeProfileLocal,
  renameProfileLocal,
  setProfileAvatarLocal
} from './profile'

function profile(name: string, overrides: Partial<ProfileInfo> = {}): ProfileInfo {
  return {
    avatar_updated_at: null,
    has_avatar: false,
    has_env: false,
    is_default: name === 'default',
    model: null,
    name,
    path: `/profiles/${name}`,
    provider: null,
    skill_count: 0,
    ...overrides
  }
}

describe('ensureProfileAvatars', () => {
  beforeEach(() => {
    $profileAvatars.set({})
    getProfileAvatar.mockReset()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('fetches and caches avatars only for profiles that have one', async () => {
    getProfileAvatar.mockResolvedValue({ data_url: 'data:image/png;base64,AAA', exists: true })

    ensureProfileAvatars([
      profile('alpha', { has_avatar: true, avatar_updated_at: 1 }),
      profile('beta') // no avatar → no fetch
    ])
    await vi.waitFor(() => expect($profileAvatars.get().alpha).toBe('data:image/png;base64,AAA'))

    expect(getProfileAvatar).toHaveBeenCalledTimes(1)
    expect(getProfileAvatar).toHaveBeenCalledWith('alpha')
    expect($profileAvatars.get().beta).toBeUndefined()
  })

  it('does not refetch when the version is unchanged but refetches when it changes', async () => {
    // Distinct name from other tests — the version counter is module-level and
    // persists across tests in this file.
    getProfileAvatar.mockResolvedValue({ data_url: 'data:image/png;base64,V1', exists: true })

    const p1 = profile('verprof', { has_avatar: true, avatar_updated_at: 1 })
    ensureProfileAvatars([p1])
    await vi.waitFor(() => expect($profileAvatars.get().verprof).toBe('data:image/png;base64,V1'))

    // Same version → cache hit, no new request.
    ensureProfileAvatars([p1])
    expect(getProfileAvatar).toHaveBeenCalledTimes(1)

    // Bumped version → refetch.
    getProfileAvatar.mockResolvedValue({ data_url: 'data:image/png;base64,V2', exists: true })
    ensureProfileAvatars([profile('verprof', { has_avatar: true, avatar_updated_at: 2 })])
    await vi.waitFor(() => expect($profileAvatars.get().verprof).toBe('data:image/png;base64,V2'))
    expect(getProfileAvatar).toHaveBeenCalledTimes(2)
  })

  it('drops cached avatars for profiles that vanish or lose their picture', async () => {
    setProfileAvatarLocal('alpha', 'data:image/png;base64,AAA')
    setProfileAvatarLocal('beta', 'data:image/png;base64,BBB')
    expect(Object.keys($profileAvatars.get()).sort()).toEqual(['alpha', 'beta'])

    // alpha is gone from the list, beta still exists but lost its picture.
    ensureProfileAvatars([profile('beta', { has_avatar: false })])

    expect($profileAvatars.get().alpha).toBeUndefined()
    expect($profileAvatars.get().beta).toBeUndefined()
  })
})

describe('removeProfileLocal', () => {
  afterEach(() => {
    $activeGatewayProfile.set('default')
    $activeProfile.set('default')
    $newChatProfile.set(null)
  })

  it('drops the profile from the cached list and clears its avatar', () => {
    $profiles.set([profile('default'), profile('guy', { has_avatar: true })])
    setProfileAvatarLocal('guy', 'data:image/png;base64,AAA')

    removeProfileLocal('guy')

    expect($profiles.get().map(p => p.name)).toEqual(['default'])
    expect($profileAvatars.get().guy).toBeUndefined()
  })

  it('falls back to default when the deleted profile was the routed one', async () => {
    // The gateway, statusbar pill, and new-chat target all point at the
    // doomed profile — deletion must unwind every one of them, or the
    // primary reconnect loop keeps dialing a backend that can't come back.
    $profiles.set([profile('default'), profile('guy')])
    $activeGatewayProfile.set('guy')
    $activeProfile.set('guy')
    $newChatProfile.set('guy')

    removeProfileLocal('guy')

    expect($activeProfile.get()).toBe('default')
    // selectProfile points new chats at default rather than clearing outright.
    expect($newChatProfile.get()).toBe('default')
    // The gateway swap settles async (socket ensure → active pointer flip).
    await vi.waitFor(() => expect($activeGatewayProfile.get()).toBe('default'))
  })

  it('leaves routing alone when the deleted profile was not active', () => {
    $profiles.set([profile('default'), profile('alpha'), profile('beta')])
    $activeGatewayProfile.set('alpha')
    $activeProfile.set('alpha')
    $newChatProfile.set('alpha')

    removeProfileLocal('beta')

    expect($activeGatewayProfile.get()).toBe('alpha')
    expect($activeProfile.get()).toBe('alpha')
    expect($newChatProfile.get()).toBe('alpha')
    expect($profiles.get().map(p => p.name)).toEqual(['default', 'alpha'])
  })
})

describe('renameProfileLocal', () => {
  afterEach(() => {
    $activeGatewayProfile.set('default')
    $activeProfile.set('default')
    $newChatProfile.set(null)
    $profileColors.set({})
    $profileOrder.set([])
  })

  it('renames the cached entry and carries color, order slot, and avatar across', () => {
    $profiles.set([profile('default'), profile('old-name', { has_avatar: true })])
    $profileColors.set({ 'old-name': 'hsl(120 68% 58%)' })
    $profileOrder.set(['old-name', 'other'])
    setProfileAvatarLocal('old-name', 'data:image/png;base64,AAA')

    renameProfileLocal('old-name', 'new-name')

    expect($profiles.get().map(p => p.name)).toEqual(['default', 'new-name'])
    expect($profileColors.get()).toEqual({ 'new-name': 'hsl(120 68% 58%)' })
    expect($profileOrder.get()).toEqual(['new-name', 'other'])
    expect($profileAvatars.get()['old-name']).toBeUndefined()
    expect($profileAvatars.get()['new-name']).toBe('data:image/png;base64,AAA')
  })

  it('follows routing to the new name when the renamed profile was active', async () => {
    $profiles.set([profile('default'), profile('old-name')])
    $activeGatewayProfile.set('old-name')
    $activeProfile.set('old-name')
    $newChatProfile.set('old-name')

    renameProfileLocal('old-name', 'new-name')

    expect($activeProfile.get()).toBe('new-name')
    expect($newChatProfile.get()).toBe('new-name')
    await vi.waitFor(() => expect($activeGatewayProfile.get()).toBe('new-name'))
  })
})
