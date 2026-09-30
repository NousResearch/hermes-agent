import { describe, expect, it, vi } from 'vitest'

import {
  createSessionProfileAppearanceResolver,
  type ExactSessionOwner,
  normalizeExactSessionOwner,
  shouldPresentSessionAppearance
} from './session-profile-appearance'

const owner = (connectionId: string, profile = 'kite', targetProfile = profile): ExactSessionOwner => ({
  connectionId,
  profile,
  targetProfile
})

function mockRequest(avatar = 'data:image/png;base64,synthetic') {
  return vi.fn(async (_connectionId: string, _profile: string, method: string) => {
    if (method === 'profiles.list') {
      return {
        profiles: [
          {
            name: 'kite',
            display_name: 'Kite',
            has_avatar: true,
            title: 'TikTok Channel Steward'
          }
        ]
      }
    }

    return { data: avatar, found: true }
  })
}

describe('session profile appearance resolver', () => {
  it('fails closed on bare or incomplete owners and preserves the backend target profile', () => {
    expect(normalizeExactSessionOwner('kite')).toBeNull()
    expect(normalizeExactSessionOwner({ profile: 'kite' })).toBeNull()
    expect(normalizeExactSessionOwner({ connectionId: 'source-a', profile: 'desktop-kite', targetProfile: 'kite' })).toEqual({
      connectionId: 'source-a',
      profile: 'desktop-kite',
      targetProfile: 'kite'
    })
  })

  it('deduplicates concurrent exact-owner requests and isolates same-named profiles by connection', async () => {
    const request = mockRequest()
    const resolver = createSessionProfileAppearanceResolver(request)

    const [first, duplicate, remote] = await Promise.all([
      resolver.resolve(owner('source-a')),
      resolver.resolve(owner('source-a')),
      resolver.resolve(owner('source-b'))
    ])

    expect(first).toEqual(duplicate)
    expect(remote?.displayName).toBe('Kite')
    expect(request.mock.calls.filter(call => call[2] === 'profiles.list')).toHaveLength(2)
    expect(request.mock.calls.filter(call => call[2] === 'profiles.get_asset')).toHaveLength(2)
  })

  it('routes a routine-owned session only through its exact source-qualified owner', async () => {
    const request = mockRequest()
    const resolver = createSessionProfileAppearanceResolver(request)

    await resolver.resolve(owner('source-a', 'desktop-kite', 'kite'))

    expect(request).toHaveBeenNthCalledWith(1, 'source-a', 'desktop-kite', 'profiles.list', {})
    expect(request).toHaveBeenNthCalledWith(2, 'source-a', 'desktop-kite', 'profiles.get_asset', {
      asset: 'avatar',
      name: 'kite'
    })
  })

  it('invalidates true-to-true avatar replacement and purges only the removed source', async () => {
    const request = mockRequest('data:image/png;base64,first')
    const resolver = createSessionProfileAppearanceResolver(request)
    const local = owner('source-a')
    const remote = owner('source-b')

    await resolver.resolve(local)
    await resolver.resolve(remote)
    request.mockImplementation(async (_connectionId, _profile, method) =>
      method === 'profiles.list'
        ? ({ profiles: [{ name: 'kite', display_name: 'Kite', has_avatar: true, title: 'TikTok Channel Steward' }] } as never)
        : ({ data: 'data:image/png;base64,second', found: true } as never)
    )

    resolver.invalidateOwner(local)
    expect((await resolver.resolve(local))?.avatarDataUrl).toContain('second')
    expect((await resolver.resolve(remote))?.avatarDataUrl).toContain('first')

    resolver.purgeConnection('source-a')
    expect(resolver.peek(local)).toBeNull()
    expect(resolver.peek(remote)?.displayName).toBe('Kite')
  })

  it('reopens with current appearance while preserving owner and message bytes', async () => {
    const request = mockRequest('data:image/png;base64,first')
    const resolver = createSessionProfileAppearanceResolver(request)
    const exactOwner = owner('source-a', 'desktop-kite', 'kite')
    const ownerBytes = JSON.stringify(exactOwner)
    const messageBytes = JSON.stringify({ id: 'assistant-1', role: 'assistant', content: 'Synthetic reply' })

    expect((await resolver.resolve(exactOwner))?.displayName).toBe('Kite')
    request.mockImplementation(async (_connectionId, _profile, method) =>
      method === 'profiles.list'
        ? ({ profiles: [{ name: 'kite', display_name: 'Kite Current', has_avatar: false, title: 'Current Role' }] } as never)
        : ({ found: false } as never)
    )

    expect((await resolver.resolve(exactOwner, { revalidate: true }))?.displayName).toBe('Kite Current')
    expect(JSON.stringify(exactOwner)).toBe(ownerBytes)
    expect(JSON.stringify({ id: 'assistant-1', role: 'assistant', content: 'Synthetic reply' })).toBe(messageBytes)
  })

  it('fails quietly and returns a non-person fallback when metadata is missing or a request fails', async () => {
    const missing = createSessionProfileAppearanceResolver(async () => ({ profiles: [] }))
    const failed = createSessionProfileAppearanceResolver(async () => {
      throw new Error('synthetic failure')
    })

    expect(await missing.resolve(owner('source-a'))).toBeNull()
    await expect(failed.resolve(owner('source-a'))).resolves.toBeNull()
  })

  it('retains exact state while disconnected but requires reconnect revalidation before presentation', () => {
    const key = JSON.stringify(['source-a', 'kite', 'kite', 'avatar'])

    expect(shouldPresentSessionAppearance('closed', key, key)).toBe(true)
    expect(shouldPresentSessionAppearance('open', '', key)).toBe(false)
    expect(shouldPresentSessionAppearance('open', key, key)).toBe(true)
    expect(shouldPresentSessionAppearance('open', key, JSON.stringify(['source-b', 'kite', 'kite', 'avatar']))).toBe(
      false
    )
  })

  it('does not resurrect purged appearance state when an older request settles', async () => {
    let releaseList!: (value: unknown) => void
    const listResult = new Promise(resolve => {
      releaseList = resolve
    })
    const request = vi.fn(async (_connectionId: string, _profile: string, method: string) => {
      if (method === 'profiles.list') {
        return listResult
      }

      return { data: 'data:image/png;base64,stale', found: true }
    })
    const resolver = createSessionProfileAppearanceResolver(request)
    const exactOwner = owner('source-a')
    const stale = resolver.resolve(exactOwner)

    resolver.purgeConnection('source-a')
    releaseList({
      profiles: [{ name: 'kite', display_name: 'Stale Kite', has_avatar: true, title: 'Stale Role' }]
    })

    await expect(stale).resolves.toBeNull()
    expect(resolver.peek(exactOwner)).toBeNull()
    expect(request.mock.calls.filter(call => call[2] === 'profiles.get_asset')).toHaveLength(0)
  })

  it('does not let a purged request delete or overwrite a fresh concurrent request', async () => {
    let releaseStale!: (value: unknown) => void
    const staleList = new Promise(resolve => {
      releaseStale = resolve
    })
    let listCalls = 0
    const request = vi.fn(async (_connectionId: string, _profile: string, method: string) => {
      if (method === 'profiles.list') {
        listCalls += 1

        if (listCalls === 1) {
          return staleList
        }

        return {
          profiles: [{ name: 'kite', display_name: 'Fresh Kite', has_avatar: false, title: 'Fresh Role' }]
        }
      }

      return { found: false }
    })
    const resolver = createSessionProfileAppearanceResolver(request)
    const exactOwner = owner('source-a')
    const stale = resolver.resolve(exactOwner)

    resolver.purgeConnection('source-a')
    const fresh = resolver.resolve(exactOwner)
    releaseStale({
      profiles: [{ name: 'kite', display_name: 'Stale Kite', has_avatar: false, title: 'Stale Role' }]
    })

    await expect(stale).resolves.toBeNull()
    await expect(fresh).resolves.toMatchObject({ displayName: 'Fresh Kite', role: 'Fresh Role' })
    expect(resolver.peek(exactOwner)).toMatchObject({ displayName: 'Fresh Kite', role: 'Fresh Role' })
    expect(await resolver.resolve(exactOwner)).toMatchObject({ displayName: 'Fresh Kite' })
    expect(listCalls).toBe(2)
  })
})
