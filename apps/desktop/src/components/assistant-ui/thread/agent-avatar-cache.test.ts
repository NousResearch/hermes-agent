import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $gateway } from '@/store/gateway'

import { agentAvatarCache, resolveAgentAvatar } from './user-message'

// The avatar cache is keyed by a handle parsed out of message text — unbounded
// distinct senders over a long session — and its hits hold base64 avatar data
// URLs, so it must not grow for the life of the renderer window. This pins the
// bound and the negative-entry TTL: an evicted or expired entry only costs a
// refetch, never correctness.
const CACHE_MAX = 128
const MISS_TTL_MS = 30_000
const HIT_TTL_MS = 60_000

const clearCache = () => {
  for (const key of [...agentAvatarCache.keys()]) {
    agentAvatarCache.delete(key)
  }
}

const withAvatar = new Set<string>()
const avatarRevs = new Map<string, string>()
let listCalls = 0
let assetCalls = 0

const stubGateway = () => {
  listCalls = 0
  assetCalls = 0
  $gateway.set({
    request: async (method: string, params: Record<string, unknown>) => {
      if (method === 'profiles.list') {
        listCalls += 1

        return {
          profiles: [...withAvatar].map(name => ({ avatar_rev: avatarRevs.get(name) ?? null, has_avatar: true, name }))
        }
      }

      if (method === 'profiles.get_asset') {
        assetCalls += 1
        const name = String(params.name)

        return { data: `data:image/png;base64,${name}${avatarRevs.get(name) ?? ''}`, found: true }
      }

      throw new Error(`unexpected request: ${method}`)
    }
  } as never)
}

describe('agent avatar cache', () => {
  beforeEach(() => {
    clearCache()
    withAvatar.clear()
    avatarRevs.clear()
    stubGateway()
  })

  afterEach(() => {
    $gateway.set(null)
    vi.restoreAllMocks()
  })

  it('holds at most 128 handles, evicting the least recently used', async () => {
    for (let i = 0; i <= CACHE_MAX; i += 1) {
      await resolveAgentAvatar(`bot${i}`)
    }

    expect(agentAvatarCache.size).toBe(CACHE_MAX)
    expect(agentAvatarCache.has('bot0')).toBe(false)
    expect(agentAvatarCache.has(`bot${CACHE_MAX}`)).toBe(true)
  })

  it('honours a negative entry inside the TTL and re-probes once it expires', async () => {
    const now = vi.spyOn(Date, 'now').mockReturnValue(1_000_000)

    expect(await resolveAgentAvatar('ghost')).toBeNull()
    expect(listCalls).toBe(1)

    // Inside the TTL the miss is served from the cache, not re-probed.
    expect(await resolveAgentAvatar('ghost')).toBeNull()
    expect(listCalls).toBe(1)

    // The art backfill lands: the expired miss must re-probe and pick it up.
    withAvatar.add('ghost')
    now.mockReturnValue(1_000_000 + MISS_TTL_MS + 1)

    expect(await resolveAgentAvatar('ghost')).toBe('data:image/png;base64,ghost')
    expect(listCalls).toBe(2)
  })

  it('revalidates a hit after its TTL and re-downloads only when avatar_rev moved', async () => {
    const now = vi.spyOn(Date, 'now').mockReturnValue(2_000_000)
    withAvatar.add('tutor')
    avatarRevs.set('tutor', 'r1')

    expect(await resolveAgentAvatar('tutor')).toBe('data:image/png;base64,tutorr1')
    expect([listCalls, assetCalls]).toEqual([1, 1])

    // Inside the hit TTL: no gateway traffic at all.
    expect(await resolveAgentAvatar('tutor')).toBe('data:image/png;base64,tutorr1')
    expect([listCalls, assetCalls]).toEqual([1, 1])

    // TTL expired, same rev: one roster probe, no image download.
    now.mockReturnValue(2_000_000 + HIT_TTL_MS + 1)
    expect(await resolveAgentAvatar('tutor')).toBe('data:image/png;base64,tutorr1')
    expect([listCalls, assetCalls]).toEqual([2, 1])

    // The avatar file was replaced: the next revalidation picks up the new art.
    avatarRevs.set('tutor', 'r2')
    now.mockReturnValue(2_000_000 + 2 * (HIT_TTL_MS + 1))
    expect(await resolveAgentAvatar('tutor')).toBe('data:image/png;base64,tutorr2')
    expect([listCalls, assetCalls]).toEqual([3, 2])
  })
})
