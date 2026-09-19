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

const clearCache = () => {
  for (const key of [...agentAvatarCache.keys()]) {
    agentAvatarCache.delete(key)
  }
}

const withAvatar = new Set<string>()
let listCalls = 0

const stubGateway = () => {
  listCalls = 0
  $gateway.set({
    request: async (method: string, params: Record<string, unknown>) => {
      if (method === 'profiles.list') {
        listCalls += 1

        return { profiles: [...withAvatar].map(name => ({ has_avatar: true, name })) }
      }

      if (method === 'profiles.get_asset') {
        return { data: `data:image/png;base64,${String(params.name)}`, found: true }
      }

      throw new Error(`unexpected request: ${method}`)
    }
  } as never)
}

describe('agent avatar cache', () => {
  beforeEach(() => {
    clearCache()
    withAvatar.clear()
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

  it('serves a resolved avatar from the cache without another request', async () => {
    withAvatar.add('hermes')

    const first = await resolveAgentAvatar('hermes')
    const calls = listCalls

    expect(first).toBe('data:image/png;base64,hermes')
    expect(await resolveAgentAvatar('Hermes')).toBe(first)
    expect(listCalls).toBe(calls)
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
})
