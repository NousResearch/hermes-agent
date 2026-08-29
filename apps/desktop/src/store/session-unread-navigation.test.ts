import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { UnreadSessionTarget } from './session-dot-state'
import {
  $openNextUnreadRequest,
  openNextValidUnread,
  ownerRouteForUnreadTarget,
  preflightCandidatesForUnreadTarget,
  profileScopeForUnreadTarget,
  requestOpenNextUnread
} from './session-unread-navigation'

beforeEach(() => {
  $openNextUnreadRequest.set(0)
})

afterEach(() => {
  $openNextUnreadRequest.set(0)
})

describe('unread session navigation requests', () => {
  it('starts inert and increments once per request', () => {
    expect($openNextUnreadRequest.get()).toBe(0)

    requestOpenNextUnread()
    requestOpenNextUnread()

    expect($openNextUnreadRequest.get()).toBe(2)
  })
})

const target = (id: string, extra: Partial<UnreadSessionTarget> = {}): UnreadSessionTarget => ({
  id,
  kind: 'session',
  ...extra
})

describe('unread target navigation', () => {
  it.each(['session', 'cron', 'messaging'] as const)('opens a valid %s target', async kind => {
    const open = vi.fn()
    const item = target(kind, { kind })

    await expect(openNextValidUnread([item], async () => null, open)).resolves.toEqual(item)
    expect(open).toHaveBeenCalledWith(item, null)
  })

  it('uses the exact connection and profile for preflight and resume', async () => {
    const item = target('remote', { connectionId: ' conn-a ', profile: ' research ' })
    const route = { connectionId: 'conn-a', profile: 'research' }
    const open = vi.fn()

    expect(profileScopeForUnreadTarget(item)).toEqual(route)
    expect(ownerRouteForUnreadTarget(item)).toEqual(route)
    await openNextValidUnread([item], async candidate => ownerRouteForUnreadTarget(candidate), open)
    expect(open).toHaveBeenCalledWith(item, route)
  })

  it('pins an untagged primary row to the active registered gateway', () => {
    const item = target('primary', { profile: 'default' })
    const route = { connectionId: 'work-vps', profile: 'default' }

    expect(profileScopeForUnreadTarget(item, 'work-vps')).toEqual(route)
    expect(ownerRouteForUnreadTarget(item, 'work-vps')).toEqual(route)
  })

  it('uses the backend target profile from the exact registered route', () => {
    const item = target('primary', { profile: 'claude-martin' })

    const routes = [
      {
        connectionId: 'work-vps',
        mode: 'remote' as const,
        profile: 'claude-martin',
        targetProfile: 'default'
      }
    ]

    expect(profileScopeForUnreadTarget(item, 'work-vps', routes)).toEqual({
      connectionId: 'work-vps',
      profile: 'default'
    })
    expect(ownerRouteForUnreadTarget(item, 'work-vps', routes)).toEqual({
      connectionId: 'work-vps',
      mode: 'remote',
      profile: 'claude-martin',
      targetProfile: 'default'
    })
  })

  it('ignores a same-named profile route from another connection', () => {
    const item = target('primary', { profile: 'work' })
    const routes = [{ connectionId: 'other', profile: 'work', targetProfile: 'default' }]

    expect(profileScopeForUnreadTarget(item, 'work-vps', routes)).toEqual({
      connectionId: 'work-vps',
      profile: 'work'
    })
  })

  it('tries profile routing before the active registry fallback for a primary row', () => {
    const item = target('primary', { profile: 'claude-martin' })

    const routes = [
      {
        connectionId: 'work-vps',
        mode: 'remote' as const,
        profile: 'claude-martin',
        targetProfile: 'default'
      }
    ]

    expect(preflightCandidatesForUnreadTarget(item, 'work-vps', routes)).toEqual([
      { ownerRoute: null, scope: 'claude-martin' },
      {
        ownerRoute: {
          connectionId: 'work-vps',
          mode: 'remote',
          profile: 'claude-martin',
          targetProfile: 'default'
        },
        scope: { connectionId: 'work-vps', profile: 'default' }
      }
    ])
  })

  it('uses only the exact owner for a tagged secondary row', () => {
    const item = target('secondary', { connectionId: 'homelab', profile: 'research' })

    expect(preflightCandidatesForUnreadTarget(item, 'work-vps')).toEqual([
      {
        ownerRoute: { connectionId: 'homelab', profile: 'research' },
        scope: { connectionId: 'homelab', profile: 'research' }
      }
    ])
  })

  it('prefers a row-owned connection over the active gateway fallback', () => {
    const item = target('secondary', { connectionId: 'homelab', profile: 'research' })
    const route = { connectionId: 'homelab', profile: 'research' }

    expect(profileScopeForUnreadTarget(item, 'work-vps')).toEqual(route)
    expect(ownerRouteForUnreadTarget(item, 'work-vps')).toEqual(route)
  })

  it('keeps a stale newest target unread and falls through to the next', async () => {
    const stale = target('stale', { kind: 'cron' })
    const next = target('next', { kind: 'messaging' })
    const open = vi.fn()

    const opened = await openNextValidUnread(
      [stale, next],
      async candidate => {
        if (candidate === stale) {
          throw new Error('not found')
        }

        return null
      },
      open
    )

    expect(opened).toEqual(next)
    expect(open).toHaveBeenCalledOnce()
    expect(open).toHaveBeenCalledWith(next, null)
  })

  it('does not open or mutate anything when every target is unavailable', async () => {
    const open = vi.fn()

    await expect(
      openNextValidUnread([target('a'), target('b')], async () => {
        throw new Error('offline')
      }, open)
    ).resolves.toBeNull()
    expect(open).not.toHaveBeenCalled()
  })

  it('does not swallow an opening failure and continue to a second target', async () => {
    const open = vi.fn(() => {
      throw new Error('open failed')
    })

    await expect(openNextValidUnread([target('first'), target('second')], async () => null, open)).rejects.toThrow(
      'open failed'
    )
    expect(open).toHaveBeenCalledOnce()
  })

  it('keeps profile-only and ambient preflight scopes distinct', () => {
    expect(profileScopeForUnreadTarget(target('profile', { profile: 'writer' }))).toBe('writer')
    expect(profileScopeForUnreadTarget(target('ambient'))).toBeUndefined()
  })
})
