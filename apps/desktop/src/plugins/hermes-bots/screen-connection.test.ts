/**
 * Two remote hosts can share the same `~/.hermes` path, so a `display.*` event
 * matched on `profile_key` alone would let host B's take-over repaint host A's
 * screen pane. The event must also have arrived on the bot's own connection.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type * as routing from './routing'
import type { RosterRow } from './types'

const routeMock = vi.fn<() => { connectionId: string; profile: string; targetProfile?: string } | null>(() => null)

vi.mock('@hermes/plugin-sdk', () => ({
  host: { requestProfile: vi.fn() },
  resolveSiblingWsUrl: vi.fn()
}))

vi.mock('./routing', async importOriginal => {
  const actual = await importOriginal<typeof routing>()

  // A mocked route stands in for a resolved registry row; without one the real resolver runs, so
  // an orphaned (`owner_removed`) row behaves exactly as it does in the app.
  const resolveBotConnectionRoute = (bot: RosterRow): ReturnType<typeof actual.resolveBotConnectionRoute> => {
    const route = routeMock()

    return route
      ? { status: 'resolved', route: { ...route, mode: 'remote', targetProfile: route.targetProfile ?? route.profile } }
      : actual.resolveBotConnectionRoute(bot)
  }

  return {
    ...actual,
    resolveBotConnectionRoute,
    botConnectionRoute: (bot: RosterRow) => {
      const resolved = resolveBotConnectionRoute(bot)

      if (resolved.status === 'owner_removed') {
        throw new Error(`Bot ${resolved.profile} has no connection owner`)
      }

      return resolved.route
    }
  }
})

import { host } from '@hermes/plugin-sdk'

import { displayRequest, isEventForBotScreen } from './screen-connection'

const bot = { name: 'ops' } as RosterRow
const orphan = { name: 'ops', remoteSource: true } as RosterRow
const key = '/home/hermes/.hermes'

describe('isEventForBotScreen', () => {
  it('ignores a same-profile-path event that arrived from another host', () => {
    routeMock.mockReturnValue({ connectionId: 'conn-a', profile: 'ops' })

    const fromB = { connectionId: 'conn-b', payload: { profile_key: key }, type: 'display.lease' as const }
    const fromA = { connectionId: 'conn-a', payload: { profile_key: key }, type: 'display.lease' as const }

    expect(isEventForBotScreen(bot, fromB, key)).toBe(false)
    expect(isEventForBotScreen(bot, fromA, key)).toBe(true)
  })

  it('still matches the untagged local socket for a local bot', () => {
    routeMock.mockReturnValue({ connectionId: 'local', profile: 'ops' })

    expect(isEventForBotScreen(bot, { payload: { profile_key: key }, type: 'display.lease' }, key)).toBe(true)
    expect(isEventForBotScreen(bot, { payload: { profile_key: '/other' }, type: 'display.lease' }, key)).toBe(false)
  })

  it('treats a row whose connection was removed as having no screen instead of throwing', () => {
    routeMock.mockReturnValue(null)

    // Runs for EVERY display.* event, so a throw here would kill the listener for a stale sidebar row.
    expect(isEventForBotScreen(orphan, { payload: { profile_key: '/x' } } as never, '/x')).toBe(false)
  })
})

describe('displayRequest', () => {
  beforeEach(() => {
    vi.mocked(host.requestProfile).mockReset()
  })

  it('sends the backend target profile and preserves other params on a resolved route', async () => {
    routeMock.mockReturnValue({ connectionId: 'conn-a', profile: 'ops', targetProfile: 'kensho-a' })

    await displayRequest(bot, 'display.observe', { viewer_id: 'viewer-1' })

    expect(host.requestProfile).toHaveBeenCalledWith(
      { connectionId: 'conn-a', mode: 'remote', profile: 'ops', targetProfile: 'kensho-a' },
      'display.observe',
      { profile: 'kensho-a', viewer_id: 'viewer-1' }
    )
  })

  it('sends the bot name as profile on a string route', async () => {
    routeMock.mockReturnValue(null)

    await displayRequest(bot, 'display.status')

    expect(host.requestProfile).toHaveBeenCalledWith('ops', 'display.status', { profile: 'ops' })
  })

  it('preserves a caller-supplied profile and defaults one when omitted', async () => {
    routeMock.mockReturnValue({ connectionId: 'conn-a', profile: 'ops', targetProfile: 'kensho-a' })

    const route = { connectionId: 'conn-a', mode: 'remote', profile: 'ops', targetProfile: 'kensho-a' }

    await displayRequest(bot, 'display.observe', { profile: 'caller-profile', viewer_id: 'viewer-2' })
    await displayRequest(bot, 'display.status')

    expect(host.requestProfile).toHaveBeenNthCalledWith(1, route, 'display.observe', {
      profile: 'caller-profile',
      viewer_id: 'viewer-2'
    })
    expect(host.requestProfile).toHaveBeenNthCalledWith(2, route, 'display.status', { profile: 'kensho-a' })
  })

  it('rejects instead of throwing synchronously for a row whose connection was removed', async () => {
    routeMock.mockReturnValue(null)

    await expect(displayRequest(orphan, 'display.status')).rejects.toThrow(/no connection owner/)
    expect(host.requestProfile).not.toHaveBeenCalled()
  })
})
