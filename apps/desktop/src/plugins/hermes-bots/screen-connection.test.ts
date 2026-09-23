/**
 * Two remote hosts can share the same `~/.hermes` path, so a `display.*` event
 * matched on `profile_key` alone would let host B's take-over repaint host A's
 * screen pane. The event must also have arrived on the bot's own connection.
 */

import { describe, expect, it, vi } from 'vitest'

import type * as routing from './routing'
import type { RosterRow } from './types'

const routeMock = vi.fn<() => { connectionId: string; profile: string } | null>(() => null)

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

    return route ? { status: 'resolved', route: { ...route, mode: 'remote', targetProfile: route.profile } } : actual.resolveBotConnectionRoute(bot)
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
  it('rejects instead of throwing synchronously for a row whose connection was removed', async () => {
    routeMock.mockReturnValue(null)

    await expect(displayRequest(orphan, 'display.status')).rejects.toThrow(/no connection owner/)
    expect(host.requestProfile).not.toHaveBeenCalled()
  })
})
