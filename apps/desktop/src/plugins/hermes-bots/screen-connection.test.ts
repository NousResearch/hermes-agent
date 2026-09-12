/**
 * Two remote hosts can share the same `~/.hermes` path, so a `display.*` event
 * matched on `profile_key` alone would let host B's take-over repaint host A's
 * screen pane. The event must also have arrived on the bot's own connection.
 */

import { describe, expect, it, vi } from 'vitest'

import type { RosterRow } from './types'

const routeMock = vi.fn<() => { connectionId: string; profile: string } | null>(() => null)

vi.mock('@hermes/plugin-sdk', () => ({
  host: { requestProfile: vi.fn() },
  resolveSiblingWsUrl: vi.fn()
}))

vi.mock('./routing', () => ({
  botConnectionRoute: () => routeMock()
}))

import { isEventForBotScreen } from './screen-connection'

const bot = { name: 'ops' } as RosterRow
const key = '/home/hermes/.hermes'

describe('isEventForBotScreen', () => {
  it('ignores a same-profile-path event that arrived from another host', () => {
    routeMock.mockReturnValue({ connectionId: 'conn-a', profile: 'ops' })

    const fromB = { connectionId: 'conn-b', payload: { profile_key: key }, type: 'display.lease' }
    const fromA = { connectionId: 'conn-a', payload: { profile_key: key }, type: 'display.lease' }

    expect(isEventForBotScreen(bot, fromB, key)).toBe(false)
    expect(isEventForBotScreen(bot, fromA, key)).toBe(true)
  })

  it('still matches the untagged local socket for a local bot', () => {
    routeMock.mockReturnValue({ connectionId: 'local', profile: 'ops' })

    expect(isEventForBotScreen(bot, { payload: { profile_key: key }, type: 'display.lease' }, key)).toBe(true)
    expect(isEventForBotScreen(bot, { payload: { profile_key: '/other' }, type: 'display.lease' }, key)).toBe(false)
  })
})
