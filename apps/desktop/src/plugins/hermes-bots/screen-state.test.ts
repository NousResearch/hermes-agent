/**
 * Lease ordering: `display.lease` events and `display.status` replies race on
 * the wire. A slower status reply describing an OLDER lease must never roll
 * back the newer event — the backend's monotonic `epoch` is the tiebreak.
 */

import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { DisplayLease, DisplayStatus } from './screen-connection'
import type { RosterRow } from './types'

vi.mock('./data', () => ({ botSelectionKey: (bot: RosterRow) => bot.name }))

import { $screenState, beginScreenStatusRequest, screenStateFor, setScreenLease, setScreenStatus } from './screen-state'

const bot: RosterRow = { name: 'ops' }

const agent: DisplayLease = { holder: 'agent', viewer_id: null, viewer_hash: null, since: 1, reason: '', epoch: 3 }
const human: DisplayLease = { ...agent, holder: 'human', viewer_hash: 'abc123abc123', epoch: 4 }

const statusWith = (lease: DisplayLease): DisplayStatus => ({
  profile: 'ops',
  profile_key: '/home/hermes/.hermes',
  supported: true,
  installed: true,
  missing: [],
  running: true,
  pid: 1,
  display: ':20',
  socket: null,
  geometry: '1440x900',
  install_command: null,
  lease
})

beforeEach(() => $screenState.set({}))

describe('lease epoch ordering', () => {
  it('a status reply carrying an older epoch does not roll back a newer lease event', () => {
    setScreenLease(bot, human)
    setScreenStatus(bot, statusWith(agent))

    expect(screenStateFor($screenState.get(), bot)?.lease).toEqual(human)
    // The status itself still lands — only its stale lease is ignored.
    expect(screenStateFor($screenState.get(), bot)?.status?.running).toBe(true)
  })

  it('a lease event with an older epoch is ignored; a newer one applies', () => {
    setScreenLease(bot, human)
    setScreenLease(bot, agent)
    expect(screenStateFor($screenState.get(), bot)?.lease).toEqual(human)

    const released = { ...agent, epoch: 5 }
    setScreenLease(bot, released)
    expect(screenStateFor($screenState.get(), bot)?.lease).toEqual(released)
  })

  it('an unchanged-looking lease still advances the cached epoch, so a delayed older takeover is rejected', () => {
    // agent@0 → agent@2 (a take-over and hand-back that both happened before we looked) must leave epoch 2
    // recorded even though nothing visible changed; otherwise the late human@1 event wins and the pane shows
    // a human holding a screen the agent already has back.
    setScreenLease(bot, { ...agent, epoch: 0 })
    setScreenLease(bot, { ...agent, epoch: 2 })
    setScreenLease(bot, { ...human, epoch: 1 })
    expect(screenStateFor($screenState.get(), bot)?.lease?.holder).toBe('agent')
    expect(screenStateFor($screenState.get(), bot)?.lease?.epoch).toBe(2)
  })
})

describe('status request ordering', () => {
  it('a slower reply from a superseded status request never overwrites newer truth', () => {
    const stale = beginScreenStatusRequest(bot)
    const fresh = beginScreenStatusRequest(bot)

    setScreenStatus(bot, { ...statusWith(agent), running: false }, fresh)
    setScreenStatus(bot, statusWith(agent), stale)
    expect(screenStateFor($screenState.get(), bot)?.status?.running).toBe(false)

    // A pushed event is newer than anything still in flight.
    const inFlight = beginScreenStatusRequest(bot)
    setScreenStatus(bot, statusWith(agent))
    setScreenStatus(bot, { ...statusWith(agent), running: false }, inFlight)
    expect(screenStateFor($screenState.get(), bot)?.status?.running).toBe(true)
  })
})
