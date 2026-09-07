import { describe, expect, it } from 'vitest'

import type { BotMetaSnapshot } from './data'
import { pinnedBotRows } from './pinboard-model'
import type { RosterRow } from './types'

const bot = (name: string, overrides: Partial<RosterRow> = {}): RosterRow => ({
  name,
  ...overrides
})

describe('pinnedBotRows', () => {
  it('keeps pinned visible bots first and uses recent visible bots as fallback entries', () => {
    const roster = [
      bot('older', { last_session: { last_active: 100 } }),
      bot('pinned', { last_session: { last_active: 200 } }),
      bot('hidden', { last_session: { last_active: 300 } }),
      bot('ghost', { ghost: true, last_session: { last_active: 500 } }),
      bot('newer', { last_session: { last_active: 400 } })
    ]

    const meta: BotMetaSnapshot = {
      pinned: { pinned: true },
      hidden: { hidden: true },
      ghost: { pinned: true }
    }

    expect(pinnedBotRows(roster, meta, 3).map(row => row.name)).toEqual(['pinned', 'newer', 'older'])
  })

  it('does not collapse source-qualified bots that share a profile name', () => {
    const roster = [
      bot('default', { connectionId: 'remote-a', remoteSource: true, sourceScoped: true }),
      bot('default', { connectionId: 'remote-b', remoteSource: true, sourceScoped: true })
    ]

    const meta: BotMetaSnapshot = {
      'remote-a::default': { pinned: true },
      'remote-b::default': { pinned: true }
    }

    expect(pinnedBotRows(roster, meta).map(row => row.connectionId)).toEqual(['remote-a', 'remote-b'])
  })

  it('uses a stable source-aware key when recent activity ties', () => {
    const roster = [
      bot('default', {
        connectionId: 'remote-b',
        remoteSource: true,
        sourceScoped: true,
        last_session: { last_active: 100 }
      }),
      bot('default', {
        connectionId: 'remote-a',
        remoteSource: true,
        sourceScoped: true,
        last_session: { last_active: 100 }
      })
    ]

    const meta: BotMetaSnapshot = {
      'remote-a::default': { pinned: true },
      'remote-b::default': { pinned: true }
    }

    expect(pinnedBotRows(roster, meta).map(row => row.connectionId)).toEqual(['remote-a', 'remote-b'])
  })

  it('returns no entries when every bot is hidden', () => {
    const roster = [bot('hidden')]
    const meta: BotMetaSnapshot = { hidden: { hidden: true, pinned: true } }

    expect(pinnedBotRows(roster, meta)).toEqual([])
  })
})
