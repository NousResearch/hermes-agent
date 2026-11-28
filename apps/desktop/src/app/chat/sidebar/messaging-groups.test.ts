import { describe, expect, it } from 'vitest'

import type { Contribution } from '@/contrib/types'
import type { SessionInfo } from '@/types/hermes'

import { buildMessagingGroups } from './messaging-groups'

const row = (id: string, source: string, fields: Partial<SessionInfo> = {}): SessionInfo => ({
  ended_at: null,
  id,
  input_tokens: 0,
  is_active: false,
  last_active: 1,
  message_count: 1,
  model: null,
  output_tokens: 0,
  preview: null,
  source,
  started_at: 1,
  title: id,
  tool_call_count: 0,
  ...fields
})

describe('buildMessagingGroups', () => {
  it('separates a contributed integration without borrowing its transport totals or pin identities', () => {
    const contributions: Contribution[] = [
      {
        area: 'session.sourceLabels',
        data: { label: 'iMessage', source: 'photon' },
        id: 'messages',
        source: 'plugin:channel-labels'
      },
      {
        area: 'session.sourceLabels',
        data: { label: 'Home Assistant', profile: 'default', sessionIdPrefix: 'ha-', source: 'api_server' },
        id: 'home-assistant',
        source: 'plugin:channel-labels'
      }
    ]

    const sessions = [
      row('photon-pinned', 'photon', { pinned: true }),
      row('message', 'photon', { last_active: 6 }),
      row('ha-pinned', 'api_server', { pinned: true }),
      row('ha-voice', 'api_server', { last_active: 4 }),
      row('compressed-tip', 'api_server', { _lineage_root_id: 'ha-original', last_active: 5 }),
      row('unrelated-api', 'api_server', { last_active: 3 }),
      row('ha-other-profile', 'api_server', { last_active: 2, profile: 'work' })
    ]

    const options = {
      contributions,
      getSourceTotal: (source: string) => ({ api_server: 8, photon: 3 })[source],
      isPinned: (session: SessionInfo) => Boolean(session.pinned),
      profile: 'all',
      sessions,
      truncated: true
    }

    const groups = buildMessagingGroups(options)

    expect(groups.map(group => [group.label, group.sessions.map(session => session.id)])).toEqual([
      ['iMessage', ['message']],
      ['Home Assistant', ['compressed-tip', 'ha-voice']],
      ['API', ['unrelated-api', 'ha-other-profile']]
    ])
    expect(groups.map(group => [group.groupId, group.sourceId, group.total, group.hasMore])).toEqual([
      ['photon', 'photon', 2, true],
      ['api_server:home-assistant', 'api_server', 2, true],
      ['api_server', 'api_server', 2, true]
    ])

    // Exhaustion is decided across every group sharing a transport, including
    // pinned rows, so neither subgroup promises rows after the source is done.
    const exhausted = buildMessagingGroups({
      ...options,
      getSourceTotal: (source: string) => sessions.filter(session => session.source === source).length
    })

    expect(exhausted.every(group => !group.hasMore)).toBe(true)

    // A partial source page containing no integration row cannot claim that
    // the whole API count belongs to its generic remainder.
    const remainderOnly = buildMessagingGroups({
      ...options,
      profile: 'default',
      sessions: [row('unrelated-api', 'api_server')]
    })

    expect(remainderOnly).toEqual([expect.objectContaining({ groupId: 'api_server', hasMore: true, total: 1 })])

    const unrelatedProfile = buildMessagingGroups({
      ...options,
      profile: 'work',
      sessions: [row('ha-other-profile', 'api_server', { profile: 'work' })]
    })

    expect(unrelatedProfile).toEqual([expect.objectContaining({ groupId: 'api_server', label: 'API', total: 8 })])

    // A whole-source label scoped to one profile still needs separate display
    // identities in the unified view, even without an id-prefix filter.
    const scopedMessages = buildMessagingGroups({
      ...options,
      contributions: [
        {
          ...contributions[0],
          data: { label: 'iMessage', profile: 'default', source: 'photon' }
        }
      ],
      sessions: [row('default-message', 'photon'), row('work-message', 'photon', { profile: 'work' })]
    })

    expect(scopedMessages.map(group => [group.groupId, group.label, group.total, group.hasMore])).toEqual([
      ['photon:messages', 'iMessage', 1, true],
      ['photon', 'Photon', 1, true]
    ])
  })
})
