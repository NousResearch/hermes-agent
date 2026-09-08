import { describe, expect, it } from 'vitest'

import { groupCreationRoster } from './roster-pane'
import type { RosterRow } from './types'

function row(overrides: Partial<RosterRow>): RosterRow {
  return {
    name: 'default',
    ...overrides
  }
}

describe('groupCreationRoster', () => {
  it('includes reachable remote bots and excludes unavailable sources', () => {
    const eligible = groupCreationRoster([
      row({ name: 'noah', connectionId: 'local' }),
      row({ name: 'archie', connectionId: 'archie', remoteSource: true, sourceReachable: true }),
      row({ name: 'maya', connectionId: 'maya', remoteSource: true, sourceReachable: true }),
      row({ name: 'romeo', connectionId: 'romeo', remoteSource: true, sourceReachable: false })
    ])

    expect(eligible.map(bot => bot.name)).toEqual(['noah', 'archie', 'maya'])
  })
})
