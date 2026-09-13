import { describe, expect, it } from 'vitest'

import { searchResultToSession } from './index'

describe('searchResultToSession owner routing', () => {
  it('preserves the registry connection and profile on a server-only search hit', () => {
    const session = searchResultToSession({
      connection_id: 'registry-remote',
      lineage_root: null,
      model: null,
      profile: 'default',
      role: null,
      session_id: 'remote-chat',
      session_started: 1,
      snippet: 'Archived project notes',
      source: 'telegram'
    })

    expect(session.connection_id).toBe('registry-remote')
    expect(session.profile).toBe('default')
  })
})
