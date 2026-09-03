import { describe, expect, it } from 'vitest'

import { sessionExportScope } from './session-export'

describe('session export routing', () => {
  it('keeps the selected aggregate row on its exact connection and profile', () => {
    expect(sessionExportScope({ connection_id: 'remote-a', profile: 'agentops' })).toEqual({
      connectionId: 'remote-a',
      profile: 'agentops'
    })
    expect(sessionExportScope({ profile: 'clientops' })).toBe('clientops')
  })
})
