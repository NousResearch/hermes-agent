import { describe, expect, it } from 'vitest'

import { resolveLogsSessionQueryValue } from './panes'

describe('LogsPane session filter routing', () => {
  it('serializes Current as the stored/logged session id, not the runtime id', () => {
    expect(
      resolveLogsSessionQueryValue({
        activeSessionId: 'runtime-session-after-resume',
        selectedStoredSessionId: 'stored-lineage-session',
        resolvedFilter: 'current'
      })
    ).toBe('stored-lineage-session')
  })

  it('falls back to the runtime id only before a stored id is known', () => {
    expect(
      resolveLogsSessionQueryValue({
        activeSessionId: 'runtime-session-during-create',
        selectedStoredSessionId: null,
        resolvedFilter: 'current'
      })
    ).toBe('runtime-session-during-create')
  })

  it('omits the query session for All and passes explicit stored sessions through', () => {
    expect(
      resolveLogsSessionQueryValue({
        activeSessionId: 'runtime-session',
        selectedStoredSessionId: 'stored-session',
        resolvedFilter: 'all'
      })
    ).toBeUndefined()

    expect(
      resolveLogsSessionQueryValue({
        activeSessionId: 'runtime-session',
        selectedStoredSessionId: 'stored-session',
        resolvedFilter: 'picked-stored-session'
      })
    ).toBe('picked-stored-session')
  })
})
