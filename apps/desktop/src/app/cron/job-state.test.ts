import { describe, expect, it } from 'vitest'

import { noAgentRunTime } from './job-state'

describe('noAgentRunTime', () => {
  it('returns the persisted execution time for a no-agent cron job', () => {
    expect(
      noAgentRunTime({
        last_run_at: '2026-08-20T08:20:18.088489+09:00',
        no_agent: true
      })
    ).toBe(Date.parse('2026-08-20T08:20:18.088489+09:00'))
  })

  it('does not treat an agent job or an invalid timestamp as a script execution', () => {
    expect(noAgentRunTime({ last_run_at: '2026-08-20T08:20:18+09:00' })).toBeNull()
    expect(noAgentRunTime({ last_run_at: 'not-a-date', no_agent: true })).toBeNull()
  })
})
