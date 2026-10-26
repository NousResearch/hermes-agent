import { describe, expect, it } from 'vitest'

import { cronEditorUpdates } from './cron-job-model'
import { catchUpChoice, catchUpValue, jobLastCancelReason, jobRunInFlight, jobRunLive } from './run-control'

const SAVE = { deliver: 'local', model: '', name: 'Nightly', prompt: 'p', provider: '', schedule: '0 3 * * *' }

describe('cron Stop visibility', () => {
  it('offers Stop only while the newest attempt is unfinished', () => {
    const at = (status: string) => ({ latest_execution: { id: 'e1', status } })

    expect(jobRunInFlight(at('claimed'))).toBe(true)
    expect(jobRunInFlight(at('running'))).toBe(true)
    expect(['completed', 'failed', 'cancelled', 'missed'].some(s => jobRunInFlight(at(s)))).toBe(false)
    expect(jobRunInFlight({ latest_execution: null })).toBe(false)
    expect(jobLastCancelReason({ latest_execution: { error: 'job removed while running', id: 'e1', status: 'cancelled' } })).toBe(
      'job removed while running'
    )
  })
})

describe('cron Stop during a Run-now', () => {
  it('stays live until an attempt claimed since the click has finished', () => {
    const click = Date.parse('2026-10-25T21:00:00Z')
    const row = (status: string, claimed_at: string) => ({ latest_execution: { claimed_at, id: 'e', status } })

    // The list has not caught up yet: the newest row is an older run.
    expect(jobRunLive(row('completed', '2026-10-25T20:00:00Z'), click)).toBe(true)
    // This click's run already ended (stopped): no flicker back to Stop while the request returns.
    expect(jobRunLive(row('cancelled', '2026-10-25T21:00:01Z'), click)).toBe(false)
    expect(jobRunLive(row('completed', '2026-10-25T20:00:00Z'))).toBe(false)
  })
})

describe('cron editor missed-run policy', () => {
  it('sends catch_up only when the choice changed, including back to the default', () => {
    const stored = catchUpChoice(false)

    expect(cronEditorUpdates({ ...SAVE }, { scriptOnlyJob: false })).not.toHaveProperty('catch_up')
    expect(cronEditorUpdates({ ...SAVE, catchUp: catchUpValue('run') }, { scriptOnlyJob: false }).catch_up).toBe(true)
    expect(stored).toBe('skip')
    expect(cronEditorUpdates({ ...SAVE, catchUp: catchUpValue('default') }, { scriptOnlyJob: true }).catch_up).toBeNull()
  })
})
