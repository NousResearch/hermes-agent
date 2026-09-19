import { describe, expect, it } from 'vitest'

import {
  cronEditorUpdates,
  jobIsScriptOnly,
  lastErrorSummary,
  parseCronDeliveryTargets,
  toggleCronDeliveryTarget,
  validateCronEditor
} from './cron-job-model'
import { nextRunOverdueMs } from './job-state'

describe('jobIsScriptOnly', () => {
  it('is true when no_agent is set and a script is present', () => {
    expect(jobIsScriptOnly({ no_agent: true, script: 'echo hi' })).toBe(true)
  })

  it('is false for agent-backed jobs', () => {
    expect(jobIsScriptOnly({ no_agent: false, script: 'echo hi' })).toBe(false)
    expect(jobIsScriptOnly({ no_agent: true, script: '' })).toBe(false)
    expect(jobIsScriptOnly({ no_agent: true, script: null })).toBe(false)
  })
})

describe('validateCronEditor', () => {
  it('requires prompt and schedule for agent-backed jobs', () => {
    expect(validateCronEditor({ prompt: '', schedule: '', scriptOnlyJob: false })).toBe('prompt_and_schedule')
    expect(validateCronEditor({ prompt: '', schedule: '0 9 * * *', scriptOnlyJob: false })).toBe('prompt')
    expect(validateCronEditor({ prompt: 'go', schedule: '', scriptOnlyJob: false })).toBe('schedule')
  })

  it('allows an empty prompt when editing a script-only job', () => {
    expect(validateCronEditor({ prompt: '', schedule: '0 9 * * 1', scriptOnlyJob: true })).toBe(null)
    expect(validateCronEditor({ prompt: 'optional note', schedule: '0 9 * * 1', scriptOnlyJob: true })).toBe(null)
  })

  it('still requires schedule for script-only jobs', () => {
    expect(validateCronEditor({ prompt: '', schedule: '', scriptOnlyJob: true })).toBe('schedule')
  })
})

describe('cron delivery targets', () => {
  it('parses comma-separated targets and removes duplicates', () => {
    expect(parseCronDeliveryTargets('local, telegram,local')).toEqual(['local', 'telegram'])
  })

  it('falls back to local for an empty stored value', () => {
    expect(parseCronDeliveryTargets('')).toEqual(['local'])
  })

  it('adds a second target in the scheduler comma-separated format', () => {
    expect(toggleCronDeliveryTarget('local', 'origin', true)).toBe('local,origin')
  })

  it('removes one target while keeping the other selection', () => {
    expect(toggleCronDeliveryTarget('local,origin', 'local', false)).toBe('origin')
  })

  it('does not allow the final delivery target to be unchecked', () => {
    expect(toggleCronDeliveryTarget('origin', 'origin', false)).toBe('origin')
  })
})

describe('lastErrorSummary', () => {
  it('strips the exception wrapper and markers, keeping only the first sentence', () => {
    const raw =
      "RuntimeError: Cron job 'x' has no model configured (job.model=None, HERMES_MODEL=''). Set a per-job model via `hermes cron edit x --model <name>`."

    expect(lastErrorSummary(raw)).toBe("Cron job 'x' has no model configured (job.model=None, HERMES_MODEL='').")
    expect(lastErrorSummary('[blocked_config:silent] ⚠️ ValueError: bad schedule\nDetails follow')).toBe('bad schedule')
  })

  it('caps very long single sentences with an ellipsis', () => {
    const summary = lastErrorSummary(`HTTPStatusError: ${'x'.repeat(400)}`)

    expect(summary.length).toBeLessThanOrEqual(200)
    expect(summary.endsWith('…')).toBe(true)
  })

  it('returns an empty string for missing input', () => {
    expect(lastErrorSummary(null)).toBe('')
    expect(lastErrorSummary(undefined)).toBe('')
    expect(lastErrorSummary('   ')).toBe('')
  })
})

describe('cronEditorUpdates', () => {
  it('omits prompt when saving a script-only job with an empty prompt', () => {
    expect(
      cronEditorUpdates(
        { deliver: 'local', model: '', name: 'Weekly', prompt: '', provider: '', schedule: '0 9 * * 1' },
        { scriptOnlyJob: true }
      )
    ).toEqual({
      deliver: 'local',
      name: 'Weekly',
      schedule: '0 9 * * 1',
      no_agent: true
    })
  })

  it('includes prompt when the user typed one on a script-only job', () => {
    expect(
      cronEditorUpdates(
        { deliver: 'email', model: '', name: 'Weekly', prompt: 'note', provider: '', schedule: '0 9 * * 1' },
        { scriptOnlyJob: true }
      ).prompt
    ).toBe('note')
  })

  it('writes the model override for agent jobs', () => {
    const updates = cronEditorUpdates(
      {
        deliver: 'local',
        model: 'claude-sonnet-4',
        name: 'Daily',
        prompt: 'go',
        provider: 'anthropic',
        schedule: '0 9 * * *'
      },
      { scriptOnlyJob: false }
    )

    expect(updates.model).toBe('claude-sonnet-4')
    expect(updates.provider).toBe('anthropic')
  })

  it('clears a previous pin when the override is reset to default', () => {
    const updates = cronEditorUpdates(
      { deliver: 'local', model: '', name: 'Daily', prompt: 'go', provider: '', schedule: '0 9 * * *' },
      { scriptOnlyJob: false }
    )

    expect(updates.model).toBe(null)
    expect(updates.provider).toBe(null)
  })

  it('never touches model fields on script-only jobs', () => {
    const updates = cronEditorUpdates(
      { deliver: 'local', model: 'x', name: 'Weekly', prompt: '', provider: 'y', schedule: '0 9 * * 1' },
      { scriptOnlyJob: true }
    )

    expect('model' in updates).toBe(false)
    expect('provider' in updates).toBe(false)
  })
})

describe('cronEditorUpdates script mode payload', () => {
  it('includes no_agent: true and script path for script-only jobs', () => {
    expect(
      cronEditorUpdates(
        {
          deliver: 'local',
          name: 'Backup',
          prompt: '',
          schedule: '0 2 * * *',
          no_agent: true,
          script: '/usr/local/bin/backup.sh'
        },
        { scriptOnlyJob: true }
      )
    ).toEqual({
      deliver: 'local',
      name: 'Backup',
      schedule: '0 2 * * *',
      no_agent: true,
      script: '/usr/local/bin/backup.sh'
    })
  })

  it('does not include no_agent or script for agent-backed jobs', () => {
    const result = cronEditorUpdates(
      {
        deliver: 'local',
        name: 'Daily Summary',
        prompt: 'Summarize today',
        schedule: '0 9 * * *',
        no_agent: false,
        script: '/some/script.sh'
      },
      { scriptOnlyJob: false }
    )

    expect(result).not.toHaveProperty('no_agent')
    expect(result).not.toHaveProperty('script')
    expect(result.prompt).toBe('Summarize today')
  })

  it('omits script from payload when script path is empty (script-only)', () => {
    const result = cronEditorUpdates(
      {
        deliver: 'local',
        name: 'EmptyScript',
        prompt: '',
        schedule: '0 9 * * 1',
        no_agent: true,
        script: ''
      },
      { scriptOnlyJob: true }
    )

    expect(result.no_agent).toBe(true)
    expect(result).not.toHaveProperty('script')
  })

  it('omits script from payload when script path is whitespace only (script-only)', () => {
    const result = cronEditorUpdates(
      {
        deliver: 'local',
        name: 'WhitespaceScript',
        prompt: '',
        schedule: '*/15 * * * *',
        no_agent: true,
        script: '   '
      },
      { scriptOnlyJob: true }
    )

    expect(result.no_agent).toBe(true)
    expect(result).not.toHaveProperty('script')
  })

  it('trims script path when included in payload', () => {
    expect(
      cronEditorUpdates(
        {
          deliver: 'local',
          name: 'Trimmed',
          prompt: '',
          schedule: '0 * * * *',
          no_agent: true,
          script: '  /path/with/spaces.sh  '
        },
        { scriptOnlyJob: true }
      ).script
    ).toBe('/path/with/spaces.sh')
  })
})

describe('nextRunOverdueMs', () => {
  const now = Date.parse('2026-09-17T20:35:00+04:00')

  it('flags an active job whose stored slot sits past the scheduler grace (#114309)', () => {
    const job = { enabled: true, next_run_at: '2026-09-17T13:34:18+04:00', state: 'scheduled' }

    expect(nextRunOverdueMs(job, now)).toBe(now - Date.parse(job.next_run_at))
  })

  it('keeps upcoming, within-grace, paused and unparseable slots as plain next runs', () => {
    expect(nextRunOverdueMs({ enabled: true, next_run_at: '2026-09-17T21:00:00+04:00' }, now)).toBeNull()
    expect(nextRunOverdueMs({ enabled: true, next_run_at: '2026-09-17T20:30:00+04:00' }, now)).toBeNull()
    expect(nextRunOverdueMs({ enabled: true, next_run_at: '2026-09-17T13:34:18+04:00', state: 'paused' }, now)).toBeNull()
    expect(nextRunOverdueMs({ enabled: false, next_run_at: '2026-09-17T13:34:18+04:00' }, now)).toBeNull()
    expect(nextRunOverdueMs({ enabled: true, next_run_at: 'not-a-date' }, now)).toBeNull()
  })
})
