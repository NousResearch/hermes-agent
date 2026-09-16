import { describe, expect, it } from 'vitest'

import {
  cronEditorUpdates,
  jobIsScriptOnly,
  lastErrorSummary,
  parseCronDeliveryTargets,
  toggleCronDeliveryTarget,
  validateCronEditor
} from './cron-job-model'

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
      schedule: '0 9 * * 1'
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
