import { describe, expect, it } from 'vitest'

import {
  cronAdvancedSummaryRows,
  cronAdvancedUpdates,
  cronEditorUpdates,
  emptyCronAdvancedValues,
  formatRepeatTimes,
  jobIsScriptOnly,
  parseCommaList,
  parseCronDeliveryTargets,
  parseRepeatTimes,
  seedCronAdvancedValues,
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

  it('rejects setting both monitor script and monitor URL', () => {
    expect(
      validateCronEditor({
        advanced: { ...emptyCronAdvancedValues(), monitorScript: 'check.sh', monitorUrl: 'https://x/y' },
        prompt: 'go',
        schedule: '0 9 * * *',
        scriptOnlyJob: false
      })
    ).toBe('monitor')
  })

  it('rejects a non-numeric repeat', () => {
    expect(
      validateCronEditor({
        advanced: { ...emptyCronAdvancedValues(), repeat: 'many' },
        prompt: 'go',
        schedule: '0 9 * * *',
        scriptOnlyJob: false
      })
    ).toBe('repeat')
    expect(
      validateCronEditor({
        advanced: { ...emptyCronAdvancedValues(), repeat: '3' },
        prompt: 'go',
        schedule: '0 9 * * *',
        scriptOnlyJob: false
      })
    ).toBe(null)
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

describe('cron advanced helpers', () => {
  it('parses comma lists with dedupe', () => {
    expect(parseCommaList('web, file,web ,, ')).toEqual(['web', 'file'])
    expect(parseCommaList('')).toEqual([])
  })

  it('round-trips stored repeat shapes', () => {
    expect(formatRepeatTimes(null)).toBe('')
    expect(formatRepeatTimes(undefined)).toBe('')
    expect(formatRepeatTimes(5)).toBe('5')
    expect(formatRepeatTimes({ completed: 2, times: 5 })).toBe('5')
    expect(formatRepeatTimes({ completed: 0, times: null })).toBe('')
    expect(parseRepeatTimes('')).toBe(null)
    expect(parseRepeatTimes('3')).toBe(3)
    expect(Number.isNaN(parseRepeatTimes('often'))).toBe(true)
  })

  it('seeds the form from a stored job', () => {
    const seed = seedCronAdvancedValues({
      attach_to_session: true,
      context_from: ['abc123'],
      enabled: true,
      enabled_toolsets: ['web'],
      id: 'j1',
      monitor_url: 'https://x/y',
      reasoning_effort: 'high',
      repeat: { completed: 1, times: 4 },
      skills: ['reporter'],
      workdir: '/tmp/p'
    })

    expect(seed).toMatchObject({
      attachToSession: true,
      contextFrom: 'abc123',
      enabledToolsets: 'web',
      monitorScript: '',
      monitorUrl: 'https://x/y',
      reasoningEffort: 'high',
      repeat: '4',
      skills: 'reporter',
      workdir: '/tmp/p'
    })
  })

  it('sends every non-empty field on create', () => {
    const updates = cronAdvancedUpdates(
      { ...emptyCronAdvancedValues(), skills: 'reporter, researcher', repeat: '3', workdir: '/tmp/p' },
      null
    )

    expect(updates).toMatchObject({
      repeat: 3,
      skills: ['reporter', 'researcher'],
      workdir: '/tmp/p'
    })
    expect('enabled_toolsets' in updates).toBe(false)
    expect('monitor_script' in updates).toBe(false)
    expect('attach_to_session' in updates).toBe(false)
  })

  it('omits untouched fields on edit so stored values survive', () => {
    const job = {
      attach_to_session: true,
      enabled: true,
      id: 'j1',
      skills: ['reporter'],
      workdir: '/tmp/p'
    }

    const updates = cronAdvancedUpdates(seedCronAdvancedValues(job), job)

    expect(updates).toEqual({})
  })

  it('sends only the changed field on edit, including clears', () => {
    const job = { enabled: true, id: 'j1', skills: ['reporter'], workdir: '/tmp/p' }
    const seed = seedCronAdvancedValues(job)

    expect(cronAdvancedUpdates({ ...seed, workdir: '/tmp/q' }, job)).toEqual({ workdir: '/tmp/q' })
    expect(cronAdvancedUpdates({ ...seed, workdir: '' }, job)).toEqual({ workdir: null })
  })

  it('summarizes only the effective settings', () => {
    expect(cronAdvancedSummaryRows({ enabled: true, id: 'j1' })).toEqual([
      { key: 'executionMode', value: 'agent' }
    ])
    expect(
      cronAdvancedSummaryRows({
        enabled: true,
        id: 'j1',
        no_agent: true,
        repeat: { completed: 0, times: 2 },
        script: 'echo hi',
        skills: ['reporter']
      })
    ).toEqual([
      { key: 'executionMode', value: 'script' },
      { key: 'repeat', value: '2' },
      { key: 'skills', value: 'reporter' }
    ])
  })
})
