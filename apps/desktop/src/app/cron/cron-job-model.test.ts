import { describe, expect, it } from 'vitest'

import { cronEditorUpdates, jobIsScriptOnly, validateCronEditor } from './cron-job-model'

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
