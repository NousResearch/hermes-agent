import { describe, expect, it } from 'vitest'

import { classifyCronReasoningEffort, cronEditorUpdates, jobIsScriptOnly, validateCronEditor } from './cron-job-model'

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

describe('classifyCronReasoningEffort', () => {
  it('keeps malformed present values distinguishable from an inherited value', () => {
    expect(classifyCronReasoningEffort({})).toEqual({ formValue: '', preserveOnSave: false })
    expect(classifyCronReasoningEffort({ reasoning_effort: '' })).toEqual({ formValue: '', preserveOnSave: true })
    expect(classifyCronReasoningEffort({ reasoning_effort: 'high' })).toEqual({
      formValue: 'high',
      preserveOnSave: false
    })
    expect(classifyCronReasoningEffort({ reasoning_effort: false })).toEqual({
      formValue: 'none',
      preserveOnSave: false
    })
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

  it('writes the reasoning effort override for agent jobs', () => {
    const updates = cronEditorUpdates(
      {
        deliver: 'local',
        model: 'claude-sonnet-4',
        name: 'Daily',
        prompt: 'go',
        provider: 'anthropic',
        reasoningEffort: 'high',
        schedule: '0 9 * * *'
      },
      { scriptOnlyJob: false }
    )

    expect(updates.reasoning_effort).toBe('high')
  })

  it('clears a previous reasoning effort when reset to inherit', () => {
    const updates = cronEditorUpdates(
      {
        deliver: 'local',
        model: '',
        name: 'Daily',
        prompt: 'go',
        provider: '',
        reasoningEffort: null,
        schedule: '0 9 * * *'
      },
      { scriptOnlyJob: false }
    )

    expect(updates.reasoning_effort).toBe(null)
  })

  it('keeps none as the explicit reasoning effort override', () => {
    const updates = cronEditorUpdates(
      {
        deliver: 'local',
        model: '',
        name: 'Daily',
        prompt: 'go',
        provider: '',
        reasoningEffort: 'none',
        schedule: '0 9 * * *'
      },
      { scriptOnlyJob: false }
    )

    expect(updates.reasoning_effort).toBe('none')
  })
  it('omits an unknown legacy reasoning value instead of resubmitting it', () => {
    // The editor maps an unchanged malformed legacy value to undefined.
    const updates = cronEditorUpdates(
      {
        deliver: 'local',
        model: '',
        name: 'Daily',
        prompt: 'go',
        provider: '',
        reasoningEffort: undefined,
        schedule: '0 9 * * *'
      },
      { scriptOnlyJob: false }
    )

    expect('reasoning_effort' in updates).toBe(false)
  })
  it('drops a malformed runtime reasoning value instead of sending it', () => {
    const updates = cronEditorUpdates(
      {
        deliver: 'local',
        model: '',
        name: 'Daily',
        prompt: 'go',
        provider: '',
        reasoningEffort: 'warp9' as never,
        schedule: '0 9 * * *'
      },
      { scriptOnlyJob: false }
    )

    expect('reasoning_effort' in updates).toBe(false)
  })

  it('guards reasoning writes for non-script no-agent jobs', () => {
    const updates = cronEditorUpdates(
      {
        deliver: 'local',
        model: 'claude-sonnet-4',
        name: 'Maintenance',
        prompt: 'go',
        provider: 'anthropic',
        reasoningEffort: 'high',
        schedule: '0 9 * * *'
      },
      { noAgentJob: true, scriptOnlyJob: false }
    )

    expect('reasoning_effort' in updates).toBe(false)
  })

  it('preserves malformed empty reasoning during an unrelated edit until explicitly cleared', () => {
    const baseValues = {
      deliver: 'local',
      model: '',
      name: 'Daily',
      prompt: 'go',
      provider: '',
      reasoningEffort: null,
      reasoningPreserveOnSave: true,
      schedule: '0 9 * * *'
    }

    const unchanged = cronEditorUpdates({ ...baseValues, reasoningChanged: false }, { scriptOnlyJob: false })
    expect('reasoning_effort' in unchanged).toBe(false)

    const cleared = cronEditorUpdates({ ...baseValues, reasoningChanged: true }, { scriptOnlyJob: false })
    expect(cleared.reasoning_effort).toBe(null)
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
    expect('reasoning_effort' in updates).toBe(false)
  })
  it('retains the configured reasoning effort for no-agent jobs', () => {
    const updates = cronEditorUpdates(
      {
        deliver: 'local',
        model: 'x',
        name: 'Weekly',
        prompt: '',
        provider: 'y',
        reasoningEffort: 'ultra',
        schedule: '0 9 * * 1'
      },
      { scriptOnlyJob: true }
    )

    expect('reasoning_effort' in updates).toBe(false)
  })
})
