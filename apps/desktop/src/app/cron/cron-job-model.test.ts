import { describe, expect, it } from 'vitest'

import {
  buildCronModelOptions,
  CRON_MODEL_DEFAULT,
  cronEditorCreatePayload,
  cronEditorUpdates,
  cronModelSelectValue,
  decodeCronModelChoice,
  encodeCronModelChoice,
  ensureCronModelOption,
  jobIsScriptOnly,
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

describe('cron model choice encoding', () => {
  it('round-trips a pinned provider/model pair', () => {
    const value = encodeCronModelChoice('openrouter', 'anthropic/claude-sonnet-4.6')

    expect(value).toBe('openrouter::anthropic/claude-sonnet-4.6')
    expect(decodeCronModelChoice(value)).toEqual({
      provider: 'openrouter',
      model: 'anthropic/claude-sonnet-4.6'
    })
  })

  it('treats the default sentinel and empty input as unpinned', () => {
    expect(decodeCronModelChoice(CRON_MODEL_DEFAULT)).toEqual({ model: null, provider: null })
    expect(decodeCronModelChoice('')).toEqual({ model: null, provider: null })
    expect(encodeCronModelChoice('', 'gpt-5.5')).toBe(CRON_MODEL_DEFAULT)
  })

  it('maps an unpinned job to the default select value', () => {
    expect(cronModelSelectValue({ model: null, provider: null })).toBe(CRON_MODEL_DEFAULT)
    expect(cronModelSelectValue({})).toBe(CRON_MODEL_DEFAULT)
  })

  it('maps a pinned job to its encoded select value', () => {
    expect(cronModelSelectValue({ provider: 'nous', model: 'hermes-4' })).toBe('nous::hermes-4')
  })
})

describe('buildCronModelOptions', () => {
  it('flattens authenticated providers and skips unavailable models', () => {
    expect(
      buildCronModelOptions([
        {
          name: 'Nous',
          slug: 'nous',
          authenticated: true,
          models: ['hermes-4', 'hermes-4-pro'],
          unavailable_models: ['hermes-4-pro']
        },
        {
          name: 'OpenRouter',
          slug: 'openrouter',
          authenticated: false,
          models: ['should-skip']
        }
      ])
    ).toEqual([
      {
        label: 'Nous · hermes-4',
        model: 'hermes-4',
        provider: 'nous',
        value: 'nous::hermes-4'
      }
    ])
  })

  it('keeps a stale pin visible via ensureCronModelOption', () => {
    const options = buildCronModelOptions([
      { name: 'Nous', slug: 'nous', authenticated: true, models: ['hermes-4'] }
    ])

    expect(
      ensureCronModelOption(options, { provider: 'openrouter', model: 'retired-model' }).map(o => o.value)
    ).toEqual(['openrouter::retired-model', 'nous::hermes-4'])
  })
})

describe('cronEditorCreatePayload', () => {
  it('omits model pins when following the default', () => {
    expect(
      cronEditorCreatePayload({
        deliver: 'local',
        model: null,
        name: 'Briefing',
        prompt: 'summarize',
        provider: null,
        schedule: '0 9 * * *'
      })
    ).toEqual({
      deliver: 'local',
      name: 'Briefing',
      prompt: 'summarize',
      schedule: '0 9 * * *'
    })
  })

  it('includes model and provider when a job is pinned', () => {
    expect(
      cronEditorCreatePayload({
        deliver: 'telegram',
        model: 'hermes-4',
        name: '',
        prompt: 'go',
        provider: 'nous',
        schedule: '0 * * * *'
      })
    ).toEqual({
      deliver: 'telegram',
      model: 'hermes-4',
      prompt: 'go',
      provider: 'nous',
      schedule: '0 * * * *'
    })
  })
})

describe('cronEditorUpdates', () => {
  it('omits prompt when saving a script-only job with an empty prompt', () => {
    expect(
      cronEditorUpdates(
        {
          deliver: 'local',
          model: null,
          name: 'Weekly',
          prompt: '',
          provider: null,
          schedule: '0 9 * * 1'
        },
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
        {
          deliver: 'email',
          model: 'hermes-4',
          name: 'Weekly',
          prompt: 'note',
          provider: 'nous',
          schedule: '0 9 * * 1'
        },
        { scriptOnlyJob: true }
      ).prompt
    ).toBe('note')
  })

  it('preserves stored inference pins when saving a script-only job', () => {
    // Dialog hides Model for script-only edits — values.model/provider are null
    // placeholders and must not be written, or a prior pin would be cleared.
    expect(
      cronEditorUpdates(
        {
          deliver: 'local',
          model: null,
          name: 'Script job',
          prompt: '',
          provider: null,
          schedule: '0 9 * * 1'
        },
        { scriptOnlyJob: true }
      )
    ).not.toHaveProperty('model')
    expect(
      cronEditorUpdates(
        {
          deliver: 'local',
          model: null,
          name: 'Script job',
          prompt: '',
          provider: null,
          schedule: '0 9 * * 1'
        },
        { scriptOnlyJob: true }
      )
    ).not.toHaveProperty('provider')
  })

  it('clears a prior model pin when switching back to default', () => {
    expect(
      cronEditorUpdates(
        {
          deliver: 'local',
          model: null,
          name: 'Briefing',
          prompt: 'summarize',
          provider: null,
          schedule: '0 9 * * *'
        },
        { scriptOnlyJob: false }
      )
    ).toMatchObject({
      model: null,
      provider: null,
      prompt: 'summarize'
    })
  })
})
