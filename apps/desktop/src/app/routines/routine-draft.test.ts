import { describe, expect, it } from 'vitest'

import type { CronJob } from '@/hermes'

import { cronModelChoiceValue } from '../cron/cron-job-model'

import { blankDraft, createPayload, draftFromJob, updatePayload, validateDraft } from './routine-draft'

const NOW = new Date(2026, 10, 12, 12, 0).getTime()

const stored = {
  catch_up: null,
  context_from: ['self'],
  deliver: 'bot-chat:default',
  enabled: true,
  id: 'd92f05770964',
  name: 'Weekly review',
  profile: 'default',
  prompt: 'Summarize what I shipped.',
  repeat: { completed: 2, times: 10 },
  schedule: { expr: '0 16 * * 5', kind: 'cron' },
  state: 'scheduled'
} as CronJob

describe('a new routine', () => {
  it('creates with only what the user set: run limit, remember last run and model when chosen', () => {
    const draft = {
      ...blankDraft('ops', NOW),
      deliver: 'telegram',
      model: cronModelChoiceValue('openrouter', 'some/model'),
      name: ' Morning brief ',
      prompt: ' Brief me ',
      remember: true,
      runLimit: '5'
    }

    expect(createPayload(draft)).toEqual({
      context_from: ['self'],
      deliver: 'telegram',
      model: 'some/model',
      name: 'Morning brief',
      prompt: 'Brief me',
      provider: 'openrouter',
      repeat: 5,
      schedule: '0 9 * * *'
    })
    expect(createPayload({ ...blankDraft('ops', NOW), prompt: 'x' })).toEqual({
      deliver: 'local',
      prompt: 'x',
      schedule: '0 9 * * *'
    })
  })

  it('asks for an instruction and a schedule', () => {
    expect(validateDraft({ ...blankDraft('ops', NOW), prompt: '' })).toBe('prompt')
    expect(validateDraft({ ...blankDraft('ops', NOW), pick: { ...blankDraft('ops', NOW).pick, custom: '', frequency: 'custom' }, prompt: 'x' })).toBe(
      'schedule'
    )
  })

  it('lets a script-only routine keep an empty instruction', () => {
    const script = { ...stored, no_agent: true, prompt: '', script: 'check.sh' } as CronJob

    expect(validateDraft(draftFromJob(script, NOW), script)).toBeNull()
  })
})

describe('editing a routine', () => {
  it('reopens every field as stored', () => {
    expect(draftFromJob(stored, NOW)).toMatchObject({
      catchUp: 'default',
      deliver: 'bot-chat:default',
      name: 'Weekly review',
      pick: { frequency: 'weekly', time: '16:00', weekday: '5' },
      profile: 'default',
      remember: true,
      runLimit: '10'
    })
  })

  it('rewrites the run limit, memory and missed-run policy only when they changed', () => {
    const untouched = updatePayload(draftFromJob(stored, NOW), stored)

    expect(untouched).not.toHaveProperty('repeat')
    expect(untouched).not.toHaveProperty('context_from')
    expect(untouched).not.toHaveProperty('catch_up')
    expect(untouched).toMatchObject({ deliver: 'bot-chat:default', name: 'Weekly review', schedule: '0 16 * * 5' })

    const changed = updatePayload(
      { ...draftFromJob(stored, NOW), catchUp: 'skip', remember: false, runLimit: '' },
      { ...stored, context_from: ['upstream', 'self'] }
    )

    expect(changed).toMatchObject({ catch_up: false, context_from: ['upstream'], repeat: null })
  })

  it('sends a changed schedule as the picker composed it', () => {
    const draft = draftFromJob(stored, NOW)

    expect(updatePayload({ ...draft, pick: { ...draft.pick, frequency: 'daily', time: '07:00' } }, stored).schedule).toBe(
      '0 7 * * *'
    )
  })
})
