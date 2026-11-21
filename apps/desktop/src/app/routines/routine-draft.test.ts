import { describe, expect, it } from 'vitest'

import type { AutomationBlueprint, CronJob } from '@/hermes'

import { cronModelChoiceValue } from '../cron/cron-job-model'

import { blankDraft, createPayload, draftFromJob, draftFromTemplate, updatePayload, validateDraft } from './routine-draft'
import { defaultPick } from './routine-schedule'

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

const daily = defaultPick(NOW)

describe('a new routine', () => {
  it('creates with only what the user set: run limit, remember last run and model when chosen', () => {
    const draft = {
      ...blankDraft('ops'),
      pick: daily,
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
    expect(createPayload({ ...blankDraft('ops'), pick: daily, prompt: 'x' })).toEqual({
      deliver: 'local',
      prompt: 'x',
      schedule: '0 9 * * *'
    })
  })

  it('starts with no frequency, and asks for a schedule, then an instruction', () => {
    expect(blankDraft('ops').pick).toBeNull()
    expect(validateDraft({ ...blankDraft('ops'), prompt: 'x' })).toBe('schedule')
    expect(validateDraft({ ...blankDraft('ops'), pick: daily, prompt: '' })).toBe('prompt')
    expect(validateDraft({ ...blankDraft('ops'), pick: { ...daily, custom: '', frequency: 'custom' }, prompt: 'x' })).toBe(
      'schedule'
    )
  })

  it('starts from a template with everything its defaults make, and keeps its skills', () => {
    const template = {
      category: 'daily',
      description: 'A weekday nudge.',
      fields: [],
      key: 'workday-start',
      prefill: {
        deliver: 'origin',
        name: 'Workday start reminder',
        prompt: 'Give the user a brief weekday start-of-day nudge.',
        schedule: '0 9 * * 1-5',
        skills: ['google-workspace']
      },
      title: 'Workday start reminder'
    } as unknown as AutomationBlueprint

    const draft = draftFromTemplate(template, blankDraft('ops'), NOW)

    expect(draft).toMatchObject({
      deliver: 'local',
      name: 'Workday start reminder',
      pick: { frequency: 'weekdays', time: '09:00' },
      profile: 'ops',
      prompt: 'Give the user a brief weekday start-of-day nudge.'
    })
    expect(createPayload(draft)).toMatchObject({ schedule: '0 9 * * 1-5', skills: ['google-workspace'] })

    // A backend from before `prefill`: its name and description, the frequency left to pick.
    const older = draftFromTemplate({ ...template, prefill: undefined }, blankDraft('ops'), NOW)

    expect(older).toMatchObject({ name: 'Workday start reminder', pick: null, prompt: 'A weekday nudge.', skills: [] })
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

    expect(updatePayload({ ...draft, pick: { ...daily, frequency: 'daily', time: '07:00' } }, stored).schedule).toBe(
      '0 7 * * *'
    )
  })
})
