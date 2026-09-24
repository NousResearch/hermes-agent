import { describe, expect, it } from 'vitest'

import { matchRecurrence } from './cron'

describe('matchRecurrence', () => {
  it.each([
    ['remind me every morning to check the board', 'every morning'],
    ['run this daily please', 'daily'],
    ['every 2 hours, poll the feed', 'every 2 hours'],
    ['each week send a summary', 'each week'],
    ['do it every friday', 'every friday'],
    ['nightly build report', 'nightly']
  ])('matches recurring phrasing: %s', (draft, phrase) => {
    expect(matchRecurrence(draft)?.toLowerCase()).toBe(phrase)
  })

  it.each([
    'remind me to call mom tomorrow',
    'this happens to everyone',
    'the everyday struggle',
    'weekly-report.pdf is attached',
    'i read the Daily Prophet'
  ])('stays quiet on one-shots and lookalikes: %s', draft => {
    expect(matchRecurrence(draft)).toBeNull()
  })
})
