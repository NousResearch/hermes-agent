import { describe, expect, it } from 'vitest'

import { sanitizeTextForSpeech } from './speech-text'

describe('sanitizeTextForSpeech', () => {
  it('summarizes fenced code blocks instead of reading them literally', () => {
    expect(sanitizeTextForSpeech('Here is code:\n```ts\nconst x = 1\n```\nDone.')).toBe(
      'Here is code: code block omitted Done.'
    )
  })

  it('still keeps normal prose and inline code readable', () => {
    expect(sanitizeTextForSpeech('Use `git status` after the change.')).toBe(
      'Use git status after the change.'
    )
  })
})
