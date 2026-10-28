import { expect, it } from 'vitest'

import { type ChatMessage, preserveLocalAssistantErrors, textPart, toChatMessages } from './index'

// The backend armed a rerun (`quota_resume.scheduled`) that only the live terminal frame reports;
// the refresh rebuilds the card from the failed-turn boundary row, which cannot know about it.
it('a refresh that rebuilds the failed tail card from its boundary row keeps the rerun the backend armed', () => {
  const stored = toChatMessages([
    { role: 'user', content: 'summarize the notes', timestamp: 1 },
    {
      role: 'assistant',
      content: 'Your request was not processed. Send it again if you still want me to carry it out.',
      timestamp: 2,
      display_kind: 'failed_turn',
      display_metadata: {
        error: 'HTTP 429',
        error_surface: { code: 'rate_limit', layer: 'provider', retryable: true, resets_at: 1000 }
      }
    }
  ])

  const local: ChatMessage[] = [
    { id: 'local-user', role: 'user', parts: [textPart('summarize the notes')] },
    {
      id: 'assistant-live',
      role: 'assistant',
      parts: [],
      error: 'HTTP 429',
      errorSurface: { code: 'rate_limit', layer: 'provider', retryable: true, resetsAt: 1000, rerunAt: 1045 }
    }
  ]

  const cards = preserveLocalAssistantErrors(stored, local).filter(message => message.error)

  expect(cards).toHaveLength(1)
  expect(cards[0].errorSurface).toMatchObject({ code: 'rate_limit', resetsAt: 1000, rerunAt: 1045 })
})
