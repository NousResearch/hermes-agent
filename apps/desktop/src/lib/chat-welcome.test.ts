import { describe, expect, it } from 'vitest'

import { resolveDraftWelcomeProfile } from './chat-welcome'

describe('draft chat welcome profile', () => {
  it.each([
    {
      activeProfile: 'previous',
      expected: 'routed',
      newChatProfile: 'intent',
      routeProfile: 'routed'
    },
    {
      activeProfile: 'previous',
      expected: 'intent',
      newChatProfile: 'intent',
      routeProfile: null
    }
  ])('keeps the pinned draft owner ahead of ambient active profile', values => {
    expect(resolveDraftWelcomeProfile(values)).toBe(values.expected)
  })
})
