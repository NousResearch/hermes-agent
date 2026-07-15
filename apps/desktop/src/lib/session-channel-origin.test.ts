import { describe, expect, it } from 'vitest'

import { sessionChannelOriginLabel } from './session-channel-origin'

describe('sessionChannelOriginLabel', () => {
  it('joins distinct safe room and topic labels', () => {
    expect(
      sessionChannelOriginLabel({
        chat_name: 'Build Room',
        chat_topic: 'Release coordination',
        chat_type: 'channel',
        display_name: 'Build Room',
        has_thread: true,
        platform: 'webhook'
      })
    ).toBe('Build Room · Release coordination')
  })

  it('omits duplicate or missing topic text', () => {
    expect(
      sessionChannelOriginLabel({
        chat_topic: 'Build Room',
        display_name: 'Build Room',
        has_thread: false
      })
    ).toBe('Build Room')
    expect(sessionChannelOriginLabel(null)).toBeNull()
  })
})
