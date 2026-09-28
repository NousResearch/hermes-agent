import { describe, expect, it } from 'vitest'

import { sessionDotClassName } from './session-status-dot'

describe('session status marks', () => {
  it('keeps finished unread sessions on the good circle shape', () => {
    const unread = sessionDotClassName('unread')

    expect(unread).toContain('rounded-full')
    expect(unread).not.toContain('rounded-[1px]')
  })
})
