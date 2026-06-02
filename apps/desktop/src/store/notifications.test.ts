import { afterEach, describe, expect, it } from 'vitest'

import { $notifications, clearNotifications, notifyError } from './notifications'

afterEach(() => {
  clearNotifications()
})

describe('notifyError', () => {
  it('keeps microphone settings guidance visible', () => {
    notifyError(
      new Error(
        'Microphone permission was denied. System Settings was opened at Privacy & Security > Microphone. Enable Hermes, then try again.'
      ),
      'Voice recording failed'
    )

    expect($notifications.get()[0]?.message).toBe(
      'Microphone permission was denied. System Settings was opened so you can allow Hermes, then try again.'
    )
  })
})
