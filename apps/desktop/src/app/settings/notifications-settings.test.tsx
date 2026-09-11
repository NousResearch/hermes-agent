// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $hapticsMuted } from '@/store/haptics'
import {
  $nativeNotifyPrefs,
  NATIVE_NOTIFICATION_KINDS,
  type NativeNotificationPrefs
} from '@/store/native-notifications'

import { NotificationsSettings } from './notifications-settings'

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        notifications: {
          title: 'Notifications',
          intro: 'OS notifications are separate from in-app feedback.',
          enableAll: 'Enable notifications',
          enableAllDesc: 'Off silences OS notifications.',
          enableAppSounds: 'Enable app sounds & haptics',
          enableAppSoundsDesc: 'Controls completion, thinking, and wake sounds separately from OS notifications.',
          focusedHint: 'Completion alerts only fire while Hermes is in the background.',
          kinds: Object.fromEntries(
            NATIVE_NOTIFICATION_KINDS.map(kind => [kind, { label: kind, description: `${kind} notifications` }])
          ),
          test: 'Send test notification',
          testTitle: 'Hermes',
          testBody: 'Notifications are working.',
          testSent: 'Test sent.',
          testUnsupported: 'Unsupported.',
          completionSoundTitle: 'Completion Sound',
          completionSoundDesc: 'Choose a completion sound.',
          completionSoundPreview: 'Preview'
        }
      }
    }
  })
}))

vi.mock('@/lib/completion-sound', () => ({
  COMPLETION_SOUND_VARIANTS: [{ id: 1, name: 'Comfort' }],
  previewCompletionSound: vi.fn()
}))

describe('NotificationsSettings', () => {
  beforeEach(() => {
    $hapticsMuted.set(false)
    $nativeNotifyPrefs.set({
      enabled: true,
      kinds: Object.fromEntries(NATIVE_NOTIFICATION_KINDS.map(kind => [kind, true])) as NativeNotificationPrefs['kinds']
    })
  })

  afterEach(() => {
    cleanup()
    vi.clearAllMocks()
  })

  it('lets users mute in-app feedback without disabling OS notifications', () => {
    render(<NotificationsSettings />)

    const soundSwitch = screen.getByRole('switch', { name: 'Enable app sounds & haptics' })
    expect(soundSwitch.getAttribute('data-state')).toBe('checked')

    fireEvent.click(soundSwitch)

    expect($hapticsMuted.get()).toBe(true)
    expect($nativeNotifyPrefs.get().enabled).toBe(true)
    expect(soundSwitch.getAttribute('data-state')).toBe('unchecked')
  })
})
