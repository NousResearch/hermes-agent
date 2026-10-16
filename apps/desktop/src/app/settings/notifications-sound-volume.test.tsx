// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const mocks = vi.hoisted(() => ({ preview: vi.fn() }))

vi.mock('@/i18n', () => ({
  useI18n: () => ({
    t: {
      settings: {
        notifications: {
          completionSoundDesc: 'Plays when an agent turn finishes.',
          completionSoundPreview: 'Preview',
          completionSoundTitle: 'Completion Sound',
          completionSoundVolumeDesc: 'Scales the turn-end cue.',
          completionSoundVolumeTitle: 'Completion Sound Volume',
          enableAll: 'Enable notifications',
          enableAllDesc: 'Banner alerts for finished work.',
          focusedHint: 'Alerts stay quiet while a window is focused.',
          intro: 'Native notifications.',
          test: 'Send test notification',
          testBody: 'Notifications are working.',
          testSent: 'Test sent.',
          testTitle: 'Hermes',
          testUnsupported: 'Unsupported here.',
          title: 'Notifications',
          kinds: { turnDone: { description: 'Turn finished.', label: 'Turn done' } }
        }
      }
    }
  })
}))

vi.mock('@/lib/completion-sound', () => ({
  COMPLETION_SOUND_VARIANTS: [{ id: 1, name: 'Two-note comfort' }],
  previewCompletionSound: mocks.preview
}))

vi.mock('@/lib/haptics', () => ({ triggerHaptic: vi.fn() }))
vi.mock('@/store/native-notifications', async () => {
  const { atom } = await import('nanostores')

  return {
    $nativeNotifyPrefs: atom({ enabled: true, kinds: { turnDone: true } }),
    NATIVE_NOTIFICATION_KINDS: ['turnDone'],
    sendTestNativeNotification: vi.fn(async () => true),
    setNativeNotifyEnabled: vi.fn(),
    setNativeNotifyKind: vi.fn()
  }
})
vi.mock('@/store/notifications', () => ({ notify: vi.fn() }))
vi.mock('./use-setting-deep-link', () => ({ useSettingDeepLink: vi.fn() }))

import { $completionSoundVolume, setCompletionSoundVolume } from '@/store/completion-sound-volume'

import { NotificationsSettings } from './notifications-settings'

describe('notifications settings — completion sound volume', () => {
  beforeEach(() => {
    mocks.preview.mockClear()
    window.localStorage.clear()
    setCompletionSoundVolume(1)
  })

  afterEach(cleanup)

  it('shows the stored volume as a percentage', () => {
    render(<NotificationsSettings subpage="sounds" />)
    expect((screen.getByLabelText('Completion Sound Volume') as HTMLInputElement).value).toBe('100')
    expect(screen.getByText('100%')).toBeTruthy()
  })

  it('writes the slider position into the volume store', () => {
    render(<NotificationsSettings subpage="sounds" />)

    fireEvent.change(screen.getByLabelText('Completion Sound Volume'), { target: { value: '150' } })

    expect($completionSoundVolume.get()).toBe(1.5)
    expect(screen.getByText('150%')).toBeTruthy()
  })

  it('auditions the cue when the slider is released', () => {
    render(<NotificationsSettings subpage="sounds" />)

    fireEvent.pointerUp(screen.getByLabelText('Completion Sound Volume'))

    expect(mocks.preview).toHaveBeenCalled()
  })
})
