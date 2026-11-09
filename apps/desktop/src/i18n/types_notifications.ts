// Notification settings types used by every desktop locale.
export interface NotificationSettingsTranslations {
  title: string
  intro: string
  enableAll: string
  enableAllDesc: string
  focusedHint: string
  kinds: Record<
    'approval' | 'backgroundDone' | 'credits' | 'input' | 'plugin' | 'turnDone' | 'turnError',
    { label: string; description: string }
  >
  browserPermission: {
    title: string
    allow: string
    states: Record<'default' | 'denied' | 'granted' | 'insecure' | 'unsupported', string>
  }
  test: string
  testTitle: string
  testBody: string
  testSent: string
  testUnsupported: string
  muteSoundsTitle: string
  muteSoundsDesc: string
  completionSoundTitle: string
  completionSoundDesc: string
  completionSoundPreview: string
}
