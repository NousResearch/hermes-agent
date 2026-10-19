// The profile delete / reset confirmation copy, composed by en.ts as `profiles.removal`.
// The default profile is reset, never removed: its home holds the install and every other profile.
import type { ProfileRemovalTranslations } from './types_profile_removal'

export const enProfileRemoval: ProfileRemovalTranslations = {
  deleteTitle: 'Delete profile?',
  deleteDescPrefix: 'This will delete ',
  deleteDescMid: ' and remove its ',
  deleteDescSuffix: ' directory. This cannot be undone.',
  deleting: 'Deleting...',
  resetTitle: 'Reset the default profile?',
  resetDesc:
    'The default profile cannot be removed: its directory holds the Hermes install and your other profiles. ' +
    'Resetting erases its memories, skills, settings, SOUL.md and chats, and removes its cron jobs. ' +
    'Your other profiles, the install, and your API keys and logins are kept. This cannot be undone.',
  resetAction: 'Reset',
  resetting: 'Resetting...',
  resetDone: 'Profile reset'
}
