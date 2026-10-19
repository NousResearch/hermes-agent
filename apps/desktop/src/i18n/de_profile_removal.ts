// The profile delete / reset confirmation copy, composed by de.ts as `profiles.removal`.
import type { TranslationOverrides } from './define-locale'

export const deProfileRemoval: NonNullable<TranslationOverrides['profiles']>['removal'] = {
  deleteTitle: 'Profil löschen?',
  deleteDescPrefix: 'Das löscht ',
  deleteDescMid: ' und entfernt sein ',
  deleteDescSuffix: ' Verzeichnis. Das kann nicht rückgängig gemacht werden.',
  deleting: 'Wird gelöscht...',
  resetTitle: 'Standardprofil zurücksetzen?',
  resetDesc:
    'Das Standardprofil kann nicht entfernt werden: Sein Verzeichnis enthält die Hermes-Installation und deine anderen Profile. ' +
    'Beim Zurücksetzen werden Erinnerungen, Skills, Einstellungen, SOUL.md und Chats gelöscht und seine Cron-Jobs entfernt. ' +
    'Deine anderen Profile, die Installation sowie API-Schlüssel und Anmeldungen bleiben erhalten. Das kann nicht rückgängig gemacht werden.',
  resetAction: 'Zurücksetzen',
  resetting: 'Wird zurückgesetzt...',
  resetDone: 'Profil zurückgesetzt'
}
