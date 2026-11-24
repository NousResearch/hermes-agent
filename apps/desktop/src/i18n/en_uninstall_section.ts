import type { UninstallSectionTranslations } from './types_uninstall_section'

// Settings > About > Uninstall, composed by en.ts as `settings.uninstallSection`.
export const enUninstallSection: UninstallSectionTranslations = {
  dangerZone: 'Danger zone',
  checkingInstalled: 'Checking what’s installed…',
  uninstallHermes: 'Uninstall Hermes',
  managedBody: 'This install is managed by your system, so Hermes cannot remove itself.',
  dataKept: path => `Your config, chats, and secrets live in ${path}. Removing the app does not delete them.`,
  openAppsSettings: 'Open Apps settings',
  chooseHowMuch:
    'Choose how much to remove. The app closes to finish the job; reopen the installer any time to come back.',
  confirmUninstall: 'Confirm uninstall',
  confirmBody: what => `This removes ${what}. This can’t be undone.`,
  appLabel: 'App:',
  couldNotStart: 'Uninstall could not start.',
  uninstalling: 'Uninstalling…',
  yesUninstall: 'Yes, uninstall',
  options: {
    gui: {
      title: 'Uninstall Chat GUI only',
      description: 'Remove this desktop app. The Hermes agent, your config, and chats all stay.',
      consequence: 'the desktop Chat GUI (this app and its data)'
    },
    lite: {
      title: 'Uninstall GUI + agent, keep my data',
      description: 'Remove the app and the Hermes agent, but keep config, chats, and secrets for a future reinstall.',
      consequence: 'the Chat GUI and the Hermes agent (config, chats, and secrets are kept)'
    },
    full: {
      title: 'Uninstall everything',
      description: 'Remove the app, the agent, and all user data — config, chats, scheduled jobs, secrets, logs.',
      consequence: 'EVERYTHING — the Chat GUI, the Hermes agent, and all of your config, chats, secrets, and logs'
    }
  }
}
