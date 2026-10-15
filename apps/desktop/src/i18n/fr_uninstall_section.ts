import type { UninstallSectionTranslations } from './types_uninstall_section'

// Settings > About > Uninstall, composed by fr.ts as `settings.uninstallSection`.
export const frUninstallSection: UninstallSectionTranslations = {
  dangerZone: 'Zone dangereuse',
  checkingInstalled: 'Vérification de ce qui est installé…',
  uninstallHermes: 'Désinstaller Hermes',
  managedBody: 'Cette installation est gérée par votre système, donc Hermes ne peut pas la supprimer.',
  dataKept: path => `Votre configuration, vos conversations et vos secrets se trouvent dans ${path}. La suppression de l\'application ne les supprime pas.`,
  openAppsSettings: 'Ouvrir les paramètres des applications',
  chooseHowMuch:
    'Choisissez ce qu\'il faut supprimer. L\'application se ferme pour terminer l\'opération; vous pouvez rouvrir l\'installateur à tout moment pour revenir.',
  confirmUninstall: 'Confirmer la désinstallation',
  confirmBody: what => `Cela supprime ${what}. Cette action ne peut pas être annulée.`,
  appLabel: 'Application :',
  couldNotStart: 'La désinstallation n\'a pas pu démarrer.',
  uninstalling: 'Désinstallation en cours…',
  yesUninstall: 'Oui, désinstaller',
  options: {
    gui: {
      title: 'Désinstaller uniquement l\'interface graphique Chat',
      description: 'Supprimez cette application de bureau. L\'agent Hermes, votre configuration, et vos conversations restent.',
      consequence: 'l\'interface graphique Chat de bureau (cette application et ses données)'
    },
    lite: {
      title: 'Désinstaller l\'interface graphique + l\'agent, conserver mes données',
      description: 'Supprimez l\'application et l\'agent Hermes, mais conservez la configuration, les conversations, et les secrets pour une future réinstallation.',
      consequence: 'l\'interface graphique Chat et l\'agent Hermes (configuration, conversations, et secrets sont conservés)'
    },
    full: {
      title: 'Désinstaller tout',
      description: 'Supprimez l\'application, l\'agent, et toutes les données de l\'utilisateur — configuration, conversations, tâches programmées, secrets, journaux.',
      consequence: 'TOUT — l\'interface graphique Chat de bureau, l\'agent Hermes, et toutes vos configurations, conversations, secrets, et journaux'
    }
  }
}