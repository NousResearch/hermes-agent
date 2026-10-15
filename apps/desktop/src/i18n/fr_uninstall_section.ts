import type { UninstallSectionTranslations } from './types_uninstall_section'

// Paramètres > À propos > Désinstaller, composé par fr.ts comme `settings.uninstallSection`.
export const frUninstallSection: UninstallSectionTranslations = {
  dangerZone: 'Zone dangereuse',
  checkingInstalled: 'Vérification des éléments installés…',
  uninstallHermes: 'Désinstaller Hermes',
  managedBody: 'Cette installation est gérée par votre système : Hermes ne peut donc pas se supprimer lui-même.',
  dataKept: path =>
    `Votre configuration, vos conversations et vos secrets se trouvent dans ${path}. La suppression de l’application ne les supprime pas.`,
  openAppsSettings: 'Ouvrir les réglages des applications',
  chooseHowMuch:
    'Choisissez ce que vous souhaitez supprimer. L’application se ferme pour terminer ; rouvrez le programme d’installation à tout moment pour revenir.',
  confirmUninstall: 'Confirmer la désinstallation',
  confirmBody: what => `Cette action supprime ${what}. Elle est irréversible.`,
  appLabel: 'Application :',
  couldNotStart: 'La désinstallation n’a pas pu démarrer.',
  uninstalling: 'Désinstallation…',
  yesUninstall: 'Oui, désinstaller',
  options: {
    gui: {
      title: 'Désinstaller uniquement l’interface de chat',
      description:
        'Supprime cette application de bureau. L’agent Hermes, votre configuration et vos conversations sont conservés.',
      consequence: 'l’interface de chat de bureau (cette application et ses données)'
    },
    lite: {
      title: 'Désinstaller l’interface et l’agent, conserver mes données',
      description:
        'Supprime l’application et l’agent Hermes, mais conserve la configuration, les conversations et les secrets pour une future réinstallation.',
      consequence:
        'l’interface de chat et l’agent Hermes (la configuration, les conversations et les secrets sont conservés)'
    },
    full: {
      title: 'Tout désinstaller',
      description:
        'Supprime l’application, l’agent et toutes les données utilisateur : configuration, conversations, tâches planifiées, secrets, journaux.',
      consequence:
        'TOUT — l’interface de chat, l’agent Hermes et l’ensemble de votre configuration, de vos conversations, secrets et journaux'
    }
  }
}
