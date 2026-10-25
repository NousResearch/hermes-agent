import type { TranslationOverrides } from './define-locale'
import { frDesktopCommands } from './fr_desktop_commands'
import { frSessionCommands } from './fr_session_commands'

// Shell notices (remote-display toast, butterbar, keep-awake offer), spread into fr.ts.
export const frNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `Rendu logiciel actif — affichage distant détecté (${reason}). L'accélération GPU est désactivée pour éviter les scintillements.`
  },
  butterbar: {
    goTo: (index, total) => `Afficher l'avis ${index} sur ${total}`,
    legal: {
      before: "L'utilisation de Hermes Agent est soumise à nos ",
      terms: "Conditions d'utilisation",
      between: ' et à notre ',
      privacy: 'Politique de confidentialité',
      after: '.'
    }
  },
  promptNotices: {
    legacySendUnconfirmed:
      "Ce serveur n'a pas pu confirmer l'envoi précédent de ce message ; il a peut-être déjà été exécuté. Vérifiez la conversation avant de le renvoyer.",
    mcpReloadDeferred:
      'Cette discussion est occupée : la modification MCP s’applique à la fin de son tour et au prochain rechargement MCP, ou à la réouverture de la discussion',
    mcpReloadUnavailable: servers => `Connexion impossible : ${servers}`,
    compactedTurnUnavailable:
      'Ce message a été résumé lors de la compression de la conversation : il ne peut plus être restauré ni modifié.',
    staleTurnUnavailable: 'Ce message ne fait plus partie de l’historique de cette discussion.',
    sessionCommands: frSessionCommands,
    desktopCommands: frDesktopCommands
  },
  keepAwakeOffer: {
    title: 'Votre ordinateur s’est mis en veille pendant que Hermes travaillait',
    message:
      'Hermes peut empêcher la mise en veille pendant qu’une tâche s’exécute. L’écran s’éteint toujours, et sur batterie l’ordinateur peut quand même se mettre en veille.',
    action: 'Empêcher la veille pendant le travail'
  },
  billingBlock: {
    titleNous: 'Plus de crédits Nous',
    titleProvider: provider => `Plus de crédits — ${provider}`,
    fallbackMessage: 'Votre compte est à court de crédits. Ajoutez-en pour continuer.',
    openBilling: 'Ouvrir la facturation',
    addCredits: 'Ajouter des crédits',
    dismiss: 'Fermer'
  },
  sendDiagnostics: {
    title: 'Envoyer les diagnostics à Nous',
    privacyNotice:
      "Cela téléverse un paquet de débogage vers un stockage interne de Nous, et non vers un service de partage public. Il contient des informations système (système d'exploitation, versions, fournisseur et clés API configurées — jamais les clés elles-mêmes) ainsi que les journaux complets de l'agent, du gateway et du Desktop (jusqu'à 512 Ko chacun), susceptibles de contenir des conversations, des résultats d'outils et des chemins de fichiers. Les secrets sont expurgés avant l'envoi. Seuls le personnel de Nous et les modérateurs Discord autorisés peuvent consulter le paquet, qui est automatiquement supprimé après 14 jours.",
    upload: 'Envoyer',
    uploading: 'Envoi en cours…',
    cancel: 'Annuler',
    close: 'Fermer',
    copyLink: 'Copier le lien',
    uploadIdFallback: id => `Aucun lien de consultation renvoyé — indiquez l'identifiant ${id} au support`,
    doneTitle: 'Diagnostics envoyés',
    doneDescription:
      "Votre paquet a été téléversé de manière privée. Partagez le lien ci-dessous dans votre fil d'assistance afin que l'équipe puisse consulter vos journaux.",
    failedTitle: "Échec de l'envoi",
    failedHint:
      'Vous pouvez également exécuter `hermes debug share --nous` dans un terminal, ou `hermes debug share --local` pour afficher le rapport sans le téléverser.',
    handoffLead: 'Poursuivez la discussion sur :',
    links: {
      github: 'Issues GitHub',
      portal: 'Assistance du portail Nous',
      discord: 'Discord'
    }
  }
} satisfies Pick<
  TranslationOverrides,
  'remoteDisplayBanner' | 'butterbar' | 'promptNotices' | 'keepAwakeOffer' | 'billingBlock' | 'sendDiagnostics'
>
