import type { TranslationOverrides } from './define-locale'
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
    mcpReloadNotInChat: server => `${server} n’est pas activé dans cette discussion`,
    mcpReloadUnavailable: servers => `Connexion impossible : ${servers}`,
    compactedTurnUnavailable:
      'Ce message a été résumé lors de la compression de la conversation : il ne peut plus être restauré ni modifié.',
    staleTurnUnavailable: 'Ce message ne fait plus partie de l’historique de cette discussion.',
    sessionCommands: frSessionCommands
  },
  keepAwakeOffer: {
    title: 'Votre ordinateur s’est mis en veille pendant que Hermes travaillait',
    message:
      'Hermes peut empêcher la mise en veille pendant qu’une tâche s’exécute. L’écran s’éteint toujours, et sur batterie l’ordinateur peut quand même se mettre en veille.',
    action: 'Empêcher la veille pendant le travail'
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'butterbar' | 'promptNotices' | 'keepAwakeOffer'>
