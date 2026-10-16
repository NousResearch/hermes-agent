import type { TranslationOverrides } from './define-locale'
import { frSessionCommands } from './fr_session_commands'

// Shell notices (remote-display toast, butterbar), spread into fr.ts.
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
    sessionCommands: frSessionCommands
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'butterbar' | 'promptNotices'>
