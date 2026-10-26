import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

// The error card's per-code title and body (lib/error-surface-copy.ts), keyed by error_surface.code.
export const frErrorCodes: Partial<ThreadCopy['errorCodes']> = {
  cron_session_read_only: {
    title: 'Cette conversation appartient à une tâche planifiée',
    body: 'Elle n’exécute que cette tâche ; votre message n’a donc pas été exécuté. Démarrez une nouvelle conversation pour l’envoyer.'
  },
  workspace_unavailable: {
    title: 'Le dossier de cette conversation a disparu',
    body: "Le dossier dans lequel travaille cette conversation a été déplacé ou supprimé ; votre message n'a donc pas été exécuté. Recréez le dossier ou utilisez « Déplacer vers un projet » pour lui en attribuer un autre, puis réessayez."
  },
  launch_credentials_unavailable: {
    title: 'La clé API de cette conversation a changé',
    body: "Cette conversation a démarré avec une clé API qui a depuis été modifiée ou supprimée ; votre message n'a donc pas été envoyé. Une conversation ouverte ne passe jamais d'elle-même à une autre clé. Utilisez « Démarrer une nouvelle session » pour utiliser votre clé actuelle."
  }
}
