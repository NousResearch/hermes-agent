import type { InboxCopy } from './types_inbox'

export const frInbox: InboxCopy = {
  title: 'Boîte de réception',
  unreadCount: (count: number) => `${count} non lus`,
  loading: 'Chargement de la boîte de réception…',
  emptyTitle: 'Aucun résultat pour l’instant',
  emptyDesc: 'Quand une tâche planifiée se termine, son résultat arrive ici.',
  running: 'En cours',
  unread: 'Non lu',
  noPreview: 'Terminée sans réponse.',
  refresh: 'Actualiser',
  refreshing: 'Actualisation…',
  markReadFailed: 'Impossible de marquer le résultat comme lu',
  newResultsTitle: (count: number) => `${count} nouveaux résultats de tâches planifiées`,
  newResultsBody: 'Ouvrez la boîte de réception pour les lire.'
}
