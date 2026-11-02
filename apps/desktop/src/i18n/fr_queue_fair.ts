import type { QueueFairCopy } from './types_queue_fair'

export const frQueueFair: QueueFairCopy = {
  holdUnknown: reason => `En attente : ${reason}`,
  hold: {
    session_busy: 'En attente du tour en cours',
    slot: 'En attente d\'un emplacement de session libre',
    editing: 'En cours de modification, ne démarrera pas encore',
    restarting: 'En attente du redémarrage de la passerelle',
    paused: 'En pause jusqu\'à la résolution du tour perdu',
    provider_limit: 'En attente de la réinitialisation de la limite du fournisseur',
    memory: 'En attente de mémoire libre'
  },
  moveUp: 'Monter',
  moveDown: 'Descendre',
  holdFailed: 'Impossible de suspendre ce message en file pendant sa modification ; il peut démarrer avant l\'enregistrement'
}
