import type { CronControlCopy } from './types_cron_control'

export const frCronControl: CronControlCopy = {
  stop: 'Arrêter',
  stopping: 'Arrêt…',
  stopped: 'Exécution annulée',
  finishedFirst: 'L’exécution s’est terminée avant l’arrêt',
  nothingToStop: 'Aucune exécution en cours',
  failedStop: 'Impossible d’arrêter l’exécution',
  lastCancelled: (reason: string) => (reason ? `Dernière exécution annulée: ${reason}` : 'Dernière exécution annulée'),
  catchUpLabel: 'Exécutions manquées',
  catchUpHint: 'Que faire des exécutions manquées pendant que Hermes ne tournait pas.',
  catchUpChoices: {
    default: 'Utiliser la valeur par défaut',
    run: 'Exécuter une fois au retour',
    skip: 'Les ignorer'
  }
}
