import type { UpdateWhenIdleCopy } from './types_update_when_idle'

export const frUpdateWhenIdle: UpdateWhenIdleCopy = {
  busyTitle: 'Hermes travaille encore',
  busyBody: 'La mise à jour redémarre Hermes, ce qui arrête :',
  chats: (n: number) => (n === 1 ? '1 conversation en cours' : `${n} conversations en cours`),
  cron: (jobs: string) => `Tâche planifiée en cours (${jobs})`,
  delegation: 'Des sous-agents tournent en arrière-plan',
  input: 'Une question attend votre réponse',
  unknown: 'Hermes n’a pas pu confirmer qu’il est inactif',
  updateNow: (n: number) => `Mettre à jour maintenant (interrompt ${n})`,
  updateWhenIdle: 'Mettre à jour une fois libre',
  waitingTitle: 'La mise à jour attend qu’Hermes ait fini',
  waitingBody:
    'La mise à jour démarre d’elle-même dès que plus rien ne tourne. Continuez à travailler ; Hermes reste utilisable d’ici là.',
  waitingFor: 'En attente de :',
  waitingIdle: 'Plus rien ne tourne. Démarrage de la mise à jour…',
  waitingUntil: (time: string) => `Si Hermes est encore occupé à ${time}, la mise à jour est abandonnée.`,
  cancel: 'Annuler la mise à jour',
  timedOut: 'Hermes est resté occupé, la mise à jour n’a donc pas eu lieu. Rien n’a changé ; réessayez plus tard.'
}
