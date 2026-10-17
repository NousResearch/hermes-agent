// Typed /undo /retry /steer /queue lines (en_session_commands.ts is the source).
export const frSessionCommands = {
  undoUsage: 'utilisation : /undo [N]',
  undoNothing: 'rien à annuler',
  undidOne: (count: string) => `${count} message annulé — le message retiré est dans la zone de saisie`,
  undidOther: (count: string) => `${count} messages annulés — le message retiré est dans la zone de saisie`,
  retryNothing: 'rien à réessayer',
  retryUnsupported: '/retry : impossible de réessayer un contenu multimédia ou des parties inconnues',
  steerUsage: 'utilisation : /steer <prompt>',
  steerQueued: (preview: string) => `orientation en file — arrive après le prochain appel d’outil : « ${preview} »`,
  steerNoActiveTurn: (preview: string) => `aucun tour actif — mis en file pour le suivant : « ${preview} »`,
  steerRejected: 'orientation refusée — aucun tour actif, mise en file pour le tour suivant',
  queued: (preview: string) => `en file : « ${preview} »`,
  queueFailed: 'Impossible de mettre le message en file',
  countOne: (count: string) => `${count} message en file`,
  countOther: (count: string) => `${count} messages en file`,
  queueEditKeptTitle: 'Message en file non modifié',
  queueEditKept:
    "Il a démarré ou a été refusé avant l'enregistrement de ta modification. Ton texte modifié est dans la zone de saisie.",
  queueEditUnconfirmedTitle: 'Modification non confirmée',
  queueEditUnconfirmed:
    "Hermes n'a pas pu confirmer ta modification. Enregistre à nouveau pour réessayer la même modification ; elle n'est jamais mise en file deux fois."
}
