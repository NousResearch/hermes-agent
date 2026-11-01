import type { QueueFairCopy } from './types_queue_fair'

export const deQueueFair: QueueFairCopy = {
  holdUnknown: reason => `Wartet: ${reason}`,
  hold: {
    session_busy: 'Wartet auf den laufenden Durchgang',
    slot: 'Wartet auf einen freien Sitzungsplatz',
    editing: 'Wird bearbeitet, startet noch nicht',
    restarting: 'Wartet auf den Neustart des Gateways',
    paused: 'Pausiert, bis der verlorene Durchgang geklärt ist',
    provider_limit: 'Wartet auf das Zurücksetzen des Anbieterlimits',
    memory: 'Wartet auf freien Speicher'
  },
  moveUp: 'Nach oben',
  moveDown: 'Nach unten',
  holdFailed: 'Diese Nachricht konnte während der Bearbeitung nicht angehalten werden; sie kann vor dem Speichern starten'
}
