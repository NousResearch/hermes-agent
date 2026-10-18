import type { SessionCommandNotices } from './types_notices'

// Righe digitate per /undo /retry /steer /queue: la stessa formulazione della TUI Ink.
export const itSessionCommands: SessionCommandNotices = {
  undoUsage: 'uso: /undo [N]',
  undoNothing: 'niente da annullare',
  undidOne: count => `annullato ${count} messaggio — il messaggio rimosso è nel compositore`,
  undidOther: count => `annullati ${count} messaggi — il messaggio rimosso è nel compositore`,
  retryNothing: 'niente da ritentare',
  retryUnsupported: '/retry: retry non supporta contenuti multimediali o parti sconosciute',
  steerUsage: 'uso: /steer <prompt>',
  steerQueued: preview => `steer in coda — arriva dopo la prossima chiamata a strumenti: "${preview}"`,
  steerNoActiveTurn: preview => `nessun turno attivo — in coda per il prossimo: "${preview}"`,
  steerRejected: 'steer rifiutato — nessun turno attivo, messo in coda per il prossimo turno',
  queued: preview => `in coda: "${preview}"`,
  queueFailed: 'impossibile mettere il messaggio in coda',
  countOne: count => `${count} messaggio in coda`,
  countOther: count => `${count} messaggi in coda`,
  queueEditKeptTitle: 'Messaggio in coda non modificato',
  queueEditKept:
    'È partito o è stato rifiutato prima che la tua modifica fosse salvata. Il testo modificato è nel compositore.',
  queueEditUnconfirmedTitle: 'Modifica non confermata',
  queueEditUnconfirmed:
    'Hermes non è riuscito a confermare la tua modifica. Salvala di nuovo per riprovare la stessa modifica; non viene mai messa in coda due volte.'
}
