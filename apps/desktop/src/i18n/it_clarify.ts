import type { ClarifyCopy } from './types_clarify'

export const itClarify: Partial<ClarifyCopy> = {
  notReady: 'La richiesta di chiarimento non è ancora pronta',
  gatewayDisconnected: 'Hermes non è connesso in questo momento. Riconnettiti e invialo di nuovo.',
  sendFailed: 'Impossibile inviare la risposta di chiarimento',
  loadingQuestion: 'Caricamento domanda…',
  other: 'Altro (scrivi la tua risposta)',
  placeholder: 'Scrivi la tua risposta…',
  skip: 'Salta',
  skipped: 'Saltato',
  noAnswer: 'Senza risposta',
  confirmAndContinueLabel: 'Conferma e continua',
  singleSelectHint: 'Scegline una',
  multiSelectHint: 'Scegli tutte quelle applicabili',
  questionProgress: (answered, total) => `${answered} di ${total} risposte`,
  oneQuestion: '1 domanda',
  notDelivered:
    'Questa domanda non è arrivata all’app, quindi non si può rispondere qui. Premi Interrompi per terminare il turno e poi rispondi in chat.'
}
