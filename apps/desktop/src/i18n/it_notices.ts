import type { TranslationOverrides } from './define-locale'
import { itSessionCommands } from './it_session_commands'

// Avvisi della shell (toast display remoto, butterbar), sparsi in it.ts.
export const itNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `Rendering software attivo — rilevato uno schermo remoto (${reason}). L’accelerazione GPU è stata disattivata per evitare lo sfarfallio.`
  },
  butterbar: {
    goTo: (index, total) => `Mostra avviso ${index} di ${total}`,
    legal: {
      before: 'L’uso di Hermes Agent è soggetto ai nostri ',
      terms: 'Termini di servizio',
      between: ' e ',
      privacy: 'Informativa sulla privacy',
      after: '.'
    }
  },
  promptNotices: {
    legacySendUnconfirmed:
      'Questo server non ha potuto confermare l’invio precedente di questo messaggio, quindi potrebbe essere già stato elaborato. Controlla la conversazione prima di inviarlo di nuovo.',
    mcpReloadDeferred:
      'Questa chat è occupata: la modifica MCP si applica quando il turno in corso termina e MCP si ricarica, o alla riapertura della chat',
    mcpReloadNotInChat: server => `${server} non è attivato in questa chat`,
    mcpReloadUnavailable: servers => `Connessione non riuscita: ${servers}`,
    sessionCommands: itSessionCommands
  },
  keepAwakeOffer: {
    title: 'Il computer si è spento mentre Hermes lavorava',
    message:
      'Hermes può mantenere il computer sveglio mentre un’attività è in esecuzione. Lo schermo si spegne comunque, e a batteria il computer potrebbe comunque andare in riposo.',
    action: 'Tieni sveglio il computer mentre lavora'
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'butterbar' | 'promptNotices' | 'keepAwakeOffer'>
