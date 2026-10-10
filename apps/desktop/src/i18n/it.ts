import { defineFieldCopy } from '@/app/settings/field-copy'

import { defineLocale, type TranslationOverrides } from './define-locale'
import { itModelMenu } from './it_model_menu'
import { introIt } from './intro-it'

export const itOverrides = {
  sharedMetrics: {
    consentTitle: 'Ci aiuti a migliorare Hermes?',
    consentBody:
      'Le metriche condivise contengono solo contatori limitati. Mai prompt, file, percorsi né testi di errore. La raccolta è locale. Inviarle a Nous è un consenso separato.',
    whatIsCollected: 'Cosa viene raccolto',
    collectedIntro: 'Solo contatori limitati:',
    collectedActivity: 'Attività, durata delle sessioni, risultati e classi di errore',
    collectedModels: 'Percorsi dei modelli e totali di token',
    collectedNames: 'Nomi di strumenti, comandi ed elementi del catalogo integrato',
    collectedMilestones: 'Conteggi di configurazione raggruppati',
    collectedReliability:
      'Risultati e durata degli aggiornamenti, errori, velocità di avvio e di risposta, stato delle piattaforme di messaggistica',
    collectedUsage:
      'Come si usa Hermes: precisione ed efficienza dell’agente (modifiche riuscite, cicli, recuperi, token e chiamate a strumenti per attività, interruzioni di cache), tempo attivo per superficie e modalità Desktop, quali aree, azioni e impostazioni dell’app vengono usate, chiuse subito o disattivate, e risultati della configurazione dei provider',
    collectedMachine:
      'Dati generali del computer: intervallo di RAM, tipo di GPU, età e canale della versione di Hermes, aggiornamenti in sospeso, se si usa un server di modelli locale',
    installId:
      'Quando invii, ogni pacchetto giornaliero viene caricato sul servizio di telemetria di Nous. I pacchetti riportano l’ID di installazione di questo profilo: un UUID casuale e stabile senza informazioni personali, che viene ripristinato eliminando la directory delle metriche condivise.',
    consentWindow:
      'Vengono inviati solo i pacchetti il cui intero periodo di raccolta rientra in una finestra di consenso registrata; i dati precedenti all’accettazione, o di qualsiasi intervallo con l’invio disattivato, restano su questo computer. Puoi disattivare di nuovo l’invio quando vuoi.',
    readDocs: 'Leggi tutti i dettagli',
    share: 'Raccogli e invia a Nous',
    local: 'Raccogli solo in locale',
    off: 'No, grazie',
    changeLater: 'Puoi cambiarlo quando vuoi in Impostazioni → Sicurezza.',
    saveFailed: 'Impossibile salvare la tua scelta',
    collectLabel: 'Raccogli statistiche di utilizzo',
    collectDesc:
      'Contatori limitati salvati su questo dispositivo. Mai prompt, file, percorsi né testi di errore.',
    sendLabel: 'Invia statistiche di utilizzo a Nous',
    sendDesc:
      'Carica ogni pacchetto giornaliero sul servizio di telemetria di Nous. Vengono inviati solo dati di una finestra di consenso. Richiede la raccolta attivata.',
    unavailable: 'Aggiorna il backend di Hermes per cambiare questa impostazione.',
    stripBody: 'Solo contatori limitati, mai prompt né file.',
    stripChoices: { share: 'Invia a Nous', local: 'Solo locale', off: 'No, grazie' },
    stripDetails: 'Dettagli'
  },
  intro: introIt,
  connectors: {
    title: 'Connetti le tue app',
    connect: 'Connetti',
    skip: 'Non ora',
    cancel: 'Smetti di attendere',
    retry: 'Riprova',
    grant: 'Connetti di nuovo',
    connected: 'Connesso',
    checking: 'Verifica delle tue app…',
    notConnected: 'Non connesso',
    skipped: 'Ignorato',
    disabled: 'Non disponibile',
    failed: 'Impossibile connettersi',
    needsAuth: 'Accesso scaduto',
    opening: 'Apertura dell’accesso…',
    waiting: 'In attesa del tuo browser…',
    timeout: 'Stiamo ancora aspettando l’autorizzazione.',
    refresh: 'Aggiorna stato',
    connectError: 'Impossibile avviare l’autorizzazione. Riprova.',
    connectErrorFor: (app: string) => `Impossibile avviare l’autorizzazione per ${app}.`,
    unavailable: 'I connettori non sono disponibili in questa sessione.',
    ownerMissing: 'Riapri questa conversazione per gestire le sue connessioni.',
    search: 'Cerca un’app',
    empty: 'Nessuna app corrisponde',
    disclaimer: 'La connessione è facoltativa. Autorizza solo le app che vuoi far usare a Hermes.',
    execution: 'Strumenti dei connettori',
    setup: server => `Configura ${server}`,
    openInBrowser: 'Apri nel browser',
    setupCancel: 'Annulla',
    authorizedToolsUnavailable: 'Autorizzato. Strumenti non disponibili.',
    required: 'Obbligatorio'
  },
  connectorsPage: {
    title: 'Connettori',
    searchPlaceholder: (count: number) => `Cerca tra ${count} app`,
    filterCategory: 'Categoria',
    categoryAll: 'Tutte le categorie',
    uncategorised: 'Senza categoria',
    residencyLocal: 'Su questo dispositivo',
    segment: {
      all: 'Tutti',
      available: 'Disponibili',
      connected: 'Connessi',
      off: 'Disattivati'
    },
    group: {
      connected: 'Connessi',
      connectedNote: 'Prima le connessioni con errori.',
      available: 'Disponibili',
      off: 'Disattivati',
      offNote: 'Gli accessi vengono conservati.'
    },
    card: {
      kindManaged: 'Gestito',
      kindCatalog: 'MCP · Catalogo',
      kindCustom: 'MCP · Personalizzato',
      kindPlugin: (plugin: string) => `MCP · Plugin ${plugin}`,
      inCatalog: 'Nel catalogo di Hermes',
      hostedTwin: 'Versione gestita disponibile',
      alsoLocal: 'Si esegue anche su questo dispositivo',
      open: (name: string) => `Apri ${name}`,
      turnServerOn: (name: string) => `Attiva ${name}`,
      turnServerOff: (name: string) => `Disattiva ${name}`,
      state: {
        accessExpired: 'Accesso scaduto',
        available: 'Disponibile',
        connected: 'Connesso',
        connecting: 'Connessione in corso',
        connectionUnknown: 'Stato sconosciuto',
        couldNotConnect: 'Impossibile connettersi',
        offByYourOrganisation: 'Disattivato dalla tua organizzazione',
        offForYou: 'Disattivato per te',
        serverConnecting: 'Connessione in corso…',
        serverError: 'Errore',
        serverNeedsAuth: 'Richiede autenticazione',
        serverOff: 'Disattivato',
        serverOn: 'Attivato',
        serverOnUnused: 'Attivato, inutilizzato'
      },
      fact: {
        tools: (count: number) => `${count} strument${count === 1 ? 'o' : 'i'}`,
        toolsOff: (count: number) =>
          `${count} strument${count === 1 ? 'o' : 'i'} disattivat${count === 1 ? 'o' : 'i'}`,
        toolsOn: (count: number) => `${count} strument${count === 1 ? 'o' : 'i'} attivat${count === 1 ? 'o' : 'i'}`,
        toolsSomeOn: (total: number, on: number) => `${total} strumenti, ${on} attivat${on === 1 ? 'o' : 'i'}`
      },
      verb: {
        authenticate: 'Autentica',
        connect: 'Connetti',
        install: 'Installa',
        openLogs: 'Apri i log',
        reconnect: 'Connetti di nuovo',
        stopWaiting: 'Smetti di attendere',
        tryAgain: 'Riprova',
        turnBackOn: 'Riattiva'
      },
      reason: {
        finishSignIn: 'Completa l’accesso nel tuo browser.',
        reconnect: 'Connetti di nuovo perché questa app continui a funzionare.',
        serverError: 'Il server ha rifiutato la connessione.',
        serverNeedsAuth: 'Accedi perché questo server possa rispondere.'
      }
    },
    page: {
      loading: 'Lettura del catalogo e dei server di questo computer',
      emptyTitle: 'Ancora nessuna app. Aggiungi un server su questo computer per iniziare.',
      noMatchTitle: 'Nessuna app corrisponde',
      noMatchBody: 'Nessuna corrispondenza. Indica a Hermes il tuo server MCP per aggiungerlo.',
      clearSearch: 'Cancella la ricerca',
      hostedFailedTitle: 'Impossibile accedere alle app ospitate.',
      hostedFailedBody: 'I server di questo computer non sono interessati e continuano a funzionare. Nulla è stato disattivato.',
      retry: 'Riprova',
      matchesElsewhere: (count: number) => `${count} coincidenz${count === 1 ? 'a' : 'e'} in più in altri gruppi.`,
      showAllMatches: 'Mostra tutte le coincidenze',
      segmentNoMatch: (segment: string) => `Nessuna corrispondenza in ${segment}, quindi vengono mostrate tutte.`,
      freeTierNote: 'Le connessioni restano su questo computer finché non accedi.',
      signInLine: 'Accedi a Nous per usare le app gestite.',
      signIn: 'Accedi',
      managedUnavailable: 'Le app gestite non sono ancora disponibili per questo account.',
      writeFailed: 'La modifica non è stata salvata.',
      refreshFailed: 'L’elenco degli strumenti non è stato aggiornato.',
      disconnectNoAccount:
        'Hermes non ha alcun account da disconnettere qui. Aggiorna la pagina e riprova.',
      disconnectRefused:
        'Nous non è riuscito a rimuovere questo accesso al momento. Disattiva l’app con l’interruttore o riprova più tardi.'
    },
    add: {
      action: 'Aggiungi il tuo',
      title: 'Connessione a un MCP personalizzato',
      hint: 'una nuova voce in mcp.json su questo dispositivo',
      pasteLabel: 'Incolla un comando o un frammento',
      pastePlaceholder: 'npx -y @modelcontextprotocol/server-filesystem /percorso/alla/cartella',
      pasteNoMatch: 'Nulla di questo sembra un server. Compila i campi qui sotto.',
      name: 'Nome',
      nameTaken: 'Questo nome è già in uso.',
      type: 'Tipo',
      typeStdio: 'STDIO',
      typeHttp: 'HTTP trasmettibile',
      command: 'Comando di avvio',
      args: 'Argomenti',
      addArg: '+ Aggiungi argomento',
      envVars: 'Variabili d’ambiente',
      addEnvVar: '+ Aggiungi variabile d’ambiente',
      passthrough: 'Passaggio di variabili d’ambiente',
      addPassthrough: '+ Aggiungi variabile',
      cwd: 'Directory di lavoro',
      url: 'URL',
      headers: 'Intestazioni',
      addHeader: '+ Aggiungi intestazione',
      auth: 'Autenticazione',
      authNone: 'Nessuna',
      authOauth: 'OAuth',
      authBearer: 'Token Bearer',
      keyPlaceholder: 'CHIAVE',
      valuePlaceholder: 'valore',
      removeRow: 'Rimuovi questa riga',
      editJson: 'Modifica mcp.json',
      saveFailed: 'Il server non è stato salvato.'
    },
    dialog: {
      disconnect: 'Disconnetti',
      disconnectTitle: (name: string) => `Disconnettere ${name}?`,
      disconnectBody: 'Hermes smette di operare con questo account. Puoi riconnetterlo quando vuoi.',
      menuRefreshTools: 'Aggiorna gli strumenti',
      moreActions: 'Altre azioni',
      removeServerTitle: (name: string) => `Rimuovere ${name}?`,
      removeServerBody: 'La voce viene rimossa da mcp.json su questo computer. Non viene eliminato altro.',
      appSwitch: (name: string) => `Hermes può usare ${name}`,
      waysTitle: (name: string) => `Dove viene eseguito ${name}`,
      wayNotConnected: (name: string) => `Non è ancora connesso. Accedi a ${name} dal tuo browser.`,
      wayHosted: 'Gestito',
      bothOn: (name: string) => `Entrambi sono attivati, quindi Hermes vede ogni strumento di ${name} due volte.`,
      turnOffLocal: 'Disattiva il server locale',
      providedByPlugin: (plugin: string) => `Fornito dal plugin ${plugin}`,
      openPlugins: 'Apri la scheda Plugin',
      nousLine: 'Le app di Nous seguono il tuo account, non il profilo.',
      rulesReadOnly: 'Le regole non si possono cambiare al momento.',
      rulesAppOff: (name: string) => `Attiva ${name} per modificare i suoi strumenti.`,
      rulesSignIn: 'Accedi per cambiare ciò che Hermes può fare qui.',
      orgNote: (count: number) => `La tua organizzazione ha disattivato ${count} strument${count === 1 ? 'o' : 'i'}.`,
      orgLink: 'Apri l’amministrazione dei connettori',
      connectEnded: 'L’accesso non è stato completato.',
      connectOpenAgain: 'Apri di nuovo il collegamento',
      tokensPerCall: 'token per chiamata',
      usesPerMonth: 'usi in 30 giorni',
      advanced: 'Avanzato',
      advancedHint: 'la voce di mcp.json e i log'
    },
    tools: {
      title: 'Strumenti',
      notInstalledBody: 'Installalo su questo dispositivo per vedere gli strumenti che include.',
      summaryTitle: (name: string) => `Cosa può fare Hermes con ${name}`,
      summaryPreviewTitle: (name: string) => `Cosa potrebbe fare Hermes con ${name} quando lo connetti`,
      summaryCount: (count: number) => `${count} strument${count === 1 ? 'o' : 'i'}`,
      summaryAllTools: 'Tutti gli strumenti',
      summaryOther: 'Altro',
      allToolsSwitch: 'Attiva o disattiva tutti gli strumenti',
      summaryAllOn: 'tutti attivati',
      summarySomeOn: (on: number, total: number) => `${on} su ${total} attivati`,
      summaryOff: 'disattivati',
      showAllTools: (count: number) =>
        count === 1 ? `Mostra ${count} strumento` : `Mostra i ${count} strumenti`,
      showSummary: 'Mostra riepilogo',
      facetSwitch: (facet: string) => `Attiva o disattiva gli strumenti di ${facet}`,
      moreHints: (count: number) => `+${count}`,
      staleSignIn: 'Accedi per leggere l’elenco degli strumenti più recente.',
      searchCountPlaceholder: (count: number) => `Cerca tra ${count} strumenti`,
      toolList: (name: string) => `Strumenti di ${name}`,
      categorySelect: (count: number) => `${count} categori${count === 1 ? 'a' : 'e'}`,
      showDeprecated: (count: number) => `Mostra ${count} obsolet${count === 1 ? 'o' : 'i'}`,
      hideDeprecated: (count: number) => `Nascondi ${count} obsolet${count === 1 ? 'o' : 'i'}`,
      quickReadOnly: 'Sola lettura',
      quickNoDestructive: 'Disattiva le distruttive',
      quickEverythingOn: 'Tutto attivato',
      lockedHint: 'disattivato dalla tua organizzazione',
      turnToolOn: (tool: string) => `Attiva ${tool}`,
      turnToolOff: (tool: string) => `Disattiva ${tool}`,
      showDetails: (tool: string) => `Mostra cosa fa ${tool}`,
      hideDetails: (tool: string) => `Nascondi cosa fa ${tool}`,
      noMatch: 'Nessuno strumento corrisponde a questi filtri.',
      loading: 'Lettura dell’elenco degli strumenti',
      unavailableLine: 'Elenco degli strumenti non disponibile.',
      needsAuthTitle: (name: string) => `Accedi a ${name} per leggere i suoi strumenti.`,
      needsAuthBody: 'L’accesso resta su questo computer. Nulla ne esce.',
      retry: 'Riprova',
      goneTitle: (name: string) => `${name} è uscito dal catalogo.`,
      goneBody: 'Hermes non può più chiamarlo. La riga resta finché non la rimuovi, quindi nulla scompare.',
      remove: 'Rimuovi',
      offTitle: (name: string) => `${name} è disattivato.`,
      offBody: 'Attivalo con l’interruttore qui sopra per leggere gli strumenti che include.',
      signedOutTitle: 'Accedi a Nous per leggere l’elenco degli strumenti.',
      signedOutBody: 'I tuoi server su questo computer non sono interessati.',
      conflictTitle: 'Qualcuno ha cambiato questa regola mentre la modificavi.',
      conflictBody: (theyOff: number, theyOn: number) => {
        const they = [
          theyOff > 0
            ? `ha disattivato ${theyOff} strument${theyOff === 1 ? 'o' : 'i'} che hai attivat${theyOff === 1 ? 'o' : 'i'}`
            : '',
          theyOn > 0
            ? `ha lasciato attivat${theyOn === 1 ? 'o' : 'i'} ${theyOn} strument${theyOn === 1 ? 'o' : 'i'} che avevi disattivato`
            : ''
        ].filter(Boolean)

        return `${they.length > 0 ? `Questa persona ${they.join(' e ')}. ` : ''}Le tue modifiche restano a schermo; non è stato scritto nulla.`
      },
      conflictReload: 'Ricarica la sua versione',
      conflictSave: 'Salva sopra la sua versione',
      saveFailed: 'Le regole degli strumenti non sono state salvate.',
      footerDirty: (off: number, backOn: number) =>
        `${off} strument${off === 1 ? 'o' : 'i'} disattivat${off === 1 ? 'o' : 'i'}, ${backOn === 0 ? 'nessuno' : backOn} riattivat${backOn === 1 ? 'o' : 'i'}`,
      discard: 'Scarta',
      save: 'Salva modifiche',
      saving: 'Salvataggio...'
    },
    vocabulary: {
      facetRead: {
        label: 'Lettura',
        long: 'Legge dati da questa app. Non cambia nulla.'
      },
      facetWrite: {
        label: 'Scrittura',
        long: 'Crea o modifica qualcosa in questa app.'
      },
      facetDestructive: {
        label: 'Distruttivo',
        long: 'Può eliminare qualcosa da questa app in modo definitivo.'
      },
      facetUnclassified: {
        label: 'Effetto sconosciuto',
        long: 'L’app non ha mai indicato cosa fa questo strumento.'
      },
      hintReadOnly: {
        label: 'Sola lettura',
        long: 'Lo strumento dichiara di leggere soltanto.'
      },
      hintCreate: {
        label: 'Crea',
        long: 'Crea qualcosa di nuovo.'
      },
      hintUpdate: {
        label: 'Aggiorna',
        long: 'Modifica qualcosa che esiste già.'
      },
      hintDelete: {
        label: 'Elimina',
        long: 'Rimuove qualcosa.'
      },
      hintDestructive: {
        label: 'Distruttivo',
        long: 'La modifica che fa non si può annullare qui.'
      },
      hintIdempotent: {
        label: 'Ripetibile',
        long: 'Eseguirla due volte dà lo stesso risultato di eseguirla una volta.'
      },
      hintOpenWorld: {
        label: 'Esterno',
        long: 'Raggiunge qualcosa al di fuori di questa app.'
      }
    }
  },
  sessionImport: {
    title: 'Continua da un’altra app',
    subtitle: 'Porta una conversazione in Hermes e riprendila da dove l’hai lasciata.',
    action: 'Importa sessione',
    readingFrom: 'Lettura da',
    connectedComputer: 'il computer connesso',
    destination: 'Importa in',
    all: 'Tutte',
    search: 'Cerca nelle sessioni caricate',
    scanning: 'Ricerca delle conversazioni',
    scanError: 'Impossibile trovare sessioni',
    scanHelp:
      'Verifica la connessione con il backend e riprova. I backend datati potrebbero richiedere un aggiornamento.',
    empty: 'Nessuna conversazione trovata',
    emptyHelp: 'Qui compariranno le sessioni di Claude Code e Codex di questo backend.',
    noMatches: 'Nessuna conversazione corrisponde',
    searchHelp: 'Prova con un altro titolo o un’altra cartella, oppure carica altre sessioni.',
    skipped: 'Alcune voci erano vuote, non si potevano leggere o erano troppo grandi per l’anteprima.',
    more: 'Carica altre sessioni',
    messages: 'messaggi',
    choose: 'Una conversazione che vale la pena continuare',
    chooseHelp: 'Scegli una sessione per leggere la sua cronologia prima di portarla in Hermes.',
    previewLoading: 'Apertura dell’anteprima',
    previewError: 'Anteprima non disponibile',
    previewHelp: 'L’origine potrebbe essere stata spostata o modificata. Aggiorna l’elenco e riprova.',
    previewLimit: 'Anteprima accorciata per facilitare la lettura. Viene importata la conversazione completa.',
    you: 'Tu',
    snapshot: 'Questa conversazione è già in Hermes. Apri la tua copia esistente per continuare.',
    copyNotice:
      'Copia il testo della conversazione. I file di origine non cambiano. L’output degli strumenti e il ragionamento non vengono trasferiti.',
    importing: 'Importazione…',
    open: 'Apri in Hermes',
    continue: 'Continua in Hermes',
    importError: 'Impossibile importare questa conversazione.'
  },
  common: {
    apply: 'Applica',
    back: 'Indietro',
    save: 'Salva',
    saving: 'Salvataggio…',
    cancel: 'Annulla',
    change: 'Modifica',
    choose: 'Scegli',
    clear: 'Azzera',
    close: 'Chiudi',
    collapse: 'Comprimi',
    confirm: 'Conferma',
    connect: 'Connetti',
    connecting: 'Connessione in corso',
    continue: 'Continua',
    bots: 'Bot',
    copied: 'Copiato',
    copy: 'Copia',
    copyFailed: 'Impossibile copiare',
    delete: 'Elimina',
    docs: 'Docs',
    done: 'Fatto',
    error: 'Errore',
    expand: 'Espandi',
    failed: 'Non riuscito',
    formatJson: 'Formatta JSON',
    free: 'Gratis',
    loading: 'Caricamento…',
    notSet: 'Non impostato',
    refresh: 'Aggiorna',
    remove: 'Rimuovi',
    replace: 'Sostituisci',
    retry: 'Riprova',
    run: 'Esegui',
    send: 'Invia',
    set: 'Imposta',
    skip: 'Salta',
    update: 'Aggiorna',
    tryHint: term => `Prova “${term}”`,
    on: 'Attivato',
    off: 'Disattivato'
  },
  fileMenu: {
    revealFinder: 'Mostra in Finder',
    revealExplorer: 'Mostra in Esplora file',
    revealFileManager: 'Apri la cartella contenitore',
    revealInSidebar: 'Mostra nell’albero dei file',
    copyPath: 'Copia percorso',
    copyRelativePath: 'Copia percorso relativo',
    download: 'Scarica',
    downloadSaved: 'Salvato',
    downloadFailed: 'Errore durante il download',
    rename: 'Rinomina…',
    delete: 'Elimina',
    renameTitle: 'Rinomina',
    renameLabel: 'Nuovo nome',
    deleteTitle: name => `Eliminare ${name}?`,
    deleteBody: 'Verrà spostato nel Cestino; potrai ripristinarlo da lì.',
    pathCopied: 'Percorso copiato',
    revealMissing: 'Questa cartella non è su questo computer',
    revealUnavailable:
      'Questo percorso non è su questo computer: si trova sulla macchina del backend. Usa “Mostra nell’albero dei file”.'
  },
  boot: {
    ready: 'Hermes Desktop è pronto',
    desktopBootFailedWithMessage: message => `Avvio del desktop non riuscito: ${message}`,
    steps: {
      connectingGateway: 'Connessione al gateway desktop live',
      loadingSettings: 'Caricamento della configurazione di Hermes',
      loadingSessions: 'Caricamento delle sessioni recenti',
      retryingRemoteBackend: 'Riconnessione al backend remoto di Hermes…',
      startingDesktopConnection: 'Avvio della connessione desktop',
      startingHermesDesktop: 'Avvio di Hermes Desktop…'
    },
    errors: {
      backgroundExited:
        'Il servizio che esegue le tue chat si è chiuso in modo imprevisto. Riavvialo per continuare; le tue chat e le impostazioni sono al sicuro.',
      backgroundExitedDuringStartup: 'Hermes si è interrotto subito dopo l’avvio.',
      backendStopped: 'Hermes ha smesso di funzionare in background',
      restartHermes: 'Riavvia Hermes',
      openLogs: 'Apri i log',
      desktopBootFailed: 'Impossibile avviare Hermes',
      gatewayConnectionLost: 'Hermes ha perso la connessione',
      gatewayConnectionLostDetail:
        'Stiamo provando a riconnetterci. Puoi continuare a leggere e a scrivere bozze. Se il problema persiste, riconnetti ora o controlla le impostazioni di connessione.',
      reconnectNow: 'Riconnetti ora',
      connectionSettings: 'Configurazione della connessione',
      gatewaySignInRequired: 'Il tuo Hermes remoto ti ha disconnesso',
      gatewaySignInRequiredDetail: 'Accedi di nuovo per riconnetterti. Le tue chat e le impostazioni sono al sicuro.',
      signInAgain: 'Accedi di nuovo',
      ipcBridgeUnavailable: 'Hermes Desktop non è riuscito a comunicare con il proprio livello in background. Riavvia l’app.'
    },
    causes: {
      exitedEarly: 'Il servizio in background di Hermes si è interrotto subito dopo l’avvio.',
      timedOut: 'Il servizio in background di Hermes non ha risposto in tempo.',
      permission: 'Hermes non è riuscito a scrivere nella propria cartella dati (problema di permessi).',
      diskFull: 'Il disco è pieno, quindi Hermes non è riuscito ad avviarsi.',
      portInUse: 'Un altro programma sta usando la porta di rete di cui Hermes ha bisogno.',
      installMissing: 'Manca una parte dell’installazione di Hermes. Scegli Ripara installazione per ripristinarla.'
    },
    failure: {
      title: 'Impossibile avviare Hermes',
      description:
        'Il servizio in background di Hermes non si è avviato. Prova una delle azioni di recupero qui sotto. Nessuna di queste elimina le tue chat o le tue impostazioni.',
      details: 'Dettagli',
      remoteTitle: 'È necessario accedere al gateway remoto',
      remoteDescription:
        'La tua sessione del gateway remoto è scaduta. Accedi di nuovo per riconnetterti. Questa operazione non elimina le tue chat né la tua configurazione.',
      retry: 'Riprova',
      repairInstall: 'Ripara installazione',
      useLocalGateway: 'Usa il gateway locale',
      gatewaySettings: 'Configurazione del gateway',
      back: 'Indietro',
      openLogs: 'Apri i log',
      repairHint: 'La riparazione esegue di nuovo il programma di installazione e può richiedere alcuni minuti su una macchina nuova.',
      remoteSignInHint: signInLabel =>
        `Disconnetti la sessione salvata del browser remoto e apri ${signInLabel}. Usa il gateway locale per passare al backend incluso.`,
      signOutAndSignIn: 'Disconnetti e accedi',
      remoteFailureHint: 'Controlla l’URL e accedi da Configurazione del gateway, oppure passa al gateway locale.',
      cloudDownTitle: 'L’agente di Nous Cloud non è disponibile',
      cloudDownDescription:
        'L’agente nel cloud gestito da Nous a cui si connette questo gateway restituisce un errore del server. Non è possibile riavviarlo da qui: controllane lo stato, passa al gateway locale o chiedi aiuto.',
      cloudDownHint:
        'I pulsanti qui sotto aprono il Nous Portal (stato e controlli dell’istanza) e il nostro Discord per ottenere aiuto.',
      cloudDownCheckPortal: 'Vedi lo stato nel Portal',
      cloudDownDiscord: 'Chiedi aiuto su Discord',
      hideRecentLogs: 'Nascondi log recenti',
      showRecentLogs: 'Mostra log recenti',
      signedInTitle: 'Accesso effettuato',
      signedInMessage: 'Riconnessione al gateway remoto…',
      signInIncompleteTitle: 'Accesso incompleto',
      signInIncompleteMessage: 'La finestra di accesso si è chiusa prima del termine dell’autenticazione.',
      signInFailed: 'Impossibile accedere',
      signInToRemoteGateway: 'Accedi al gateway remoto',
      signInWithProvider: provider => `Accedi con ${provider}`,
      identityProvider: 'il tuo provider di identità'
    }
  },
  notifications: {
    region: 'Notifiche',
    hide: 'Nascondi',
    show: 'Mostra',
    more: count => `${count} ${count === 1 ? 'notifica in più' : 'notifiche in più'}`,
    clearAll: 'Cancella tutto',
    dismiss: 'Ignora notifica',
    details: 'Dettagli',
    copyDetail: 'Copia dettagli',
    copyDetailFailed: 'Impossibile copiare i dettagli della notifica',
    backendOutOfDateTitle: 'Backend non aggiornato',
    backendOutOfDateMessage:
      'Il tuo backend Hermes è più vecchio di questa build desktop e potrebbe non funzionare correttamente. Aggiornalo per allinearli.',
    desktopOutOfDateTitle: 'App di Hermes non aggiornata',
    desktopOutOfDateMessage:
      'Questa app di Hermes è più vecchia del backend a cui è connessa e potrebbe non funzionare correttamente. Aggiorna l’app per allineare le versioni.',
    updateDesktopApp: 'Aggiorna app',
    installMethodUnsupportedTitle: 'Metodo di installazione non supportato',
    updateHermes: 'Aggiorna Hermes',
    updateReadyTitle: 'Aggiornamento pronto',
    updateReadyMessage: count => `${count} ${count === 1 ? 'nuova modifica disponibile' : 'nuove modifiche disponibili'}.`,
    updateReadyMessageUnknown: 'È disponibile un nuovo aggiornamento.',
    seeWhatsNew: 'Vedi le novità',
    mcp: {
      needsAuthTitle: 'Il server MCP richiede una nuova autenticazione',
      needsAuthMessage: name => `${name} MCP richiede una nuova autenticazione.`,
      errorTitle: 'Server MCP irraggiungibile',
      errorMessage: name => `${name} MCP non ha superato il controllo di integrità.`,
      signIn: 'Accedi',
      view: 'Vedi',
      disable: 'Disattiva',
      disabledMessage: name => `MCP ${name} disattivato. Riattivalo quando vuoi da Capacità → MCP.`,
      disableFailed: name => `Impossibile disattivare il MCP ${name}.`
    },
    errors: {
      elevenLabsNeedsKey: 'L’input vocale richiede una chiave ElevenLabs. Aggiungine una in Impostazioni → Chiavi.',
      elevenLabsRejectedKey:
        'ElevenLabs non ha accettato la tua chiave API. Aggiornala in Impostazioni → Chiavi e riprova.',
      diskFull: 'Disco pieno — libera spazio e riprova.',
      storageFailure: 'Hermes non è riuscito a salvare nella propria cartella dati. Apri Manutenzione per verificarla e ripararla.',
      gatewayAuthFailed:
        'Questo Hermes non accetta più il tuo accesso salvato. Apri Gateways ed effettua di nuovo l’accesso (o incolla un nuovo token di accesso) e riprova.',
      methodNotAllowed:
        'Il servizio in background di Hermes non è sincronizzato con l’app, probabilmente dopo un aggiornamento. Riavvialo per risolvere.',
      microphonePermission: 'Il permesso del microfono è stato negato.',
      openaiRejectedApiKey:
        'OpenAI non ha accettato la tua chiave API. Aggiornala in Impostazioni → Chiavi e riprova.',
      openaiTtsNeedsKey: 'La voce richiede una chiave OpenAI. Aggiungine una in Impostazioni → Chiavi.',
      codeSkewRestartRequired:
        'Hermes si è aggiornato, ma esegue ancora la versione precedente. Riavvialo per completare l’aggiornamento.',
      rpcOutOfSync: 'L’app e il backend sono su versioni diverse. Aggiorna entrambi.',
      restartHermesFailed: 'Impossibile riavviare Hermes'
    },
    actions: {
      restartHermes: 'Riavvia Hermes',
      openKeys: 'Apri Chiavi',
      openGateways: 'Apri Gateways',
      openMaintenance: 'Apri Manutenzione'
    },
    voice: {
      configureSpeechToText: 'Configura la conversione da voce a testo per usare la modalità vocale.',
      couldNotStartSession: 'Impossibile avviare la sessione vocale',
      microphoneAccessDenied: 'Accesso al microfono negato.',
      microphoneConstraintsUnsupported: 'Questo dispositivo non supporta i vincoli del microfono.',
      microphoneFailed: 'Errore del microfono',
      microphoneInUse: 'Il microfono è già in uso da un’altra app.',
      microphonePermissionDenied: 'Il permesso del microfono è stato negato.',
      microphoneStartFailed: 'Impossibile avviare la registrazione dal microfono.',
      microphoneUnsupported: 'Questo runtime non supporta la registrazione dal microfono.',
      noMicrophone: 'Nessun microfono trovato.',
      noSpeechDetected: 'Nessuna voce rilevata',
      playbackFailed: 'Riproduzione vocale non riuscita',
      recordingFailed: 'Registrazione vocale non riuscita',
      sayStopToEnd: phrase => `Di "${phrase}" per terminare la chat vocale.`,
      transcriptionFailed: 'Trascrizione vocale non riuscita',
      transcriptionUnavailable: 'La trascrizione vocale non è ancora disponibile.',
      tryRecordingAgain: 'Prova a registrare di nuovo.',
      unavailable: 'Voce non disponibile',
      liveEnded: 'Sessione vocale live terminata',
      liveEndedConnectionLost: 'La sessione vocale live ha perso la connessione.',
      liveEndedClosed: 'Il servizio ha chiuso la sessione vocale live.',
      liveError: 'Voce live',
      liveDelegationFailed: 'Impossibile passare la richiesta a Hermes',
      liveUnavailable: reason =>
        `La chat vocale GPT-Live non è disponibile: ${reason}. Verrà usata la conversione da voce a testo al suo posto.`
    },
    native: {
      approvalTitle: 'Approvazione necessaria',
      approvalTitleNamed: session => `Approvazione necessaria — ${session}`,
      approveAction: 'Approva',
      rejectAction: 'Rifiuta',
      inputTitle: 'Informazioni necessarie',
      inputTitleNamed: session => `È necessaria una risposta — ${session}`,
      inputBody: 'Hermes è in attesa della tua risposta.',
      turnDoneTitle: 'Hermes ha terminato',
      turnDoneBody: '',
      turnErrorTitle: 'Il turno non è riuscito',
      backgroundDoneTitle: 'Attività in background completata',
      backgroundFailedTitle: 'Attività in background non riuscita',
      creditsTitle: 'Crediti'
    }
  },
  remoteDisplayBanner: {
    message: reason =>
      `Rendering software attivo — rilevato uno schermo remoto (${reason}). L’accelerazione GPU è stata disattivata per evitare lo sfarfallio.`
  },
  billingBlock: {
    titleNous: 'Senza crediti Nous',
    titleProvider: provider => `Senza crediti — ${provider}`,
    fallbackMessage: 'Il tuo account ha esaurito i crediti. Aggiungi crediti per continuare.',
    openBilling: 'Apri la fatturazione',
    addCredits: 'Aggiungi crediti',
    dismiss: 'Ignora'
  },
  sendDiagnostics: {
    title: 'Invia la diagnostica a Nous',
    privacyNotice:
      'Questo carica un pacchetto di debug in un’archiviazione interna di Nous (non su un sito pubblico). Include informazioni di sistema (SO, versioni, provider e quali chiavi API sono configurate, mai le chiavi stesse) e i log completi dell’agente, del gateway e dell’app desktop (fino a 512 KB ciascuno), che probabilmente contengono contenuti di conversazioni, output di strumenti e percorsi di file. I segreti vengono oscurati prima del caricamento. Solo il personale di Nous e i moderatori autorizzati di Discord possono vedere il pacchetto, che viene eliminato automaticamente dopo 14 giorni.',
    upload: 'Carica',
    uploading: 'Caricamento…',
    cancel: 'Annulla',
    close: 'Chiudi',
    copyLink: 'Copia link',
    uploadIdFallback: id => `Non è stato restituito un link di visualizzazione: comunica l’ID di caricamento ${id} al team di supporto`,
    doneTitle: 'Diagnostica inviata',
    doneDescription:
      'Il tuo pacchetto è stato caricato in privato. Condividi il link qui sotto nel tuo thread di assistenza perché il team possa vedere i tuoi log.',
    failedTitle: 'Errore durante il caricamento',
    failedHint:
      'Puoi anche eseguire `hermes debug share --nous` da un terminale, oppure `hermes debug share --local` per mostrare il report senza caricarlo.',
    handoffLead: 'Continua la conversazione su:',
    links: {
      github: 'Issues di GitHub',
      portal: 'Supporto di Nous Portal',
      discord: 'Discord'
    }
  },
  titlebar: {
    hideSidebar: 'Nascondi barra laterale',
    showSidebar: 'Mostra barra laterale',
    search: 'Cerca',
    searchTitle: 'Cerca sessioni, viste e azioni',
    swapSidebarSides: 'Scambia i lati delle barre laterali',
    hideRightSidebar: 'Nascondi barra laterale destra',
    showRightSidebar: 'Mostra barra laterale destra',
    unreadSessions: count => (count === 1 ? '1 sessione non letta' : `${count} sessioni non lette`),
    muteHaptics: 'Silenzia feedback aptico',
    unmuteHaptics: 'Attiva feedback aptico',
    openSettings: 'Apri la configurazione',
    openStarmap: 'Apri il grafo della memoria',
    enterHud: 'Modalità HUD',
    exitHud: 'Esci dalla modalità HUD',
    resetHudLayout: 'Ripristina dimensioni e posizione dell’HUD',
    layoutEditor: 'Editor del layout',
    layoutEditorTitle: mod => `Editor del layout — clic ${mod} ripristina il layout`
  },
  keybinds: {
    title: 'Scorciatoie da tastiera',
    subtitle: open => `Fai clic su una scorciatoia per riassegnarla · ${open} riapre questo pannello.`,
    search: 'Cerca scorciatoie…',
    rebind: 'Riassegna',
    reset: 'Ripristina predefinito',
    resetAll: 'Ripristina tutto',
    pressKey: 'Premi un tasto…',
    set: 'definito',
    conflictWith: label => `Assegnato anche a “${label}”`,
    categories: {
      composer: 'Compositore',
      profiles: 'Profili',
      session: 'Sessione',
      navigation: 'Navigazione',
      view: 'Vista'
    },
    actions: {
      'keybinds.openPanel': 'Apri scorciatoie da tastiera',
      'nav.commandPalette': 'Apri palette dei comandi',
      'nav.commandCenter': 'Apri Centro comandi',
      'nav.settings': 'Apri configurazione',
      'nav.profiles': 'Apri profili',
      'nav.capabilities': 'Apri skills',
      'nav.messaging': 'Apri messaggistica',
      'nav.artifacts': 'Apri artefatti',
      'nav.cron': 'Apri attività pianificate',
      'nav.agents': 'Apri agenti',
      'session.new': 'Nuova sessione',
      'session.newTab': 'Nuova scheda di sessione',
      'session.newWindow': 'Nuova finestra',
      'session.next': 'Sessione successiva',
      'session.prev': 'Sessione precedente',
      'session.slot.1': 'Vai alla sessione recente 1',
      'session.slot.2': 'Vai alla sessione recente 2',
      'session.slot.3': 'Vai alla sessione recente 3',
      'session.slot.4': 'Vai alla sessione recente 4',
      'session.slot.5': 'Vai alla sessione recente 5',
      'session.slot.6': 'Vai alla sessione recente 6',
      'session.slot.7': 'Vai alla sessione recente 7',
      'session.slot.8': 'Vai alla sessione recente 8',
      'session.slot.9': 'Vai alla sessione recente 9',
      'session.focusSearch': 'Cerca sessioni',
      'session.togglePin': 'Fissa / sfissa la sessione corrente',
      'session.archive': 'Archivia la sessione corrente',
      'workspace.newWorktree': 'Nuovo worktree',
      'workspace.openFolder': 'Apri cartella come progetto',
      'composer.focus': 'Attiva il compositore',
      'composer.modelPicker': 'Apri selettore modello',
      'composer.voice': 'Avvia / interrompi conversazione vocale',
      'composer.reasoningUp': 'Aumenta livello di ragionamento',
      'composer.reasoningDown': 'Riduci livello di ragionamento',
      'view.toggleSidebar': 'Mostra/Nascondi barra laterale delle sessioni',
      'view.cycleSidebarGrouping': 'Cambia il raggruppamento delle sessioni',
      'view.toggleRightSidebar': 'Mostra/Nascondi esploratore di file',
      'view.toggleReview': 'Mostra/Nascondi pannello di revisione',
      'view.toggleStatusbar': 'Mostra/Nascondi barra di stato',
      'view.toggleTabStrip': 'Mostra o nascondi le schede',
      'view.toggleProfileRail': 'Mostra o nascondi la barra dei profili',
      'view.toggleSimpleMode': 'Attiva o disattiva la modalità semplice',
      'view.showFiles': 'Mostra esploratore di file',
      'view.showBrowser': 'Mostra/Nascondi browser',
      'view.toggleHud': 'Mostra/Nascondi modalità HUD',
      'hud.snapToPointer': 'Sposta l’HUD sul puntatore (globale, mentre l’HUD è aperto)',
      'view.showTerminal': 'Mostra terminale',
      'view.newTerminal': 'Nuovo terminale',
      'view.nextTerminal': 'Terminale successivo',
      'view.prevTerminal': 'Terminale precedente',
      'view.closeTerminal': 'Chiudi terminale',
      'view.selectionToComposer': 'Invia la selezione al compositore',
      'view.terminalCopy': 'Copia la selezione dal terminale',
      'view.terminalPaste': 'Incolla nel terminale',
      'view.closeTab': 'Chiudi scheda',
      'view.reopenTab': 'Riapri scheda chiusa',
      'view.flipPanes': 'Inverti i lati delle barre laterali',
      'view.findInPage': 'Trova nella pagina',
      'view.findNext': 'Corrispondenza successiva',
      'view.findPrevious': 'Corrispondenza precedente',
      'view.tabSlot.1': 'Vai alla scheda 1',
      'view.tabSlot.2': 'Vai alla scheda 2',
      'view.tabSlot.3': 'Vai alla scheda 3',
      'view.tabSlot.4': 'Vai alla scheda 4',
      'view.tabSlot.5': 'Vai alla scheda 5',
      'view.tabSlot.6': 'Vai alla scheda 6',
      'view.tabSlot.7': 'Vai alla scheda 7',
      'view.tabSlot.8': 'Vai alla scheda 8',
      'view.tabSlot.9': 'Vai alla scheda 9',
      'appearance.toggleMode': 'Attiva/disattiva chiaro / scuro',
      'profile.default': 'Passa al profilo predefinito',
      'profile.switch.1': 'Passa al profilo 1',
      'profile.switch.2': 'Passa al profilo 2',
      'profile.switch.3': 'Passa al profilo 3',
      'profile.switch.4': 'Passa al profilo 4',
      'profile.switch.5': 'Passa al profilo 5',
      'profile.switch.6': 'Passa al profilo 6',
      'profile.switch.7': 'Passa al profilo 7',
      'profile.switch.8': 'Passa al profilo 8',
      'profile.switch.9': 'Passa al profilo 9',
      'profile.switch.10': 'Passa al profilo 10',
      'profile.switch.11': 'Passa al profilo 11',
      'profile.switch.12': 'Passa al profilo 12',
      'profile.switch.13': 'Passa al profilo 13',
      'profile.switch.14': 'Passa al profilo 14',
      'profile.switch.15': 'Passa al profilo 15',
      'profile.switch.16': 'Passa al profilo 16',
      'profile.switch.17': 'Passa al profilo 17',
      'profile.switch.18': 'Passa al profilo 18',
      'profile.next': 'Profilo successivo',
      'profile.prev': 'Profilo precedente',
      'profile.toggleAll': 'Mostra/Nascondi la vista di tutti i profili',
      'profile.create': 'Crea profilo',
      'composer.send': 'Invia messaggio',
      'composer.newline': 'Inserisci nuova riga',
      'composer.steer': 'Guida il turno in corso',
      'composer.queue': 'Metti il messaggio in coda',
      'composer.sendQueued': 'Invia il prossimo turno in coda',
      'composer.mention': 'Menziona file, cartelle e URL',
      'composer.slash': 'Palette dei comandi slash',
      'composer.help': 'Aiuto rapido',
      'composer.history': 'Scorri popover / cronologia',
      'composer.cancel': 'Chiudi popover · annulla esecuzione'
    }
  },
  findInPage: {
    next: 'Corrispondenza successiva',
    previous: 'Corrispondenza precedente'
  },
  language: {
    label: 'Lingua',
    description: 'Scegli la lingua dell’interfaccia desktop.',
    saving: 'Salvataggio lingua…',
    saveError: 'Impossibile aggiornare la lingua',
    switchTo: 'Cambia lingua',
    searchPlaceholder: 'Cerca lingue…',
    noResults: 'Nessuna lingua trovata'
  },
  settings: {
    subpages: {
      appearanceTheme: 'Tema',
      appearanceTypography: 'Tipografia',
      appearanceWindowLayout: 'Finestra e layout',
      appearanceChatDisplay: 'Visualizzazione della chat',
      appearancePet: 'Mascotte',
      appearanceGeneral: 'Generale',
      modelMain: 'Modello principale',
      modelAuxiliary: 'Modelli ausiliari',
      modelMoa: 'Mixture of Agents',
      modelFallbacks: 'Modelli di riserva',
      chatBehavior: 'Comportamento',
      chatAttachments: 'Allegati',
      workspaceProjects: 'Progetti e rilevamento',
      workspaceShell: 'Ambiente della shell',
      workspaceFiles: 'File ed esecuzione',
      safetyApprovals: 'Approvazioni',
      safetyPrivacy: 'Privacy e rete',
      safetyCheckpoints: 'Checkpoint',
      browserProfile: 'Profilo del browser',
      browserNetwork: 'URL locali e private',
      memoryPersistent: 'Memoria persistente',
      memoryContext: 'Contesto e compressione',
      voiceConversation: 'Conversazione vocale',
      voiceTranscription: 'Voce in testo',
      voiceSpeech: 'Testo in voce',
      advancedRuntime: 'Limiti dell’agente',
      advancedTools: 'Accesso agli strumenti',
      advancedTerminal: 'Backend del terminale',
      advancedOutput: 'Limiti di output',
      advancedDelegation: 'Subagenti',
      advancedDesktop: 'Desktop e avvio',
      gatewayConnection: 'Questa finestra',
      gatewayDevices: 'Connessioni salvate',
      gatewayManagedUpdates: 'Aggiornamenti remoti',
      gatewayManagedUpdatesUnavailable:
        'Gli aggiornamenti remoti richiedono una versione desktop compatibile con aggiornamenti SSH gestiti.',
      gatewayManagedUpdatesEmpty:
        'Aggiungi una connessione SSH in Connessioni salvate per gestirne qui gli aggiornamenti.',
      keyboardShortcuts: 'Scorciatoie da tastiera',
      hudGesture: 'Gesto dell’HUD',
      screenCapture: 'Acquisizione schermo',
      notificationAlerts: 'Avvisi del desktop',
      notificationSounds: 'Suoni',
      archivedSessions: 'Archiviazione e conservazione',
      defaultDirectory: 'Cartella di progetto predefinita',
      vaultCredentials: 'Credenziali salvate',
      vaultSources: 'Gestori di password',
      appUpdates: 'Versione e aggiornamenti',
      uninstall: 'Disinstalla',
      billingOverview: 'Riepilogo',
      billingPlans: 'Piani'
    },
    closeSettings: 'Chiudi configurazione',
    exportConfig: 'Esporta configurazione',
    importConfig: 'Importa configurazione',
    resetToDefaults: 'Ripristina valori predefiniti',
    resetConfirm: 'Ripristinare tutta la configurazione ai valori predefiniti di Hermes?',
    exportFailed: 'Esportazione non riuscita',
    resetFailed: 'Ripristino non riuscito',
    pluginPages: {
      blurb:
        'Opzioni aggiunte dai plugin installati. Ogni plugin ha una propria pagina e alcuni aggiungono sottopagine.',
      empty: 'Nessun plugin ha ancora impostazioni.',
      manage: 'Gestisci plugin',
      agentSettings: 'Impostazioni dell’agente',
      pageCount: (n: number) => (n === 1 ? '1 pagina' : `${n} pagine`),
      missing: 'Questo plugin non ha una pagina di impostazioni. Potrebbe essere disattivato o disinstallato.'
    },
    nav: {
      providers: 'Provider',
      providerAccounts: 'Account',
      providerApiKeys: 'Chiavi API',
      providerCustomEndpoints: 'Endpoint personalizzati',
      providerLocalModels: 'Modelli locali',
      gateway: 'Gateway',
      apiKeys: 'Strumenti e chiavi',
      keybinds: 'Scorciatoie da tastiera',
      keysTools: 'Strumenti',
      keysSettings: 'Configurazione',
      mcp: 'MCP',
      archivedChats: 'Chat archiviate',
      sessions: 'Sessioni',
      about: 'Informazioni',
      billing: 'Fatturazione',
      notifications: 'Notifiche',
      vault: 'Password e accessi'
    },
    plugins: {
      title: 'Plugin desktop',
      openFolder: 'Apri la cartella dei plugin desktop',
      rescan: 'Ripeti la ricerca',
      reveal: 'Mostra nel gestore di file',
      failed: 'fallito',
      kinds: {
        bundled: 'incluso',
        disk: 'su disco',
        runtime: 'in esecuzione'
      },
      installModal: {
        installFromGit: 'Installa da Git',
        reviewRepository: 'Esamina il repository',
        repoPlaceholder: 'https://github.com/propietario/repo',
        title: 'Installa plugin',
        description: 'Controlla cosa contiene questo repository prima di installare qualcosa.',
        repoLabel: 'Repository',
        includesHeading: 'Questo pacchetto include',
        agentLabel: 'Plugin dell’agente',
        desktopLabel: 'Interfaccia desktop',
        profileLabel: 'Installa per il profilo',
        agentTargetLocal: (profile, dir) => `Si installa nel backend ${profile} (${dir})`,
        agentTargetRemote: profile => `Si installa nel backend ${profile} connesso`,
        catalogPinned: (name, sha) =>
          `Voce del catalogo di Hermes “${name}”: il componente dell’agente viene installato alla versione fissata e verificata${sha ? ` ${sha}` : ''}, non alla punta del branch.`,
        reviewedHeading: 'Voce del catalogo verificata',
        reviewedIntro:
          'Una persona ha verificato questa voce al suo commit fissato. Puoi comunque ispezionare il codice esatto qui sotto.',
        toolsConnected: n => (n === 1 ? '1 strumento connesso' : `${n} strumenti connessi`),
        skillsReady: names => (names.length === 1 ? `skill ${names[0]} pronta` : `${names.length} skills pronte`),
        nextChat: 'altri strumenti disponibili nella tua prossima chat',
        serverNotConnected: (server, reason) =>
          `Il server MCP ${server} non è connesso${reason ? `: ${reason}` : '.'}`,
        missingEnvAction: 'Configuralo',
        alreadyInstalled: (name: string) => `${name} è già installato.`,
        desktopTarget: 'Si installa nella cartella locale desktop-plugins di questa app',
        desktopTargetFromPackage: 'Viene caricato in questa app dal pacchetto qui sopra; è uguale per tutti i profili',
        desktopOnlyNote: 'I pacchetti solo desktop non installano un plugin dell’agente nel backend.',
        insecureWarning:
          'Questa URL usa uno schema non sicuro o locale. Per le installazioni di produzione, usa https:// o git@.',
        securityHeading: 'Prima di installare',
        securityIntro:
          'Installa solo da fonti attendibili; controlla il repository qui sotto se vuoi vedere cosa verrà aggiunto.',
        sourceHeading: 'Codice sorgente',
        viewRepository: 'Visualizza repository',
        viewPluginFiles: 'Visualizza i file del plugin',
        gitCloneLabel: 'URL di git clone',
        enableAgent: 'Attiva il plugin dell’agente dopo l’installazione',
        forceReinstall: 'Forza reinstallazione (sostituisci se già installato)',
        pinToCommit: 'Fissa a un commit (opzionale)',
        pinToCommitPlaceholder: 'SHA di commit completo di 40 caratteri',
        pinToCommitHint:
          'Chiunque installi questo SHA ottiene lo stesso codice; poi il plugin rifiuta gli aggiornamenti finché non viene fissato di nuovo. Lascialo vuoto per usare l’ultimo commit.',
        pinToCommitInvalid: 'Deve essere uno SHA di commit completo di 40 caratteri (non sono accettati branch né tag).',
        install: 'Installa',
        installing: 'Installazione…',
        probing: 'Ispezione del repository…',
        probeUnavailable: 'L’ispezione dei plugin non è disponibile in questo ambiente.',
        desktopUnavailable: 'L’installazione dei plugin desktop non è disponibile in questo ambiente.',
        selectComponent: 'Seleziona almeno un componente da installare.',
        agentSuccess: name => `Plugin dell’agente ${name} installato`,
        desktopSuccess: name => `Plugin desktop ${name} installato`,
        agentFailed: 'Errore durante l’installazione del plugin dell’agente',
        installUncertain:
          'Hermes ha smesso di attendere l’esito dell’installazione, ma è possibile che il plugin sia ancora in installazione. Chiudi questa finestra e aggiorna l’elenco dei plugin prima di reinstallarlo.',
        desktopFailed: 'Errore durante l’installazione del plugin desktop',
        missingEnv: (name, vars) =>
          `${name} è installato, ma richiede una chiave per funzionare: ${vars}. Aggiungila ora o gli strumenti del plugin falliranno.`
      }
    },
    vault: {
      title: 'Password e accessi',
      blurb:
        'Di “accedi a GitHub” e l’agente lo farà per te. La prima volta che incontra una pagina di accesso, ti chiederà i dati subito lì; dopo, funzionerà semplicemente. Le password vengono cifrate su questo computer e inserite direttamente nella pagina: il modello non le vede mai.',
      count: n => `${n} item${n === 1 ? '' : 's'}`,
      loadFailed: 'Impossibile caricare gli elementi salvati',
      empty: 'Non c’è ancora niente salvato',
      emptyDesc:
        'Non serve aggiungere nulla qui. Chiedi all’agente di accedere a un sito e ti chiederà i dati una sola volta, in quel momento. Usa Aggiungi se preferisci inserirli in anticipo.',
      add: 'Aggiungi',
      addTitle: 'Aggiungi un accesso, una carta o un indirizzo',
      addDescription: 'Viene salvato in forma cifrata su questo computer. L’agente non vede mai la password.',
      added: 'Salvato.',
      adding: 'Salvataggio…',
      addConfirm: 'Salva',
      kindField: 'Tipo',
      kinds: {
        login: 'Accesso',
        payment: 'Carta di pagamento',
        address: 'Indirizzo'
      },
      labelField: 'Etichetta',
      labelPlaceholder: 'ad es., account di lavoro di GitHub',
      labelRequired: 'L’etichetta è obbligatoria.',
      originField: 'Origine del sito',
      originPlaceholder: 'https://github.com',
      originPlaceholderCheckout: 'https://tienda.example.com',
      originInvalid: 'Inserisci una URL valida, come https://example.com.',
      identifierTypeField: 'Tipo di identificatore',
      identifierTypes: {
        email: 'Email',
        phone: 'Telefono',
        username: 'Nome utente'
      },
      identifierField: 'Identificatore',
      identifierShown: identifier => identifier,
      passwordField: 'Password',
      loginFieldsRequired: 'Identificatore e password sono obbligatori.',
      cardNumberField: 'Numero della carta',
      cardNameField: 'Nome sulla carta',
      expMonthField: 'Mese di scad.',
      expYearField: 'Anno di scad.',
      cvcField: 'CVC',
      postalField: 'CAP',
      addressLine1Field: 'Indirizzo, riga 1',
      addressLine2Field: 'Indirizzo, riga 2',
      cityField: 'Città',
      stateField: 'Provincia / regione',
      countryField: 'Paese',
      optional: '(opzionale)',
      createdOn: date => `Aggiunto il ${date}`,
      deleteAction: 'Rimuovi elemento salvato',
      otpField: 'Chiave dell’autenticatore',
      otpPlaceholder: 'Segreto Base32 o link otpauth://',
      otpHint:
        'La “chiave di configurazione” che il sito mostra all’attivazione della 2FA. Se la salvi, Hermes genera i codici da solo.',
      twoFactorBadge: '2FA automatica',
      deleteTitle: 'Eliminare questo elemento?',
      deleteDescription: label => `Verrà rimosso “${label}”. Questa azione non può essere annullata.`,
      deleteConfirm: 'Elimina',
      sources: {
        title: 'Gestori di password',
        blurb:
          'I gestori di password installati vengono rilevati automaticamente. L’agente ti chiede di sbloccarne uno la prima volta che ha bisogno di un accesso da esso (una volta per sessione); in memoria viene salvato solo un token di sessione, e l’agente non vede mai la tua password principale né alcun accesso.',
        toggleFailed: 'Impossibile aggiornare il gestore di password',
        notInstalled: name =>
          `Non rilevato. Installa lo strumento a riga di comando di ${name} e accedi ad esso; Hermes lo rileverà automaticamente.`,
        disabledDesc: 'Rilevato, ma disattivato per Hermes.',
        lockedDesc:
          'Rilevato. L’agente ti chiederà di sbloccarlo quando avrà bisogno di un accesso, oppure puoi sbloccarlo ora.',
        unlockedDesc:
          'Sbloccato per questa sessione. Si blocca automaticamente dopo 30 minuti di inattività o alla chiusura di Hermes.',
        statusLocked: 'Bloccato',
        statusNotDetected: 'Non rilevato',
        statusOff: 'Disattivato',
        statusUnlocked: 'Sbloccato',
        unlock: 'Sblocca',
        unlocking: 'Sblocco…',
        lock: 'Blocca',
        unlocked: name => `${name} sbloccato per questa sessione.`,
        unlockTitle: name => `Sblocca ${name}`,
        unlockDescription:
          'Inserisci la tua password principale. Viene consegnata al gestore di password di questo computer e poi scartata: non viene mai salvata, registrata nei log né mostrata all’agente.',
        masterPasswordPlaceholder: 'Password principale'
      }
    },
    notifications: {
      title: 'Notifiche',
      intro: 'Notifiche del sistema operativo (non avvisi all’interno dell’app). Per dispositivo.',
      enableAll: 'Attiva notifiche',
      enableAllDesc: 'Se disattivato, silenzia tutte le notifiche seguenti.',
      focusedHint: 'Gli avvisi di completamento si attivano solo quando Hermes è in secondo piano.',
      kinds: {
        approval: {
          label: 'Approvazione necessaria',
          description: 'Un comando attende che tu lo approvi o lo rifiuti.'
        },
        input: {
          label: 'Informazioni richieste',
          description: 'Hermes ha fatto una domanda o ha bisogno di una password o di un segreto.'
        },
        turnDone: {
          label: 'Risposta pronta',
          description: 'Un turno è terminato mentre Hermes era in secondo piano.'
        },
        turnError: {
          label: 'Il turno non è riuscito',
          description: 'Errori di turni in secondo piano.'
        },
        backgroundDone: {
          label: 'Attività in secondo piano completata',
          description: 'È stato completato un comando di terminale in secondo piano.'
        },
        credits: {
          label: 'Avvisi sui crediti',
          description: 'L’accesso ai crediti viene messo in pausa o ripristinato.'
        },
        plugin: {
          label: 'Notifiche dei plugin',
          description: 'Un plugin desktop ha inviato una notifica mentre Hermes era in secondo piano.'
        }
      },
      test: 'Invia notifica di test',
      testTitle: 'Hermes',
      testBody: 'Le notifiche funzionano.',
      testSent:
        'Test inviato. Se non compare nulla, controlla i permessi di notifica del sistema operativo e la modalità Concentrazione o Non disturbare.',
      testUnsupported: 'Questo sistema non supporta le notifiche native.',
      completionSoundTitle: 'Suono di completamento',
      completionSoundDesc:
        'Riproduce un suono quando termina il turno di un agente. Scegli un’impostazione predefinita e provala qui.',
      completionSoundPreview: 'Anteprima'
    },
    sections: {
      model: 'Modello',
      chat: 'Chat',
      appearance: 'Aspetto',
      workspace: 'Workspace',
      safety: 'Sicurezza',
      memory: 'Memoria e contesto',
      voice: 'Voce',
      advanced: 'Avanzate'
    },
    searchPlaceholder: {
      about: 'Informazioni su Hermes Desktop',
      config: 'Cerca configurazione...',
      gateway: 'Connessione del gateway...',
      keys: 'Cerca chiavi API...',
      mcp: 'Cerca server MCP...',
      sessions: 'Cerca sessioni archiviate...'
    },
    modeOptions: {
      light: {
        label: 'Chiaro',
        description: 'Superfici desktop luminose'
      },
      dark: {
        label: 'Scuro',
        description: 'Workspace con luminosità ridotta'
      },
      system: {
        label: 'Sistema',
        description: 'Segui l’aspetto del SO'
      }
    },
    appearance: {
      chatTextScaleTitle: 'Dimensione del testo della chat',
      chatTextScaleDesc:
        'Regola il testo della conversazione e dell’editor rispetto alla scala dell’interfaccia. Le barre laterali e i controlli mantengono la loro dimensione.',
      title: 'Aspetto',
      intro: 'Solo desktop. La modalità è la luminosità; il tema è la palette e la cornice della chat.',
      colorMode: 'Modalità colore',
      colorModeDesc: 'Scegli una modalità fissa o lascia che Hermes segua la configurazione del sistema.',
      toolViewTitle: 'Visualizzazione delle chiamate agli strumenti',
      toolViewDesc: 'Prodotto nasconde i payload grezzi; Tecnico mostra input/output completi.',
      hideCodeDiffsTitle: 'Nascondi i diff del codice',
      hideCodeDiffsDesc:
        'Mostra le modifiche ai file come righe di strumento con il conteggio delle righe aggiunte/eliminate, senza il codice.',
      hideThreadTimelineTitle: 'Nascondi le barre della linea temporale',
      hideThreadTimelineDesc: 'Nasconde le barre di navigazione sul bordo destro di ogni conversazione.',
      reasoningCollapsedTitle: 'Comprimi il ragionamento per impostazione predefinita',
      reasoningCollapsedDesc: 'Mantiene disponibile il ragionamento in streaming senza espanderlo finché non lo apri.',
      uiScaleTitle: 'Scala dell’interfaccia',
      uiScaleDesc: percent =>
        `Ridimensiona il testo e i controlli di tutta l’app. Funziona anche Cmd/Ctrl con +, - e 0. Attuale: ${percent}%.`,
      sessionDensityTitle: 'Densità dell’elenco delle sessioni',
      sessionDensityDesc: 'Scegli quanto contesto appare sotto i titoli delle sessioni nella barra laterale.',
      sessionDensityCompact: 'Compatta',
      sessionDensityComfortable: 'Comoda',
      sessionDensityDetailed: 'Dettagliata',
      tabStripTitle: 'Barra delle schede',
      tabStripDesc:
        'Mostra le schede sopra una zona. Auto le nasconde se c’è un solo pannello, a meno che ci sia un’altra zona di chat o di mosaico aperta.',
      tabStripAuto: 'Auto',
      tabStripAlways: 'Sempre',
      tabStripNever: 'Mai',
      appActionsTitle: 'Azioni dell’app',
      appActionsDesc:
        'Dove vengono posizionati Impostazioni, Layout e HUD nella barra del titolo. A destra lascia spazio per le schede a sinistra.',
      appActionsLeft: 'Sinistra',
      appActionsRight: 'Destra',
      terminalFontTitle: 'Font del terminale',
      terminalFontDesc:
        'Scegli un font installato per i terminali del desktop. Le Nerd Fonts renderizzano Powerlevel10k e le icone della shell; lascialo vuoto per usare la JetBrains Mono inclusa.',
      terminalFontPlaceholder: 'MesloLGS NF o una pila di font CSS',
      terminalFontPreview: 'Anteprima dei glifi',
      terminalFontReset: 'Usa quella predefinita',
      chatFontTitle: 'Font della chat',
      chatFontDesc:
        'Scegli un font installato per la chat e il resto dell’app. Utile per font di lettura come OpenDyslexic; lascialo in bianco per usare il font del tema.',
      chatFontPlaceholder: 'OpenDyslexic o una pila di font CSS',
      chatFontPreview: 'Anteprima',
      chatFontSample: 'Ma la volpe, col suo balzo, ha raggiunto il quieto Fido. 0123456789',
      chatFontReset: 'Usa il font del tema',
      translucencyTitle: 'Trasparenza della finestra',
      translucencyDesc: 'Vedrai il tuo desktop attraverso tutta la finestra. Solo macOS e Windows.',
      translucencyGlassDesc:
        'Vetro satinato: il desktop si vede attraverso con una sfocatura leggera mentre il testo resta nitido. Regolabile separatamente per chiaro e scuro.',
      translucencyModeClear: 'Trasparente',
      translucencyModeGlass: 'Vetro',
      translucencyTintTitle: 'Tinta',
      translucencyFadeTitle: 'Dissolvenza',
      translucencyFrostTitle: 'Brina',
      translucencyFrost: {
        'under-window': 'Profonda',
        popover: 'Leggera',
        titlebar: 'Luminosa',
        header: 'Bagliore'
      },
      translucencyScopeTitle: 'Area',
      translucencyScope: {
        window: 'Tutta la finestra',
        sidebar: 'Solo la barra laterale'
      },
      backdropTitle: 'Sfondo della chat',
      backdropDesc: 'La tenue immagine della statua dietro la conversazione.',
      userBubbleTitle: 'Bolla del messaggio',
      userBubbleDesc: 'Quanta trasparenza hanno i tuoi messaggi. Opaca a 0; a 100 resta solo il contorno.',
      textDirectionTitle: 'Direzione del testo',
      textDirectionDesc:
        'Come scelgono la propria direzione i messaggi della chat e il campo di scrittura. Auto segue la prima lettera di ogni paragrafo; scegli una direzione quando un testo misto si allinea male. Il codice va sempre da sinistra a destra.',
      textDirection: { auto: 'Auto', rtl: 'Da destra a sinistra', ltr: 'Da sinistra a destra' },
      introSplashTitle: 'Schermata di benvenuto',
      introSplashDesc: 'Il logotipo e la dicitura mostrati in una chat vuota.',
      modelPricingTitle: 'Prezzi dei modelli',
      modelPricingDesc:
        'Mostra i prezzi di input, output e lettura della cache per milione di token nel selettore dei modelli.',
      reactionsTitle: 'Reazioni ai messaggi',
      reactionsDesc:
        'Reazioni emoji in stile iMessage — reagisci ai messaggi, e Hermes può reagire ai tuoi.',
      tipsTitle: 'Consigli nell’app',
      tipsDesc:
        'Suggerimenti occasionali dell’app e di Hermes. Ogni consiglio appare una volta sola. Si disattiva automaticamente dopo i tuoi primi 30 giorni; puoi riattivarlo.',
      tipsReset: (count: number) => `Mostra di nuovo ${count} ${count === 1 ? 'consiglio' : 'consigli'}`,
      toursTitle: 'Tour guidati',
      toursDesc:
        'Lascia che Hermes evidenzi ogni passaggio mentre ti guida nell’app. Si disattiva automaticamente dopo i tuoi primi 30 giorni; puoi riattivarlo.',
      composerPopoutTitle: 'Compositore flottante',
      composerPopoutDesc:
        'Consente di trascinare il compositore fuori dalla posizione fissa. Se disattivato, resta ancorato in basso.',
      fileBrowserTitle: 'Esplora file',
      fileBrowserDesc:
        'Mostra l’esplora file accanto alla chat quando c’è un workspace aperto. Il pulsante nella barra del titolo modifica anche questa impostazione.',
      vibeHeartsTitle: 'Cuori di vibe',
      vibeHeartsDesc:
        'Cuori fluttuanti quando dici grazie, ti voglio bene, bel bot o invii un cuore. Indipendente dalle reazioni ai messaggi di sopra.',
      embedsTitle: 'Contenuto incorporato',
      embedsDesc:
        'Le anteprime avanzate vengono caricate da siti di terze parti (YouTube, X, …). Chiedi mostra un segnaposto finché non li consenti singolarmente; Sempre li carica automaticamente; Disattivato mantiene i link semplici.',
      embedsAsk: 'Chiedi',
      embedsAlways: 'Sempre',
      embedsOff: 'Disattivato',
      embedsReset: count => `Ripristina ${count} ${count === 1 ? 'servizio consentito' : 'servizi consentiti'}`,
      resumeLastSessionTitle: 'Riapri l’ultima chat all’avvio',
      resumeLastSessionDesc:
        'Se attivato, l’app riapre la chat più recente all’avvio a freddo. Disattivalo per iniziare sempre con una chat nuova.',
      product: 'Prodotto',
      productDesc: 'Attività degli strumenti leggibile con riepiloghi concisi.',
      technical: 'Tecnico',
      technicalDesc: 'Include argomenti/risultati grezzi e dettagli di basso livello.',
      themeTitle: 'Tema',
      themeDesc: 'Palette solo per desktop. Si applicano sopra la modalità selezionata.',
      themeSearchPlaceholder: 'Cerca nei tuoi temi o nel VS Code Marketplace…',
      themeProfileNote: profile => `Salvato per il profilo ${profile}; ogni profilo conserva il proprio tema.`,
      installTitle: 'Installa da VS Code',
      installDesc:
        'Incolla un ID di estensione del Marketplace (ad esempio, dracula-theme.theme-dracula) per convertire il suo tema di colore in una palette desktop.',
      installPlaceholder: 'publisher.extension',
      installButton: 'Installa',
      installing: 'Installazione…',
      installError: 'Impossibile installare quel tema.',
      installed: name => `Installato “${name}”.`,
      removeTheme: 'Rimuovi tema',
      importedBadge: 'Importato',
      pet: {
        title: 'Mascotte',
        intro:
          'Adotta una mascotte animata di petdex che fluttua sopra l’app e reagisce a ciò che fa Hermes: corre mentre vengono eseguiti gli strumenti, festeggia i successi e si rattrista per gli errori.',
        restartHint:
          'Le mascotte richiedono un rapido riavvio: l’applicazione in esecuzione è stata avviata prima che questa funzione venisse aggiunta. Chiudi e riapri Hermes, poi torna qui.',
        scaleTitle: 'Dimensione',
        scaleDesc: 'Cambia la dimensione della mascotte flottante. Si applica all’istante ovunque.',
        roamTitle: 'Movimento libero',
        roamDesc: 'Consente alla mascotte di girovagare per la finestra per conto suo quando è inattiva.',
        chooseTitle: 'Scegli una mascotte',
        chooseDesc: 'Scegliendone una, viene installata se necessario e resta attiva.',
        searchPlaceholder: 'Cerca mascotte…',
        unreachable: 'Impossibile accedere alla galleria di petdex. Controlla la tua connessione e riapri questa pagina.',
        noMatch: query => `Nessuna mascotte corrisponde a “${query}”.`,
        installedTag: 'installata',
        generatedTag: 'Generata',
        countCapped: (cap, total) => `Vengono mostrate ${cap} di ${total}; digita per restringere l’elenco.`,
        count: n => `${n} ${n === 1 ? 'mascotte' : 'mascotte'}.`,
        uninstall: name => `Disinstalla ${name}`,
        delete: name => `Elimina ${name}`,
        deleteTitle: name => `Eliminare ${name}?`,
        deleteBody: 'Questa operazione elimina la mascotte in modo permanente; non potrà essere reinstallata.',
        deleteConfirm: 'Elimina',
        rename: name => `Rinomina ${name}`,
        renameTitle: 'Rinomina mascotte',
        renamePlaceholder: 'Dai un nome alla tua mascotte',
        renameSave: 'Salva',
        exportPet: name => `Esporta ${name}`,
        adoptFailed: slug => `Impossibile adottare ${slug}`,
        uninstallFailed: slug => `Impossibile disinstallare ${slug}`,
        renameFailed: slug => `Impossibile rinominare ${slug}`,
        exportFailed: slug => `Impossibile esportare ${slug}`,
        noneAvailable: 'Non ci sono mascotte disponibili da attivare al momento.',
        turnOnFailed: 'Impossibile attivare la mascotte.',
        turnOffFailed: 'Impossibile disattivare la mascotte.'
      }
    },
    fieldLabels: defineFieldCopy({
      model: 'Modello predefinito',
      modelContextLength: 'Finestra di contesto del modello principale (forzata)',
      fallbackProviders: 'Modelli di riserva',
      toolsets: 'Set di strumenti attivati',
      timezone: 'Fuso orario',
      display: {
        personality: 'Personalità',
        showReasoning: 'Blocchi di ragionamento'
      },
      desktop: {
        repoScanEnabled: 'Rilevamento automatico dei repository',
        repoScanRoots: 'Cartelle di ricerca dei repository',
        repoScanExcludePaths: 'Percorsi di repository esclusi'
      },
      agent: {
        maxTurns: 'Passi massimi dell’agente',
        imageInputMode: 'Allegati immagine',
        apiMaxRetries: 'Nuovi tentativi API',
        serviceTier: 'Livello di servizio',
        toolUseEnforcement: 'Applicazione dell’uso degli strumenti'
      },
      terminal: {
        cwd: 'Directory di lavoro',
        backend: 'Backend di esecuzione',
        timeout: 'Timeout dei comandi',
        persistentShell: 'Shell persistente',
        envPassthrough: 'Pass-through delle variabili d’ambiente',
        dockerImage: 'Immagine Docker',
        singularityImage: 'Immagine Singularity',
        modalImage: 'Immagine Modal',
        daytonaImage: 'Immagine Daytona'
      },
      fileReadMaxChars: 'Limite di lettura dei file',
      toolOutput: {
        maxBytes: 'Limite di output del terminale',
        maxLines: 'Limite di pagine di file',
        maxLineLength: 'Limite di lunghezza riga'
      },
      codeExecution: {
        mode: 'Modalità di esecuzione del codice'
      },
      approvals: {
        mode: 'Modalità di approvazione',
        timeout: 'Timeout di approvazione',
        mcpReloadConfirm: 'Conferma dei ricaricamenti di MCP'
      },
      commandAllowlist: 'Elenco dei comandi consentiti',
      security: {
        redactSecrets: 'Maschera i segreti',
        allowPrivateUrls: 'Consenti URL privati'
      },
      browser: {
        allowPrivateUrls: 'URL privati del browser',
        autoLocalForPrivateUrls: 'Browser locale per URL privati',
        useRealProfile: 'Usa il mio profilo reale del browser'
      },
      checkpoints: {
        enabled: 'Checkpoint dei file',
        maxSnapshots: 'Limite di checkpoint'
      },
      voice: {
        maxRecordingSeconds: 'Durazione massima della registrazione',
        autoTts: 'Leggi le risposte ad alta voce',
        voiceChatMode: 'Modalità chat vocale',
        gptLive: {
          voice: 'Voce di GPT-Live',
          instructions: 'Personalità di GPT-Live'
        }
      },
      stt: {
        enabled: 'Voce in testo',
        echoTranscripts: 'Eco delle trascrizioni',
        provider: 'Provider di voce in testo',
        local: {
          model: 'Modello locale di trascrizione',
          language: 'Lingua della trascrizione'
        },
        openai: {
          model: 'Modello STT di OpenAI'
        },
        groq: {
          model: 'Modello STT di Groq'
        },
        mistral: {
          model: 'Modello STT di Mistral'
        },
        elevenlabs: {
          modelId: 'Modello STT di ElevenLabs',
          languageCode: 'Lingua di ElevenLabs',
          tagAudioEvents: 'Etichetta gli eventi audio',
          diarize: 'Diarizzazione dei parlanti'
        }
      },
      tts: {
        provider: 'Provider di testo in voce',
        edge: {
          voice: 'Voce di Edge'
        },
        openai: {
          model: 'Modello TTS di OpenAI',
          voice: 'Voce di OpenAI'
        },
        elevenlabs: {
          voiceId: 'Voce di ElevenLabs',
          modelId: 'Modello di ElevenLabs'
        },
        xai: {
          voiceId: 'Voce di xAI (Grok)',
          language: 'Lingua di xAI',
          speed: 'Velocità di riproduzione di xAI',
          autoSpeechTags: 'Tag vocali automatici di xAI',
          optimizeStreamingLatency: 'Ottimizzazione della latenza di streaming di xAI',
          sampleRate: 'Frequenza di campionamento di xAI',
          bitRate: 'Bitrate di xAI'
        },
        minimax: {
          model: 'Modello TTS di MiniMax',
          voiceId: 'Voce di MiniMax'
        },
        mistral: {
          model: 'Modello TTS di Mistral',
          voiceId: 'Voce di Mistral'
        },
        gemini: {
          model: 'Modello TTS di Gemini',
          voice: 'Voce di Gemini'
        },
        neutts: {
          model: 'Modello di NeuTTS',
          device: 'Dispositivo di NeuTTS'
        },
        kittentts: {
          model: 'Modello di KittenTTS',
          voice: 'Voce di KittenTTS'
        },
        piper: {
          voice: 'Voce di Piper'
        },
        deepinfra: {
          model: 'Modello TTS di DeepInfra',
          voice: 'Voce di DeepInfra'
        }
      },
      memory: {
        memoryEnabled: 'Memoria persistente',
        userProfileEnabled: 'Profilo utente',
        memoryCharLimit: 'Budget di memoria',
        userCharLimit: 'Budget del profilo',
        provider: 'Provider di memoria'
      },
      context: {
        engine: 'Motore di contesto'
      },
      compression: {
        enabled: 'Compressione automatica',
        threshold: 'Soglia di compressione',
        codexGpt55Autoraise: 'Aumento automatico della compressione di Codex',
        targetRatio: 'Obiettivo di compressione',
        protectLastN: 'Messaggi recenti protetti'
      },
      auxiliary: {
        compression: {
          timeout: 'Tempo di attesa del modello di compressione (s)'
        }
      },
      delegation: {
        model: 'Modello del subagente',
        provider: 'Provider del subagente',
        maxIterations: 'Limite di turni del subagente',
        maxConcurrentChildren: 'Subagenti paralleli',
        childTimeoutSeconds: 'Timeout del subagente',
        reasoningEffort: 'Sforzo di ragionamento del subagente'
      },
      updates: {
        nonInteractiveLocalChanges: 'Modifiche locali durante l’aggiornamento dall’app'
      }
    }),
    fieldDescriptions: defineFieldCopy({
      model: 'Usato per le nuove chat a meno che tu non scelga un altro modello nel compositore.',
      modelContextLength:
        'Sostituisce la finestra di contesto rilevata solo del modello di chat PRINCIPALE (in token). Lascialo a 0 per usare il valore rilevato del modello selezionato. Non influenza i modelli ausiliari/MoA.',
      fallbackProviders: 'Voci provider:model di riserva da provare se il modello predefinito fallisce.',
      display: {
        personality: 'Stile predefinito dell’assistente per le sessioni nuove.',
        showReasoning: 'Mostra le sezioni di ragionamento quando il backend le fornisce.'
      },
      desktop: {
        repoScanEnabled: 'Cerca repository Git nelle cartelle locali per mostrarli in Progetti.',
        repoScanRoots: 'Cartelle da cercare. Lascialo vuoto per cercare nella tua directory home.',
        repoScanExcludePaths:
          'Cartelle da omettere, insieme a tutte le loro sottodirectory, durante il rilevamento dei repository.'
      },
      timezone: 'Identificatore di fuso orario IANA. Vuoto usa il fuso orario di sistema.',
      browser: {
        useRealProfile:
          'La navigazione locale usa i tuoi accessi reali. Hermes copia il profilo del tuo browser predefinito (cookie, accessi, preferenze) in uno snapshot gestito e lo controlla con il suo Chromium integrato: il tuo profilo attivo non viene mai aperto direttamente e la copia viene aggiornata da lui a ogni esecuzione. Consente anche all’agente di aprire su richiesta una sessione locale con il tuo profilo reale, anche se è configurato un backend di browser nel cloud. Sono supportati solo i browser Chromium (Chrome, Edge, Brave, Brave Origin, Chromium); un browser predefinito che non sia Chromium fallisce con un messaggio chiaro. Disattivato per impostazione predefinita.'
      },
      agent: {
        imageInputMode: 'Controlla come vengono inviati gli allegati immagine al modello.',
        maxTurns: 'Limite superiore di turni con chiamate agli strumenti prima che Hermes interrompa un’esecuzione.'
      },
      terminal: {
        cwd: 'Cartella di progetto predefinita per strumenti e terminale.',
        persistentShell: 'Mantiene lo stato della shell tra i comandi quando il backend lo supporta.',
        envPassthrough: 'Variabili d’ambiente passate all’esecuzione degli strumenti.',
        dockerImage: 'Immagine contenitore usata quando il backend di esecuzione è Docker.',
        singularityImage: 'Immagine usata quando il backend di esecuzione è Singularity.',
        modalImage: 'Immagine usata quando il backend di esecuzione è Modal.',
        daytonaImage: 'Immagine usata quando il backend di esecuzione è Daytona.'
      },
      codeExecution: {
        mode: 'Quanto rigorosamente l’esecuzione del codice viene limitata al progetto corrente.'
      },
      fileReadMaxChars: 'Numero massimo di caratteri che Hermes può leggere in una richiesta di file.',
      approvals: {
        mode: 'Come gestisce Hermes i comandi che richiedono un’approvazione esplicita.',
        timeout: 'Quanto attendono i prompt di approvazione prima di scadere.'
      },
      security: {
        redactSecrets: 'Nasconde i segreti rilevati dal contenuto visibile al modello quando possibile.'
      },
      checkpoints: {
        enabled: 'Crea snapshot di ripristino prima di modificare i file.'
      },
      memory: {
        memoryEnabled: 'Salva memorie durature che possono aiutare nelle sessioni future.',
        userProfileEnabled: 'Mantiene un profilo compatto delle preferenze dell’utente.'
      },
      context: {
        engine: 'Strategia per gestire le conversazioni lunghe vicine al limite di contesto.'
      },
      compression: {
        enabled: 'Riassume il contesto precedente quando le conversazioni crescono.',
        codexGpt55Autoraise: 'Porta la compressione all’85% per i modelli compatibili con ChatGPT Codex OAuth.'
      },
      auxiliary: {
        compression: {
          timeout:
            'Secondi di attesa del modello ausiliario di compressione per chiamata (120 per impostazione predefinita). Aumentalo per modelli locali lenti.'
        }
      },
      voice: {
        autoTts: 'Legge automaticamente ad alta voce le risposte dell’assistente.',
        voiceChatMode:
          'chained: voce in testo → Hermes → testo in voce con i provider qui sotto. gpt-live: un modello vocale full-duplex di OpenAI (gpt-live-1) ascolta e parla, e passa ogni richiesta reale a Hermes; il modello che hai selezionato risponde con tutti gli strumenti. Richiede una chiave API di OpenAI; il layer vocale costa 0,05 US$ al minuto.',
        gptLive: {
          voice: 'Voce della modalità GPT-Live. Sono accettati ID voce personalizzati.',
          instructions:
            'Frasi aggiuntive per la personalità della voce live (tono, ritmo, lingua). Hermes mantiene il proprio prompt di sistema.'
        }
      },
      tts: {
        xai: {
          voiceId: 'ID voce di xAI (ad esempio, eve) o un ID voce personalizzato.',
          language: 'Codice della lingua parlata (p. es., en, pt-BR) o "auto" per rilevarla automaticamente.',
          speed: 'Velocità di riproduzione. 0.7 = più lento, 1.0 = normale, 1.5 = più veloce.',
          autoSpeechTags:
            'Consente a un LLM di inserire tag audio espressivi ([laughing], [sighs]) nel copione prima di sintetizzarlo.',
          optimizeStreamingLatency: 'Equilibrio tra latenza e qualità. 0 = qualità migliore, 2 = latenza minore.',
          sampleRate:
            'Frequenza di campionamento dell’audio in Hz. Un valore maggiore offre più qualità e file più grandi.',
          bitRate: 'Bitrate MP3 in bps. Si applica solo quando il codec è mp3.'
        },
        neutts: {
          device: 'Dispositivo di inferenza locale per NeuTTS.'
        }
      },
      stt: {
        enabled: 'Attiva la trascrizione vocale locale o basata su provider.',
        echoTranscripts: 'Pubblica la trascrizione grezza 🎙️ dei messaggi vocali nella chat.',
        elevenlabs: {
          languageCode: 'Codice lingua ISO-639-3 opzionale. Se in bianco, ElevenLabs lo rileva automaticamente.'
        }
      },
      updates: {
        nonInteractiveLocalChanges:
          'Quando Hermes si aggiorna dall’app senza prompt da terminale, conserva le modifiche locali del codice sorgente (stash) o scartale. Gli aggiornamenti da terminale chiedono sempre.'
      }
    }),
    uninstallSection: {
      dangerZone: 'Zona di pericolo',
      checkingInstalled: 'Controllo di ciò che è installato…',
      uninstallHermes: 'Disinstalla Hermes',
      chooseHowMuch:
        'Scegli quanto vuoi rimuovere. L’app si chiude per completare l’operazione; riapri il programma di installazione quando vuoi per tornare.',
      confirmUninstall: 'Conferma disinstallazione',
      confirmBody: what => `Questo rimuove ${what}. Non è possibile annullare.`,
      appLabel: 'App:',
      couldNotStart: 'Impossibile avviare la disinstallazione.',
      uninstalling: 'Disinstallazione in corso…',
      yesUninstall: 'Sì, disinstalla',
      options: {
        gui: {
          title: 'Disinstalla solo l’interfaccia di chat',
          description: 'Rimuove questa app desktop. L’agente di Hermes, la tua configurazione e le tue chat vengono conservati.',
          consequence: 'l’interfaccia di chat desktop (questa app e i suoi dati)'
        },
        lite: {
          title: 'Disinstalla l’interfaccia e l’agente, conservando i miei dati',
          description:
            'Rimuove l’app e l’agente di Hermes, ma conserva la configurazione, le chat e i segreti per una futura reinstallazione.',
          consequence:
            'l’interfaccia di chat e l’agente di Hermes (vengono conservati la configurazione, le chat e i segreti)'
        },
        full: {
          title: 'Disinstalla tutto',
          description:
            'Rimuove l’app, l’agente e tutti i dati utente: configurazione, chat, attività pianificate, segreti e log.',
          consequence:
            'TUTTO: l’interfaccia di chat, l’agente di Hermes e tutta la tua configurazione, chat, segreti e log'
        }
      }
    },
    poolLimits: {
      warmBotBackendsAria: 'Backend di bot a caldo',
      warmBotBackendsTitle: 'Backend di bot a caldo',
      backendIdleTimeoutAria: 'Tempo di inattività del backend in millisecondi',
      backendIdleTimeoutTitle: 'Tempo di inattività del backend'
    },
    customEndpoints: {
      active: 'Attivo',
      apiKeySet: 'Chiave API configurata',
      use: 'Usa',
      editTitle: 'Modifica endpoint',
      addTitle: 'Aggiungi endpoint',
      fields: {
        name: 'Nome',
        providerId: 'ID del provider',
        endpointUrl: 'URL dell’endpoint',
        defaultModel: 'Modello predefinito',
        context: 'Contesto',
        apiKey: 'Chiave API',
        apiKeyNewPlaceholder: 'Lascia vuoto per conservare la chiave attuale',
        apiKeyPlaceholder: 'Facoltativo',
        useNewChats: 'Usa nelle nuove chat',
        discoverModels: 'Scopri modelli'
      },
      test: 'Prova',
      save: 'Salva',
      newEndpoint: 'Nuovo endpoint',
      apiMode: 'Modalità API',
      autoDetect: 'Rilevamento automatico',
      couldNotLoad: 'Impossibile caricare gli endpoint personalizzati',
      endpointSaved: 'Endpoint personalizzato salvato.',
      saveFailed: 'Errore durante il salvataggio',
      endpointReachable: 'L’endpoint è raggiungibile.',
      endpointReachableTransport: transport => `L’endpoint è raggiungibile (percorso ${transport} servito).`,
      endpointReachableModels: (reachable, count) => `${reachable} Trovati ${count} modelli.`,
      endpointValidationFailed: 'Validazione dell’endpoint non riuscita.',
      validationFailed: 'Validazione non riuscita',
      activationFailed: 'Attivazione non riuscita',
      deleteConfirm: name => `Eliminare ${name}?`,
      deleteFailed: 'Errore durante l’eliminazione',
      title: 'Endpoint personalizzati',
      deleteEndpoint: 'Elimina endpoint',
      emptyDescription: 'Aggiungi qui sotto un endpoint compatibile con OpenAI.',
      emptyTitle: 'Nessun endpoint personalizzato',
      namePlaceholder: 'Axet Proxy',
      contextPlaceholder: 'Auto'
    },
    computerUse: {
      accessibility: 'Accessibilità',
      screenRecording: 'Registrazione dello schermo',
      driverHealth: 'Stato del driver'
    },
    about: {
      updates: 'Aggiornamenti'
    },
    config: {
      minimizeToTrayTitle: 'Minimizza nell’area di notifica',
      minimizeToTrayDesc:
        'Riducendo a icona le finestre o chiudendo la finestra principale, vengono nascoste nell’area di notifica di sistema (barra dei menu su macOS) e Hermes continua a essere eseguito. Usa Esci da Hermes nel menu dell’area di notifica oppure Cmd+Q per uscire. Disattivato per impostazione predefinita; si applica solo a questo dispositivo.',
      minimizeToTrayUnavailable:
        'L’area di notifica di sistema non è disponibile. Le finestre verranno ridotte a icona e chiuse normalmente. Disattiva e riattiva questa opzione per riprovare.',
      none: 'Nessuno',
      noneParen: '(nessuno)',
      builtinOnly: 'Solo integrate',
      notSet: 'Non impostato',
      commaSeparated: 'valori separati da virgole',
      searchPlaceholder: 'Cerca…',
      noResults: 'Nessun risultato trovato',
      systemDefault: 'Valore di sistema',
      loading: 'Caricamento della configurazione di Hermes...',
      emptyTitle: 'Niente da configurare',
      emptyDesc: 'Questa sezione non ha impostazioni configurabili.',
      failedLoad: 'Impossibile caricare la configurazione',
      autosaveFailed: 'Salvataggio automatico non riuscito',
      imported: 'Configurazione importata',
      invalidJson: 'JSON di configurazione non valido',
      toolsetsWipeConfirm:
        'Rimuovere tutti i set di strumenti attivati? Questo disattiva la memoria, il terminale, la ricerca web, la delega e la maggior parte degli altri strumenti finché non li riattivi.',
      keepAwakeTitle: 'Mantieni il computer attivo',
      keepAwakeDesc:
        'Impedisce a questo computer di entrare in riposo. «Mentre lavora» si applica solo mentre c’è un turno in corso: le esecuzioni notturne continuano senza tenere sveglio il portatile tutta la settimana. Lo schermo può continuare a oscurarsi.',
      keepAwakeOff: 'Disattivato',
      keepAwakeWhileWorking: 'Mentre lavora',
      keepAwakeAlways: 'Sempre',
      disableF12Title: 'Disattiva DevTools con F12',
      disableF12Desc:
        'Impedisce a F12 di aprire gli strumenti per sviluppatori. Ctrl+Shift+I (o Cmd+Opt+I su Mac) continua a funzionare.',
      attachmentSizeTitle: 'Dimensione massima di anteprima / caricamento immagini',
      attachmentSizeDesc:
        'Dimensione massima del file locale che il desktop caricherà per anteprime e allegati immagine, in MB. Il valore predefinito è 16. Gli allegati remoti non immagine usano un limite separato di 256 MB. Un valore troppo alto carica l’intero file in memoria e può congelare o bloccare l’app.',
      attachmentSizeUnit: 'MB',
      attachmentSizeLabel: 'Dimensione massima di anteprima / caricamento immagini in megabyte',
      showOptions: 'Mostra opzioni'
    },
    hudModifier: {
      title: 'Premi per mostrare l’HUD',
      description:
        'Premi e rilascia ⌘ + Opzione su Mac, oppure Ctrl + Alt su Windows/Linux, per portare l’HUD in primo piano da qualsiasi app. Disattivato per impostazione predefinita; si applica solo a questo dispositivo.',
      permission:
        'Consenti a Hermes in Impostazioni di Sistema → Privacy e sicurezza → Monitoraggio input e riprova. Questo gesto non registra le pressioni dei tasti né acquisisce il tuo schermo.',
      unavailable:
        'L’assistente del gesto dell’HUD non si è avviato o si è interrotto in modo imprevisto. Riprova o riavvia Hermes. La scorciatoia HUD esistente continua a funzionare all’interno di Hermes.',
      missingHelper:
        'A questa installazione di Hermes manca l’assistente del gesto dell’HUD. Aggiorna o reinstalla Hermes e riprova.',
      unsupportedSession:
        'Questa sessione desktop non supporta la pressione globale di tasti modificatori. Linux richiede X11; Wayland non è supportato.'
    },
    screenshot: {
      enabledTitle: 'Scorciatoia per l’acquisizione dello schermo',
      enabledDesc:
        'Premi insieme i due tasti Comando da qualsiasi app per acquisire la sua finestra in primo piano e allegarla alla tua bozza corrente di Hermes. Non viene mai inviata automaticamente. Disattivato per impostazione predefinita; si applica solo a questo Mac. Il contenuto della finestra può essere confidenziale: controlla l’allegato prima di inviarlo.',
      statusTitle: 'Stato della scorciatoia di acquisizione',
      checking: 'Verifica della scorciatoia di acquisizione…',
      disabled: 'La scorciatoia di acquisizione è disattivata.',
      starting: 'Avvio dell’ascolto della scorciatoia. Non è ancora pronta.',
      ready: 'La scorciatoia è pronta. Le acquisizioni vengono allegate alla tua bozza corrente senza essere inviate.',
      inputPermission:
        'Il permesso di Monitoraggio input consente a Hermes di rilevare i due tasti Comando mentre un’altra app è attiva. Consenti a Hermes in Impostazioni di Sistema → Privacy e sicurezza → Monitoraggio input, torna qui e riprova.',
      screenPermission:
        'Il permesso di Registrazione dello schermo consente a Hermes di acquisire la finestra in primo piano quando usi questa scorciatoia. Consenti a Hermes in Impostazioni di Sistema → Privacy e sicurezza → Registrazione dello schermo, torna qui e riprova. Riavvia Hermes se macOS te lo chiede.',
      openSettings: 'Apri Impostazioni di Sistema',
      retry: 'Riprova',
      unavailable: 'La scorciatoia di acquisizione non è disponibile. Riprova o disattivala.',
      errorTitle: 'Errore della scorciatoia di acquisizione',
      loadFailed: 'Impossibile leggere lo stato della scorciatoia. Riprova per verificare la relativa impostazione attuale.',
      saveFailed: 'Impossibile confermare la modifica della scorciatoia. Riprova per verificare la relativa impostazione attuale.',
      permissionFailed:
        'Impossibile aprire Impostazioni di Sistema. Apri Privacy e sicurezza manualmente e riprova.',
      captureFailed: 'Impossibile acquisire la finestra in primo piano. Nulla è stato allegato né inviato.',
      contextChanged: 'La bozza corrente è cambiata durante l’acquisizione. L’acquisizione non è stata allegata né inviata.'
    },
    quickEntry: {
      enabledTitle: 'Ingresso rapido',
      enabledDesc:
        'Richiama un piccolo composer da qualsiasi punto con una scorciatoia globale e invia un prompt senza aprire Hermes.',
      shortcutTitle: 'Scorciatoia di ingresso rapido',
      shortcutDesc: 'Richiede almeno un modificatore, ad es. CommandOrControl+Shift+Spazio.',
      active: 'La scorciatoia è attiva.',
      takenBy: 'Un’altra app usa già questa scorciatoia — scegline una diversa.',
      invalidShortcut: 'Non è una scorciatoia valida. Includi almeno un tasto modificatore.'
    },
    credentials: {
      pasteKey: 'Incolla chiave',
      pasteLabelKey: label => `Incolla chiave di ${label}`,
      optional: 'Facoltativo',
      enterValueFirst: 'Inserisci prima un valore.',
      couldNotSave: 'Impossibile salvare la credenziale.',
      remove: 'Rimuovi',
      getKey: 'Ottieni una chiave',
      saving: 'Salvataggio'
    },
    envActions: {
      actions: 'Azioni',
      manageInKeys: 'Gestisci in Chiavi API',
      docs: 'Docs',
      hideValue: 'Nascondi valore',
      revealValue: 'Mostra valore',
      replace: 'Sostituisci',
      set: 'Definisci',
      clear: 'Cancella'
    },
    connections: {
      title: 'Gateway registrati',
      intro:
        'Gestisci questo dispositivo e ogni gateway di Hermes raggiungibile tramite connessioni remote, SSH o Cloud.',
      stagedNote:
        'Cambia gateway da Sessioni. Profili, chat, messaggistica e attività cron restano al loro gateway; il lavoro sugli altri gateway continua a essere eseguito.',
      launchModeTitle: 'All’avvio, torna a Sessioni sull’ultimo gateway usato',
      launchModeDesc: 'Se disattivato, Sessioni si apre sul gateway principale.',
      searchPlaceholder: 'Cerca gateway…',
      noSearchResults: 'Nessun gateway corrisponde alla tua ricerca.',
      loadFailed: 'Impossibile caricare le connessioni',
      currentPill: 'Attuale',
      primaryPill: 'Principale',
      managedPill: 'Gestito dall’app',
      addConnection: 'Aggiungi connessione',
      editConnection: 'Modifica',
      removeConnection: 'Rimuovi',
      removeConfirmTitle: 'Rimuovere questa connessione?',
      removeConfirmDesc: (label: string) =>
        `“${label}” verrà rimossa da questa app. L’istanza in sé non viene toccata; puoi aggiungerla di nuovo quando vuoi.`,
      makePrimary: 'Rendi principale',
      testConnection: 'Prova',
      testOk: 'Raggiungibile',
      testFailed: 'Test di connessione non riuscito',
      saveFailed: 'Impossibile salvare la connessione',
      removeFailed: 'Impossibile rimuovere la connessione',
      updateAll: 'Aggiorna tutte le istanze',
      updateAllRunning: 'Aggiornamento di tutte le istanze…',
      updateAllDone: 'Aggiornamenti inviati',
      updateAllFailed: 'Invio degli aggiornamenti non riuscito',
      updateSkippedCloud: 'Gestito da Hermes Cloud',
      kindLocal: 'Locale',
      kindRemote: 'Gateway remoto',
      kindCloud: 'Hermes Cloud',
      kindSsh: 'SSH',
      kindLocalDesc: 'L’ambiente di esecuzione di Hermes gestito da questa app.',
      kindRemoteDesc: 'Un gateway di Hermes raggiungibile via HTTP(S): LAN, Tailscale o internet.',
      kindCloudDesc: 'Un’istanza ospitata rilevata tramite il tuo account Hermes Cloud.',
      kindSshDesc: 'Un’installazione di Hermes raggiungibile via SSH.',
      labelTitle: 'Nome',
      labelDesc:
        'Obbligatorio. Viene mostrato ovunque compaia questa istanza; deve essere univoco (ad es. “Homelab”, “Portatile del lavoro”).',
      labelPlaceholder: 'Homelab',
      urlTitle: 'URL del gateway',
      sshHostTitle: 'Host SSH',
      headersTitle: 'Intestazioni aggiuntive del gateway',
      headersDesc:
        'Vengono inviate con ogni richiesta HTTP e WebSocket a questo gateway, per proxy di accesso come Cloudflare Access (CF-Access-Client-Id / CF-Access-Client-Secret). I valori vengono salvati cifrati. Le intestazioni gestite da Hermes (Authorization, Cookie, Host…) vengono ignorate.',
      headerValuePlaceholder: 'Valore',
      headerValueSaved: 'Salvato: lascia vuoto per conservarlo',
      headerAdd: 'Aggiungi intestazione',
      headerRemove: 'Rimuovi',
      duplicateLocal: 'Questa app gestisce già una connessione locale; può essercene solo una.',
      duplicateUrl: (label: string) => `Esiste già una connessione all’URL di questo gateway (“${label}”).`,
      duplicateSsh: (label: string) => `Esiste già una connessione a questo host SSH (“${label}”).`,
      sameBackendHint: (label: string) => `Stesso backend di “${label}”`,
      localAddHint: 'Locale non è disponibile: la connessione locale gestita esiste già (può essercene solo una).',
      cloudAddHint:
        'Suggerimento: accedendo a Hermes Cloud qui sopra, i tuoi agenti vengono rilevati automaticamente; usa questo modulo solo per registrare manualmente l’URL di un’istanza nota.',
      save: 'Salva connessione',
      saving: 'Salvataggio…',
      cancel: 'Annulla',
      empty: 'Non ci sono ancora connessioni registrate.'
    },
    managedUpdates: {
      title: 'Aggiornamenti gestiti',
      intro:
        'Aggiorna in modo transazionale le installazioni SSH gestite dall’app: le sessioni vengono svuotate, la copia remota viene aggiornata e ogni profilo viene ripristinato con una ricevuta correlata.',
      sshConnection: 'Installazione SSH gestita dall’app',
      update: 'Aggiorna',
      updating: 'Aggiornamento…',
      progress: 'Svuotamento delle sessioni, aggiornamento dell’installazione remota e ripristino dei profili…',
      updated: 'Aggiornato',
      partial: 'Aggiornato, ma il ripristino non è riuscito',
      refused: 'Rifiutato',
      failed: 'Aggiornamento non riuscito',
      alreadyRunning: 'C’è già un aggiornamento in corso',
      receipt: (id: string, outcome: string) => `Ricevuta ${id} · ${outcome}`,
      receiptVersions: (pre: string, post: string) => `${pre} → ${post}`,
      scopesRestored: (profiles: string) => `Profili ripristinati: ${profiles}`,
      scopeNotRestored: (profile: string, error: string) => `Profilo “${profile}” non ripristinato: ${error}`
    },
    gateway: {
      loading: 'Caricamento delle impostazioni del gateway...',
      unavailableTitle: 'Impostazioni del gateway non disponibili',
      unavailableDesc:
        'Le impostazioni di connessione si possono modificare solo dall’app Hermes Desktop sul computer su cui è in esecuzione.',
      title: 'Connessione del gateway',
      envOverride: 'override dell’ambiente',
      intro:
        'Locale per impostazione predefinita. Usa remoto quando questa app deve controllare un backend di Hermes altrove. Override per profilo di seguito.',
      envOverrideTitle: 'Questa connessione è stata fissata dal modo in cui Hermes è stato avviato.',
      envOverrideDesc:
        'Un’impostazione di avvio esterna all’app ha scelto questa connessione, quindi le opzioni qui sotto sono di sola lettura. Riavvia Hermes senza quell’impostazione (o chiedi a chi l’ha configurata) per modificarla qui.',
      modeTitle: 'Modalità di connessione',
      localTitle: 'Gateway locale',
      localDesc:
        'Avvia un backend privato di Hermes su localhost. È il valore predefinito e funziona offline.',
      remoteTitle: 'Gateway remoto',
      remoteDesc: 'Collega questa shell desktop a un backend remoto di Hermes.',
      remoteAuthHint:
        'I gateway ospitati usano OAuth oppure nome utente e password; quelli self-hosted possono usare un token di sessione.',
      cloudTitle: 'Hermes Cloud',
      cloudDesc:
        'Accedi una sola volta a Hermes Cloud e scegli uno degli agenti del tuo account; non devi incollare nessun URL.',
      cloudSignInTitle: 'Hermes Cloud',
      cloudSignIn: 'Accedi a Hermes Cloud',
      cloudSignedIn: 'Accesso effettuato a Hermes Cloud',
      cloudNeedsSignIn: 'Accedi a Hermes Cloud per individuare gli agenti del tuo account.',
      cloudSignedInDesc: 'Accesso effettuato. Scegli un agente qui sotto; la sessione si aggiorna automaticamente.',
      cloudAgentsTitle: 'I tuoi agenti',
      cloudOrgPickerTitle: 'Scegli un’organizzazione',
      cloudOrgSelect: 'Seleziona',
      cloudOrgChange: 'Cambia organizzazione',
      cloudOrgRole: role => `Ruolo: ${role}`,
      cloudLoadingAgents: 'Caricamento dei tuoi agenti…',
      cloudNoAgents: {
        before: 'Non sono stati trovati agenti in questo account. Creane uno nel ',
        linkText: 'Portale di Nous',
        after: ', poi aggiorna.'
      },
      cloudRefresh: 'Aggiorna',
      cloudConnect: 'Connetti',
      cloudSavedTitle: 'Gateway di Cloud salvati',
      cloudSavedDesc:
        'Usa un gateway salvato senza modificare quello predefinito. Accedi qui sotto per aggiungere istanze. Gestisci i nomi e l’accesso nell’elenco delle connessioni salvate.',
      cloudUseSaved: 'Usa gateway',
      cloudActive: 'Attivo in questa finestra',
      cloudConnecting: 'Connessione…',
      cloudDiscoverFailed: 'Impossibile caricare i tuoi agenti di Hermes Cloud',
      cloudConnectFailed: 'Impossibile connettersi a quell’agente',
      cloudSignInFailed: 'Accesso a Hermes Cloud non riuscito',
      cloudSignedOutTitle: 'Disconnessione da Hermes Cloud',
      cloudSignedOutMessage: 'La sessione Hermes Cloud è stata eliminata.',
      cloudConnectedTitle: 'Connesso',
      cloudConnectedPill: 'Connesso',
      cloudConnectedTo: name => `Connesso a ${name}.`,
      cloudAgentProvisioning: 'Preparazione…',
      cloudStatusLabel: status => `Stato: ${status}`,
      remoteUrlTitle: 'URL remoto',
      remoteUrlDesc: 'URL di base del backend della dashboard remota. Sono supportati i prefissi di percorso, ad esempio /hermes.',
      probing: 'Verifica di come si autentica questo gateway…',
      probeError:
        'Hermes non riesce a raggiungere quell’indirizzo. Controlla l’URL e che l’altro computer stia eseguendo Hermes; le opzioni di accesso compaiono quando risponde.',
      signedIn: 'Accesso effettuato',
      signIn: 'Accedi',
      signOut: 'Disconnetti',
      signInWith: provider => `Accedi con ${provider}`,
      authTitle: 'Autenticazione',
      authSignedInPassword:
        'Questo gateway usa nome utente e password. Accesso già effettuato; la sessione si aggiorna automaticamente.',
      authSignedInOauth: 'Questo gateway usa OAuth. Accesso già effettuato; la sessione si aggiorna automaticamente.',
      authNeedsPassword: 'Questo gateway usa nome utente e password. Accedi per autorizzare questa app desktop.',
      authNeedsOauth: provider => `Questo gateway usa OAuth. Accedi con ${provider} per autorizzare questa app.`,
      tokenTitle: 'Token di sessione',
      tokenDesc: 'Token di sessione della dashboard usato per REST e WebSocket. Lascialo vuoto per conservare quello salvato.',
      existingToken: value => `Token esistente ${value}`,
      savedToken: 'salvato',
      pasteSessionToken: 'Incolla token di sessione',
      plainTextConfirmTitle: 'Salvare il token del gateway in chiaro?',
      plainTextConfirmDesc:
        'In questo computer non è stato trovato alcun servizio di keychain di sistema, quindi il token verrebbe salvato senza cifratura nel file delle impostazioni di connessione dell’app, leggibile da qualsiasi processo eseguito con questo utente. Installa o attiva il keychain di sistema (GNOME Keyring o KWallet su Linux) per salvarlo cifrato.',
      plainTextConfirmAction: 'Salva come testo in chiaro',
      plainTextStoredTitle: 'Token salvato in chiaro',
      plainTextStoredDesc:
        'L’archiviazione sicura non è disponibile, quindi il token salvato è in chiaro nel file delle impostazioni di connessione dell’app su questo computer. Installa o attiva il keychain di sistema (GNOME Keyring o KWallet su Linux) per cifrarlo.',
      keychainEncryptionTitle: 'Cifra i segreti salvati con il keychain di sistema',
      keychainEncryptionDesc:
        'Disattivato per impostazione predefinita. Se attivato, i token del gateway e le credenziali di accesso vengono cifrati con il keychain di sistema (Accesso Portachiavi, GNOME Keyring o DPAPI di Windows); il sistema potrebbe chiedere un permesso o una password. Se disattivato, vengono salvati come file normali che solo il tuo account utente può leggere.',
      keychainEncryptionFailed: 'Impossibile modificare la cifratura dei segreti',
      testRemote: 'Prova remoto',
      saveForRestart: 'Salva per il prossimo riavvio',
      saveAndReconnect: 'Salva e riconnetti',
      diagnostics: 'Diagnostica',
      diagnosticsDesc: 'Mostra desktop.log nel tuo gestore di file; utile quando il gateway non si avvia.',
      openLogs: 'Apri i log',
      incompleteTitle: 'Gateway remoto incompleto',
      incompleteSignIn: 'Inserisci un URL remoto e accedi prima di passare a remoto.',
      incompleteToken: 'Inserisci un URL remoto e un token di sessione prima di passare a remoto.',
      incompleteSignInTest: 'Inserisci un URL remoto e accedi prima di eseguire il test.',
      incompleteTokenTest: 'Inserisci un URL remoto e un token di sessione prima di eseguire il test.',
      enterUrlFirst: 'Inserisci prima un URL remoto.',
      restartingTitle: 'Riavvio della connessione del gateway',
      savedTitle: 'Impostazioni del gateway salvate',
      restartingMessage: 'Hermes Desktop si riconnetterà con le impostazioni salvate.',
      savedMessage: 'Salvato per il prossimo riavvio.',
      connectedTo: (baseUrl, version) => `Connesso a ${baseUrl}${version ? ` · Hermes ${version}` : ''}`,
      reachableTitle: 'Gateway remoto raggiungibile',
      signedOutTitle: 'Disconnessione effettuata',
      signedOutMessage: 'La sessione del gateway remoto è stata eliminata.',
      failedLoad: 'Impossibile caricare le impostazioni del gateway',
      signInFailed: 'Accesso non riuscito',
      signOutFailed: 'Disconnessione non riuscita',
      testFailed: 'Test del gateway remoto non riuscito',
      applyFailed: 'Impossibile applicare le impostazioni del gateway',
      saveFailed: 'Impossibile salvare le impostazioni del gateway',
      sshTitle: 'Connetti via SSH',
      sshDesc:
        'Hermes viene avviato sul computer remoto tramite SSH e si connette a questa app attraverso un tunnel; non devi avviare né esporre nulla per conto tuo. Richiede un accesso SSH tramite chiavi che funzioni già con l’host.',
      sshTrustHint:
        'La prima chiave host presentata viene accettata e fissata; se cambia in seguito, la connessione viene rifiutata.',
      sshHostTitle: 'Host',
      sshHostDesc: 'utente@host o un alias Host di ~/.ssh/config.',
      sshHostPick: 'Seleziona un host…',
      sshHostPickTitle: 'Host',
      sshHostPickDesc: 'Un alias Host di ~/.ssh/config, oppure Personalizzato per digitarne uno.',
      sshHostCustom: 'Personalizzato (inserimento manuale)…',
      sshUserTitle: 'Utente',
      sshUserDesc: 'Vuoto = ~/.ssh/config o il tuo utente attuale.',
      sshUserPlaceholder: 'da ~/.ssh/config',
      sshPortTitle: 'Porta',
      sshPortDesc: 'Vuoto = 22 o la porta di ~/.ssh/config.',
      sshKeyTitle: 'File di identità',
      sshKeyDesc: 'Percorso della chiave privata. Vuoto = ssh-agent o ~/.ssh/config.',
      sshHermesPathTitle: 'Percorso di Hermes (facoltativo)',
      sshHermesPathDesc: 'Percorso completo al binario remoto di Hermes. Vuoto = rilevamento automatico.',
      sshHermesPathPlaceholder: 'rilevamento automatico',
      sshTestConnection: 'Prova SSH',
      sshConnect: 'Connetti',
      sshButtonsHint: 'Salva si applica al prossimo avvio. Connetti si riconnette subito.',
      sshReachable: (host, platform) => `Raggiungibile: ${host} (${platform}) — Hermes trovato`,
      sshIncompleteHost: 'Inserisci un host SSH prima di connetterti.',
      sshErrUnreachable: 'Impossibile accedere a quell’host via SSH. Controlla host, porta e rete.',
      sshErrAuth:
        'Autenticazione SSH non riuscita. Carica la tua chiave in ssh-agent (ssh-add) o configura un IdentityFile in ~/.ssh/config; Hermes esegue SSH in modo non interattivo.',
      sshErrHostKey:
        'La chiave dell’host È CAMBIATA dall’ultima connessione. Conferma che si tratti di una modifica prevista, esegui ssh-keygen -R <host> e riconnetti.',
      sshErrNotInstalled:
        'Hermes non è installato sull’host remoto. Installalo lì (curl -fsSL https://hermes-agent.nousresearch.com/install.sh | sh) oppure indica il percorso di Hermes.',
      sshErrPlatform:
        'Piattaforma remota non supportata. La modalità SSH di Hermes Desktop supporta host remoti Linux, macOS e Windows.',
      sshErrTimeout: 'La connessione SSH è andata in timeout. L’host potrebbe non rispondere oppure essere in riposo.',
      sshErrUpdateRequired: 'Aggiorna Hermes sull’host remoto prima di connetterti con Desktop SSH.',
      sshErrInteractiveAuth:
        'Tailscale SSH richiede una verifica interattiva nel browser. Esegui `ssh <host> true` nel terminale, completa la verifica e riprova; Hermes esegue SSH in modo non interattivo.',
      sshErrUnknown: 'Connessione SSH non riuscita.'
    },
    keys: {
      loading: 'Caricamento delle chiavi API e delle credenziali...',
      failedLoad: 'Impossibile caricare le chiavi API',
      empty: 'Non c’è ancora niente configurato in questa categoria.'
    },
    search: {
      placeholder: 'Cerca in tutte le impostazioni…',
      pill: 'Cerca'
    },
    profileScope: {
      appliesTo: 'Si applica a',
      editsProfile: profile => `Le modifiche di questa pagina si applicano al profilo “${profile}”.`
    },
    mcp: {
      loading: 'Caricamento dei server MCP...',
      invalidJson: 'JSON MCP non valido',
      saveFailed: 'Impossibile salvare',
      removeFailed: 'Impossibile rimuovere',
      reloadFailed: 'Ricarica di MCP non riuscita',
      savedTitle: 'Server MCP salvato',
      savedMessage: name => `${name} si applica dopo aver ricaricato MCP.`,
      disabled: 'disabilitato',
      name: 'Nome',
      serverJson: 'JSON del server',
      remove: 'Rimuovi',
      test: 'Prova connessione',
      catalogLoading: 'Caricamento del catalogo MCP...',
      catalogInstallFailed: name => `Impossibile installare ${name}`,
      catalogEnvRequired: 'Compila i valori obbligatori prima di installare.',
      capabilitySummary: (tools, prompts, resources) =>
        `Abilitato: ${[`${tools} strumenti`, ...(prompts ? [`${prompts} prompts`] : []), ...(resources ? [`${resources} risorse`] : [])].join(', ')}`,
      costTokens: tokens => `~${tokens} tok/chiamata`,
      usage30d: uses => `${uses} usi/30 gg`,
      statusConnecting: 'Connessione…',
      statusNeedsAuth: 'Richiede autenticazione',
      statusError: 'Errore',
      statusOff: 'Disattivato',
      allServers: 'Tutti i server',
      authenticatedTitle: 'Autenticato',
      authenticatedMessage: (server, count) => `${server}: ${count} strumenti`,
      authenticate: 'Autentica',
      noOutput: 'Ancora nessun output.',
      deepLinkTitle: 'Aggiungere il server MCP?',
      deepLinkDescription:
        'Un link ha chiesto di aggiungere questo server MCP a Hermes. Controlla la configurazione esatta qui sotto: viene dal link, non da Hermes.',
      deepLinkStdioWarning:
        'Questo server esegue un processo locale sul tuo computer con il comando mostrato qui sotto. Prosegui solo se ti fidi della sua origine.',
      deepLinkConfirm: 'Aggiungi server',
      deepLinkNameInvalid: 'I nomi usano da 1 a 64 lettere, cifre, punti, trattini o trattini bassi.',
      deepLinkNameConflict: name => `Esiste già un server di nome ${name}: scegli un altro nome o annulla.`,
      deepLinkErrorTitle: 'Link di installazione di MCP rifiutato',
      deepLinkErrorName: 'Manca il nome del server nel link oppure non è valido.',
      deepLinkErrorConfig: 'La configurazione del link non è JSON valido codificato in base64.',
      deepLinkErrorShape: 'La configurazione deve essere un oggetto JSON con un campo `url` o `command` di tipo stringa.',
      deepLinkErrorUrl: 'Sono consentiti solo URL di server http:// e https://.',
      deepLinkErrorTooLarge: 'La configurazione supera il limite di 32 KB.'
    },
    model: {
      setupProviderFallback: 'provider',
      setUpProvider: name => `Configura ${name}`,
      staleAuxBefore: (count, names) =>
        count === 1
          ? `${count} attività ausiliaria (${names}) è ancora in esecuzione su `
          : `${count} attività ausiliarie (${names}) sono ancora in esecuzione su `,
      staleAuxAfter: ', non sul tuo modello principale.',
      staleAuxOtherProviders: 'altri provider',
      moaEnabled: 'Attivato',
      moaSetDefault: 'Imposta come predefinito',
      moaNewPresetPlaceholder: 'nuovo preset',
      moaAddPreset: 'Aggiungi preset',
      customModel: 'Modello personalizzato…',
      customModelPlaceholder: 'ID del modello',
      chooseFromList: 'Scegli dalla lista',
      moaDefault: 'Predefinito:',
      moaReferenceToggle: (enabled, index) => `${enabled ? 'Disattiva' : 'Attiva'} il riferimento ${index}`,
      moaReferenceTitle: index => `Riferimento ${index}`,
      moaAddReference: 'Aggiungi modello di riferimento',
      loading: 'Caricamento della configurazione del modello...',
      appliesDesc:
        'Si applica alle nuove sessioni. Usa il selettore del modello nel composer per cambiare la chat attiva.',
      provider: 'Provider',
      model: 'Modello',
      applying: 'Applicazione...',
      mainAppliedTitle: 'Modello principale aggiornato',
      mainAppliedMessage: model => `Le nuove sessioni useranno ${model}.`,
      defaultsLabel: 'Valori predefiniti',
      reasoning: 'Ragionamento',
      reasoningOff: 'Disattivato',
      speed: 'Velocità',
      speedStandard: 'Standard',
      defaultsFailed: 'Impossibile salvare i valori predefiniti del modello',
      loadFailed: 'Impossibile caricare i modelli',
      restartRequired:
        'Questo backend esegue codice obsoleto dopo un aggiornamento. Riavvialo per caricare il codice nuovo.',
      restartBackend: 'Riavvia backend',
      restartingBackend: 'Riavvio del backend...',
      restartFailed: 'Impossibile riavviare il backend',
      auxiliaryTitle: 'Modelli ausiliari',
      resetAllToMain: 'Ripristina tutti sul principale',
      staleAuxDismiss: 'Non mostrare più',
      auxiliaryDesc:
        'Le attività ausiliarie usano il modello principale per impostazione predefinita. Assegna un modello dedicato a qualsiasi attività per cambiarlo.',
      setToMain: 'Usa principale',
      change: 'Cambia',
      autoUseMain: 'auto · usa il modello principale',
      inheritMainEffort: 'eredita · effort del modello principale',
      providerDefault: '(predefinito del provider)',
      fallbackAdd: 'Aggiungi fallback',
      fallbackEmpty: 'Nessun modello di fallback; viene usato il modello predefinito a meno che non fallisca.',
      notInCatalog: 'non è nella lista dei modelli di questo provider; le chiamate possono ripiegare su un fallback.',
      moaTitle: 'Mixture of Agents',
      moaPreset: 'Preset',
      moaDescription:
        'Configura preset con nome che appaiono come modelli del provider Mixture of Agents. L’aggregatore è il modello che agisce: esegue ogni passaggio del ciclo di strumenti, e quasi tutto il costo dell’esecuzione viene fatturato al suo provider. Per impostazione predefinita, i riferimenti consigliano solo una volta per turno dell’utente.',
      moaAggregator: 'Aggregatore',
      moaAggregatorBilled: 'modello che agisce · fatturato per l’esecuzione',
      moaReferenceHint: 'consiglia una volta per turno per impostazione predefinita',
      tasks: {
        vision: {
          label: 'Visione',
          hint: 'Analisi delle immagini'
        },
        compression: {
          label: 'Compressione',
          hint: 'Compattazione del contesto'
        },
        skills_hub: {
          label: 'Hub delle skill',
          hint: 'Ricerca delle skill'
        },
        approval: {
          label: 'Approvazione',
          hint: 'Approvazione automatica intelligente'
        },
        mcp: {
          label: 'MCP',
          hint: 'Instradamento degli strumenti MCP'
        },
        title_generation: {
          label: 'Generazione dei titoli',
          hint: 'Titoli delle sessioni'
        },
        review: {
          label: 'Revisione',
          hint: 'subagente revisore di /review'
        },
        triage_specifier: {
          label: 'Specificatore di triage',
          hint: 'Dettaglio delle specifiche di Kanban'
        },
        kanban_decomposer: {
          label: 'Decompositore di Kanban',
          hint: 'Scomposizione delle attività'
        },
        profile_describer: {
          label: 'Descrittore dei profili',
          hint: 'Descrizioni automatiche dei profili'
        },
        curator: {
          label: 'Curatore',
          hint: 'Revisione dell’uso delle skill'
        }
      }
    },
    localModels: {
      connectionChanged: 'La connessione dei modelli locali è cambiata',
      title: 'Modelli locali',
      runtimeTitle: 'Runtime locale',
      runtimeReady: backend => `Pronto · ${backend}`,
      serverRunning: 'In esecuzione',
      runtimeInstalled: 'Runtime llama.cpp installato',
      runtimeInstalledDetail: (tag, backend) =>
        `Build ${tag}, backend ${backend}. Hermes avvia e gestisce il server per te.`,
      installTitle: 'Installa il runtime locale',
      installDetail:
        'Scarica il motore di inferenza llama.cpp (alcune centinaia di MB). I modelli che scarichi vengono eseguiti interamente su questo computer: senza account e senza che nulla esca dal tuo computer.',
      installAction: 'Installa runtime',
      installing: 'Installazione del runtime…',
      installFailed: 'Installazione del runtime non riuscita',
      hardwareTitle: 'Questo computer',
      hardwareLoading: 'Verifica del tuo hardware…',
      vram: label => `${label} di memoria GPU`,
      ram: label => `${label} di RAM`,
      unifiedMemory: 'Memoria unificata',
      modelsTitle: 'Modelli',
      recommended: 'Consigliato',
      recommendedReason: {
        'best-quality-resident':
          'Il modello di qualità più alta che gira interamente sulla tua GPU alla massima velocità. La selezione bilancia la qualità con la velocità prevista su questo hardware.',
        'speed-gated-quality':
          'Su questo computer entrerebbe un modello di qualità più alta, ma risponderebbe troppo lentamente per la larghezza di banda della sua memoria; questo è il miglior modello che resta veloce.',
        'fastest-resident':
          'Nessun modello raggiunge la massima velocità su questo hardware; questo è quello che si avvicina di più, eseguito interamente nella memoria della GPU.'
      },
      noRecommendationTitle: 'Nessuna raccomandazione automatica per questo computer',
      noRecommendationDetail:
        'La configurazione automatica richiede un modello selezionato che entri interamente nella memoria della GPU o unificata. Puoi comunque scegliere un modello qui sotto o esplorarne altri.',
      noRecommendationAction: 'Esplora modelli',
      downloaded: 'Scaricato',
      downloadAction: size => `Scarica · ${size}`,
      downloadProgress: (done, total) => `Scaricamento ${done} di ${total}`,
      downloadDoneToast: model => `${model} è pronto.`,
      installDoneToast: 'Runtime locale installato e pronto.',
      quickstartTitle: 'Esegui un modello su questo computer',
      quickstartDetail: (model, size) =>
        `Un clic configura tutto: il motore locale, ${model} (${size} di download) e il tuo modello predefinito per le nuove chat. Nulla esce da questo computer.`,
      quickstartDetailReady: model =>
        `Un clic rende ${model} il tuo modello predefinito per le nuove chat. Tutto viene eseguito su questo computer.`,
      quickstartAction: 'Configuralo per me',
      quickstartConfigure: 'Preferisco scegliere',
      quickstartDoneToast: model => `${model} è configurato: le nuove chat vengono eseguite su questo computer.`,
      quickstartFailed: 'Configurazione del modello locale non riuscita',
      quickstartStageEngine: 'Motore',
      quickstartStageModel: 'Modello',
      quickstartStageFinish: 'Fine',
      useAction: 'Usa',
      activePill: 'Predefinito',
      updateTitle: 'È disponibile un aggiornamento del motore',
      updateDetail: (next, current) =>
        `È disponibile una build più recente di llama.cpp (${next}) pronta da installare; hai ${current}. I modelli continuano a funzionare durante lo scaricamento.`,
      updateAction: 'Aggiorna motore',
      updating: 'Aggiornamento del motore…',
      upToDateTitle: 'Motore aggiornato',
      upToDateDetail: (tag, backend) => `Esecuzione di llama.cpp ${tag} (${backend}), la build configurata.`,
      activeDetail: 'Le nuove chat usano questo modello; viene caricato quando invii il tuo primo messaggio',
      activeNotLoaded: 'Viene caricato con il tuo primo messaggio',
      loadedPill: 'In memoria',
      placementResident: 'tutto in GPU',
      placementSpilled: 'parte in RAM',
      placementResidentTip:
        'Viene eseguito interamente nella memoria della GPU con questa finestra di contesto: velocità massima.',
      placementSpilledTip:
        'Parte di questo modello viene eseguita dalla RAM di sistema: funziona, ma più lentamente. Una build più compatta o un contesto minore entrerebbero interamente.',
      loadingPill: 'Caricamento…',
      ejectTip: 'Libera la memoria GPU (verrà ricaricato con il prossimo messaggio)',
      ejected: 'Modello rimosso dalla memoria: memoria GPU liberata.',
      ejectFailed: 'Impossibile rimuovere il modello dalla memoria',
      stopServer: 'Spegni',
      startServer: 'Accendi',
      runtimeRunningDetail:
        'Il server locale è in esecuzione. Spegnerlo libera tutta la memoria GPU e impedisce alle nuove chat di usare modelli locali finché non lo riaccendi.',
      serverStopped: 'Server locale arrestato: memoria GPU liberata.',
      serverStarted: 'Server locale in esecuzione.',
      serverStopFailed: 'Impossibile arrestare il server locale',
      serverStartFailed: 'Impossibile avviare il server locale',
      activating: 'Avvio…',
      activateFailed: model => `Impossibile passare a ${model}`,
      activateDoneToast: model => `Le nuove chat usano ${model}.`,
      downloadFailed: model => `Scaricamento di ${model} non riuscito`,
      pillFitsGpu: 'Entra nella tua GPU',
      pillUsesRam: 'Usa la RAM di sistema',
      pillTooBig: 'Troppo grande per questo computer',
      browseTitle: 'Trova altri modelli',
      browseHint:
        'Cerca in tutto Hugging Face. I modelli che scarichi qui vengono adattati automaticamente al tuo computer, ma non li abbiamo testati.',
      browsePlaceholder: 'Cerca modelli per nome o autore…',
      browseSearching: 'Ricerca in Hugging Face',
      browseListing: 'Lettura dei file del modello',
      browseShowFiles: 'Mostra file',
      browseRefresh: 'Aggiorna',
      browseDownloads: 'download',
      browseLikes: 'mi piace',
      browseGated: 'richiede l’accesso a Hugging Face',
      browseNoGguf: 'Nessun file di modello compatibile trovato.',
      browseFitUnknown: 'Compatibilità sconosciuta',
      browseAlreadyDownloaded: 'Già scaricato.',
      addedByYou: 'Aggiunto da te',
      browseDownloadStarted: 'Scaricamento di {name}',
      browseDownloadAria: 'Scarica {name}',
      sideloadButton: 'Aggiungi file di modello',
      sideloadTitle: 'Scegli un file di modello GGUF',
      sideloadDone: '{name} aggiunto.',
      sideloadAlreadyPresent: 'È già nella tua libreria.',
      pillFullContext: (max: string) => `Contesto completo di ${max}`,
      pillFullContextTip: 'Viene eseguito con la finestra di contesto completa del modello fin dall’inizio',
      pillUpTo: (max: string) => `Fino a ${max} di contesto`,
      pillGrowsTip: 'Cresce automaticamente quando la tua conversazione richiede più spazio',
      pillVision: 'Vede le immagini',
      deleteAction: 'Elimina modello',
      deleteConfirm: (model: string) => `Eliminare ${model} dal disco?`,
      deleted: (model: string) => `${model} eliminato.`,
      deleteFailed: 'Errore durante l’eliminazione'
    },
    billing: {
      perMonth: (amount: string) => `${amount}/mese`,
      creditsPerMonth: (amount: string) => `${amount} crediti/mese`,
      usageLabel: (label: string) => `Utilizzo di ${label}`,
      freeTier: {
        signIn: 'Accedi',
        title: 'Sei nel piano gratuito di Nous',
        message: 'Accedi con un account Nous per sbloccare più modelli e strumenti.',
        caption:
          'Funziona con nous/welcome, con connettori inclusi. Accedendo mantieni i tuoi connettori e vengono aggiunti gli strumenti che richiedono un account e tutti gli altri modelli.',
        name: 'Nous · piano gratuito',
        footnote:
          'Il piano gratuito non ha saldo né nulla da pagare. Pagamento e utilizzo compaiono dopo l’accesso con un account Nous.',
        plan: 'Piano gratuito',
        model: 'Modello',
        connectors: 'Connettori',
        included: 'Inclusi'
      },
      amountValidation: {
        reloadTo: 'Ricarica fino a',
        greaterThanThreshold: 'L’importo della ricarica deve essere superiore alla soglia.',
        decimal: (label: string) => `${label}: inserisci un importo in dollari con al massimo 2 decimali.`,
        positive: (label: string) => `${label}: l’importo deve essere maggiore di 0 $.`,
        minimum: (label: string, amount: string) => `${label}: il minimo è ${amount}.`,
        maximum: (label: string, amount: string) => `${label}: il massimo è ${amount}.`
      },
      stepUp: {
        openVerification: 'Apri la pagina di verifica',
        dismiss: 'Ignora',
        waiting: 'In attesa del link di verifica…',
        verify: 'Verifica per continuare',
        deniedTitle: 'La verifica non è stata approvata',
        deniedBody: 'La verifica è terminata senza consentire la spesa remota per questo terminale.',
        successTitle: 'Verifica completata',
        successBody: 'La spesa remota è consentita per questo terminale.'
      },
      charge: {
        added: (amount?: string) => (amount ? `Aggiunti ${amount} $.` : 'Crediti aggiunti.'),
        failedTitle: 'Addebito non riuscito',
        unconfirmedTitle: 'Risultato dell’addebito non confermato',
        unconfirmedBody: (message: string) =>
          `${message} Il risultato del tuo ultimo addebito non è confermato: controlla il tuo saldo/cronologia prima di riprovare.`,
        checkTitle: 'Impossibile verificare l’addebito',
        checkBody: 'Impossibile verificare l’addebito.',
        untrackedTitle: 'Impossibile tracciare l’addebito',
        untrackedBody: 'Il servizio di fatturazione ha accettato la richiesta, ma non ha restituito un ID addebito.',
        timeoutTitle: 'Ancora in elaborazione dopo 5 minuti',
        timeoutBody: 'L’addebito potrebbe comunque completarsi. Controlla il portale prima di riprovare.',
        authenticationRequired:
          'La tua banca richiede una verifica (3DS). Completala nel portale per terminare questo acquisto.',
        expired: 'La tua carta è scaduta. Aggiornala nel portale.',
        declined: 'La tua carta è stata rifiutata. Prova un’altra carta nel portale.',
        failedBody: (reason: string) => `L’addebito non è stato completato (${reason}).`
      },
      title: 'Fatturazione',
      preview: 'anteprima',
      summary: {
        balance: 'Saldo',
        plan: 'Piano',
        autoRefill: 'Ricarica automatica'
      },
      sections: {
        invoices: 'Fatture',
        plan: 'Piano',
        paymentAndCredits: 'Pagamento e crediti',
        usage: 'Utilizzo'
      },
      usage: {
        title: 'Utilizzo'
      },
      buyCredits: {
        customAmount: 'Importo crediti personalizzato',
        title: 'Acquista crediti ora',
        buyButton: 'Acquista',
        processing: 'Elaborazione… verifica della liquidazione',
        added: (amount: string) => `Aggiunti ${amount}. Il saldo si sta aggiornando.`,
        retry: 'Riprova',
        openPortal: 'Apri il portale'
      },
      plan: {
        title: 'Piani',
        changePlan: 'Cambia piano',
        viewPlans: 'Vedi piani',
        backAria: 'Torna alla fatturazione',
        current: 'Piano attuale',
        scheduled: 'Programmato',
        empty: 'Al momento non ci sono piani disponibili a cui passare.',
        undo: 'Annulla',
        undoing: 'Annullamento…',
        downgrade: 'Passa a un piano inferiore',
        confirmDowngrade: 'Conferma il passaggio a un piano inferiore',
        tryAgain: 'Riprova',
        checkingChange: 'Verifica di questa modifica…',
        cannotChange: 'Questa modifica non si può fare qui.',
        alreadyOn: (name: string) => `Hai già il piano ${name}; non c’è nulla da cambiare.`,
        notScheduleable: 'Questa modifica non può essere programmata qui.',
        scheduling: 'Programmazione…',
        cancel: 'Annulla',
        effectScheduled: (targetName: string, effectiveAt: string, creditsDelta?: string) =>
          `Passaggio a ${targetName}: entra in vigore ${effectiveAt}. Non viene addebitato nulla ora; mantieni il tuo piano attuale fino ad allora.${creditsDelta ? ` Variazione dei crediti mensili: ${creditsDelta}.` : ''}`
      },
      autoReload: {
        threshold: 'Soglia',
        thresholdAria: 'Soglia di ricarica automatica',
        reloadTo: 'Ricarica fino a',
        reloadToAria: 'Importo obiettivo della ricarica automatica',
        turnOffConfirm: 'Disattivare la ricarica automatica?',
        turnOff: 'Disattiva',
        disable: 'Disattiva',
        updated: 'Ricarica automatica aggiornata.',
        turnedOff: 'Ricarica automatica disattivata.',
        manage: 'Gestisci',
        save: 'Salva',
        saving: 'Salvataggio…',
        cancel: 'Annulla'
      },
      state: {
        notice: {
          loggedOut: {
            title: 'Collega il tuo account Nous',
            message: 'Accedi con il tuo account Nous per vedere qui saldo, piano e utilizzo.',
            action: 'Accedi'
          },
          openPortal: 'Apri il portale ↗',
          noCard: {
            title: 'Nessun metodo di pagamento registrato',
            message:
              'L’acquisto di crediti e la ricarica automatica restano disattivati finché non registri una carta. Aggiungine una nel portale.',
            action: 'Aggiungi carta ↗'
          }
        },
        paymentMethod: {
          title: 'Metodo di pagamento',
          description: 'Gestisci la carta usata per le ricariche e i rinnovi dell’abbonamento.',
          addAction: 'Aggiungi metodo di pagamento',
          updateAction: 'Aggiorna',
          provenance: {
            autoRefill: 'carta di ricarica automatica',
            customerDefault: 'predefinita del cliente',
            subPin: 'carta dell’abbonamento',
            suffix: (label: string) => ` - ${label}`
          }
        },
        buyCredits: {
          description: 'Un singolo addebito sulla tua carta che si aggiunge oggi al tuo saldo.'
        },
        autoRefill: {
          title: 'Ricarica quando il saldo è basso',
          genericDescription: 'Mantieni il saldo ricaricato quando scende sotto la soglia.',
          offPill: 'Disattivata',
          enabledPill: 'Attivata',
          notAvailablePill: '—',
          manageCaption: 'Gestisci la ricarica automatica dal portale.',
          turnOnCaption: 'Attiva la ricarica automatica dal portale',
          chargesDescription: (reloadTo: string, threshold: string) =>
            `Addebita ${reloadTo} automaticamente quando il tuo saldo scende sotto ${threshold}.`,
          distinctCardCaption: (cardLabel: string) =>
            `La ricarica automatica addebita su ${cardLabel}: riconcilialo nel portale`,
          distinctCardFallback: 'un’altra carta',
          reconcileAction: 'Riconcila ↗'
        },
        usage: {
          subscriptionCredits: {
            title: 'Crediti dell’abbonamento',
            barLabel: 'Crediti dell’abbonamento rimanenti',
            captionResets: (date: string) => `Si azzera ${date}`,
            valueOf: (remaining: string, monthly: string) => `Rimangono ${remaining} su ${monthly}`,
            valueOver: (remaining: string, monthly: string, over: string) =>
              `Rimangono ${remaining} su ${monthly} · ${over} di eccedenza`
          },
          topupCredits: {
            title: 'Crediti di ricarica',
            caption: 'Non scadono'
          },
          monthlyCap: {
            title: 'Limite di spesa mensile',
            barLabel: 'Limite di spesa mensile utilizzato',
            captionDefault: 'Limite predefinito',
            captionSpending: 'Spesa remota mensile',
            valueUsed: (spent: string, limit: string) => `${spent} su ${limit} utilizzati`
          }
        },
        planCard: {
          freeTier: 'Gratis',
          chooseAction: 'Scegli ↗',
          adjustPlanAction: 'Modifica piano ↗',
          unavailableCaption: 'I dettagli dell’abbonamento non sono disponibili; puoi comunque aprire il portale.',
          downgradeCaption: (tierName: string, when: string) => `Passa a ${tierName} il ${when}.`,
          cancellationCaption: (when: string) => `Verrà annullato il ${when}.`,
          renewsCaption: (date: string) => `Si rinnova ${date}`,
          noSubscriptionCaption: 'Nessun abbonamento attivo: i modelli a pagamento consumano crediti di ricarica.'
        }
      },
      errors: {
        consentRequired: {
          title: 'È necessario confermare la carta',
          message: 'Conferma questa carta per gli addebiti del terminale nel portale'
        },
        insufficientScope: {
          title: 'La spesa remota richiede approvazione',
          message: 'Questo richiede di consentire la spesa remota. Avvia una ricarica per consentirla e riprova.'
        },
        remoteSpendingRevoked: {
          title: 'La spesa remota è stata interrotta',
          messageByAdmin: 'Un amministratore ha interrotto la spesa remota per questo terminale.',
          messageBySelf: 'Hai interrotto la spesa remota per questo terminale.'
        },
        remoteSpendingReconnect: (who: string) =>
          `${who} Riconnettiti da Impostazioni -> Gateway per autorizzare di nuovo questo dispositivo.`,
        sessionRevoked: {
          title: 'Sessione terminata',
          message: 'La tua sessione è stata chiusa. Accedi di nuovo da Impostazioni → Gateway.'
        },
        cliBillingDisabled: {
          title: 'La spesa remota è disattivata',
          message:
            'La spesa remota è disattivata per questo account; un amministratore della fatturazione può attivarla dalla pagina Hermes Agent del portale.'
        },
        roleRequired: {
          title: 'Richiesto ruolo di amministratore',
          message:
            'Per aggiungere fondi serve un amministratore o un proprietario dell’organizzazione. Chiedi aiuto a un amministratore oppure gestiscilo nel portale.'
        },
        idempotencyConflict: {
          title: 'Avvia una nuova ricarica',
          message: '🔴 Quella chiave di addebito è già stata usata per un altro importo. Avvia una nuova ricarica.'
        },
        noPaymentMethod: {
          title: 'Nessuna carta salvata',
          message:
            '💳 Non c’è ancora una carta salvata per gli addebiti del terminale. Configurane una nel portale ' +
            '(gli acquisti singoli di crediti non salvano una carta riutilizzabile).'
        },
        orgAccessDenied: {
          title: 'Accesso all’organizzazione negato',
          message: 'Questo token non è collegato a un’organizzazione che puoi gestire'
        },
        monthlyCapExceeded: {
          title: 'Raggiunto il limite di spesa mensile',
          messageReached: '🔴 Limite di spesa mensile raggiunto.',
          messageHeadroom: (remaining: string) =>
            `🔴 Limite di spesa mensile raggiunto: restano ${remaining} $ di margine.`
        },
        rateLimited: {
          title: 'Troppi addebiti in questo momento',
          message: (mins: number) =>
            mins > 0
              ? `🟡 Troppi addebiti in questo momento (riprova tra ~${mins} min). Non è un errore di pagamento.`
              : '🟡 Troppi addebiti in questo momento. Non è un errore di pagamento.'
        },
        stripeUnavailable: {
          title: 'Stripe ha problemi',
          message: (mins: number) =>
            mins > 0
              ? `Stripe ha problemi: riprova tra ~${mins} min`
              : 'Stripe ha problemi: riprova a breve'
        },
        upgradeCapExceeded: {
          title: 'Raggiunto il limite giornaliero di cambi di piano',
          message: 'Raggiunto il limite giornaliero di cambi di piano: riprova domani'
        },
        endpointUnavailable: {
          title: 'Endpoint di fatturazione non disponibile',
          message:
            'L’endpoint di fatturazione ha restituito una risposta che non è JSON (potrebbe non essere disponibile in questo deployment).'
        },
        timeout: {
          title: 'Timeout della richiesta di fatturazione',
          message: 'La richiesta di fatturazione è andata in timeout.'
        },
        transport: {
          title: 'Connessione di fatturazione non riuscita',
          message: 'La richiesta di fatturazione non è riuscita prima di raggiungere il gateway.'
        },
        default: {
          title: 'Richiesta di fatturazione non riuscita',
          message: 'La richiesta di fatturazione non è riuscita.'
        }
      }
    },
    providers: {
      connectAccount: 'Collega un account',
      haveApiKey: 'Hai una chiave API?',
      intro:
        'Accedi con un abbonamento, senza copiare chiavi API. Hermes esegue l’accesso del browser per te, proprio qui nell’app.',
      connected: 'Connesso',
      collapse: 'Comprimi',
      connectAnother: 'Collega un altro provider',
      otherProviders: 'Altri provider',
      disconnect: 'Disconnetti',
      disconnectInTerminal: 'Disconnetti (esegue il comando di rimozione nel terminale)',
      removeConfirm: provider => `Rimuovere ${provider}?`,
      removeExternalGeneric: provider => `${provider} è gestito con la sua CLI; rimuovilo da lì.`,
      removeKeyManaged: provider => `${provider} si configura con una chiave API. Rimuovilo in Chiavi API.`,
      removeTerminalConfirm: (provider, command) =>
        `Disconnettere ${provider}? Verrà eseguito “${command}” nel terminale per cancellare la credenziale.`,
      removeTerminalRunning: provider => `Disconnessione di ${provider} in esecuzione nel terminale…`,
      removedTitle: 'Account eliminato',
      removedMessage: provider => `${provider} è stato eliminato.`,
      failedRemove: provider => `Impossibile eliminare ${provider}`,
      noProviderKeys: 'Nessuna chiave API di provider disponibile.',
      searchKeys: 'Cerca provider…',
      noKeysMatch: 'Nessun provider corrisponde alla tua ricerca.',
      localEndpoint: {
        title: 'Endpoint locale o personalizzato',
        description:
          'Collega Hermes a qualsiasi endpoint compatibile con OpenAI (Zyphra, vLLM, llama.cpp, Ollama, ecc.).'
      },
      loading: 'Caricamento provider...'
    },
    sessions: {
      loading: 'Caricamento sessioni archiviate…',
      archivedTitle: 'Sessioni archiviate',
      archivedIntro:
        'Le chat archiviate vengono nascoste dalla barra laterale, ma conservano tutti i loro messaggi. Fai Alt/⌥+Maiusc/⇧-clic su una chat nella barra laterale per archiviarla.',
      emptyArchivedTitle: 'Niente archiviato',
      emptyArchivedDesc: 'Archivia una chat per nasconderla qui.',
      unarchive: 'Annulla archiviazione',
      deletePermanently: 'Elimina definitivamente',
      messages: count => `${count} ${count === 1 ? 'messaggio' : 'messaggi'}`,
      restored: 'Ripristinato',
      deleteConfirm: title => `Eliminare definitivamente "${title}"? Questa operazione non può essere annullata.`,
      autoArchiveTitle: 'Archivia automaticamente le chat inattive',
      autoArchiveDesc:
        'Archivia automaticamente le chat che non usi da un po’ di tempo. Le chat fissate non vengono mai archiviate e non viene eliminato nulla; le chat archiviate vengono solo spostate qui.',
      autoArchiveDaysLabel: 'Archivia dopo',
      autoArchiveDaysUnit: 'giorni di inattività',
      autoArchiveFailed: 'Impossibile aggiornare l’archiviazione automatica',
      defaultDirTitle: 'Directory di progetto predefinita',
      defaultDirDesc:
        'Le nuove sessioni iniziano in questa cartella a meno che tu ne scelga un’altra. Lasciala non definita per usare la tua directory home.',
      defaultDirUpdated:
        'Directory di progetto predefinita aggiornata; avvia una nuova chat (Ctrl/⌘+N) perché la modifica abbia effetto.',
      defaultsTo: label => `Predefinito: ${label}.`,
      change: 'Cambia',
      choose: 'Scegli',
      clear: 'Cancella',
      notSet: 'Non definita',
      failedLoad: 'Impossibile caricare le sessioni archiviate',
      unarchiveFailed: 'Impossibile annullare l’archiviazione',
      deleteFailed: 'Impossibile eliminare',
      updateDirFailed: 'Impossibile aggiornare la directory predefinita',
      clearDirFailed: 'Impossibile cancellare la directory predefinita'
    },
    toolsets: {
      loadingConfig: 'Caricamento della configurazione',
      savedTitle: 'Credenziale salvata',
      savedMessage: key => `${key} aggiornata.`,
      removedTitle: 'Credenziale rimossa',
      removedMessage: key => `${key} rimossa.`,
      failedSave: key => `Impossibile salvare ${key}`,
      failedRemove: key => `Impossibile rimuovere ${key}`,
      failedReveal: key => `Impossibile mostrare ${key}`,
      removeConfirm: key => `Rimuovere ${key} da .env?`,
      set: 'Definisci',
      notSet: 'Non definito',
      selectedTitle: 'Provider selezionato',
      selectedMessage: provider => `${provider} è attivo.`,
      failedSelect: provider => `Impossibile selezionare ${provider}`,
      failedLoad: 'Impossibile caricare la configurazione degli strumenti',
      noProviderOptions:
        'Questo set di strumenti non ha opzioni di provider; attivalo e funzionerà con la tua configurazione attuale.',
      noProviders: 'Al momento non ci sono provider disponibili per questo set di strumenti.',
      ready: 'Pronto',
      needsSignIn: 'Accesso richiesto',
      needsSetup: 'Configurazione richiesta',
      activeBackend: 'Attivo',
      activeBackendHint: 'Questo è il tuo backend attivo',
      useBackend: 'Usa questo backend',
      nousIncluded: 'Incluso con un abbonamento Nous: accedi con il tuo account Nous per attivarlo.',
      nousAuthNeededTitle: 'Accedi con il tuo account Nous',
      nousAuthNeededMessage: (provider: string) =>
        `${provider} è salvato, ma funzionerà solo dopo l’accesso con il tuo account Nous.`,
      nousAuthSignIn: 'Accedi',
      nousAuthDoneTitle: 'Account Nous collegato',
      nousAuthDoneMessage: 'I backend del tuo abbonamento sono già attivi.',
      nousAuthFailed: 'Accesso a Nous non completato',
      nousAuthFailedMessage: 'Riprova.',
      nousAuthTryAgain: 'Riprova',
      noApiKeyRequired: 'Non è richiesta una chiave API.',
      postSetupHint: step =>
        `Questo backend richiede un’installazione una tantum (${step}). Viene eseguita su questa macchina e può richiedere alcuni minuti.`,
      postSetupInstalledHint: 'Installato. Ripeti la configurazione solo se qualcosa non funziona.',
      postSetupRun: 'Esegui installazione',
      postSetupRerun: 'Ripeti installazione',
      postSetupInstalled: 'Installato',
      postSetupRunning: 'Installazione…',
      postSetupStarting: 'Avvio…',
      postSetupCompleteTitle: 'Installazione completata',
      postSetupCompleteMessage: step => `${step} installato.`,
      postSetupErrorTitle: 'L’installazione è terminata con errori',
      postSetupErrorMessage: (step: string) =>
        `La configurazione di ${step} non è terminata. Apri i log per vedere perché ed esegui di nuovo la configurazione.`,
      postSetupOpenLogs: 'Apri i log',
      postSetupRunAgain: 'Esegui di nuovo',
      postSetupFailed: step => `Impossibile eseguire l’installazione di ${step}`,
      webSearchActive: backend => `Ricerca: ${backend}`,
      webExtractActive: backend => `Estrazione: ${backend}`,
      webCapabilityUnset: 'non definito',
      webUseForSearch: 'Usa per la ricerca',
      webUseForExtract: 'Usa per l’estrazione',
      webUsedForSearch: 'Backend di ricerca',
      webUsedForExtract: 'Backend di estrazione',
      webCapabilitySelectedMessage: (provider, capability) =>
        `${provider} ora si occupa di ${capability === 'search' ? 'delle ricerche web' : 'dell’estrazione di contenuti web'}.`,
      failedSelectCapability: provider => `Impossibile configurare ${provider}`,
      loadingModels: 'Caricamento del catalogo modelli...',
      modelSectionTitle: 'Modello',
      modelCount: count => `${count} ${count === 1 ? 'modello' : 'modelli'}`,
      modelInUse: 'In uso',
      modelDefault: 'predefinito',
      modelInactiveHint: 'Seleziona prima questo backend per cambiarne il modello.',
      modelSelectedTitle: 'Modello selezionato',
      modelSelectedMessage: model => `${model} si applica alle nuove sessioni.`,
      failedSelectModel: model => `Impossibile selezionare ${model}`,
      terminalBackend: {
        sectionTitle: 'Backend di esecuzione',
        loading: 'Verifica dei backend di esecuzione…',
        failedLoad: 'Impossibile caricare i backend del terminale',
        ready: 'Pronto',
        needsSetup: 'Richiede configurazione',
        unavailable: 'Non disponibile',
        inUse: 'In uso',
        selectedTitle: 'Backend selezionato',
        selectedMessage: backend =>
          `I comandi del terminale ora vengono eseguiti tramite ${backend}. Si applica alle nuove sessioni.`,
        failedSelect: backend => `Impossibile selezionare ${backend}`,
        needsSetupHint:
          'Questo backend è selezionato senza configurazione completa: i comandi falliranno finché la configurazione non sarà terminata.',
        needsSetupConfirmTitle: (backend: string) => `Selezionare ${backend} comunque?`,
        needsSetupConfirmDescription: (detail: string) =>
          `${detail} Le sessioni avviate dopo questa modifica non avranno strumenti di terminale né di file finché la configurazione non sarà terminata.`,
        needsSetupConfirmDescriptionGeneric:
          'Questo backend non è ancora configurato. Le sessioni avviate dopo questa modifica non avranno strumenti di terminale né di file finché la configurazione non sarà terminata.',
        needsSetupConfirmAction: 'Seleziona comunque',
        unavailableTitle: 'I comandi del terminale non sono disponibili',
        unavailableMessage: (backend: string) =>
          `Hermes non può eseguire comandi di shell al momento: ${backend} non è pronto. Passa a Local oppure termina la configurazione di ${backend} e riprova.`,
        openBackendSettings: 'Apri le impostazioni del terminale',
        useLocal: 'Usa Local',
        switchedToLocal: 'I comandi del terminale ora vengono eseguiti localmente. Si applica alle nuove sessioni.'
      },
      browserRealProfile: {
        label: 'Usa il mio profilo reale del browser',
        description:
          'Copia gli accessi e i cookie del tuo browser predefinito in uno snapshot gestito con cui naviga l’agente. Il tuo profilo in uso non viene mai aperto direttamente. Si applica alle nuove sessioni.',
        enabledTitle: 'Navigazione con profilo reale attivata',
        enabledMessage: 'Le nuove sessioni navigheranno con uno snapshot del tuo profilo predefinito del browser.',
        disabledTitle: 'Navigazione con profilo reale disattivata',
        disabledMessage: 'Lo snapshot del profilo verrà eliminato; le nuove sessioni usano un browser pulito.',
        failedSave: 'Impossibile salvare l’impostazione del profilo reale',
        prompt: {
          title: 'Mantieni l’accesso ai tuoi siti',
          body: 'Lascia che Hermes navighi con uno snapshot del tuo profilo predefinito del browser, così i siti si aprono con l’accesso già effettuato.',
          bulletSnapshot: 'Cookie e accessi vengono copiati in uno snapshot gestito.',
          bulletLiveProfile: 'Il tuo profilo del browser in uso non viene mai aperto direttamente.',
          bulletLocal: 'Nulla esce da questo computer.',
          dontShowAgain: 'Non mostrare più',
          notNow: 'Non ora',
          enable: 'Usa il mio profilo'
        }
      }
    }
  },
  skills: {
    tabSkills: 'Skills',
    tabToolsets: 'Set di strumenti',
    configuringProfile: 'Configurazione in corso:',
    all: 'Tutto',
    searchSkills: 'Cerca skills...',
    searchToolsets: 'Cerca set di strumenti...',
    refresh: 'Aggiorna skills',
    refreshing: 'Aggiornamento skills',
    loading: 'Caricamento delle capacità...',
    noSkillsTitle: 'Nessuna skill trovata',
    noSkillsDesc: 'Prova una ricerca più ampia o un’altra categoria.',
    noToolsetsTitle: 'Nessun set di strumenti trovato',
    noToolsetsDesc: 'Prova una ricerca più ampia.',
    noDescription: 'Senza descrizione.',
    configured: 'Configurato',
    needsKeys: 'Richiede chiavi',
    visionModelHint:
      'Vision usa la configurazione del tuo modello ausiliario; il modello compatibile con le immagini si sceglie lì, non qui per ogni provider.',
    visionModelLink: 'Scegli modello di visione in Impostazioni → Modelli',
    toolsetsEnabled: (enabled, total) => `${enabled}/${total} set attivati`,
    configureToolset: label => `Configura ${label}`,
    toggleToolset: (label, enabled) =>
      `Attiva/disattiva set di strumenti ${label} ${enabled ? 'attivato' : 'disattivato'}`,
    skillsLoadFailed: 'Impossibile caricare gli skills',
    toolsetsRefreshFailed: 'Impossibile aggiornare i set di strumenti',
    skillEnabled: 'Skill attivato',
    skillDisabled: 'Skill disattivato',
    toolsetEnabled: 'Set attivato',
    toolsetDisabled: 'Set disattivato',
    appliesToNewSessions: name => `${name} si applica alle nuove sessioni.`,
    failedToUpdate: name => `Impossibile aggiornare ${name}`,
    sortMostUsed: 'Più usate',
    sortAlpha: 'A–Z',
    sortMostUsedDesc: '↓ Più usate',
    sortLeastUsedAsc: '↑ Meno usate',
    enableAll: 'Attiva tutto',
    disableAll: 'Disattiva tutto',
    disableUnused: 'Disattiva quelli inutilizzati',
    bulkUpdated: count =>
      `${count === 1 ? 'Aggiornato' : 'Aggiornati'} ${count} ${count === 1 ? 'elemento' : 'elementi'} per le nuove sessioni.`,
    bulkNoChange: 'Nulla da modificare.',
    usageCount: count => `usato ${count}×`,
    provenance: {
      agent: 'Appreso',
      bundled: 'Integrato',
      hub: 'Hub'
    },
    emptyNoneFound: noun => `Non sono stati trovati ${noun}`,
    emptyNothingMatches: query => `Nessuna corrispondenza per “${query}”.`,
    emptyNoneAvailable: noun => `Non ci sono ancora ${noun} disponibili.`,
    changesApplyNewSessions: 'Le modifiche si applicano alle nuove sessioni.',
    skillUpdated: 'Skill aggiornato',
    edit: 'Modifica',
    archive: 'Archivia',
    skillArchivedTitle: 'Skill archiviato',
    skillArchivedMessage: 'Puoi ripristinarlo con hermes curator restore.',
    tabPlugins: 'Plugin',
    plugins: {
      agentTitle: 'Plugin dell’agente',
      agentBlurb:
        'Ampliano l’agente del profilo selezionato: strumenti, hook, provider. Si applicano dopo il riavvio del gateway.',
      pageBlurb: 'Un plugin può estendere questa app, l’agente o entrambi; ogni metà ha il proprio interruttore.',
      halfDesktop: 'Desktop',
      halfDesktopHint: 'questa app, uguale per tutti i profili',
      halfAgent: 'Agente',
      halfAgentIn: (profile: string) => `Agente in ${profile}`,
      defaultProfile: 'Hermes (predefinito)',
      kindAgent: 'Agente',
      kindDesktop: 'Desktop',
      kindBoth: 'Agente + Desktop',
      installAgentHere: 'Installa qui',
      installAgentHereTip: (profile: string) =>
        `La metà desktop è caricata in questa app, ma la metà dell’agente non è installata in ${profile}. Installala lì.`,
      installAgentHereNoOrigin:
        'La metà dell’agente non è installata in questo profilo e questo pacchetto è stato copiato a mano (senza voce di catalogo né remote git), quindi non può essere installato da qui. Copia la sua cartella nel profilo o reinstallalo da Git.',
      desktopHalfPending: 'copia in corso…',
      desktopHalfPendingTip:
        'Questo pacchetto include una metà desktop che non è ancora stata copiata nell’app. Usa Ripeti scansione o riavvia l’app.',
      desktopHalfRemote: 'non disponibile (backend remoto)',
      desktopHalfRemoteTip:
        'La metà desktop di questo pacchetto si trova sul disco del backend remoto, che questa app non può leggere. Per usarla qui, esegui Installa da Git con l’URL del repository del pacchetto e la destinazione Desktop selezionata; così clona la metà desktop su questo computer.',
      emptyAll: 'Ancora nessun plugin.',
      empty: 'Nessun plugin dell’agente installato per questo profilo.',
      emptyHint: 'Esplora il catalogo qui sotto e installa un plugin verificato con un clic.',
      loadFailed: 'Impossibile caricare i plugin dell’agente',
      toggleFailed: (name: string) => `Impossibile modificare ${name}`,
      toolsetOn: (name: string, profile: string) => `Strumenti agente di ${name} attivati per ${profile}`,
      toolsetOff: (name: string, profile: string) => `Strumenti agente di ${name} disattivati per ${profile}`,
      toolsetToggleFailed: (name: string) =>
        `Impossibile modificare gli strumenti agente di ${name}; il pannello Desktop non è stato modificato`,
      legacyBackend:
        'Questo backend è precedente agli interruttori dei plugin per chiave: aggiorna Hermes per gestirli qui.',
      portableBadge: 'portatile',
      serverStates: {
        connected: 'connesso',
        app_not_running: 'l’app non è in esecuzione',
        hermes_not_connected: 'connessione MCP mancante',
        endpoint_unavailable: 'endpoint non disponibile',
        no_interactive_session: 'nessuna sessione interattiva',
        version_too_old: 'versione troppo vecchia',
        missing_app: 'app mancante',
        unsupported_gpu: 'GPU non compatibile',
        unknown: 'stato sconosciuto'
      },
      catalogTitle: 'Catalogo dei plugin',
      catalogBrowse: 'Esplora',
      catalogHide: 'Nascondi l’esploratore del catalogo',
      catalogHint:
        'Premi "+ Aggiungi a questo agente" su qualsiasi plugin: le voci verificate vengono installate al loro commit fissato nel profilo selezionato. I plugin agente+desktop inclusi offrono entrambe le metà.',
      alreadyInstalled: (name: string) => `${name} è già installato in questo profilo.`,
      catalogProvenance: (sha: string) =>
        `Installato dal catalogo di Hermes${sha ? ` al commit fissato ${sha}` : ''}.`,
      pinnedProvenance: (sha: string) =>
        `Fissato al commit ${sha}. Gli aggiornamenti vengono rifiutati finché non viene reinstallato con un nuovo commit fissato.`,
      pinnedBadge: (sha: string) => `fissato @ ${sha}`,
      tierOfficial: 'ufficiale',
      tierCommunity: 'community',
      updateToPin: (sha: string) => `Aggiorna a ${sha}`,
      updateFailed: (name: string) => `Impossibile aggiornare ${name}`,
      updated: (name: string) =>
        `${name} è stato aggiornato al commit fissato attuale del catalogo. Riavvia il gateway per applicarlo.`,
      updateConsentTitle: (name: string) => `${name} chiede di più`,
      updateConsentBody: (name: string, sha: string) =>
        `Il nuovo commit fissato di ${name} nel catalogo (${sha}) aggiunge superfici che la versione installata non ha. Applicalo solo se ti fidi di esse:`,
      updateConsentConfirm: 'Applica aggiornamento',
      uninstall: 'Disinstalla',
      uninstallTip: (name: string, profile: string) => `Disinstalla ${name} da ${profile}`,
      uninstallConfirmTitle: (name: string) => `Disinstallare ${name}?`,
      uninstallConfirmBody: (name: string, profile: string) =>
        `Questa operazione rimuove i file del plugin dal profilo ${profile}. Eventuali metà desktop incluse vengono rimosse con esso. Puoi reinstallarlo dal catalogo o da Git quando vuoi.`,
      uninstallFailed: (name: string) => `Impossibile disinstallare ${name}`,
      uninstalled: (name: string) => `${name} disinstallato. Riavvia il gateway per rimuoverlo.`,
      uninstallDesktopTip: (name: string) => `Disinstalla ${name} da questa app`,
      uninstallDesktopConfirmBody: (name: string) =>
        `Questa operazione elimina ${name} dalla cartella desktop-plugins di questo computer e lo disattiva subito. Puoi reinstallarlo da Git o rimettere la cartella quando vuoi.`,
      uninstalledDesktop: (name: string) => `${name} disinstallato.`,
      deepLinkErrorTitle: 'Link di installazione del plugin rifiutato',
      deepLinkCatalogInvalidName: 'Nome del catalogo nel link mancante o non valido.',
      deepLinkCatalogUnknown: (name: string) =>
        `\u201C${name}\u201D non è nel catalogo dei plugin di Hermes. Non è stato installato nulla.`,
      deepLinkCatalogUnavailable:
        'Impossibile caricare il catalogo dei plugin di Hermes. Controlla la tua connessione e riapri il link.',
      settingsToggle: (name: string) => `Impostazioni: ${name}`,
      settingsForm: {
        save: 'Salva configurazione',
        saved: (name: string) => `Configurazione di ${name} salvata.`,
        saveFailed: (name: string) => `Impossibile salvare la configurazione di ${name}`,
        required: 'Obbligatorio',
        secretSet: '•••••••• (configurato)',
        secretStoredAs: (env: string) =>
          `Viene salvato nel .env del profilo come ${env}, mai in config.yaml; lascialo vuoto per conservare il valore attuale.`
      }
    },
    officialCatalog: 'Disponibili per l’installazione',
    officialPill: 'Ufficiale',
    hub: {
      searchPlaceholder: 'Cerca nell’hub degli skills',
      search: 'Cerca',
      searching: 'Ricerca…',
      connectingHubs: 'Connessione agli hub degli skills…',
      connectedHubs: 'Hub connessi:',
      featured: 'Skills in evidenza',
      landingHint:
        'Cerca nell’hub per esplorare skills installabili dall’indice ufficiale, da GitHub e da fonti della community.',
      noResults: 'Nessuna skill corrispondente trovata nell’hub.',
      resultCount: (count, ms) =>
        `${count} ${count === 1 ? 'risultato' : 'risultati'}${ms !== null ? ` in ${ms} ms` : ''}`,
      timedOut: sources => `Tempo di attesa scaduto: ${sources}`,
      installed: 'Installato',
      install: 'Installa',
      installing: 'Installazione…',
      uninstall: 'Disinstalla',
      uninstalling: 'Disinstallazione…',
      updateAll: 'Aggiorna le skills installate',
      updating: 'Aggiornamento…',
      preview: 'Anteprima',
      scan: 'Analizza',
      scanning: 'Analisi…',
      close: 'Chiudi',
      files: 'File',
      noReadme: 'Questa skill non ha un’anteprima di SKILL.md.',
      trust: {
        builtin: 'integrato',
        trusted: 'affidabile',
        community: 'community'
      },
      verdictSafe: 'Sicuro',
      verdictCaution: 'Attenzione',
      verdictDangerous: 'Pericoloso',
      policyAllow: 'Installazione consentita',
      policyAsk: 'Verifica prima di installare',
      policyBlock: 'Installazione bloccata dalla policy',
      findings: count => `${count} ${count === 1 ? 'riscontro' : 'riscontri'}`,
      noFindings: 'Nessun riscontro di sicurezza trovato.',
      installStarted: name => `Installazione di ${name}…`,
      uninstallStarted: name => `Disinstallazione di ${name}…`,
      updateStarted: 'Aggiornamento delle skills installate…',
      actionFailed: 'Azione sulla skill non riuscita',
      installBlockedTitle: (name: string) => `Impossibile installare ${name}`,
      installBlockedMessage: (findings: number, unverified: boolean) =>
        `L’analisi di sicurezza ha segnalato ${findings > 0 ? `${findings} ${findings === 1 ? 'elemento' : 'elementi'}` : 'pattern di rischio'} da esaminare${unverified ? ' e la skill proviene da una fonte non verificata' : ''}. Leggi l’analisi prima di decidere se ti fidi dell’autore.`,
      viewScan: 'Vedi analisi',
      openLog: 'Apri log',
      actionLog: 'Log delle azioni',
      alreadyInstalled: (name: string) => `"${name}" è già installata`,
      pickerTitle: 'Skills Hub',
      pickerBrowse: 'Esplora tutto l’hub',
      pickerHide: 'Nascondi l’esploratore dell’hub',
      pickerHint: 'Premi "+ Aggiungi a questo agente" su qualsiasi skill: viene installata e appare nell’elenco qui sopra.',
      loadFailed: 'Impossibile caricare l’hub degli skills',
      previewFailed: 'Impossibile caricare l’anteprima della skill',
      scanFailed: 'Analisi di sicurezza non riuscita',
      searchFailed: 'Ricerca nell’hub non riuscita'
    }
  },
  starmap: {
    title: 'Grafo della memoria',
    subtitle: (nodes, clusters) => `${nodes} skills in ${clusters} categorie`,
    close: 'Chiudi grafo della memoria',
    refresh: 'Aggiorna',
    memory: 'Memoria',
    filterAll: 'Tutto',
    filterUsed: 'Usato',
    filterLearned: 'Appreso',
    viewGraph: 'Grafo',
    loadFailed: 'Impossibile caricare il grafo della memoria',
    loading: 'Caricamento…',
    emptyTitle: 'Non è ancora stato appreso nulla',
    emptyDesc: 'Man mano che Hermes crea skills e memorie per il tuo lavoro, appariranno qui.',
    share: 'Condividi mappa',
    shareHint:
      'Copia il codice per condividere questa mappa o incollane uno per caricarla. Include solo il layout, non il testo delle tue memorie né delle skills.',
    shareTitle: 'Importa / esporta mappa',
    sharePlaceholder: 'Incolla un codice mappa…',
    copy: 'Copia codice mappa',
    copied: 'Copiato!',
    importMap: 'Importa una mappa',
    importBtn: 'Carica',
    importEmpty: 'Incolla un codice mappa per caricarlo.',
    importSuccess: nodes => `È stata caricata una mappa con ${nodes} ${nodes === 1 ? 'nodo' : 'nodi'}.`,
    importedBadge: 'mappa importata',
    resetToMine: 'Torna alla mia mappa'
  },
  agents: {
    extendedTranscript: 'Trascrizione estesa',
    transcriptTruncated: 'Mostra gli ultimi 16 KiB',
    transcriptUnavailable: 'Trascrizione live non disponibile',
    close: 'Chiudi agenti',
    title: 'Albero di generazione',
    subtitle: 'Attività live dei subagenti per il turno attuale.',
    emptyTitle: 'Nessun subagente attivo',
    emptyDesc: 'Quando un turno delegherà lavoro, gli agenti figli mostreranno qui i loro progressi.',
    running: 'In esecuzione',
    failed: 'Non riuscito',
    done: 'Completato',
    streaming: 'In streaming',
    files: 'File',
    moreFiles: count => `+${count} altri file`,
    moreAgents: (count: number) => `+${count} altri agenti`,
    queued: 'In coda',
    waitingActivity: 'In attesa di attività',
    steer: 'Indirizza',
    steerPlaceholder: 'Istruzioni per questo subagente',
    steerQueued: 'In coda per il prossimo checkpoint',
    stopRequested: 'Arresto richiesto',
    requestRejected: 'Il subagente non ha accettato la richiesta',
    delegation: index => `Delegazione ${index}`,
    workers: count => `${count} ${count === 1 ? 'worker' : 'worker'}`,
    workersActive: count => `${count} attivi`,
    agentsCount: count => `${count} ${count === 1 ? 'agente' : 'agenti'}`,
    activeCount: count => `${count} attivi`,
    failedCount: count => `${count} non riusciti`,
    toolsCount: count => `${count} strumenti`,
    filesCount: count => `${count} file`,
    updatedAgo: age => `aggiornato ${age}`,
    ageNow: 'adesso',
    ageSeconds: seconds => `${seconds}s fa`,
    ageMinutes: minutes => `${minutes}m fa`,
    ageHours: hours => `${hours}h fa`,
    ageDays: days => `${days}d fa`,
    durationSeconds: seconds => `${seconds}s`,
    durationMinutes: (minutes, seconds) => `${minutes}m ${seconds}s`,
    tokens: value => `${value} tok`
  },
  commandCenter: {
    close: 'Chiudi Centro comandi',
    paletteTitle: 'Palette dei comandi',
    back: 'Indietro',
    searchPlaceholder: 'Cerca sessioni, viste e azioni',
    goTo: 'Vai a',
    goToSession: 'Vai alla sessione',
    branches: 'Branch',
    projects: 'Progetti',
    openFolder: 'Apri cartella come progetto…',
    openFolderAt: path => `Apri cartella come progetto — ${path}`,
    newSessionInProject: project => `Nuova sessione in ${project}`,
    commands: 'Comandi',
    startInBranch: branch => `Nuova conversazione in ${branch}`,
    commandCenter: 'Centro comandi',
    appearance: 'Aspetto',
    settings: 'Impostazioni',
    changeTheme: 'Cambia tema...',
    changeColorMode: 'Cambia modalità colore...',
    pets: {
      title: 'Mascotte',
      placeholder: 'Cerca mascotte…',
      loading: 'Caricamento della galleria petdex…',
      error: 'Impossibile accedere alla galleria petdex.',
      staleBackend: 'Riavvia Hermes per usare le mascotte; il backend è precedente a questa funzione.',
      empty: 'Nessuna mascotta corrispondente.',
      turnOff: 'Disattiva',
      turnOn: 'Attiva',
      installed: 'Installata',
      generatedTag: 'Generata',
      adoptFailed: 'Impossibile adottare quella mascotta.',
      toggleFailed: enabled => `Impossibile ${enabled ? 'accendere' : 'spegnere'} la mascotta.`,
      noneAvailable: 'Non ci sono mascotte disponibili. Scegline una qui sotto per installarla.'
    },
    generatePet: {
      title: 'Genera una mascotta',
      placeholder: 'Descrivi una mascotta da generare…',
      promptHint: 'Scrivi una descrizione e premi Invio per creare quattro versioni.',
      readyHint: 'Premi Invio per creare quattro versioni dalla tua descrizione.',
      generate: 'Genera',
      generating: 'Generazione…',
      retry: 'Riprova',
      hatch: 'Schiudi',
      spawning: 'Creazione…',
      hatching: 'La tua mascotta si sta schiudendo…',
      hatchingSub: 'Dandole vita…',
      hatched: 'È schiusa!',
      hatchRow: (_state, done, total) => `Disegno del fotogramma ${done} di ${total}…`,
      hatchComposing: 'Unione dei pezzi…',
      hatchSaving: 'Ci siamo quasi…',
      namePlaceholder: 'Dai un nome alla tua mascotta',
      staleBackend: 'Aggiorna Hermes per generare le mascotte.',
      backgroundHint: 'Puoi chiudere questa finestra; Hermes ti avviserà al termine.',
      slowProviderHint: 'Può richiedere alcuni minuti',
      remix: 'Remix',
      remixConfirmTitle: 'Remixare questo aspetto?',
      remixConfirmBody:
        'Genera un nuovo set di bozze usando questo come punto di partenza. Può richiedere alcuni minuti.',
      genericError: 'La generazione non è riuscita. Riprova o scegli un suggerimento.',
      referenceImageTooLarge: 'L’immagine di riferimento è troppo grande. Usane una da meno di 16 MB.',
      referenceImageInvalid: 'Impossibile leggere quell’immagine di riferimento. Prova con PNG, JPG, WebP o GIF.',
      adopt: 'Adotta',
      startOver: 'Ricomincia'
    },
    installTheme: {
      title: 'Installa tema…',
      pageTitle: 'Installa tema',
      placeholder: 'Cerca su VS Code Marketplace…',
      loading: 'Ricerca nel Marketplace…',
      error: 'Impossibile accedere al Marketplace.',
      empty: 'Nessun tema corrispondente.',
      install: 'Installa',
      installing: 'Installazione…',
      installed: 'Installato',
      installs: count => `${count} installazioni`
    },
    settingsFields: 'Campi di configurazione',
    mcpServers: 'Server MCP',
    archivedChats: 'Chat archiviate',
    sections: {
      maintenance: 'Manutenzione',
      sessions: 'Sessioni',
      system: 'Sistema',
      usage: 'Utilizzo'
    },
    nav: {
      newChat: {
        title: 'Nuova sessione',
        detail: 'Avvia una nuova sessione'
      },
      settings: {
        title: 'Impostazioni',
        detail: 'Configura Hermes Desktop'
      },
      capabilities: {
        title: 'Capacità',
        detail: 'Skills, strumenti, server MCP e plugin'
      },
      messaging: {
        title: 'Messaggistica',
        detail: 'Configura Telegram, Slack, Discord e altro'
      },
      artifacts: {
        title: 'Artefatti',
        detail: 'Esplora gli output generati'
      }
    },
    sectionEntries: {
      sessions: {
        title: 'Pannello delle sessioni',
        detail: 'Cerca, fissa e gestisci le sessioni'
      },
      system: {
        title: 'Pannello di sistema',
        detail: 'Stato del gateway, log, riavvio/aggiornamento'
      },
      usage: {
        title: 'Pannello di utilizzo',
        detail: 'Attività di token, costi e skills'
      }
    },
    providerNavigate: 'Naviga',
    providerSessions: 'Sessioni',
    refresh: 'Aggiorna',
    refreshing: 'Aggiornamento...',
    noResults: 'Nessun risultato trovato.',
    pinSession: 'Fissa sessione',
    unpinSession: 'Sblocca sessione',
    exportSession: 'Esporta sessione',
    deleteSession: 'Elimina sessione',
    noSessions: 'Ancora nessuna sessione.',
    gatewayRunning: 'Gateway di messaggistica in esecuzione',
    gatewayStopped: 'Gateway di messaggistica arrestato',
    hermesActiveSessions: (version, count) => `Hermes ${version} · Sessioni attive ${count}`,
    restartGateway: 'Riavvia gateway',
    openBrowser: 'Alterna browser',
    toggleBrowser: 'Alterna browser',
    gatewayRestartFailed: 'Impossibile riavviare il gateway.',
    sharedGatewayRestartTitle: 'Riavviare il gateway condiviso?',
    sharedGatewayRestartDescription: (bots: string) => `Tutti i bot di questo dispositivo si riconnettono: ${bots}`,
    sharedGatewayRestartConfirm: 'Riavvia tutto',
    sharedGatewayRestarted: (count: number) =>
      `Gateway condiviso riavviato (${count} ${count === 1 ? 'bot' : 'bot'})`,
    updateHermes: 'Aggiorna Hermes',
    reloadWindow: 'Ricarica finestra',
    actionRunning: 'in esecuzione',
    actionDone: 'completato',
    actionFailed: 'non riuscito',
    actionStartedWaiting: 'Azione avviata, in attesa dello stato...',
    loadingStatus: 'Caricamento stato...',
    recentLogs: 'Log recenti',
    noLogs: 'Ancora nessun log caricato.',
    days: count => `${count}d`,
    statSessions: 'Sessioni',
    statApiCalls: 'Chiamate API',
    statTokens: 'Token ingresso/uscita',
    statCost: 'Costo stim.',
    actualCost: cost => `reale ${cost}`,
    loadingUsage: 'Caricamento utilizzo...',
    noUsage: period => `Nessun utilizzo negli ultimi ${period} giorni.`,
    retry: 'Riprova',
    dailyTokens: 'Token giornalieri',
    input: 'ingresso',
    output: 'uscita',
    noDailyActivity: 'Nessuna attività giornaliera.',
    topModels: 'Modelli principali',
    noModelUsage: 'Ancora nessun utilizzo dei modelli.',
    topSkills: 'Skills principali',
    noSkillActivity: 'Ancora nessuna attività delle skills.',
    actions: count => `${count} azioni`,
    logFile: 'File di log',
    logLevel: 'Livello',
    logSearchPlaceholder: 'Cerca nei log…',
    maintenance: {
      runOps: 'Diagnostica',
      doctor: 'Esegui diagnostica',
      doctorDesc: 'Verifica lo stato dell’installazione, della configurazione e dei provider',
      securityAudit: 'Audit di sicurezza',
      securityAuditDesc: 'Analizza la configurazione e le skills alla ricerca di impostazioni rischiose',
      backup: 'Crea backup',
      backupDesc: 'Comprimi configurazione, memorie, skills e sessioni in un file ZIP',
      debugShare: 'Condividi dati di debug',
      debugShareDesc:
        'Carica un rapporto e log con i dati sensibili oscurati e ottieni link da condividere (eliminati automaticamente dopo 6 h)',
      debugShareRunning: 'Caricamento del rapporto di debug…',
      debugShareLinks: 'Link da condividere',
      debugShareFailed: 'Impossibile condividere i dati di debug',
      copyLink: 'Copia link',
      linkCopied: 'Link copiato',
      curator: 'Curatore delle skills',
      curatorDesc: 'Revisione in background che archivia le skills create dagli agenti e non più usate',
      curatorPaused: 'In pausa',
      curatorActive: 'Attivo',
      curatorDisabled: 'Disattivato',
      curatorLastRun: when => `Ultima esecuzione: ${when}`,
      curatorNeverRan: 'Non è mai stato eseguito',
      pause: 'Metti in pausa',
      resume: 'Riprendi',
      runNow: 'Esegui ora',
      memoryData: 'Dati di memoria',
      memoryDataDesc: 'File di memoria integrati inclusi in ogni sessione',
      memoryProvider: name => `Provider attivo: ${name}`,
      builtinMemory: 'integrata',
      memoryFile: 'Memoria dell’agente (MEMORY.md)',
      userFile: 'Profilo utente (USER.md)',
      bytes: size => size,
      empty: 'vuoto',
      resetMemory: 'Ripristina memoria',
      resetUser: 'Ripristina profilo',
      resetAll: 'Ripristina entrambi',
      resetConfirm: target => `Eliminare ${target}? Questa azione non può essere annullata.`,
      resetDone: files => `Eliminato: ${files}.`,
      resetFailed: 'Impossibile ripristinare la memoria',
      actionStarted: name => `${name} avviato; segui il log…`,
      actionFailed: name => `Impossibile avviare ${name}`,
      running: 'In esecuzione…',
      viewLog: 'Log delle azioni'
    }
  },
  messaging: {
    search: 'Cerca messaggistica...',
    statusFilter: {
      all: 'Tutti',
      bad: 'Errori',
      good: 'Connessi',
      muted: 'Inattivi',
      warn: 'Richiede attenzione'
    },
    loading: 'Caricamento piattaforme di messaggistica...',
    loadFailed: 'Impossibile caricare le piattaforme di messaggistica',
    states: {
      connected: 'Connesso',
      connecting: 'Connessione in corso',
      disabled: 'Disabilitato',
      fatal: 'Errore',
      gateway_stopped: 'Gateway di messaggistica arrestato',
      not_configured: 'Richiede configurazione',
      pending_restart: 'Riavvio necessario',
      retrying: 'Nuovo tentativo',
      startup_failed: 'Avvio non riuscito'
    },
    unknown: 'Sconosciuto',
    hintPendingRestart: 'Riavvia il gateway dalla barra di stato per applicare questa modifica.',
    sharedListenerUrl: 'Servito sul listener del gateway condiviso su',
    hintGatewayStopped: 'Avvia il gateway dalla barra di stato per connetterti.',
    credentialsSet: 'Credenziali definite',
    needsSetup: 'Richiede configurazione',
    gatewayStopped: 'Gateway di messaggistica arrestato',
    getCredentials: 'Ottieni credenziali',
    openSetupGuide: 'Apri guida alla configurazione',
    required: 'Obbligatorio',
    recommended: 'Consigliato',
    advanced: count => `Avanzate (${count})`,
    noTokenNeeded:
      'Questa piattaforma non ha bisogno di un token qui. Usa la guida alla configurazione qui sopra e attivala qui sotto.',
    enabled: 'Attivato',
    disabled: 'Disattivato',
    unsavedChanges: 'Modifiche non salvate',
    saving: 'Salvataggio...',
    saveChanges: 'Salva modifiche',
    saved: 'Salvato',
    replaceValue: 'Sostituisci valore attuale',
    openDocs: 'Apri docs',
    clearField: key => `Svuota ${key}`,
    addListEntry: 'Aggiungi un altro',
    removeListEntry: 'Rimuovi',
    listEntryPlaceholder: 'Inserisci un ID',
    enableAria: name => `Attiva ${name}`,
    disableAria: name => `Disattiva ${name}`,
    platformEnabled: name => `${name} attivato`,
    platformDisabled: name => `${name} disattivato`,
    restartToApply: 'Riavvia il gateway perché questa modifica abbia effetto.',
    setupSaved: name => `Configurazione di ${name} salvata`,
    restartToReconnect: 'Riavvia il gateway per riconnetterti con le nuove credenziali.',
    appliedLive: 'Applicato al gateway in esecuzione.',
    connectingLive: 'Il gateway in esecuzione si sta connettendo con le nuove credenziali.',
    keyCleared: key => `${key} svuotato`,
    setupUpdated: name => `La configurazione di ${name} è stata aggiornata.`,
    failedUpdate: name => `Impossibile aggiornare ${name}`,
    failedSave: name => `Impossibile salvare ${name}`,
    failedClear: key => `Impossibile svuotare ${key}`,
    pendingRequests: count => `Richieste in attesa (${count})`,
    pendingAria: count =>
      `${count} richiesta${count === 1 ? '' : 'e'} di accoppiamento pendente${count === 1 ? '' : 'i'}`,
    approvedUsers: count => `Utenti approvati (${count})`,
    approve: 'Approva',
    approving: 'Approvazione...',
    revoke: 'Revoca',
    revoking: 'Revocazione...',
    revokeAria: name => `Revoca ${name}`,
    revokeTitle: 'Revoca accesso',
    revokeDesc: name => `${name} perderà l’accesso e non sarà più riconosciuto al prossimo messaggio.`,
    approvedUser: name => `${name} approvato`,
    approvedHint: 'Verrà riconosciuto automaticamente al prossimo messaggio.',
    revokedUser: name => `${name} revocato`,
    failedApprove: name => `Impossibile approvare ${name}`,
    failedRevoke: name => `Impossibile revocare ${name}`,
    pairingLockedOut: 'Troppi errori di approvazione — questa piattaforma è bloccata. Riprova più tardi.',
    waitingSince: minutes => (minutes < 1 ? 'proprio ora' : `${minutes}m fa`),
    restartNeeded: 'Salvato. Riavvia il gateway di messaggistica perché la nuova configurazione abbia effetto.',
    restartNow: 'Riavvia ora',
    restarting: 'Riavvio…',
    restartFailedManual: 'Hermes non è riuscito a riavviarsi per applicare la tua configurazione di messaggistica',
    restartFailedManualDetail:
      'Premi di nuovo Riavvia; se continua a fallire, apri i log e invia una diagnosi.',
    restartAgain: 'Riavvia di nuovo',
    openLogs: 'Apri i log',
    telegramQr: {
      title: 'Scegli come collegare il tuo bot di Telegram',
      subtitle:
        'Entrambe le opzioni collegano un bot che controlli e salvano le sue credenziali solo in questa installazione di Hermes.',
      quickSetup: 'Configurazione rapida',
      recommended: 'Consigliato',
      quickHelp:
        'Scansiona un codice QR e conferma su Telegram. Hermes crea il bot e rileva automaticamente il tuo ID utente di Telegram.',
      createWithQr: 'Crea con QR',
      starting: 'Avvio…',
      replaceWarning:
        'Ci sono già credenziali di Telegram configurate. Una nuova configurazione tramite QR o un nuovo token del bot sostituirà il bot attuale quando salvi.',
      scanHint: 'Scansionalo con l’app di Telegram sul telefono o apri il link su questo computer.',
      waiting: 'In attesa di Telegram…',
      expiresIn: (remaining: string) => `Scade tra ${remaining}`,
      expired: 'Scaduto',
      openTelegram: 'Apri Telegram',
      ready: 'Bot creato',
      allowedUsers: 'Utenti consentiti',
      ownerDetected: 'Proprietario rilevato',
      addAtLeastOne: 'Aggiungi almeno un ID utente di Telegram.',
      userIdPlaceholder: 'ID utente di Telegram',
      add: 'Aggiungi',
      numericOnly: 'Gli ID utente di Telegram consentiti devono essere numerici.',
      saveAndRestart: 'Salva e riavvia',
      applying: 'Salvataggio…',
      pairingExpired:
        'L’accoppiamento con Telegram è scaduto. Avvia una nuova configurazione tramite QR per riprovare.',
      stillWaiting: (detail: string) => `Stiamo ancora aspettando Telegram. Nuovo tentativo dopo: ${detail}`,
      savedRestarting: 'Telegram salvato; riavvio del gateway…',
      savedRestartFailed: (detail: string) => `Telegram salvato; riavvio del gateway non riuscito${detail}`
    },
    fieldCopy: {
      TELEGRAM_BOT_TOKEN: {
        label: 'Token del bot',
        help: 'Crea un bot con @BotFather e incolla il token che ti fornisce.',
        placeholder: 'Incolla il token del bot di Telegram'
      },
      TELEGRAM_ALLOWED_USERS: {
        label: 'ID utente di Telegram consentiti',
        help: 'Consigliato. ID numerici (uno per casella) da @userinfobot. Senza questo, chiunque può inviare DM al tuo bot.'
      },
      TELEGRAM_PROXY: {
        label: 'URL del proxy',
        help: 'Necessario solo nelle reti in cui Telegram è bloccato.'
      },
      DISCORD_BOT_TOKEN: {
        label: 'Token del bot',
        help: 'Crea un’app su Discord Developer Portal, aggiungi un bot e incolla il suo token.'
      },
      DISCORD_ALLOWED_USERS: {
        label: 'ID utente di Discord consentiti',
        help: 'Consigliato. ID utente di Discord (uno per casella).'
      },
      DISCORD_REPLY_TO_MODE: {
        label: 'Stile di risposta',
        help: '`first`, `all` oppure `off`.'
      },
      DISCORD_ALLOW_ALL_USERS: {
        label: 'Consenti tutti gli utenti di Discord',
        help: 'Solo sviluppo. Se è true, chiunque può inviare DM al bot senza allowlist.'
      },
      DISCORD_HOME_CHANNEL: {
        label: 'ID del canale principale',
        help: 'Canale in cui il bot invia messaggi proattivi (output di cron, promemoria).'
      },
      DISCORD_HOME_CHANNEL_NAME: {
        label: 'Nome del canale principale',
        help: 'Nome visibile del canale principale nei log e nell’output di stato.'
      },
      BLUEBUBBLES_ALLOW_ALL_USERS: {
        label: 'Consenti tutti gli utenti di iMessage',
        help: 'Se è true, ignora la allowlist di BlueBubbles.'
      },
      MATTERMOST_ALLOW_ALL_USERS: {
        label: 'Consenti tutti gli utenti di Mattermost'
      },
      MATTERMOST_HOME_CHANNEL: {
        label: 'Canale principale'
      },
      QQ_ALLOW_ALL_USERS: {
        label: 'Consenti tutti gli utenti di QQ'
      },
      QQBOT_HOME_CHANNEL: {
        label: 'Canale principale di QQ',
        help: 'Canale o gruppo predefinito per la consegna cron.'
      },
      QQBOT_HOME_CHANNEL_NAME: {
        label: 'Nome del canale principale di QQ'
      },
      SLACK_BOT_TOKEN: {
        label: 'Token del bot di Slack',
        help: 'Usa il token del bot di OAuth & Permissions dopo aver installato la tua app di Slack.',
        placeholder: 'Incolla il token del bot di Slack'
      },
      SLACK_APP_TOKEN: {
        label: 'Token app di Slack',
        help: 'Usa il token a livello di app richiesto per Socket Mode.',
        placeholder: 'Incolla il token app di Slack'
      },
      SLACK_ALLOWED_USERS: {
        label: 'ID utente di Slack consentiti',
        help: 'Consigliato. ID di Slack (uno per casella).'
      },
      MATTERMOST_URL: {
        label: 'URL del server',
        placeholder: 'https://mattermost.example.com'
      },
      MATTERMOST_TOKEN: {
        label: 'Token del bot'
      },
      MATTERMOST_ALLOWED_USERS: {
        label: 'ID utente consentiti',
        help: 'Consigliato. ID di Mattermost (uno per casella).'
      },
      MATRIX_HOMESERVER: {
        label: 'URL dell’homeserver',
        placeholder: 'https://matrix.org'
      },
      MATRIX_ACCESS_TOKEN: {
        label: 'Token di accesso'
      },
      MATRIX_USER_ID: {
        label: 'ID utente del bot',
        placeholder: '@hermes:example.org'
      },
      MATRIX_ALLOWED_USERS: {
        label: 'ID utente di Matrix consentiti',
        help: 'Consigliato. ID (uno per casella) nel formato @utente:server.'
      },
      SIGNAL_HTTP_URL: {
        label: 'URL del bridge Signal',
        placeholder: 'http://127.0.0.1:8080',
        help: 'URL di un bridge REST signal-cli in esecuzione.'
      },
      SIGNAL_ACCOUNT: {
        label: 'Numero di telefono',
        help: 'Il numero registrato con il tuo bridge signal-cli.'
      },
      SIGNAL_ALLOWED_USERS: {
        label: 'Utenti di Signal consentiti',
        help: 'Consigliato. Identificativi di Signal (uno per casella).'
      },
      WHATSAPP_ENABLED: {
        label: 'Attiva bridge di WhatsApp',
        help: 'Viene impostato automaticamente con l’interruttore qui sotto. Non toccarlo a meno che tu non sappia di averne bisogno.'
      },
      WHATSAPP_MODE: {
        label: 'Modalità del bridge'
      },
      WHATSAPP_ALLOWED_USERS: {
        label: 'Utenti di WhatsApp consentiti',
        help: 'Consigliato. Numeri di telefono o ID di WhatsApp (uno per casella).'
      }
    },
    platformIntro: {}
  },
  webhooks: {
    search: 'Cerca webhook…',
    loading: 'Caricamento webhook…',
    loadFailed: 'Impossibile caricare i webhook',
    subscriptions: count => `Abbonamenti (${count})`,
    hint: 'Le modifiche agli abbonamenti vengono ricaricate all’istante quando il ricevitore è in esecuzione. Gli abbonamenti disattivati rifiutano gli eventi in arrivo.',
    empty: 'Non ci sono ancora abbonamenti webhook.',
    disabledTitle: 'Ricevitore di webhook disattivato',
    disabledBody:
      'I webhook hanno la loro piattaforma di gateway. Attivali qui per accettare eventi HTTP in arrivo; i canali di chat sono necessari solo quando un abbonamento consegna contenuto a Telegram, Discord, Slack o un altro canale.',
    enable: 'Attiva webhook',
    enabling: 'Attivazione…',
    enabled: name => `Attivato: "${name}"`,
    disabled: name => `Disattivato: "${name}"`,
    enableRow: 'Attiva',
    disableRow: 'Disattiva',
    delete: 'Elimina',
    deleting: 'Eliminazione…',
    deleted: 'Webhook eliminato',
    deleteTitle: 'Elimina webhook',
    deleteDescPrefix: 'Questo eliminerà definitivamente ',
    deleteDescSuffix: '. Non è possibile annullare.',
    deleteFailed: name => `Impossibile eliminare "${name}"`,
    toggleFailed: (name, enabled) => `Impossibile ${enabled ? 'attivare' : 'disattivare'} "${name}"`,
    newSubscription: 'Nuovo abbonamento',
    restarting: 'Riavvio del gateway…',
    restartNeeded:
      'I webhook sono attivati, ma il gateway deve ancora essere riavviato perché il ricevitore possa andare online.',
    restartGateway: 'Riavvia gateway',
    restartingGateway: 'Riavvio…',
    restartFailed: detail => `Riavvio del gateway non riuscito${detail}`,
    enabledRestarting: 'Webhook attivati; riavvio del gateway…',
    all: '(tutti)',
    deliverOnly: 'solo consegna',
    createdTitle: 'Abbonamento creato',
    createdSecretHint: 'Copia il segreto ora; viene mostrato una sola volta.',
    webhookUrl: 'URL del webhook',
    secretOnce: 'Segreto (mostrato una sola volta)',
    done: 'Fatto',
    fieldName: 'Nome',
    fieldNamePlaceholder: 'ad es. github-push',
    fieldDescription: 'Descrizione',
    fieldDescriptionPlaceholder: 'Cosa fa questo webhook (facoltativo)',
    fieldEvents: 'Eventi',
    fieldEventsPlaceholder: 'separati da virgole; vuoto per tutti',
    fieldSkills: 'Skills',
    fieldSkillsPlaceholder: 'nomi delle skill separati da virgole (facoltativo)',
    fieldDeliver: 'Consegna a',
    fieldDeliverOnly: 'Consegna solo il payload',
    fieldPrompt: 'Prompt',
    fieldPromptPlaceholder: 'Istruzioni per l’agente quando questo webhook si attiva (facoltativo)',
    nameRequired: 'Nome obbligatorio',
    create: 'Crea',
    creating: 'Creazione…',
    created: 'Creato',
    createFailed: detail => `Impossibile creare: ${detail}`,
    copy: 'Copia',
    deliverOptions: {
      log: 'Log',
      telegram: 'Telegram',
      discord: 'Discord',
      slack: 'Slack',
      email: 'Email',
      github_comment: 'Commento GitHub'
    }
  },
  profiles: {
    close: 'Chiudi profili',
    nameHint: 'Lettere minuscole, cifre, trattini e trattini bassi. Deve iniziare con una lettera o una cifra.',
    title: 'Profili',
    count: count => `${count} ${count === 1 ? 'profilo' : 'profili'}`,
    search: 'Cerca profili…',
    loading: 'Caricamento profili...',
    newProfile: 'Nuovo profilo',
    importProfile: 'Importa profilo…',
    exportProfile: 'Esporta profilo…',
    imported: 'Profilo importato',
    exported: 'Profilo esportato',
    failedImport: 'Impossibile importare il profilo',
    failedExport: 'Impossibile esportare il profilo',
    allProfiles: 'Tutti i profili',
    showAllProfiles: 'Mostra tutti i profili',
    switchToProfile: name => `Passa a ${name}`,
    switchToConnection: name => `Passa a ${name}`,
    switchConnectionFailed: name => `Impossibile connettersi a ${name}`,
    manageProfiles: 'Gestisci profili...',
    connectGateway: 'Gestisci gateway…',
    fleet: {
      allOnGateway: 'Tutti i profili di questo gateway',
      gateway: (gateway: string) => `Profili su ${gateway}`,
      gatewayUnreachable: (gateway: string) => `${gateway} · irraggiungibile`,
      onGateway: (name: string, gateway: string) => `${name} · ${gateway}`,
      switchTo: (name: string, gateway: string) => `Passa a ${name} su ${gateway}`,
      deleteOn: (gateway: string) => ` su ${gateway}`,
      localDevice: 'Questo dispositivo (backend locale: installa Hermes se manca; altrimenti apri una nuova sessione)',
      switchDeviceTitle: 'Passare a questo dispositivo?',
      switchDeviceDesc:
        'Apre una nuova sessione su questo computer. La conversazione attuale resta sull’altro gateway.',
      switchDeviceConfirm: 'Passa',
      installDeviceTitle: 'Passare a questo dispositivo?',
      installDeviceDesc:
        'Installerà Hermes in locale e poi aprirà una nuova sessione su questo computer. Nulla viene installato finché non confermi.',
      installDeviceConfirm: 'Installa in locale',
      connectExistingInstead: 'Collega invece uno esistente'
    },
    status: {
      unread: (count: number) => (count === 1 ? '1 sessione non letta' : `${count} sessioni non lette`),
      needsInput: (count: number) =>
        count === 1 ? '1 sessione attende la tua risposta' : `${count} sessioni attendono la tua risposta`,
      working: (count: number) => (count === 1 ? '1 sessione in esecuzione' : `${count} sessioni in esecuzione`)
    },
    remoteOverride: {
      menuItem: 'Connetti a un host remoto…',
      badge: (host: string) => `In esecuzione su ${host}`,
      title: (profile: string) => `Connetti ${profile} a un host remoto`,
      description:
        'Le sessioni di questo profilo verranno eseguite sull’Hermes remoto che indichi, invece che su questo computer.',
      urlLabel: 'Indirizzo remoto',
      urlPlaceholder: 'https://hermes.example.com',
      urlInvalid: 'Inserisci un indirizzo completo che inizi con http:// o https://',
      tokenLabel: 'Token di accesso',
      tokenPlaceholder: 'Incolla il token di sessione remota',
      tokenSavedHint: 'C’è già un token salvato. Lascia vuoto per conservarlo.',
      plainTextOptIn:
        'Questo computer non ha un’archiviazione sicura delle chiavi, quindi il token verrebbe salvato senza cifratura sul disco. Salvalo comunque.',
      collisionWarning: (label: string) =>
        `Esiste già un gateway chiamato “${label}” in Impostazioni. Questa connessione del profilo è indipendente e non lo modificherà.`,
      confirmTitle: 'Connettere questo profilo a un host remoto?',
      confirmNote: (profile: string, host: string) =>
        `Le nuove chat di ${profile} verranno eseguite su ${host}. Quel computer eseguirà comandi e leggerà file lì, non su questo. Connettiti solo a un host affidabile.`,
      confirmBack: 'Indietro',
      connect: 'Connetti',
      connecting: 'Connessione…',
      disconnect: 'Rimuovi connessione remota',
      savedTitle: 'Profilo connesso',
      savedMessage: (profile: string, host: string) => `${profile} ora viene eseguito su ${host}`,
      removedTitle: 'Connessione remota rimossa',
      removedMessage: (profile: string) => `${profile} ora viene eseguito su questo computer`,
      removeFailed: 'Impossibile rimuovere la connessione remota',
      authFailedTitle: 'L’host remoto ha rifiutato il token salvato',
      authFailedMessage: (profile: string, host: string) =>
        `${host} ha rifiutato il token salvato per ${profile}. Potrebbe essere stato modificato dal lato remoto.`,
      updateToken: 'Inserisci nuovo token…'
    },
    actions: 'Azioni',
    color: 'Colore...',
    colorFor: 'Colore',
    openInNewWindow: 'Apri in una nuova finestra',
    setAsDefault: 'Imposta come predefinito',
    defaultProfile: 'Profilo predefinito',
    defaultSet: (name: string) => `${name} è ora il predefinito`,
    defaultDescription:
      'Usato all’apertura di Hermes e per le nuove chat. Le sessioni esistenti restano nei loro profili.',
    failedSetDefault: 'Impossibile impostare il profilo predefinito',
    setColor: color => `Imposta colore ${color}`,
    autoColor: 'Auto',
    noProfiles: 'Non ci sono ancora profili.',
    selectPrompt: 'Seleziona un profilo per vederne i dettagli.',
    refresh: 'Aggiorna profili',
    refreshing: 'Aggiornamento profili',
    default: 'predefinito',
    skills: count => `${count} ${count === 1 ? 'skill' : 'skills'}`,
    env: 'env',
    defaultBadge: 'Predefinito',
    rename: 'Rinomina',
    renameMenu: 'Rinomina…',
    exportMenu: 'Esporta…',
    editSoul: 'Modifica SOUL.md…',
    copySetup: 'Copia configurazione',
    copying: 'Copia in corso...',
    modelLabel: 'Modello',
    skillsLabel: 'Skills',
    notSet: 'Non impostato',
    soulDesc: 'Il prompt di sistema e le istruzioni di persona integrate in questo profilo.',
    soulOptional: 'facoltativo',
    soulPlaceholder: mode =>
      `Il prompt di sistema / persona per questo profilo.\nLascia vuoto per conservare il ${mode} predefinito.`,
    soulPlaceholderCloned: 'clonato',
    soulPlaceholderEmpty: 'vuoto',
    unsavedChanges: 'Modifiche non salvate',
    loadingSoul: 'Caricamento di SOUL.md...',
    emptySoul: 'SOUL.md vuoto; inizia a scrivere la persona...',
    saving: 'Salvataggio...',
    saveSoul: 'Salva SOUL.md',
    deleteTitle: 'Eliminare il profilo?',
    deleteDescPrefix: 'Questo eliminerà ',
    deleteDescMid: ' e rimuoverà la sua directory ',
    deleteDescSuffix: '. Questa operazione non si può annullare.',
    deleting: 'Eliminazione...',
    createDesc: 'I profili sono ambienti indipendenti di Hermes: configurazione, skill e SOUL.md separati.',
    nameLabel: 'Nome',
    cloneFrom: 'Clona da',
    cloneFromNone: 'Nessuno (vuoto)',
    cloneFromDesc: 'Copia la configurazione, le skill e SOUL.md del profilo di origine selezionato.',
    cloneFromDefault: 'Clona dal predefinito',
    cloneFromDefaultDesc: 'Copia configurazione, skill e SOUL.md dal tuo profilo predefinito.',
    invalidName: hint => `Nome non valido. ${hint}`,
    nameRequired: 'Il nome è obbligatorio.',
    creating: 'Creazione...',
    createAction: 'Crea profilo',
    renameTitle: 'Rinomina profilo',
    renameDescPrefix: 'Rinominare aggiorna la directory del profilo ed eventuali script wrapper in ',
    renameDescSuffix: '.',
    displayNameTitle: 'Dai un nome a questo agente',
    displayNameDesc:
      'Definisci un nome visibile mostrato in tutta l’app. L’ID interno del profilo resta "default".',
    displayNameLabel: 'Nome visibile',
    newNameLabel: 'Nuovo nome',
    renaming: 'Rinomina...',
    created: 'Profilo creato',
    renamed: 'Profilo rinominato',
    deleted: 'Profilo eliminato',
    setupCopied: 'Comando di configurazione copiato',
    soulSaved: 'SOUL.md salvato',
    failedLoad: 'Impossibile caricare i profili',
    failedDelete: 'Impossibile eliminare il profilo',
    failedCopy: 'Impossibile copiare il comando di configurazione',
    failedLoadSoul: 'Impossibile caricare SOUL.md',
    failedSaveSoul: 'Impossibile salvare SOUL.md',
    failedCreate: 'Impossibile creare il profilo',
    failedRename: 'Impossibile rinominare il profilo'
  },
  modelAssignment: {
    saveFailed: 'Hermes non ha salvato quella modifica al modello.',
    confirmTitle: 'Avviso sulla selezione del modello',
    confirmDetail: 'Conferma solo se accetti questo compromesso.',
    confirmAction: 'Conferma',
    declined: 'Cambio del modello annullato: hai rifiutato l’avviso del livello con addestramento sui dati.'
  },
  cron: {
    close: 'Chiudi cron',
    title: 'Attività pianificate',
    count: count => `${count} ${count === 1 ? 'attività' : 'attività'}`,
    search: 'Cerca attività cron...',
    loading: 'Caricamento attività cron...',
    states: {
      enabled: 'attivata',
      scheduled: 'pianificata',
      running: 'in esecuzione',
      paused: 'in pausa',
      disabled: 'disabilitata',
      error: 'ultima esecuzione fallita',
      completed: 'completata'
    },
    lastRunFailed: 'Ultima esecuzione fallita:',
    editJob: 'Modifica attività',
    runAgain: 'Esegui di nuovo',
    deliveryLabels: {
      local: 'Questo desktop',
      telegram: 'Telegram',
      discord: 'Discord',
      slack: 'Slack',
      email: 'Email'
    },
    scheduleLabels: {
      daily: 'Giornaliera',
      weekdays: 'Giorni feriali',
      weekly: 'Settimanale',
      monthly: 'Mensile',
      hourly: 'Ogni ora',
      'every-15-minutes': 'Ogni 15 minuti',
      custom: 'Personalizzata'
    },
    scheduleHints: {
      daily: 'Tutti i giorni alle 9:00',
      weekdays: 'Da lunedì a venerdì alle 9:00',
      weekly: 'Ogni lunedì alle 9:00',
      monthly: 'Il primo giorno di ogni mese alle 9:00',
      hourly: 'All’inizio di ogni ora',
      'every-15-minutes': 'Ogni 15 minuti',
      custom: 'Sintassi cron o linguaggio naturale'
    },
    days: {
      '0': 'Domenica',
      '1': 'Lunedì',
      '2': 'Martedì',
      '3': 'Mercoledì',
      '4': 'Giovedì',
      '5': 'Venerdì',
      '6': 'Sabato',
      '7': 'Domenica'
    },
    dayFallback: value => `giorno ${value}`,
    everyDayAt: time => `Tutti i giorni alle ${time}`,
    weekdaysAt: time => `Giorni feriali alle ${time}`,
    everyDayOfWeekAt: (day, time) => `Ogni ${day} alle ${time}`,
    monthlyOnDayAt: (dayOfMonth, time) => `Mensile il giorno ${dayOfMonth} alle ${time}`,
    topOfHour: 'All’inizio di ogni ora',
    everyHourAt: minute => `Ogni ora a :${minute}`,
    newCron: 'Nuova attività cron',
    emptyDescNew:
      'Pianifica un prompt da eseguire con un’espressione cron. Hermes lo eseguirà e consegnerà i risultati alla destinazione che scegli.',
    emptyDescSearch: 'Prova una ricerca più ampia.',
    emptyTitleNew: 'Ancora nessuna attività pianificata',
    emptyTitleSearch: 'Nessuna corrispondenza',
    last: 'Ultima:',
    next: 'Prossima:',
    overdueSince: 'In ritardo dal:',
    noRuns: 'Ancora nessuna esecuzione',
    manage: 'Gestisci',
    showRuns: 'Mostra esecuzioni',
    hideRuns: 'Nascondi esecuzioni',
    runHistory: 'Cronologia esecuzioni',
    actionsTitle: 'Azioni attività cron',
    resume: 'Riprendi cron',
    pause: 'Metti in pausa cron',
    resumeTitle: 'Riprendi',
    pauseTitle: 'Metti in pausa',
    triggerNow: 'Esegui ora',
    edit: 'Modifica cron',
    deleteTitle: 'Eliminare attività cron?',
    deleteDescPrefix: 'Questo eliminerà ',
    deleteDescSuffix: ' in modo permanente. Smetterà di essere eseguita immediatamente.',
    deleting: 'Eliminazione...',
    resumed: 'Cron ripreso',
    paused: 'Cron in pausa',
    triggered: 'Cron eseguito',
    deleted: 'Cron eliminato',
    created: 'Cron creato',
    updated: 'Cron aggiornato',
    failedLoad: 'Impossibile caricare le attività cron',
    failedUpdate: 'Impossibile aggiornare l’attività cron',
    failedTrigger: 'Impossibile eseguire l’attività cron',
    failedDelete: 'Impossibile eliminare l’attività cron',
    failedSave: 'Impossibile salvare l’attività cron',
    editTitle: 'Modifica attività cron',
    createTitle: 'Nuova attività cron',
    editDesc: 'Aggiorna la pianificazione, il prompt o la destinazione. Le modifiche si applicano alla prossima esecuzione.',
    createDesc:
      'Pianifica un prompt da eseguire automaticamente. Usa la sintassi cron o una frase come "ogni 15 minuti".',
    nameLabel: 'Nome',
    namePlaceholder: 'Riepilogo mattutino',
    promptLabel: 'Prompt',
    scriptLabel: 'Script',
    scriptBadge: 'script',
    promptPlaceholder: 'Riassumi i miei thread di Slack non letti e inviami per email i 5 principali...',
    frequencyLabel: 'Frequenza',
    deliverLabel: 'Consegna a',
    deliverNeedsHomeChannel: 'configura prima un canale principale',
    modelLabel: 'Modello',
    modelDefault: 'Predefinito (modello globale)',
    customScheduleLabel: 'Pianificazione personalizzata',
    customPlaceholder: '0 9 * * * o giorni feriali alle 9',
    customHint: 'Espressione cron, o frasi come "ogni ora" o "giorni feriali alle 9".',
    optional: 'Facoltativo',
    promptRequired: 'Il prompt è obbligatorio.',
    promptScheduleRequired: 'Il prompt e la pianificazione sono obbligatori.',
    scheduleRequired: 'La pianificazione è obbligatoria.',
    scriptOnlyEditHint: 'Attività solo con script (senza prompt IA). ID dell’attività:',
    saveChanges: 'Salva modifiche',
    createAction: 'Crea cron',
    tabs: {
      jobs: 'Attività',
      blueprints: 'Modelli'
    },
    blueprints: {
      tab: 'Modelli',
      startFrom: 'Inizia con',
      custom: 'Personalizzata',
      subtitle: 'Automazioni pronte all’uso',
      dialogDesc: 'Completa i dettagli e pianificala.',
      scheduleIt: 'Pianifica',
      scheduling: 'Pianificazione…',
      scheduled: 'Modello pianificato',
      loading: 'Caricamento modelli…',
      failedLoad: 'Impossibile caricare i modelli',
      emptyTitle: 'Nessun modello disponibile',
      emptyDesc: 'Questo backend non offre modelli di automazione.'
    }
  },
  artifacts: {
    search: 'Cerca artefatti...',
    refresh: 'Aggiorna artefatti',
    refreshing: 'Aggiornamento artefatti',
    indexing: 'Indicizzazione artefatti recenti delle sessioni',
    tabAll: 'Tutto',
    tabImages: 'Immagini',
    tabFiles: 'File',
    tabLinks: 'Link',
    noArtifactsTitle: 'Nessun artefatto trovato',
    noArtifactsDesc:
      'Le immagini generate e gli output di file appariranno qui quando le sessioni li producono.',
    failedLoad: 'Impossibile caricare gli artefatti',
    openFailed: 'Impossibile aprire',
    itemsImage: 'immagini',
    itemsLink: 'link',
    itemsFile: 'file',
    itemsGeneric: 'elementi',
    zero: '0',
    rangeOf: (start, end, total) => `${start}-${end} di ${total}`,
    goToPage: (itemLabel, page) => `Vai alla pagina ${page} di ${itemLabel}`,
    colTitleLink: 'Titolo del link',
    colTitleFile: 'Nome',
    colTitleDefault: 'Titolo / nome',
    colLocationLink: 'URL',
    colLocationFile: 'Percorso',
    colLocationDefault: 'Posizione',
    colSession: 'Sessione',
    kindImage: 'immagine',
    kindFile: 'file',
    kindLink: 'link',
    chat: 'Chat',
    copyUrl: 'Copia URL',
    copyPath: 'Copia percorso'
  },
  artifactCard: {
    kind: {
      code: 'Codice',
      html: 'Pagina interattiva',
      svg: 'Grafico'
    },
    generating: lines => `Generazione… ${lines} righe`,
    versionBadge: count => `${count} versioni`,
    open: 'Apri'
  },
  artifactPreview: {
    versionOf: (current, total) => `v${current} di ${total}`,
    olderVersion: 'Versione precedente',
    newerVersion: 'Versione più recente',
    latest: 'Ultima',
    copyContent: 'Copia contenuto',
    download: 'Scarica',
    openInBrowser: 'Apri nel browser',
    openInBrowserFailed: 'Impossibile aprire nel browser',
    missingTitle: 'Artefatto non disponibile',
    missingBody: 'Questo artefatto non è più nel log locale.'
  },
  sidebar: {
    filter: {
      grouping: 'Raggruppamento',
      ordering: 'Ordinamento',
      show: 'Mostra',
      filters: 'Filtri',
      status: 'Stato',
      pullRequest: 'Pull request',
      profile: 'Profilo',
      project: 'Progetto',
      archived: 'Archiviate',
      resetToDefaults: 'Ripristina valori predefiniti',
      expandAll: 'Espandi tutto',
      collapseAll: 'Comprimi tutto',
      inboxStyle: 'Stile posta in arrivo',
      updated: 'Aggiornato',
      created: 'Creato',
      tokens: 'Token',
      cost: 'Costo',
      manual: 'Manuale',
      preview: 'Anteprima',
      pr: 'PR',
      needsInput: 'Richiede risposta',
      working: 'Al lavoro',
      unread: 'Non letti',
      draft: 'Bozza',
      idle: 'Inattivo',
      open: 'Aperta',
      merged: 'Unita',
      closed: 'Chiusa',
      noPR: 'Senza PR'
    },
    gatewayGroups: {
      grouping: 'Gateway e profilo',
      rename: 'Rinomina gruppo',
      aliasLabel: 'Nome visibile',
      aliasHint: 'Solo il nome visibile; i nomi del gateway e del profilo non cambiano.',
      resetName: 'Ripristina nome',
      moveUp: 'Sposta su',
      moveDown: 'Sposta giù',
      reorder: 'Riordina gruppo',
      actions: 'Azioni del gruppo'
    },
    profileRail: 'Barra dei profili',
    nav: {
      'new-session': 'Nuova sessione',
      capabilities: 'Funzionalità',
      messaging: 'Messaggistica',
      artifacts: 'Artefatti',
      cron: 'Attività pianificate'
    },
    searchAria: 'Cerca sessioni',
    searchPlaceholder: 'Cerca sessioni…',
    clearSearch: 'Cancella ricerca',
    noMatch: query => `Nessuna sessione corrisponde a “${query}”.`,
    results: 'Risultati',
    pinned: 'Fissate',
    sessions: 'Sessioni',
    terminal: 'Terminale',
    files: 'File',
    review: 'Revisione',
    logs: 'Log',
    cronJobs: 'Attività cron',
    groupAriaGrouped: 'Mostra le sessioni come un’unica lista',
    groupAriaUngrouped: 'Raggruppa le sessioni per workspace',
    showProjects: 'Mostra progetti',
    showSessions: 'Mostra sessioni',
    groupTitleGrouped: 'Annulla raggruppamento sessioni',
    groupTitleUngrouped: 'Raggruppa per workspace',
    allPinned: 'Tutto qui è fissato. Sfissa una chat per mostrarla nei recenti.',
    shiftClickHint: 'Shift-clic su una chat per fissarla',
    noWorkspace: 'Senza workspace',
    projectEmpty: 'Ancora nessuna sessione',
    projectLoadFailed: 'Impossibile caricare le sessioni',
    noSessions: 'Ancora nessuna sessione',
    storageCorrupt: {
      title: 'Il database delle sessioni è danneggiato',
      body: (profiles: string) =>
        `Hermes non riesce a leggere tutta la cronologia delle sessioni di ${profiles}. Le chat mancanti in questo elenco non sono state eliminate; il file in cui sono salvate è danneggiato.`,
      action:
        'Esci da Hermes in questo profilo e poi esamina il file senza modificarlo, oppure ripristina uno snapshot:',
      guide: 'Guida al ripristino'
    },
    noFilterMatches: 'Nessuna sessione corrisponde a questi filtri',
    projects: {
      showAllSessions: 'Mostra tutte le sessioni',
      sectionLabel: 'Progetti',
      home: 'Home',
      autoDiscovered: 'Rilevato automaticamente',
      newButton: 'Nuovo progetto',
      createTitle: 'Nuovo progetto',
      createDesc: 'Assegna un nome a un workspace e aggiungi una o più cartelle.',
      renameTitle: 'Rinomina progetto',
      addFolderTitle: 'Aggiungi cartella',
      namePlaceholder: 'es. Skunkworks',
      foldersLabel: 'Cartelle',
      ideaLabel: 'Idea',
      ideaPlaceholder: 'Di cosa si occupa questo progetto? (salvato in IDEA.md)',
      ideaGenerate: 'Genera idea',
      ideaGenerating: 'Generazione…',
      ideaShuffle: 'Modelli casuali',
      noFolders: 'Ancora nessuna cartella aggiunta.',
      addFolder: 'Aggiungi cartella',
      primaryBadge: 'principale',
      removeFolder: 'Elimina',
      create: 'Crea',
      menu: 'Azioni',
      menuRename: 'Rinomina',
      menuAppearance: 'Aspetto',
      noColor: 'Nessun colore',
      menuAddFolder: 'Aggiungi cartella',
      menuSetActive: 'Imposta come attivo',
      menuDelete: 'Elimina',
      moveToProject: 'Sposta in progetto',
      movedTo: name => `Spostato in ${name}`,
      moveFailed: 'Impossibile spostare la sessione',
      moveNoFolder: 'Quel progetto non ha una cartella in cui spostarla',
      moveNoProjects: 'Non ci sono altri progetti',
      reveal: 'Mostra nella cartella',
      copyPath: 'Copia percorso',
      removeFromSidebar: 'Nascondi dalla barra laterale',
      createdInPreviousContext:
        'Il progetto è stato creato nella connessione o nel profilo precedente. Torna lì; IDEA.md non è stato scritto.',
      createFailed: 'Impossibile creare il progetto',
      staleBackend:
        'Aggiorna il backend di Hermes per creare progetti: il tuo backend è più vecchio di questa applicazione desktop (Impostazioni → Aggiornamenti → Backend).',
      deleteConfirm:
        'Questo elimina il progetto salvato in Hermes. I file, i repository git e i worktree restano intatti.',
      startWork: 'Nuovo worktree',
      newWorktreeTitle: 'Nuovo worktree',
      newWorktreeDesc: 'Assegna un nome al branch di questo worktree.',
      branchPlaceholder: 'es. la-mia-feature',
      branchOff: () => ({ after: '', before: 'crea branch da ' }),
      baseBranchPlaceholder: 'Cerca branch…',
      baseBranchNone: 'Nessun branch trovato',
      startWorkFailed: 'Impossibile creare il worktree',
      worktreeStaleBackend:
        'Aggiorna il backend di Hermes per creare worktrees tramite questa connessione remota: è precedente all’API git worktree.',
      worktreeProjectLabel: 'Progetto',
      worktreeProjectPlaceholder: 'Cerca progetti…',
      worktreeProjectNone: 'Nessun progetto con cartella',
      convertBranch: 'Converti un branch…',
      convertBranchTitle: 'Converti un branch',
      convertBranchDesc: 'Apri branch già attivi o crea un worktree per un branch disponibile.',
      convertBranchPlaceholder: 'Cerca branch…',
      convertBranchInstead: 'Converti un branch esistente',
      branchOpenExisting: 'apri',
      branchSwitchHome: 'passa al principale',
      branchCreateWorktree: 'nuovo worktree',
      branchTrackRemote: 'segui remota',
      branchesLoading: 'Caricamento branch…',
      noBranches: 'Nessun branch trovato',
      removeWorktree: 'Elimina worktree',
      removeWorktreeFailed: 'Impossibile eliminare il worktree (ci sono modifiche senza commit?)',
      removeWorktreeConfirm:
        'Eliminalo da Git (la cartella del worktree viene eliminata; il branch viene conservato) o semplicemente nascondilo dalla barra laterale e lascia il worktree su disco.',
      removeWorktreeDirty:
        'Questo worktree ha modifiche senza commit. Forza l’eliminazione (le modifiche verranno scartate) o semplicemente nascondilo e conservalo su disco.',
      forceRemove: 'Forza eliminazione',
      enter: label => `Apri ${label}`,
      reorder: label => `Riordina ${label}`,
      toggle: (label, open) => `${open ? 'Mostra' : 'Nascondi'} le sessioni di ${label}`,
      showAllCount: (count: number) => `Mostra tutte le ${count} sessioni`,
      back: 'Tutti i progetti'
    },
    newSessionIn: label => `Nuova sessione in ${label}`,
    showMoreIn: (count, label) => `Mostra altri ${count} in ${label}`,
    loading: 'Caricamento…',
    loadMore: 'Carica altri',
    loadCount: step => `Carica altri ${step}`,
    messageCount: (count: number) => `${count} ${count === 1 ? 'messaggio' : 'messaggi'}`,
    toolCallCount: (count: number) => `${count} ${count === 1 ? 'chiamata a strumento' : 'chiamate a strumenti'}`,
    row: {
      pin: 'Fissa',
      unpin: 'Sfissa',
      markUnread: 'Segna come non letto',
      markRead: 'Segna come letto',
      unreadFailed: 'Impossibile aggiornare lo stato non letto',
      copyId: 'Copia ID',
      export: 'Esporta',
      branchFrom: 'Branch',
      rename: 'Rinomina',
      archive: 'Archivia',
      unarchive: 'Annulla archiviazione',
      newWindow: 'Nuova finestra',
      openInTerminal: 'Apri nel terminale',
      hideTabBar: 'Nascondi barra delle schede',
      openInNewTab: 'Apri in una nuova scheda',
      openInSplit: 'Apri in vista divisa',
      copyIdFailed: 'Impossibile copiare l’ID di sessione',
      sessionActions: 'Azioni sessione',
      sessionRunning: 'Sessione in esecuzione',
      needsInput: 'Richiede la tua risposta',
      waitingForAnswer: 'In attesa della tua risposta',
      finishedUnread: 'Terminata — da leggere',
      backgroundRunning: 'Attività in background in esecuzione',
      draftSession: 'Bozza — non è ancora stato inviato nulla',
      handoffOrigin: platform => `Transferito da ${platform}`,
      continuationOrigin: 'Continuazione automatica: questa conversazione è stata compressa e proseguita',
      ownedByProfile: profile => `Profilo: ${profile}`,
      renamed: 'Rinominata',
      renameFailed: 'Impossibile rinominare',
      renameTitle: 'Rinomina sessione',
      renameDesc: 'Lascialo vuoto per cancellarlo.',
      untitledPlaceholder: 'Sessione senza titolo',
      deleteTitle: 'Eliminare la sessione?',
      deleteDesc: (title: string) => `“${title}” verrà eliminato in modo permanente. Impossibile annullare.`,
      deleting: 'Eliminazione…',
      deleted: 'Sessione eliminata',
      untitledChat: id => `Chat ${id}`,
      messageCount: count => `${count} ${count === 1 ? 'messaggio' : 'messaggi'}`,
      todoProgress: 'Attività completate',
      ageNow: 'adesso',
      ageDay: 'g',
      ageHour: 'h',
      ageMin: 'm'
    },
    dateDivider: {
      today: 'Oggi, prima',
      yesterday: 'Ieri',
      thisWeek: 'Questa settimana',
      lastWeek: 'La settimana scorsa',
      thisMonth: 'Questo mese'
    },
    statusDivider: {
      working: 'In corso',
      done: 'Completato'
    },
    markAllRead: 'Segna tutto come letto'
  },
  composer: {
    message: 'Messaggio',
    wakingProfile: profile => `Risveglio di ${profile}…`,
    placeholderStarting: 'Avvio di Hermes...',
    placeholderReconnecting: 'Riconnessione a Hermes…',
    placeholderFollowUp: 'Inviare un follow-up',
    newSessionPlaceholders: [
      'Cosa costruiamo?',
      'Dai un’attività a Hermes',
      'Cosa hai in mente?',
      'Descrivi ciò di cui hai bisogno',
      'Cosa affrontiamo?',
      'Chiedi quello che vuoi',
      'Inizia con un obiettivo'
    ],
    followUpPlaceholders: [
      'Inviare un follow-up',
      'Aggiungere altro contesto',
      'Affinare la richiesta',
      'Cosa viene dopo?',
      'Continuiamo',
      'Portiamolo oltre',
      'Modificare o continuare'
    ],
    startVoice: 'Avvia la conversazione vocale',
    openDirective: 'Apri',
    queueMessage: 'Metti in coda il messaggio',
    steer: 'Guida l’esecuzione corrente',
    stop: 'Ferma',
    send: 'Inviare',
    speaking: 'Parlando',
    transcribing: 'Trascrivendo',
    thinking: 'Pensando',
    muted: 'Silenziato',
    listening: 'In ascolto',
    muteMic: 'Silenzia il microfono',
    unmuteMic: 'Attiva il microfono',
    stopListening: 'Smetti di ascoltare e invia',
    stopShort: 'Ferma',
    endConversation: 'Termina la conversazione vocale',
    endShort: 'Termina',
    stopDictation: 'Ferma il dettato',
    transcribingDictation: 'Trascrivendo il dettato',
    voiceControls: 'Voce',
    voiceEngine: 'Motore della chat vocale',
    voiceEngineChained: 'Voce in testo + voce di Hermes',
    voiceEngineLive: 'GPT-Live (full-duplex, delega a Hermes)',
    voiceEngineLiveNeedsKey: 'Richiede una chiave API di OpenAI',
    voiceEngineChangeFailed: 'Non è stato possibile cambiare il motore della chat vocale',
    voiceEngineChainedShort: 'voce in testo',
    voiceEngineLiveShort: 'GPT-Live',
    voiceDictation: 'Dettato vocale',
    speakReplies: 'Leggi le risposte ad alta voce',
    stopSpeakingReplies: 'Smetti di leggere le risposte ad alta voce',
    wakeWord: (phrase: string) => `Parola di attivazione "${phrase}"`,
    wakeWordListening: phrase => `Parola di attivazione: "${phrase}" — in ascolto`,
    wakeWordOff: phrase => `Parola di attivazione: "${phrase}" — disattivata`,
    wakeWordPausedVoice: phrase => `Parola di attivazione: "${phrase}" — in pausa durante la chat vocale`,
    lookupLoading: 'Ricerca…',
    lookupNoMatches: 'Nessuna corrispondenza.',
    lookupTry: 'Prova',
    lookupOr: 'o',
    commonCommands: 'Comandi comuni',
    hotkeys: 'Scorciatoie',
    helpFooter: 'apre il pannello completo · backspace scarta',
    commandDescs: {
      '/help': 'Mostra i comandi slash del desktop',
      '/clear': 'avvia una nuova sessione',
      '/resume': 'Riprendi una sessione salvata',
      '/details': 'controlla il livello di dettaglio della trascrizione',
      '/copy': 'copia la selezione o l’ultimo messaggio dell’assistente',
      '/quit': 'esci da hermes',
      '/start': 'Conferma i ping di avvio della piattaforma senza rispondere',
      '/new': 'Avvia una nuova chat desktop',
      '/topic': 'Attiva o ispeziona le sessioni per argomento degli MD di Telegram',
      '/save': 'Salva la trascrizione corrente in JSON',
      '/retry': 'Riprova l’ultimo messaggio (reinvialo all’agente)',
      '/prompt': 'Scrivi il tuo prossimo prompt in $EDITOR (markdown) e invialo',
      '/undo': 'Torna indietro di N turni dell’utente e chiedi di nuovo (1 per impostazione predefinita)',
      '/title': 'Rinomina la sessione corrente',
      '/handoff': 'Passa questa sessione a una piattaforma di messaggistica',
      '/branch': 'Ramifica l’ultimo messaggio in una nuova chat',
      '/worktree': 'Mostra, elenca, crea o rimuovi worktree git isolati',
      '/compress': 'Comprimi il contesto di questa conversazione',
      '/rollback':
        'Elenca o ripristina i checkpoint del file system (i ripristini conservano le tue modifiche manuali; --all le sovrascrive)',
      '/export': 'Esporta un profilo (configurazione, skill, tema) in un file condivisibile',
      '/import': 'Importa un file di profilo condiviso come nuovo profilo',
      '/stop': 'Ferma il turno attivo e i processi in background',
      '/pause': "Metti in pausa a livello globale il nuovo lavoro (arresto di emergenza); '/pause off' lo riattiva",
      '/bg': 'Esegui un prompt in una sessione indipendente in background',
      '/btw': 'Fai una domanda a margine su questa conversazione senza interromperla',
      '/agents': 'Mostra gli agenti attivi e le attività in corso',
      '/journey': 'Apri il grafo della memoria: skill e ricordi nel corso del tempo',
      '/queue':
        'Metti in coda un prompt per il prossimo turno, oppure elenca/modifica/rimuovi/sposta/svuota i prompt in coda',
      '/steer': 'Inserisci un messaggio dopo la prossima chiamata allo strumento senza interrompere',
      '/goal': 'Fissa un obiettivo permanente su cui Hermes lavora per diversi turni finché non lo completa',
      '/heartbeat': 'Configura un prompt ricorrente che torna su questa sessione quando è inattiva',
      '/refine': 'Rivedi questa conversazione ora e salva ciò che hai imparato in memoria/skill',
      '/review':
        'Avvia un subagente indipendente per rivedere il lavoro appena discusso (PR, codice, docs)',
      '/loop': 'Esegui di nuovo un prompt a intervalli regolari in questa sessione',
      '/plan': 'Scrivi un piano di implementazione in markdown in .hermes/plans/ senza eseguire nulla',
      '/moa': 'Esegui un prompt con il preset predefinito di Mixture of Agents e poi ripristina il tuo modello',
      '/subgoal': 'Aggiungi o gestisci criteri aggiuntivi dell’obiettivo attivo',
      '/status': 'Mostra lo stato della sessione corrente',
      '/egress': 'Mostra lo stato del proxy di uscita di Docker',
      '/context':
        'Mostra la vista dettagliata della finestra di contesto con indicatore di utilizzo, suddivisione per categoria, statistiche di compressione e prestazioni',
      '/whoami': 'Mostra il tuo accesso ai comandi slash (admin / utente)',
      '/profile': 'Cambia il profilo attivo di Hermes',
      '/codex-runtime': 'Attiva o disattiva il runtime codex app-server per i modelli OpenAI/Codex',
      '/personality': 'Imposta una personalità predefinita',
      '/battery': 'Mostra o nascondi un indicatore di batteria a colori nella barra di stato',
      '/timestamps': 'Mostra o nascondi le marche temporali [HH:MM] nei messaggi e in /history',
      '/diff': 'Mostra le modifiche git nella directory di lavoro',
      '/focus': 'Attiva o disattiva la vista focus: solo il tuo prompt e la risposta finale',
      '/yolo': 'Attiva o disattiva YOLO: approva automaticamente i comandi pericolosi',
      '/approvals': 'Mostra o imposta la modalità persistente di approvazione dei comandi pericolosi',
      '/reasoning': 'Sforzo o visualizzazione del ragionamento [<level> [--global]|show|hide|full|clamp]',
      '/skin': 'Cambia il tema del desktop o passa al successivo',
      '/wake': 'Controlla l’ascolto della parola di attivazione del desktop [on|off|status]',
      '/tools': 'Gestisci gli strumenti: /tools [list|disable|enable] [name...]',
      '/memory': 'Rivedi le scritture di memoria in sospeso / attiva o disattiva l’approvazione',
      '/bundles': 'Elenca i pacchetti di skill (alias /<name> per più skill)',
      '/pet': 'Mostra, nascondi o adotta una mascotte di petdex (/pet, /pet list, /pet boba)',
      '/hatch': 'Genera una nuova mascotte (apre il generatore)',
      '/learn': 'Impara una skill riutilizzabile da ciò che descrivi (cartelle, URL, questa chat, note)',
      '/init': 'Genera o aggiorna le istruzioni di progetto AGENTS.md a partire da un’analisi del repository',
      '/suggestions': 'Rivedi le automazioni suggerite (accetta/scarta)',
      '/blueprint': 'Configura un’automazione a partire da un template',
      '/browser': 'Gestisci il browser dell’agente [connect|disconnect|status|use]',
      '/palette': 'Apri la palette dei comandi fuzzy (anche Ctrl+P)',
      '/usage':
        'Mostra l’uso dei token e i limiti di frequenza; `reset` riscatta un ripristino del limite di Codex accumulato',
      '/subscription': 'Vedi il tuo piano Nous e cambialo nel browser',
      '/topup': 'Mostra il tuo saldo Nous e gestisci la fatturazione nel portale',
      '/platform': 'Metti in pausa, riprendi o elenca una piattaforma del gateway in errore',
      '/version': 'Mostra la versione di Hermes Agent',
      '/debug': 'Carica un report di debug (informazioni di sistema + log) e ottieni link da condividere',
      '/model': 'Cambia il modello di questa sessione'
    },
    hotkeyDescs: {
      'composer.mention': 'menziona file, cartelle, URL e Git',
      'composer.slash': 'palette dei comandi con /',
      'composer.help': 'questo aiuto rapido (elimina per scartarlo)',
      'composer.sendNewline': 'invia · Shift+Enter per inserire una nuova riga',
      'composer.sendQueued': 'invia il prossimo turno in coda',
      'keybinds.openPanel': 'tutte le scorciatoie da tastiera',
      'composer.cancel': 'chiudi il menu a comparsa · annulla l’esecuzione',
      'composer.history': 'naviga il menu a comparsa o la cronologia'
    },
    attachUrlTitle: 'Allega una URL',
    attachUrlDesc: 'Hermes scaricherà la pagina e la includerà come contesto per questo turno.',
    urlPlaceholder: 'https://example.com/post',
    urlHintPre: 'Includi l’URL completa, ad es. ',
    attach: 'Allega',
    queued: count => `${count} in coda`,
    queuedPaused: count => `${count} in coda — in pausa`,
    attachmentOnly: 'Turno con solo allegato',
    emptyTurn: 'Turno vuoto',
    hiddenQueued: 'Nota di configurazione',
    attachments: count => `${count} ${count === 1 ? 'allegato' : 'allegati'}`,
    editingInComposer: 'Modifica nella casella di composizione',
    editingQueuedInComposer: 'Modifica del turno in coda nella casella di composizione',
    restoredDraftNotice: 'Il tuo messaggio non inviato è stato ripristinato',
    restoredDraftUndo: 'Annulla',
    queueEdit: 'Modifica',
    queueExpand: 'Espandi',
    queueCollapse: 'Riduci',
    queueSendNext: 'Prossimo',
    queueSteer: 'Reindirizza — incanala subito il turno in corso',
    queueSend: 'Inviare',
    queueDelete: 'Elimina',
    queueResume: 'Riprendi',
    queueResumeTip: 'La coda è stata messa in pausa all’arresto; riprendi l’invio dei turni in coda',
    queueStuckTitle: 'Messaggio in coda non inviato',
    queueStuckBody: 'Un turno in coda non è stato inviato. È ancora in coda; riprova.',
    queueDroppedTitle: 'Voce in coda scartata',
    queueDroppedBody:
      'Questa voce è stata scartata in background perché la sua sessione non è stata ripristinata dopo diversi tentativi. Il resto della coda non è stato influenzato.',
    previewUnavailable: 'Anteprima non disponibile',
    previewLabel: label => `Anteprima di ${label}`,
    couldNotPreview: label => `Impossibile mostrare l’anteprima di ${label}`,
    removeAttachment: label => `Rimuovi ${label}`,
    dictating: 'Dettatura in corso',
    preparingAudio: 'Preparazione audio',
    speakingResponse: 'Risposta vocale',
    readingAloud: 'Lettura ad alta voce',
    themeSuggestions: 'Suggerimenti per il tema del desktop',
    noMatchingThemes: 'Nessun tema corrispondente.',
    themeTryPre: 'Prova ',
    themeTryPost: '.',
    attachLabel: 'Allega',
    files: 'File…',
    folder: 'Cartella…',
    images: 'Immagini…',
    pasteImage: 'Incolla immagine',
    url: 'URL…',
    promptSnippets: 'Frammenti di prompt…',
    tipPre: 'Suggerimento: scrivi ',
    tipPost: ' per menzionare i file inline.',
    snippetsTitle: 'Frammenti di prompt',
    snippetsDesc: 'Scegli un prompt iniziale da inserire nella casella di composizione.',
    dropFiles: 'Rilascia i file per allegarli',
    dropSession: 'Rilascia per collegare questa chat',
    mcpSuggestions: {
      label: server => `Aggiungi ${server}`,
      tip: keyword => `Suggerito perché hai menzionato "${keyword}" — clicca per connetterti`,
      connecting: server => `Connessione a ${server}…`,
      cancelTip: 'Clicca per annullare',
      added: server => `${server} aggiunto`,
      addedTip: 'Connesso — i suoi strumenti sono pronti in questa chat',
      connectFailed: server => `Impossibile connettersi a ${server}`
    },
    skillSuggestions: {
      label: skill => `Usa la skill: ${skill}`,
      tip: skill => `Hai menzionato "${skill}" — clicca per iniziare con quella skill`,
      done: skill => `Aggiunta /${skill}`,
      doneTip: 'La skill viene caricata quando invii'
    },
    githubSuggestions: {
      label: 'Configura GitHub',
      tip: 'GitHub funziona qui tramite le skill della CLI gh: clicca per collegare il tuo account',
      done: 'È stato aggiunto /github-auth',
      doneTip: 'Invia il messaggio e l’agente ti guiderà per accedere a GitHub'
    },
    repairSuggestions: {
      label: server => `Riconnetti ${server}`,
      tip: server => `Una chiamata a ${server} è appena fallita con un errore di connessione`,
      working: server => `Riconnessione di ${server}…`,
      workingTip: 'Clicca per annullare',
      done: server => `${server} riconnesso`,
      doneTip: 'Le nuove credenziali sono attive in questa chat',
      failed: server => `Impossibile riconnettere ${server}`
    },
    cronSuggestions: {
      label: 'Programma questo',
      tip: phrase => `"${phrase}" sembra ricorrente — eseguilo invece con una pianificazione`,
      prefix: 'Configuralo come attività pianificata:',
      done: 'Contrassegnato per la pianificazione',
      doneTip: 'Invialo e l’agente creerà l’attività'
    },
    snippets: {
      codeReview: {
        label: 'Revisione del codice',
        description: 'Analizza la modifica corrente alla ricerca di regressioni, casi limite tralasciati e test mancanti.',
        text: 'Rivedi questo per individuare bug, regressioni e test mancanti.'
      },
      implementationPlan: {
        label: 'Piano di implementazione',
        description: 'Delinea un approccio prima di toccare il codice per mantenere il diff focalizzato.',
        text: 'Crea un piano di implementazione conciso prima di modificare il codice.'
      },
      explainThis: {
        label: 'Spiega questo',
        description: 'Illustra come funziona il codice selezionato e collega i file chiave.',
        text: 'Spiega come funziona questo e indica i file chiave.'
      }
    }
  },
  statusStack: {
    hideStack: 'Nascondi lo stack di stato',
    showStack: 'Mostra lo stack di stato',
    agents: 'Agenti',
    background: count => `${count} in background`,
    goalActive: 'Obiettivo attivo',
    goalBlocked: 'Obiettivo bloccato',
    goalDone: 'Obiettivo completato',
    goalPaused: 'Obiettivo in pausa',
    goalWaiting: 'Obiettivo in attesa',
    subagents: count => `${count} subagent${count === 1 ? 'e' : 'i'}`,
    todos: (done, total) => `Attività ${done}/${total}`,
    previousTodos: (done, total) => `Attività precedenti ${done}/${total}`,
    running: 'In esecuzione',
    stop: 'Ferma',
    dismiss: 'Scarta',
    exit: code => `uscita ${code}`,
    control: {
      goalActiveTurns: (turn: number, maxTurns: number) => `Turno ${turn}/${maxTurns}`,
      goalDoneTurns: (turns: number) => `${turns} ${turns === 1 ? 'turno' : 'turni'}`,
      goalTurn: (turn: number) => `Turno ${turn}`,
      goalActions: 'Azioni dell’obiettivo',
      viewDetails: 'Vedi dettagli',
      addCriterion: 'Aggiungi criterio',
      addCriterionDialogTitle: 'Aggiungi criterio',
      addCriterionPlaceholder: 'Scrivi il testo del criterio...',
      criterionLabel: 'Criterio',
      pauseGoal: 'Metti in pausa l’obiettivo',
      resumeGoal: 'Riprendi l’obiettivo',
      resumeNow: 'Riprendi ora',
      clearGoal: 'Cancella obiettivo',
      clearGoalConfirmTitle: 'Cancellare l’obiettivo?',
      clearGoalConfirmBody: 'Sicuro di voler cancellare l’obiettivo attivo? Non si può annullare.',
      copyCriterion: (index: number) => `Copia criterio ${index}`,
      removeCriterion: (index: number) => `Rimuovi criterio ${index}`,
      removeCriterionConfirmTitle: (index: number) => `Rimuovere il criterio ${index}?`,
      removeCriterionConfirmBody: (index: number) => `Sicuro di voler rimuovere il criterio ${index}?`,
      clearCriteria: 'Cancella tutti i criteri',
      clearCriteriaConfirmTitle: 'Cancellare tutti i criteri?',
      clearCriteriaConfirmBody: 'Sicuro di voler rimuovere tutti i criteri di questo obiettivo?',
      criteriaHeader: (count: number) => `Criteri · ${count}`,
      noCriteria: 'Nessun criterio',
      goalDetailsTitle: 'Dettagli dell’obiettivo',
      objectiveLabel: 'Obiettivo',
      contractOutcome: 'Risultato',
      contractVerification: 'Verifica',
      contractConstraints: 'Vincoli',
      contractBoundaries: 'Limiti',
      contractStopWhen: 'Ferma quando',
      waitBarrierTitle: 'Condizione di attesa',
      waitUntil: (target: string) => `In attesa fino a ${target}`,
      waitSession: (target: string) => `In attesa della sessione ${target}`,
      waitPid: (pid: number) => `In attesa del processo ${pid}`,
      qualityGatesTitle: 'Controlli di qualità',
      gateCommand: 'Comando',
      gateAttempts: (attempts: number, max: number) => `${attempts}/${max} tentativi`,
      gateTimeout: (seconds: number) => `timeout di ${seconds} s`,
      gateLastExit: (code: number | null) => (code === null ? 'In attesa' : `Codice di uscita: ${code}`),
      loopActive: 'Loop attivo',
      loopPaused: 'Loop in pausa',
      loopDeferred: 'Loop differito',
      loopFinished: 'Loop terminato',
      loopRuns: (runs: number) => `${runs} ${runs === 1 ? 'esecuzione' : 'esecuzioni'}`,
      loopRunCount: (current: number, total: number) => `Esecuzione ${current}/${total}`,
      loopNext: (time: string) => `prossimo ${time}`,
      loopEverySeconds: (seconds: number) => `ogni ${seconds} s`,
      loopEveryMinutes: (minutes: number) => `ogni ${minutes} min`,
      loopEveryHours: (hours: number) => `ogni ${hours} h`,
      loopSelfPaced: 'al proprio ritmo',
      loopActions: 'Azioni del loop',
      pauseLoop: 'Metti in pausa il loop',
      resumeLoop: 'Riprendi il loop',
      stopLoop: 'Ferma il loop',
      stopLoopConfirmTitle: 'Fermare il loop?',
      stopLoopConfirmBody: 'Sicuro di voler fermare questo loop?',
      dismissLoop: 'Scarta il loop',
      loopPromptLabel: 'Prompt',
      loopCadenceLabel: 'Frequenza',
      loopUntilLabel: 'Condizione di fine',
      loopDeferredNotice: 'Un obiettivo attivo controlla la sessione in questo momento.',
      loopAwaitingResponse: 'In attesa di risposta',
      heartbeatActive: 'Heartbeat attivo',
      heartbeatPaused: 'Heartbeat in pausa',
      heartbeatEveryMinutes: (minutes: number) => `ogni ${minutes} min`,
      heartbeatEveryHours: (hours: number) => `ogni ${hours} h`,
      heartbeatEverySeconds: (seconds: number) => `ogni ${seconds} s`,
      heartbeatNext: (time: string) => `prossimo ${time}`,
      heartbeatDueWaitingForIdle: 'in sospeso: in attesa di inattività',
      heartbeatActions: 'Azioni del heartbeat',
      pauseHeartbeat: 'Metti in pausa il heartbeat',
      resumeHeartbeat: 'Riprendi il heartbeat',
      clearHeartbeat: 'Cancella il heartbeat',
      clearHeartbeatConfirmTitle: 'Cancellare il heartbeat?',
      clearHeartbeatConfirmBody: 'Sicuro di voler cancellare questo heartbeat?',
      heartbeatFiredCount: (count: number) => `Si è attivato ${count} ${count === 1 ? 'volta' : 'volte'}`,
      actionFailed: (msg: string) => `Azione non riuscita: ${msg}`,
      actionSucceeded: 'Azione completata',
      copySuccess: 'Criterio copiato negli appunti',
      copyFailure: 'Impossibile copiare il criterio negli appunti',
      continuationFailed: 'Impossibile inviare la continuazione dell’obiettivo',
      continuationQueued: 'Obiettivo ripreso: la continuazione resta in coda fino al termine del turno corrente',
      continuationBusy:
        'Obiettivo ripreso: la sessione è occupata; ferma prima la risposta corrente (pulsante Stop o Esc) per continuare',
      controlUnavailable: (msg: string) => `Controlli di sessione non disponibili: ${msg}`,
      dismissError: 'Scarta l’errore',
      add: 'Aggiungi'
    },
    coding: {
      title: 'Albero di lavoro',
      noBranch: 'Nessun branch',
      detached: 'staccata',
      clean: 'Pulito',
      changed: count => `${count} cambiamento${count === 1 ? '' : 'i'}`,
      ahead: count => `${count} in anticipo`,
      behind: count => `${count} in ritardo`,
      review: 'Rivedi',
      close: 'Chiudi',
      openChanges: 'Apri modifiche',
      openFile: 'Apri file',
      stage: 'Esegui staging',
      unstage: 'Rimuovi dallo staging',
      stageAll: 'Esegui staging di tutto',
      viewAsTree: 'Visualizza come albero',
      viewAsList: 'Visualizza come elenco',
      revert: 'Ripristina',
      revertAll: 'Ripristina tutto',
      revertConfirm:
        'Scartare le modifiche in questo file e ripristinarlo allo stato di commit? L’operazione non può essere annullata.',
      revertAllConfirm:
        'Scartare tutte le modifiche e ripristinare i file allo stato di commit? L’operazione non può essere annullata.',
      staged: 'In staging',
      noChanges: 'Nessuna modifica',
      notRepo: 'Non è un repository Git',
      noDiff: 'Non ci sono differenze da mostrare',
      scopeUncommitted: 'Non committato',
      scopeBranch: 'Branch',
      scopeLastTurn: 'Ultimo turno',
      commit: 'Esegui commit',
      commitAndPush: 'Esegui commit e push',
      commitPlaceholder: (shortcut: string) => `Messaggio (${shortcut} per eseguire il commit)`,
      generateCommitMessage: 'Genera messaggio di commit',
      stopGenerating: 'Interrompi la generazione',
      createPr: 'Crea PR',
      openPr: 'Apri PR',
      ghMissing: 'Installa GitHub CLI (gh) e accedi per aprire una PR',
      agentShip: 'Chiedi a Hermes di aprire una PR',
      agentShipUnavailable: 'La chat a cui appartengono queste modifiche non è sullo schermo.',
      agentShipPrompt:
        'Rivedi le modifiche correnti, esegui un commit con un messaggio convenzionale chiaro, fai push del branch e apri una pull request.',
      newBranch: 'Nuovo branch',
      branchOffFrom: base => `Nuovo branch da ${base}`,
      switchTo: branch => `Passa a ${branch}`,
      switchFailed: branch => `Impossibile passare a ${branch}`,
      worktrees: 'Worktree'
    }
  },
  updates: {
    discontinuedTitle: 'Questa versione di Hermes non è più supportata',
    discontinuedBody:
      'Questa versione di Hermes non è più supportata e potrebbe smettere di funzionare; disinstallala. I tuoi dati restano sul disco.',
    channels: { stable: 'Stabile', canary: 'Canary' },
    appName: 'Hermes',
    availableBodyRelease: tag => `La versione ${tag} è pronta per essere installata.`,
    releaseAvailable: tag => `La versione ${tag} è disponibile.`,
    checkingShort: 'Verifica…',
    availableBodyAppInstaller:
      'È disponibile una nuova versione di Hermes. Hermes si chiuderà, Windows completerà l’aggiornamento e Hermes si riaprirà automaticamente.',
    applyingBodyAppInstaller:
      'Hermes si chiuderà e Windows completerà l’aggiornamento. Hermes si riaprirà al termine; non devi fare nulla.',
    applyingCloseAppInstaller:
      'Questa finestra si chiuderà; Windows completerà l’aggiornamento e Hermes si riaprirà automaticamente.',
    checkUnknownTitleAppInstaller: 'Non è stato possibile cercare aggiornamenti',
    checkUnknownBodyAppInstaller:
      'Windows non è riuscito a cercare aggiornamenti in questo momento. Vengono installati automaticamente anche al riavvio di Hermes.',
    versionDetailsTitle: 'Dettagli della versione',
    versionDetailsBody:
      'Questa installazione è gestita al di fuori dell’app. Aggiornala allo stesso modo in cui l’hai installata.',
    versionDetailsVersion: 'Versione',
    versionDetailsCommit: 'Commit',
    versionDetailsBuildOrigin: 'Origine della compilazione',
    versionDetailsDistribution: 'Distribuzione',
    versionDetailsDistributionDesktop: 'Applicazione desktop',
    versionDetailsDistributionDesktopMsix: 'Applicazione desktop (MSIX)',
    versionDetailsDistributionDesktopInstaller: 'Applicazione desktop (installer)',
    versionDetailsDistributionSourceInstaller: 'Codice sorgente (script di installazione)',
    versionDetailsDistributionSourceInstallerDesktop: 'Codice sorgente (script di installazione) + hermes desktop',
    versionDetailsDistributionSource: 'Codice sorgente',
    versionDetailsDistributionSourceDesktop: 'Codice sorgente + hermes desktop',
    versionDetailsDistributionStore: 'Microsoft Store',
    versionDetailsRuntime: 'Runtime',
    versionDetailsRuntimeEmbedded: 'Runtime integrato',
    versionDetailsRuntimeExternal: 'Esterno (usa il runtime del computer)',
    versionDetailsInstallId: 'ID di installazione',
    versionDetailsUncommittedChanges: 'modifiche non committate',
    version: value => `Versione ${value}`,
    versionUnavailable: 'Versione non disponibile',
    bundleOutOfSync: 'La compilazione dell’app non è aggiornata',
    bundleOutOfSyncDesc:
      'Il runtime di Hermes è stato aggiornato, ma l’app desktop resta una compilazione precedente: le nuove funzioni dell’interfaccia (come la modalità Bot) mancheranno finché non la aggiorni. Esegui l’aggiornamento qui sotto per ricompilare l’app. Se così l’avviso non scompare, reinstalla dall’installer desktop più recente.',
    bundleOutOfSyncAction: 'Ottieni l’installer',
    bundleSwapPending: 'Riavvia per completare l’aggiornamento',
    bundleSwapPendingDesc:
      'L’app aggiornata è già installata; Hermes deve solo riavviarsi per caricarla. Chat e impostazioni restano intatte.',
    bundleSwapPendingAction: 'Riavvia Hermes',
    checkNow: 'Verifica ora',
    seeWhatsNew: 'Vedi le novità',
    releaseNotes: 'Note di versione',
    onLatest: 'Hai già la versione più recente.',
    installing: 'È in corso l’installazione di un aggiornamento.',
    cantReach: 'Non siamo riusciti a contattare il server degli aggiornamenti.',
    tapCheck: 'Premi "Verifica ora" per cercare aggiornamenti.',
    updateReady: count =>
      `C’è un aggiornamento pronto (${count} ${count === 1 ? 'modifica inclusa' : 'modifiche incluse'}).`,
    updateReadyUnknown: 'C’è un nuovo aggiornamento pronto.',
    lastChecked: age => `Ultima verifica ${age}`,
    justNowSuffix: ' · proprio ora',
    never: 'mai',
    justNow: 'proprio ora',
    minAgo: count => `${count} min fa`,
    hoursAgo: count => `${count} h fa`,
    daysAgo: count => `${count} g fa`,
    stages: {
      idle: 'Preparazione…',
      prepare: 'Preparazione…',
      fetch: 'Scaricamento…',
      pull: 'Quasi pronto…',
      pydeps: 'Finalizzazione…',
      update: 'Aggiornamento di Hermes…',
      rebuild: 'Ricompilazione dell’app desktop…',
      restart: 'Riavvio di Hermes…',
      done: 'Aggiornamento completato',
      manual: 'Aggiorna dal terminale',
      guiSkew: 'Aggiorna l’app desktop',
      error: 'Aggiornamento in pausa'
    },
    checking: 'Ricerca di aggiornamenti…',
    checkFailedTitle: 'Non è stato possibile cercare aggiornamenti',
    tryAgain: 'Riprova',
    notAvailableTitle: 'Aggiornamento non disponibile',
    unsupportedMessage: 'Questa versione di Hermes non può essere aggiornata dall’app.',
    connectionRetry:
      'Hermes non è riuscito a raggiungere il server degli aggiornamenti. Controlla la tua connessione a internet e riprova. Se usi un Hermes remoto, assicurati che sia online.',
    gitUnusable: 'Hermes non è riuscito a eseguire Git su questo computer, quindi non ha potuto cercare aggiornamenti.',
    connectionSettings: 'Configurazione di connessione',
    openDownloadPage: 'Apri la pagina di download',
    latestBody: 'Stai usando la versione più recente.',
    latestBodyBackend: 'Il backend sta eseguendo la versione più recente.',
    allSetTitle: 'Tutto pronto',
    availableTitle: 'Nuovo aggiornamento disponibile',
    availableBody: 'Una nuova versione di Hermes è pronta per l’installazione.',
    availableTitleBackend: 'Aggiornamento del backend disponibile',
    availableBodyBackend:
      'Una versione più recente del backend di Hermes a cui sei connesso è pronta per l’installazione.',
    availableBodyNoChangelog:
      'Una versione più recente è pronta. Le note di versione non sono disponibili per questo tipo di installazione.',
    updateNow: 'Aggiorna ora',
    maybeLater: 'Forse più tardi',
    moreChanges: count => `+ ${count} ${count === 1 ? 'modifica inclusa' : 'modifiche incluse'}.`,
    copyFullLog: 'Copia il log completo delle modifiche',
    manualTitle: 'Aggiorna dal terminale',
    manualUnavailableTitle: 'Impossibile aggiornare da qui',
    manualBody:
      'Hai installato Hermes dalla riga di comando, quindi anche gli aggiornamenti si eseguono lì. Incolla questo nel tuo terminale:',
    manualPickedUp: 'Hermes userà la nuova versione la prossima volta che lo apri.',
    manualBodyBackend: 'Il backend di Hermes è gestito al di fuori di questa app. Esegui questo sul server che lo ospita:',
    manualPickedUpBackend: 'Il backend caricherà la nuova versione al termine dell’aggiornamento.',
    guiSkewTitle: 'Aggiorna l’app desktop',
    guiSkewBody:
      'Il backend è stato aggiornato, ma il pacchetto di questa app desktop non è cambiato. Aggiorna o reinstalla l’app desktop di Hermes (la tua AppImage / .deb / .rpm) per farli corrispondere.',
    copy: 'Copia',
    copied: 'Copiato',
    done: 'Fatto',
    applyingBody:
      'L’updater di Hermes prenderà il controllo in una finestra dedicata e riaprirà Hermes al termine.',
    applyingBodyBackend:
      'Il backend remoto sta applicando l’aggiornamento e si riavvierà. Hermes si riconnetterà automaticamente quando sarà di nuovo disponibile.',
    applyingClose: 'Hermes si chiuderà per applicare l’aggiornamento.',
    errorTitle: 'L’aggiornamento non è terminato',
    errorBody: 'Tranquillo: non si è perso nulla. Puoi riprovare ora.',
    blockerTitle: 'Chiudere le anteprime locali per aggiornare Hermes?',
    blockerBody:
      'Hermes deve fermare queste anteprime locali prima di aggiornare. Questo non modifica né elimina i tuoi file.',
    foreignBlockerTitle: 'Chiudi gli altri processi per aggiornare Hermes',
    foreignBlockerBody:
      'Hermes non può chiudere questi processi automaticamente in modo sicuro. Chiudi l’app, il terminale o il servizio a cui appartiene ciascuno e riprova l’aggiornamento.',
    mixedBlockerBody:
      'Hermes può chiudere le anteprime locali indicate qui sotto. Gli altri processi devono essere chiusi manualmente prima di continuare con l’aggiornamento.',
    closePreviewsAndUpdate: 'Chiudi le anteprime e aggiorna',
    closePreviewsAndCheckAgain: 'Chiudi le anteprime e verifica di nuovo',
    localPreview: 'Anteprima locale',
    portLabel: (port: number) => `Porta ${port}`,
    pidLabel: (pid: number) => `PID ${pid}`,
    technicalDetails: 'Dettagli tecnici',
    notNow: 'Non ora',
    clientAlsoBehindTitle: 'L’app desktop non è aggiornata',
    clientAlsoBehindMessage:
      'Il backend è aggiornato, ma questa app desktop resta a una versione precedente. Aggiornala per ottenere le ultime correzioni.',
    clientAlsoBehindAction: 'Aggiorna l’app desktop',
    everythingDispatched: 'Aggiornamento inviato',
    everythingSkipped: 'Saltato',
    everythingRowFailed: 'Aggiornamento non riuscito',
    everythingFanoutFailedTitle: 'Impossibile aggiornare le altre istanze',
    changeLogNew: 'Novità',
    changeLogFixed: 'Corretto',
    changeLogFaster: 'Più veloce',
    changeLogImproved: 'Migliorato',
    changeLogOther: 'Altri miglioramenti',
    changeLogFallbackLabel: 'In questo aggiornamento',
    changeLogFallbackItem: 'Miglioramenti e correzioni',
    applyStatus: {
      preparing: 'Aggiornamento del backend…',
      pulling: 'Aggiornamento del backend…',
      restarting: 'Riavvio del backend per caricare l’aggiornamento…',
      notAvailable: 'L’aggiornamento non è disponibile per questo backend.',
      failed: 'Aggiornamento del backend non riuscito.',
      noReturn:
        'Il backend non è tornato disponibile. L’aggiornamento potrebbe non essere stato completato; controlla l’host del backend.'
    }
  },
  handoffTour: {
    profileTitle: 'La tua prima attività viene eseguita nel profilo predefinito',
    profileText:
      'Questa barra cambia profilo. Quello illuminato ora è il predefinito, dove si trova la sessione dell’attività. L’altro è il profilo di configurazione, dove si trova la chat di benvenuto.',
    sessionsTitle: 'Ogni profilo ha le proprie sessioni',
    sessionsText:
      'Questa lista appartiene al profilo predefinito. Nuova sessione ne crea una nel profilo selezionato. Cambia profilo nella barra e la lista cambia di conseguenza.',
    stayTitle: 'Hermes è a un clic',
    stayText: 'Passa al profilo di configurazione e apri Benvenuto in Hermes ogni volta che hai bisogno di aiuto. Resta lì.'
  },
  guidedGreeting: {
    line: 'Ciao, entra pure. Sono Hermes. Dammi due minuti per preparare tutto su misura per te e poi ci metteremo al lavoro su qualcosa che vorrai davvero fare.\n\nMa prima, come vuoi che ti chiami?',
    nameSuggestion: (name: string) => `(Posso anche chiamarti semplicemente ${name}, se preferisci.)`
  },
  install: {
    stageStates: {
      pending: 'In attesa',
      running: 'Installazione in corso',
      succeeded: 'Completato',
      skipped: 'Saltato',
      failed: 'Non riuscito'
    },
    oneTimeTitle: 'Hermes richiede una singola installazione',
    unsupportedDesc: platform =>
      `L’installazione automatica al primo avvio non è ancora disponibile su ${platform}. Apri Terminale ed esegui il comando qui sotto; poi riapri l’app. I successivi avvii salteranno questo passaggio.`,
    installCommand: 'Comando di installazione',
    copyCommand: 'Copia comando',
    viewDocs: 'Vedi i docs di installazione',
    installTo: 'Verrà installato in',
    retryAfterRun: 'L’ho già eseguito -- riprova',
    setupChoiceTitle: 'Configura Hermes Desktop',
    setupChoiceDesc:
      'Collega questa app a un gateway di Hermes già in esecuzione oppure installa Hermes localmente su questo computer.',
    connectExistingTitle: 'Connetti a un Hermes esistente',
    connectExistingShort: 'Connetti esistente',
    connectExistingDesc:
      'Usa un backend remoto con un token di sessione o l’accesso nel browser. Non verrà avviata alcuna installazione locale.',
    installLocalTitle: 'Installa Hermes localmente',
    installLocalDesc: 'Scarica Hermes, crea il suo ambiente Python ed esegui il backend su questo computer.',
    localStartUnavailable: 'Impossibile avviare l’installazione locale. Riavvia Hermes Desktop e riprova.',
    remoteSetupTitle: 'Connetti a un Hermes esistente',
    remoteSetupDesc:
      'Inserisci l’URL del tuo gateway. Hermes Desktop rileverà se serve un token o l’accesso nel browser.',
    remoteUrlTitle: 'URL del gateway',
    remoteUrlDesc: 'Usa l’URL di base del gateway di Hermes e includi https:// se è remoto.',
    remoteUrlPlaceholder: 'https://gateway.example.com/hermes',
    probing: 'Rilevamento dell’autenticazione del gateway…',
    probeError:
      'Hermes non riesce a raggiungere quell’indirizzo. Controlla l’URL e che l’altro computer stia eseguendo Hermes; le opzioni di accesso compaiono quando risponde.',
    probeErrorDetails: 'Dettagli',
    identityProvider: 'il tuo provider di identità',
    authTitle: 'Autenticazione',
    authNeedsOauth: provider => `Accedi con ${provider} prima di testare questo gateway.`,
    authSignedIn: 'Accesso nel browser completato.',
    connected: 'Connesso',
    signIn: 'Accedi',
    signInWith: provider => `Accedi con ${provider}`,
    enterUrlFirst: 'Inserisci prima l’URL di un gateway.',
    signInIncomplete: 'La finestra di accesso si è chiusa prima di completare l’autenticazione.',
    tokenTitle: 'Token di sessione',
    tokenDesc: 'Incolla il token di sessione dal file .env del gateway remoto.',
    pasteSessionToken: 'Incolla token di sessione',
    incompleteSignInTest: 'Accedi prima di testare questo gateway protetto da OAuth.',
    incompleteTokenTest: 'Inserisci un token di sessione prima di testare questo gateway.',
    testConnection: 'Prova connessione',
    testSucceeded: (baseUrl, version) => `Connesso a ${baseUrl}${version ? ` (${version})` : ''}.`,
    applyRemote: 'Applica e riconnetti',
    backToSetup: 'Indietro',
    failedTitle: 'Installazione non riuscita',
    settingUpTitle: 'Configurazione di Hermes Agent',
    finishingTitle: 'Completamento',
    failedDesc:
      'Uno dei passaggi di configurazione non è andato a buon fine. Può succedere se c’è un’altra copia di Hermes in esecuzione, la connessione a internet si è interrotta o un antivirus ha bloccato il programma di installazione. Chiudi le altre finestre di Hermes e scegli Ricarica e riprova. Se fallisce di nuovo, apri i log e inviali all’assistenza.',
    activeDesc:
      'Questa configurazione viene eseguita una sola volta. Il programma di installazione di Hermes sta scaricando le dipendenze e configurando la tua macchina. I successivi avvii salteranno questo passaggio.',
    progress: (completed, total) => `${completed} di ${total} passaggi completati`,
    currentStage: stage => ` -- ora: ${stage}`,
    fetchingManifest: 'Recupero del manifesto del programma di installazione...',
    error: 'Errore',
    hideOutput: 'Nascondi output del programma di installazione',
    showOutput: 'Mostra output del programma di installazione',
    lines: count => `${count} ${count === 1 ? 'riga' : 'righe'}`,
    noOutput: 'Ancora nessun output.',
    cancelling: 'Annullamento...',
    cancelInstall: 'Annulla installazione',
    transcriptSaved: 'Trascrizione completa salvata in',
    copiedOutput: 'Copiato!',
    copyOutput: 'Copia output',
    reloadRetry: 'Ricarica e riprova',
    openLogs: 'Apri i log'
  },
  onboarding: {
    headerTitle: 'Configuriamo Hermes Agent',
    headerDesc: 'Collega un provider di modelli per iniziare a chattare. La maggior parte delle opzioni richiede un clic.',
    preparingInstall: 'Hermes sta completando l’installazione. Al primo avvio di solito richiede meno di un minuto.',
    starting: 'Avvio di Hermes…',
    lookingUpProviders: 'Ricerca dei provider...',
    collapse: 'Comprimi',
    otherProviders: 'Altri provider',
    haveApiKey: 'Ho una chiave API',
    chooseLater: 'Sceglierò un provider più tardi',
    recommended: 'Consigliato',
    connected: 'Connesso',
    featuredPitch: 'Un abbonamento, oltre 300 modelli frontier: il modo consigliato di usare Hermes',
    fireworksPitch: 'API diretta ai modelli: modelli frontier ospitati su Fireworks',
    localModelsTitle: 'Esegui modelli localmente',
    localModelsPitch: 'Senza account: scarica un modello ed eseguilo su questo computer',
    openRouterPitch: 'Una chiave, centinaia di modelli: un buon valore predefinito',
    apiKeyOptions: {
      fireworks: {
        short: 'API diretta al modello',
        description: 'Accesso diretto ai modelli ospitati su Fireworks AI.'
      },
      openrouter: {
        short: 'una chiave, tanti modelli',
        description:
          'Ospita centinaia di modelli dietro un’unica chiave. Buon valore predefinito per le nuove installazioni.'
      },
      openai: {
        short: 'modelli tipo GPT',
        description: 'Accesso diretto ai modelli di OpenAI.'
      },
      gemini: {
        short: 'modelli Gemini',
        description: 'Accesso diretto ai modelli di Google Gemini.'
      },
      xai: {
        short: 'modelli Grok',
        description: 'Accesso diretto ai modelli Grok di xAI.'
      },
      local: {
        short: 'self-hosted',
        description:
          'Punta Hermes a un endpoint locale o self-hosted compatibile con OpenAI (vLLM, llama.cpp, Ollama, ecc.).'
      }
    },
    backToSignIn: 'Torna all’accesso',
    getKey: 'Ottieni una chiave',
    replaceCurrent: 'Sostituisci valore attuale',
    pasteApiKey: 'Incolla chiave API',
    localApiKeyPlaceholder: 'Chiave API (opzionale; solo se il tuo endpoint la richiede)',
    localModelNamePlaceholder: 'Nome del modello (ad es. command-a-plus-05-2026)',
    couldNotSave: 'Impossibile salvare la credenziale.',
    connecting: 'Connessione in corso',
    update: 'Aggiorna',
    flowSubtitles: {
      pkce: 'Apri il browser per accedere e poi continua qui',
      device_code: 'Apre una pagina di verifica nel tuo browser; Hermes si connette automaticamente',
      external: 'Accedi una volta dal tuo terminale e torna qui per chattare'
    },
    startingSignIn: provider => `Avvio dell’accesso con ${provider}...`,
    verifyingCode: provider => `Verifica del tuo codice con ${provider}...`,
    connectedProvider: provider => `${provider} connesso`,
    connectedPicking: provider => `${provider} connesso. Scelta del modello predefinito in corso...`,
    signInFailed: 'Accesso non riuscito. Riprova.',
    signInExpired:
      'La pagina di accesso è scaduta prima che finissi. Riprova e completa il passaggio nel browser entro pochi minuti, oppure usa una chiave API.',
    signInDidNotFinish: (provider: string) =>
      `L’accesso con ${provider} non è stato completato. Controlla la tua connessione a internet e riprova, oppure scegli un altro provider.`,
    tryAgain: 'Riprova',
    useApiKeyInstead: 'Usa una chiave API',
    errorDetails: 'Dettagli',
    pickDifferentProvider: 'Scegli un altro provider',
    signInWith: provider => `Accedi con ${provider}`,
    openedBrowser: provider => `Abbiamo aperto ${provider} nel tuo browser.`,
    authorizeThere: 'Autorizza Hermes lì.',
    copyAuthCode: 'Copia il codice di autorizzazione e incollalo qui sotto.',
    pasteAuthCode: 'Incolla codice di autorizzazione',
    reopenAuthPage: 'Riapri pagina di autorizzazione',
    autoBrowser: provider =>
      `Abbiamo aperto ${provider} nel tuo browser. Autorizza Hermes lì e ti connetterai automaticamente; non c’è niente da copiare o incollare.`,
    reopenSignInPage: 'Riapri pagina di accesso',
    waitingAuthorize: 'In attesa della tua autorizzazione...',
    externalPending: provider =>
      `${provider} effettua l’accesso con la propria CLI. Esegui questo comando in un terminale e poi torna qui e scegli "Accesso già effettuato":`,
    signedIn: 'Accesso già effettuato',
    deviceCodeOpened: provider => `Abbiamo aperto ${provider} nel tuo browser. Inserisci questo codice lì:`,
    reopenVerification: 'Riapri pagina di verifica',
    copy: 'Copia',
    defaultModel: 'Modello predefinito',
    freeTier: 'Piano gratuito',
    pro: 'Pro',
    free: 'Gratis',
    price: (input, output) => `${input} in ingresso / ${output} in uscita per Mtok`,
    change: 'Cambia',
    startChatting: 'Inizia',
    docs: provider => `Docs di ${provider}`
  },
  freeTier: {
    providerRowTitle: 'Nous · piano gratuito',
    providerRowPitch: 'Accedi con un account Nous per sbloccare più modelli e strumenti.',
    readyTitle: 'Hermes è pronto.',
    readyCaption: 'Gratis · connettori inclusi',
    begin: 'Inizia',
    signInInstead: 'Accedi con un account Nous',
    otherProviders: 'Altri provider',
    stripTitle: 'L’inferenza e i connettori gratuiti di Nous sono già disponibili.',
    stripBody: 'Apri il selettore dei modelli per provarli oppure accedi con un account Nous.',
    openModelPicker: 'Apri selettore modelli',
    dismiss: 'Ignora',
    providerName: 'Nous',
    statusLabel: (model: string) => `Nous · ${model}`,
    signIn: 'Accedi',
    signInHeading: 'Accedi con un account Nous per sbloccare più modelli e strumenti.',
    settingUp: 'Configurazione dell’inferenza gratuita…',
    codeBody: 'Inserisci questo codice nel tuo browser per completare l’accesso.',
    copyLink: 'Copia link',
    doNotShare: 'Non condividere questo codice.',
    waiting: 'In attesa dell’accesso…',
    finishingHeading: 'Completamento dell’accesso…',
    finishingBody: 'Approvato nel browser. Recupero dei token del tuo account.',
    signedInAs: (email: string) => `Sessione avviata come ${email}`,
    signedIn: 'Sessione avviata.',
    completedBody: 'Il tuo account include già inferenza e strumenti.',
    defaultModel: 'Modello predefinito',
    change: 'Cambia',
    done: 'Fatto',
    notNow: 'Non ora',
    tryAgain: 'Riprova',
    startAgain: 'Ricomincia',
    didNotComplete: 'L’accesso non è stato completato',
    rejectedBody: 'Nessun problema, resti nel servizio gratuito di Nous. Accedi quando vuoi.',
    supersededBody:
      'Un codice di accesso più recente ha sostituito questo. Usa quello più recente o ricomincia.',
    timedOutHeading: 'Quel link di accesso è scaduto',
    timedOutBody: 'Ricomincia quando vuoi. Resti nel servizio gratuito di Nous.',
    retiredBody:
      'La tua sessione è terminata prima di completare l’accesso. Hermes ne avvierà una nuova; poi accedi di nuovo quando vuoi.',
    errorBody: 'L’accesso non è stato completato. Riprova quando vuoi.',
    busyHeading: 'Ci siamo quasi',
    busyBody: (wait: string) =>
      `Hermes non è riuscito a completare l’avvio della tua sessione perché il servizio di Nous è occupato. Riprova tra ${wait}. Nel frattempo, la tua sessione resta qui.`,
    unreachableBody:
      'Hermes non è riuscito a raggiungere il servizio di Nous per completare l’avvio della tua sessione. Controlla la tua connessione a internet e riprova. La tua sessione resta qui.',
    alreadySignedInHeading: 'Hai già effettuato l’accesso.',
    alreadySignedInBody: 'Questo Hermes ha già una sessione attiva su un account Nous.',
    setupFailed: {
      gateClosed:
        'Questa versione di Hermes non può avviarsi senza un account Nous. Accedi o creane uno: è gratis e richiede solo un minuto.',
      paused:
        'L’uso di Hermes senza accesso è in pausa per il momento. Hermes continuerà a controllare. Accedere è gratis e ti permette di iniziare subito.',
      rateLimited: (wait: string) =>
        `Molta gente sta iniziando proprio ora, quindi Hermes riproverà tra ${wait}. Accedere è gratis e ti evita l’attesa.`,
      unreachable:
        'Hermes non è riuscito a raggiungere il servizio di Nous. Controlla la tua connessione a internet e premi Riprova. Oppure collega un altro provider per ora.',
      serverError:
        'Il servizio di Nous ha avuto un malfunzionamento. Premi Riprova tra poco oppure collega un altro provider per ora.',
      powRequired:
        'Il server di Nous ha richiesto una proof of work, ma il tuo agente non la implementa ancora. Accedi o crea un account gratuito di Nous per continuare.',
      locked:
        'Questa sessione non può continuare senza effettuare l’accesso. Accedi o crea un account gratuito di Nous per proseguire.',
      generic:
        'Hermes non è riuscito a configurare l’accesso gratuito senza login. Accedere è gratis; puoi anche collegare un altro provider.',
      signInBelow: 'Accedere è gratis. Scegli Nous qui sotto.',
      tryAgain: 'Riprova',
      retrying: 'Nuovo tentativo…'
    }
  },
  modelPicker: {
    title: 'Cambia modello',
    current: 'attuale:',
    unknown: '(sconosciuto)',
    search: 'Filtra provider e modelli...',
    noModels: 'Nessun modello trovato.',
    addProvider: 'Aggiungi provider',
    loadFailed: 'Impossibile caricare i modelli',
    loadingIntoMemory: 'Caricamento in memoria',
    downloading: 'Download in corso',
    localDownloadsHeading: 'Locali',
    noAuthenticatedProviders: 'Nessun provider autenticato.',
    pro: 'Pro',
    proNeedsSubscription: 'I modelli Pro richiedono un abbonamento a pagamento a Nous.',
    free: 'Gratis',
    freeTier: 'Piano gratuito',
    priceTitle: 'Prezzo in ingresso / in uscita per milione di token',
    wasPrice: 'prima',
    customModel: 'Modello personalizzato',
    addCustomModelAction: 'Aggiungi modello personalizzato…',
    customModelPlaceholder: 'Scrivi un ID modello, ad es. openai/gpt-5'
  },
  modelVisibility: {
    title: 'Modelli',
    search: 'Cerca modelli',
    noAuthenticatedProviders: 'Nessun provider autenticato.',
    addProvider: 'Aggiungi provider…',
    addCustomModel: 'Aggiungi modello personalizzato',
    removeCustomModel: 'Rimuovi modello personalizzato',
    resetToDefaults: 'Ripristina valori predefiniti',
    resetConfirm: 'Ripristinare la visibilità dei modelli?',
    resetDescription:
      'Le tue scelte di modelli visibili e nascosti vengono cancellate e ogni provider torna alla sua lista predefinita. I modelli personalizzati che hai aggiunto vengono conservati e mostrati.',
    resetAction: 'Ripristina'
  },
  shell: {
    windowControls: 'Controlli finestra',
    paneControls: 'Controlli pannello',
    appControls: 'Controlli app',
    modelMenu: itModelMenu,
    modelOptions: {
      noOptions: 'Nessuna opzione per questo modello',
      options: 'Opzioni',
      thinking: 'Ragionamento',
      fast: 'Veloce',
      ultrafast: 'Ultrafast',
      useStandardSpeed: 'Usa velocità standard',
      effort: 'Sforzo',
      minimal: 'Minimo',
      low: 'Basso',
      medium: 'Medio',
      high: 'Alto',
      xhigh: 'Extra alto',
      max: 'Massimo',
      ultra: 'Ultra',
      sendsOnRoute: (level: string) => `invia ${level} su questa rotta`,
      updateFailed: 'Impossibile aggiornare l’opzione del modello',
      fastFailed: 'Impossibile aggiornare la modalità veloce'
    },
    gatewayMenu: {
      gateway: 'Gateway',
      connected: 'Connesso',
      connecting: 'Connessione in corso',
      offline: 'Senza connessione',
      inferenceReady: 'Inferenza pronta',
      inferenceNotReady: 'Inferenza non pronta',
      checkingInference: 'Verifica dell’inferenza',
      disconnected: 'Disconnesso',
      reconnectGateway: 'Riconnetti il gateway',
      openSystem: 'Apri pannello di sistema',
      connection: label => `Connessione: ${label}`,
      recentActivity: 'Attività recente',
      viewAllLogs: 'Vedi tutti i log →',
      messagingPlatforms: 'Piattaforme di messaggistica'
    },
    approvalMode: {
      title: 'Modalità di approvazione',
      ariaLabel: mode => `Modalità di approvazione: ${mode}`,
      manual: 'Manuale',
      manualDescription: 'Chiedi conferma prima delle azioni che richiedono approvazione',
      smart: 'Intelligente',
      smartDescription: 'Valuta automaticamente le azioni e chiedi solo quando necessario',
      off: 'Disattivato',
      offDescription: 'Esegui senza richieste di approvazione'
    },
    statusbar: {
      unknown: 'sconosciuto',
      restart: 'riavvia',
      update: 'aggiorna',
      updateInProgress: 'Aggiornamento in corso',
      commitsBehind: (count, branch) => `${count} ${count === 1 ? 'commit' : 'commits'} indietro rispetto a ${branch}`,
      desktopVersion: version => `Hermes Desktop v${version}`,
      backendVersion: version => `backend v${version}`,
      clientLabel: version => `client v${version}`,
      connectionSsh: host => `SSH: ${host}`,
      connectionRemote: host => `Remoto: ${host}`,
      connectionCloud: host => `Cloud: ${host}`,
      connectionCloudTooltip: host => `Hermes Cloud · ${host}`,
      connectionSshTooltip: host => `SSH · ${host}`,
      connectionRemoteTooltip: host => `Remoto · ${host}`,
      backendLabel: version => `backend v${version}`,
      commit: sha => `commit ${sha}`,
      branch: branch => `branch ${branch}`,
      closeCommandCenter: 'Chiudi Centro comandi',
      openCommandCenter: 'Apri Centro comandi',
      showTerminal: 'Mostra terminale',
      hideTerminal: 'Nascondi terminale',
      gateway: 'Gateway',
      gatewayReady: 'pronto',
      gatewayNeedsSetup: 'richiede configurazione',
      gatewayUnavailable: 'inferenza non disponibile',
      gatewayChecking: 'verifica in corso',
      gatewayConnecting: 'connessione in corso',
      gatewayOffline: 'senza connessione',
      gatewayRestarting: 'riavvio in corso…',
      gatewayTitle: 'Gateway',
      customizeTitle: 'Mostra nella barra di stato',
      hideStatusbar: 'Nascondi barra di stato',
      resetStatusbar: 'Ripristina i valori predefiniti',
      toggleApprovalMode: 'Approvazioni',
      toggleBackendVersion: 'Versione del backend',
      toggleCacheHitRate: 'Tasso di hit della cache',
      toggleCommandCenter: 'Centro comandi',
      toggleContextUsage: 'Misuratore di contesto',
      toggleRunningTimer: 'Timer di turno',
      toggleSessionTimer: 'Timer di sessione',
      toggleTerminal: 'Terminale',
      toggleTokensPerSecond: 'Token al secondo',
      toggleVersion: 'Versione e aggiornamenti',
      toggleFreeTier: 'Piano gratuito',
      toggleWorkspace: 'Workspace',
      cacheHitRateTitle:
        'Tasso di hit della cache dei prompt in questa sessione: i token in cache costano meno, quindi più è alto, più risparmi',
      tokensPerSecondTitle: 'Token di output al secondo, mediati sulle ultime 10 chiamate al modello',
      agents: 'Agenti',
      closeAgents: 'Chiudi agenti',
      openAgents: 'Apri agenti',
      subagents: count => `${count} ${count === 1 ? 'subagente' : 'subagenti'}`,
      failed: count => `${count} falliti`,
      running: count => `${count} in esecuzione`,
      cron: 'Cron',
      openCron: 'Apri attività cron',
      webhooks: 'Webhooks',
      openWebhooks: 'Apri webhooks',
      starmap: 'Grafo della memoria',
      openStarmap: 'Apri grafo della memoria',
      turnRunning: 'In esecuzione',
      contextUsage: 'Uso del contesto',
      systemResources: {
        title: 'Risorse di sistema',
        loading: 'Risorse…',
        gpuUtilization: 'Utilizzo GPU',
        gpuMemory: 'Memoria GPU',
        ram: 'RAM',
        unifiedNote: 'Memoria unificata: la GPU e il sistema condividono questo spazio.',
        toggle: 'Risorse di sistema'
      },
      contextUsagePanel: {
        categories: {
          conversation: 'Conversazione',
          mcp: 'MCP',
          memory: 'Memoria',
          rules: 'Regole',
          skills: 'Skills',
          subagent_definitions: 'Definizioni dei subagenti',
          system_prompt: 'Prompt di sistema',
          tool_definitions: 'Definizioni degli strumenti'
        },
        empty: 'Ancora nessun dato di contesto',
        loading: 'Caricamento del dettaglio…',
        percentFull: percent => `${percent}% pieno`,
        title: 'Uso del contesto',
        tokenSummary: (used, max) => `${used} / ${max} token`
      },
      focusedSince: 'A fuoco da',
      focusedSinceTitle: 'Tempo da quando questa chat è a fuoco — non da quanto è in corso un turno',
      yoloOn: 'YOLO attivo — approvazione automatica dei comandi pericolosi. Shift+clic lo attiva/disattiva globalmente.',
      yoloOff: 'YOLO disattivato. Shift+clic lo attiva/disattiva globalmente.',
      modelNone: 'nessuno',
      noModel: 'senza modello',
      switchModel: 'Cambia modello',
      openModelPicker: 'Apri selettore modello',
      modelPinned: 'fissato da te; le nuove chat lo usano al posto di quello predefinito delle Impostazioni',
      modelTitle: (provider, model) => `Modello · ${provider}: ${model}`,
      providerModelTitle: (provider, model) => `${provider} · ${model}`
    }
  },
  rightSidebar: {
    aria: 'Barra laterale destra',
    panelsAria: 'Pannelli della barra laterale destra',
    files: 'File system',
    terminal: 'Terminale',
    noFolderSelected: 'Nessuna cartella selezionata',
    changeCwdTitle: 'Cambia directory di lavoro',
    remotePickerTitle: 'Scegli una cartella remota',
    remotePickerDescription: 'Esplora le cartelle nel backend connesso.',
    remotePickerSelect: 'Seleziona cartella',
    remotePickerNewFolder: 'Nuova cartella',
    remotePickerFolderName: 'Nome della cartella',
    remotePickerCreateFolder: 'Crea cartella',
    remotePickerInvalidFolderName: 'Scrivi un solo nome di cartella, senza slash.',
    remotePickerCreateFolderFailed: error => `Impossibile creare la cartella (${error}).`,
    folderTip: cwd => cwd,
    openFolder: 'Apri cartella',
    refreshTree: 'Aggiorna albero',
    collapseAll: 'Comprimi tutte le cartelle',
    showIgnored: 'Mostra i file ignorati da git',
    hideIgnored: 'Nascondi i file ignorati da git',
    previewUnavailable: 'Anteprima non disponibile',
    couldNotPreview: path => `Impossibile mostrare l’anteprima di ${path}`,
    noProjectTitle: 'Nessun progetto',
    noProjectBody: 'Definisci una directory di lavoro dalla barra di stato per esplorare i file.',
    noProjectOpen: 'Nessun progetto aperto',
    noDiffs: 'Nessun diff',
    unreadableTitle: 'Non leggibile',
    unreadableBody: error => `Impossibile leggere questa cartella (${error}).`,
    emptyTitle: 'Vuoto',
    emptyBody: 'Questa cartella è vuota.',
    treeErrorTitle: 'Errore dell’albero',
    treeErrorBody: 'L’albero dei file ha riscontrato un errore durante il rendering di questa cartella.',
    tryAgain: 'Riprova',
    loadingTree: 'Caricamento dell’albero dei file',
    loadingFiles: 'Caricamento dei file',
    terminalHide: 'Nascondi terminale',
    terminalsAria: 'Terminali',
    terminalNew: 'Nuovo terminale',
    terminalCloseOthers: 'Chiudi le altre',
    terminalCloseAll: 'Chiudi tutto',
    addToChat: 'Aggiungi alla chat'
  },
  preview: {
    tab: 'Anteprima',
    pin: 'Fissa nel workspace',
    unpin: 'Sblocca dal workspace',
    closePane: 'Chiudi pannello anteprima',
    loading: 'Caricamento anteprima',
    unavailable: 'Anteprima non disponibile',
    opening: 'Apertura...',
    hide: 'Nascondi',
    openPreview: 'Apri anteprima',
    openInBrowser: 'Apri nel browser',
    openInExternal: 'Apri esternamente',
    popIn: 'Aggancia',
    popOut: 'Sgancia',
    linkHint: '⌘/Ctrl-clic per accedere al pannello anteprima',
    sourceLineTitle: 'Clic per selezionare · Maiusc-clic per estendere · trascina nel compositore',
    source: 'SORGENTE',
    renderedPreview: 'ANTEPRIMA',
    diff: 'Diff',
    unknownSize: 'dimensione sconosciuta',
    binaryTitle: 'Sembra un file binario',
    binaryBody: label => `Visualizzare l’anteprima di ${label} può mostrare testo illeggibile.`,
    largeTitle: 'Questo file è grande',
    largeBody: (label, size) => `${label} pesa ${size}. Hermes mostrerà solo i primi 512 KB.`,
    previewAnyway: 'Mostra comunque l’anteprima',
    truncated: 'Vengono mostrati i primi 512 KB.',
    noInlineTitle: 'Nessuna anteprima inline',
    noInlineBody: mimeType => `${mimeType || 'Questo tipo di file'} può comunque essere allegato come contesto.`,
    edit: 'Modifica',
    editing: 'In modifica',
    unsavedChanges: 'Modifiche non salvate',
    saveFailed: message => `Impossibile salvare: ${message}`,
    saveScopeChanged: 'Torna alla connessione e al profilo originali per salvare questa bozza.',
    diskChangedTitle: 'File modificato sul disco',
    diskChangedBody:
      'Questo file è cambiato da quando l’hai aperto. Vuoi sovrascriverlo con la tua versione oppure scartare le tue modifiche e ricaricare?',
    overwrite: 'Sovrascrivi',
    discardReload: 'Scarta e ricarica',
    console: {
      deselect: 'Deseleziona voce',
      select: 'Seleziona voce',
      copyFailed: 'Impossibile copiare l’output della console',
      copyEntry: 'Copia questa voce',
      sendEntry: 'Invia questa voce alla chat',
      messages: count => `${count} ${count === 1 ? 'messaggio di console' : 'messaggi di console'}`,
      resize: 'Ridimensiona la console dell’anteprima',
      title: 'Console dell’anteprima',
      selected: count => `${count} selezionati`,
      sendToChat: 'Invia alla chat',
      copySelected: 'Copia la selezione negli appunti',
      copyAll: 'Copia tutto negli appunti',
      copy: 'Copia',
      clear: 'Svuota',
      empty: 'Ancora nessun messaggio di console.',
      promptHeader: 'Console dell’anteprima:',
      sentTitle: 'Inviato alla chat',
      sentMessage: count =>
        `${count} ${count === 1 ? 'voce di log aggiunta' : 'voci di log aggiunte'} al compositore`
    },
    web: {
      appFailedToBoot: 'L’app di anteprima non si è avviata',
      serverNotFound: 'Server non trovato',
      remoteLoopback:
        'Questo indirizzo punta alla macchina che esegue il tuo agente, non a questa. Il pannello del browser carica le pagine localmente, quindi un server di sviluppo remoto richiede un port forwarding o un nome host raggiungibile.',
      failedToLoad: 'Impossibile caricare l’anteprima',
      tryAgain: 'Riprova',
      restarting: 'Hermes si sta riavviando...',
      askRestart: 'Chiedi a Hermes di riavviare il server',
      lookingRestart: taskId => `Hermes sta cercando un server di anteprima da riavviare (${taskId})`,
      restartingTitle: 'Riavvio del server di anteprima',
      restartingMessage:
        'Hermes sta lavorando in background. Guarda la console dell’anteprima per vedere l’avanzamento.',
      startRestartFailed: message => `Impossibile avviare il riavvio del server: ${message}`,
      restartFailed: 'Riavvio del server non riuscito',
      hideConsole: 'Nascondi la console dell’anteprima',
      showConsole: 'Mostra la console dell’anteprima',
      hideDevTools: 'Nascondi DevTools dell’anteprima',
      openDevTools: 'Apri DevTools dell’anteprima',
      goBack: 'Indietro',
      goForward: 'Avanti',
      reload: 'Ricarica pagina',
      address: 'Indirizzo',
      addressPlaceholder: 'Inserisci un indirizzo',
      blankPageBody: 'Scrivi un indirizzo qui sopra per navigare o chiedi a Hermes di aprire una pagina.',
      finishedRestarting: message =>
        `Hermes ha terminato di riavviare il server di anteprima${message ? `: ${message}` : ''}`,
      failedRestarting: message => `Riavvio del server non riuscito: ${message}`,
      unknownError: 'errore sconosciuto',
      restartedTitle: 'Server di anteprima riavviato',
      reloadingNow: 'Ricaricamento dell’anteprima in corso.',
      restartFailedTitle: 'Riavvio dell’anteprima non riuscito',
      restartFailedMessage: 'Hermes non è riuscito a riavviare il server.',
      stillWorking:
        'Hermes continua a lavorare, ma non è ancora arrivato nessun risultato del riavvio. È possibile che il comando del server sia ancora in primo piano.',
      workspaceReloading: 'Il workspace è cambiato, ricaricamento dell’anteprima',
      fileChanged: url => `File modificato, ricaricamento dell’anteprima: ${url}`,
      filesChanged: (count, url) => `${count} modifiche ai file, ricaricamento dell’anteprima: ${url}`,
      watchFailed: message => `Impossibile monitorare il file dell’anteprima: ${message}`,
      moduleMimeDescription:
        'Gli script dei moduli vengono serviti con il tipo MIME sbagliato. Di solito significa che un server di file statici sta servendo un’app Vite/React al posto del dev server del progetto.',
      loadFailedConsole: (code, message) => `Caricamento non riuscito${code ? ` (${code})` : ''}: ${message}`,
      unreachableDescription: 'Impossibile accedere alla pagina di anteprima.',
      openTarget: url => `Apri ${url}`,
      fallbackTitle: 'Anteprima',
      annotate: 'Annota',
      annotateOn: 'Smetti di annotare',
      annotateNeedPage: 'Prima apri una pagina nel browser integrato.',
      annotateFailed: 'Impossibile avviare la modalità di annotazione',
      commenting: 'Commento in corso',
      addComments: (count: number) => (count === 1 ? 'Aggiungi 1 commento' : `Aggiungi ${count} commenti`),
      commentPlaceholder: 'Aggiungi un commento...',
      commentTitle: (n: number) => `Commento ${n}`,
      saveComment: 'Salva',
      cancelComment: 'Annulla commento'
    }
  },
  interfaceMode: {
    title: 'Modalità interfaccia',
    hint: 'Cambia ciò che viene mostrato, non ciò che Hermes può fare.',
    sessionNote:
      'Definito dalla modalità Semplice. Una modifica qui dura per questa sessione; passa ad Avanzato per renderla permanente.',
    simple: {
      label: 'Semplice',
      description: 'Per parlare con Hermes. Barra laterale e chat; senza pannelli per terminale, file o diff.'
    },
    advanced: {
      label: 'Avanzato',
      description:
        'Per sviluppatori. Terminale, file, diff, barra di stato e layout, come li configuri.'
    }
  },
  zones: {
    showTabStrip: 'Mostra le schede',
    hideTabStrip: 'Nascondi le schede',
    showStripTab: title => `Mostra ${title}`,
    hideStripTab: title => `Nascondi ${title}`,
    zoneMenuLabel: title => `Opzioni zona per ${title}`,
    lastTabKeptTitle: 'L’ultima scheda rimane',
    lastTabKeptBody:
      'Questa zona richiede almeno una scheda visibile. Mostra prima un’altra scheda, oppure comprimi l’intera barra laterale.',
    toggleStripTab: title => `Attiva/disattiva la scheda ${title}`,
    minimize: 'Riduci a icona',
    restore: 'Ripristina',
    closeRunningTitle: 'Chiudere la scheda in esecuzione?',
    closeRunningBody:
      'Questa chat sta ancora lavorando o attende la tua risposta. Chiudere la scheda la nasconde; la sessione conserva i progressi e può essere riaperta dalla barra laterale.',
    closeRunningConfirm: 'Chiudi scheda',
    reload: 'Ricarica',
    closeOthers: 'Chiudi le altre',
    closeToRight: 'Chiudi quelle a destra',
    closeAll: 'Chiudi tutto',
    newSessionTab: 'Nuova scheda di sessione',
    newTab: 'Nuova scheda',
    pluginDisabled: pluginId => `Plugin "${pluginId}" disattivato`,
    pluginDisabledBody: 'Riattivalo in Capacità → Plugin per recuperare il pannello.',
    missingPane: paneId => `Pannello mancante: ${paneId}`,
    editTitle: 'Layout',
    editHint: 'Scegli un layout, oppure trascina i pannelli tra le zone.',
    reset: 'Ripristina',
    templates: 'Modelli',
    custom: 'Personalizzato',
    newGridLayout: 'Nuovo layout a griglia',
    saveCurrentAs: 'Salva il layout attuale come modello',
    nameLayoutPlaceholder: 'Nome del layout…',
    deletePreset: name => `Elimina ${name}`,
    zoneEditorTitle: 'Editor delle zone',
    editorHintPre: 'clic per dividere · ',
    editorHintPost:
      ' inverte la linea · trascina tra le zone per unirle · trascina i bordi condivisi per ridimensionare',
    templateColumns: 'Colonne',
    templateRows: 'Righe',
    templateGrid: 'Griglia',
    templatePriority: 'Priorità',
    zoneTag: index => `zona ${index}`,
    mergeZones: count => `Unisci ${count} zone`,
    customZoneName: count => `Layout personalizzato di ${count} zone`,
    layoutNamePlaceholder: fallback => `Nome del layout (${fallback})`,
    saveApply: 'Salva e applica',
    notExpressible: 'questo layout è intrecciato (a girandola) e non può ancora essere espresso come divisioni annidate',
    zoneCount: count => `${count} zone`,
    tabCount: count => `${count} schede`
  },
  contextMenu: {
    link: {
      openInApp: 'Apri nel browser integrato',
      openExternal: 'Apri nel browser esterno',
      copyUrl: 'Copia URL',
      copyResolvedUrl: 'Copia URL risolto'
    },
    image: {
      copyImage: 'Copia immagine',
      copyImageAddress: 'Copia indirizzo immagine',
      saveImageAs: 'Salva immagine come…'
    },
    edit: {
      cut: 'Taglia',
      paste: 'Incolla',
      selectAll: 'Seleziona tutto',
      addToDictionary: 'Aggiungi al dizionario'
    },
    page: {
      copyPageUrl: 'Copia URL della pagina',
      inspectElement: 'Analizza elemento'
    }
  },
  assistant: {
    thread: {
      loadingSession: 'Caricamento sessione',
      showEarlier: 'Mostra messaggi precedenti',
      loadingResponse: 'Hermes sta caricando una risposta',
      loadingLocalModel: (model: string) => `Caricamento di ${model} in memoria`,
      processingPrompt: 'Elaborazione del prompt',
      resumeWhenBackgroundDone: count =>
        count === 1
          ? 'Riprenderà al termine dell’attività in background.'
          : `Riprenderà al termine delle ${count} attività in background.`,
      thinking: 'Sta pensando',
      thought: 'Pensiero',
      thoughtBriefly: 'Ha riflettuto brevemente',
      thoughtFor: duration => `Ha riflettuto per ${duration}`,
      turnDuration: duration => `Questo turno ha richiesto ${duration}`,
      today: time => `Oggi, ${time}`,
      yesterday: time => `Ieri, ${time}`,
      copy: 'Copia',
      refresh: 'Aggiorna',
      moreActions: 'Altre azioni',
      branchNewChat: 'Ramifica in una nuova chat',
      react: 'Reagisci',
      dismissError: 'Ignora errore',
      responseStopped: 'Risposta interrotta',
      errorLayers: {
        auth: 'Problema di accesso',
        billing: 'Crediti esauriti',
        disk: 'Disco pieno',
        endpoint: 'Impossibile connettersi al tuo server di modelli',
        gateway: 'Hermes ha avuto un problema',
        generic: 'Hermes non è riuscito a completare questa risposta',
        provider: 'Il servizio di IA ha restituito un errore',
        runtime: 'Hermes ha avuto un problema',
        streaming: 'La risposta è stata interrotta'
      },
      errorLayerBodies: {
        auth: 'Il servizio di IA ha rifiutato il tuo accesso. Controlla le credenziali di questo provider e reinvia il messaggio.',
        billing: 'Il tuo account non ha crediti per questo provider. Ricarica o cambia provider e reinvialo.',
        disk: 'Il tuo disco è pieno, quindi Hermes non è riuscito a salvare questa conversazione. Libera spazio e riprova.',
        endpoint:
          'Hermes non riesce a connettersi al tuo server di modelli personalizzato. Verifica che sia in esecuzione e reinvia il messaggio.',
        gateway:
          'Hermes ha riscontrato un problema interno all’avvio di questa risposta. Reinvia il messaggio; se il problema persiste, invia una diagnostica.',
        generic: 'Qualcosa è andato storto mentre Hermes rispondeva. Riprova o copia i dettagli se il problema persiste.',
        provider:
          'Il servizio di IA non è riuscito a completare questa richiesta. Riprova tra poco o cambia provider.',
        runtime:
          'Hermes ha riscontrato un problema interno all’avvio di questa risposta. Reinvia il messaggio; se il problema persiste, invia una diagnostica.',
        streaming: 'La connessione si è interrotta prima che la risposta fosse completata. Riprova per inviarla di nuovo.'
      },
      errorCodes: {
        auth: {
          title: (provider: string) => `${provider} ha rifiutato il tuo accesso`,
          body: (provider: string) =>
            `Le credenziali salvate per ${provider} non sono state accettate. Correggile in Impostazioni o cambia provider e reinvia il messaggio.`
        },
        auth_permanent: {
          title: (provider: string) => `${provider} ha rifiutato il tuo accesso`,
          body: (provider: string) =>
            `Le credenziali salvate per ${provider} non sono valide o sono state revocate. Aggiornale o cambia provider e reinvia il messaggio.`
        },
        billing: {
          title: 'Crediti esauriti',
          body: (provider: string) =>
            `Il tuo account ${provider} non ha crediti. Ricarica o cambia provider e reinvialo.`
        },
        rate_limit: {
          title: 'Il servizio di IA è occupato',
          body: (provider: string) =>
            `${provider} sta limitando le richieste in questo momento. Attendi un minuto e riprova.`
        },
        upstream_rate_limit: {
          title: 'Il servizio di IA è occupato',
          body: (provider: string) =>
            `${provider} sta limitando le richieste in questo momento. Attendi un minuto e riprova.`
        },
        overloaded: {
          title: 'Il servizio di IA è sovraccarico',
          body: (provider: string) =>
            `${provider} ha problemi in questo momento. Riprova tra poco o cambia provider.`
        },
        server_error: {
          title: 'Il servizio di IA ha riscontrato un problema',
          body: (provider: string) =>
            `${provider} ha restituito un errore del server. Riprova tra poco o cambia provider.`
        },
        timeout: {
          title: 'Impossibile connettersi al servizio di IA',
          body: (provider: string) =>
            `Impossibile connettersi a ${provider} oppure non ha risposto in tempo. Controlla la tua connessione a internet e riprova.`
        },
        no_reply: {
          title: 'La risposta non è stata completata',
          body: 'Hermes ha completato questo turno senza risposta. Riprova per inviarla di nuovo.'
        },
        stream_drop: {
          title: 'La risposta è stata interrotta',
          body: 'La connessione si è interrotta prima che la risposta fosse completata. Riprova per inviarla di nuovo.'
        },
        upstream_blocked: {
          title: 'Un firewall ha bloccato la richiesta',
          body: (provider: string) =>
            `Un firewall o una CDN davanti a ${provider} ha bloccato la richiesta prima che arrivasse al modello; probabilmente la tua chiave va bene. Definisci un header User-Agent tramite gli extra_headers del provider in Impostazioni oppure cambia provider e reinvia il messaggio.`
        },
        ssl_cert_verification: {
          title: 'Connessione sicura non riuscita',
          body: (provider: string) =>
            `Hermes non è riuscito a verificare la connessione sicura con ${provider}. Controlla la configurazione di rete o del proxy, oppure cambia provider, e reinvia il messaggio.`
        },
        context_overflow: {
          title: 'Questa conversazione è troppo lunga',
          body: 'La conversazione non entra più nel modello. Comprimila o inizia una nuova chat e reinvia il messaggio.'
        },
        payload_too_large: {
          title: 'Questo messaggio è troppo grande',
          body: 'La richiesta era troppo grande per il modello. Comprimi la conversazione o inizia una nuova chat e reinvia il messaggio.'
        },
        model_not_found: {
          title: 'Questo modello non è disponibile',
          body: (provider: string) =>
            `${provider} non offre questo modello sul tuo account. Scegli un altro modello e reinvia il messaggio.`
        },
        provider_policy_blocked: {
          title: 'La configurazione del tuo account blocca questo modello',
          body: (provider: string) =>
            `${provider} non instraderebbe questa richiesta con la configurazione di dati o privacy del tuo account. Scegli un altro modello o cambia provider.`
        },
        content_policy_blocked: {
          title: 'Il servizio di IA ha rifiutato questa richiesta',
          body: (provider: string) => `${provider} non risponderebbe a questo messaggio. Modificalo e reinvialo.`
        },
        format_error: {
          title: 'Il servizio di IA ha rifiutato la richiesta',
          body: (provider: string) =>
            `${provider} non ha accettato il modo in cui questa richiesta è stata costruita. Cambia provider o invia una diagnostica per farla esaminare.`
        },
        truncated: {
          title: 'La risposta è rimasta incompleta',
          body: 'Il modello si è fermato prima di terminare. Riprova per ottenere una risposta completa.'
        },
        invalid_response: {
          title: 'Il servizio di IA ha inviato una risposta illeggibile',
          body: (provider: string) => `${provider} ha restituito qualcosa che Hermes non è riuscito a leggere. Riprova tra poco.`
        },
        empty_response: {
          title: 'Il servizio di IA ha inviato una risposta vuota',
          body: (provider: string) => `${provider} non ha restituito nulla per questo messaggio. Riprova tra poco.`
        },
        loop_error: {
          title: 'Hermes è rimasto bloccato in un ciclo',
          body: 'La risposta ripeteva gli stessi passaggi, quindi Hermes l’ha interrotta. Riprova o inizia una nuova chat se si ripete.'
        },
        SESSION_NOT_OWNED: {
          title: 'Questa chat è aperta altrove',
          body: 'Questa chat è aperta in un’altra finestra di Hermes o in un terminale. Chiudila lì e reinvia il messaggio, oppure inizia una nuova chat qui.'
        },
        disk_full: {
          title: 'Disco pieno',
          body: 'Il tuo disco è pieno, quindi Hermes non è riuscito a salvare questa conversazione. Libera spazio e riprova.'
        },
        free_tier_disabled: {
          title: 'L’uso di Hermes senza accesso è attualmente disattivato',
          body: 'Accedi con un account Nous per continuare a chattare; è gratis.'
        },
        free_tier_rate_limited: {
          title: 'Hai esaurito la quota per chattare senza accesso',
          body: 'Si rinnova a breve. Accedi con un account Nous per avere una quota maggiore; è gratis.'
        },
        free_tier_at_capacity: {
          title: 'Chattare senza accesso è molto richiesto in questo momento',
          body: 'Accedi per saltare la coda (è gratis) o riprova tra un po’.'
        },
        free_tier_model_not_free: {
          title: 'Quel modello non è disponibile senza accesso',
          body: 'Per ora Hermes usa il modello gratuito. Accedi con un account Nous per avere più modelli; è gratis.'
        },
        free_tier_route: {
          title: 'Hermes non è riuscito a raggiungere il modello gratuito tramite questa route',
          body: 'Accedi con un account Nous (è gratis) o controlla l’impostazione NOUS_INFERENCE_BASE_URL.'
        },
        free_tier_outage: {
          title: 'Il modello gratuito ha problemi a rispondere in questo momento',
          body: 'Reinvia il messaggio tra un minuto.'
        },
        free_tier_refused: {
          title: 'Hermes non è riuscito a inviarlo senza accesso',
          body: 'Accedere con un account Nous è gratis.'
        }
      },
      errorAuthKinds: {
        api_key: {
          title: (provider: string) => `${provider} ha rifiutato la tua chiave API`,
          body: (provider: string) =>
            `La chiave salvata per ${provider} non è valida o è stata revocata. Aggiornala e riprova.`
        },
        oauth: {
          title: (provider: string) => `La tua sessione di ${provider} è scaduta`
        }
      },
      errorDetails: 'Dettagli',
      errorGenericProvider: 'Il servizio di IA',
      errorToastTitle: 'Hermes non è riuscito a completare la risposta',
      errorRetry: 'Riprova',
      errorLimitResets: (time: string) => `Il limite verrà ripristinato alle ${time}`,
      errorRetryAtReset: (time: string) => `Riprova quando il limite verrà ripristinato (${time})`,
      errorRetryScheduled: (time: string, wait: string) => `Nuovo tentativo alle ${time}, tra ${wait}`,
      errorRetryScheduledCancel: 'Annulla',
      errorStartNewSession: 'Avvia una nuova sessione',
      errorSwitchProvider: 'Cambia provider',
      errorChooseModel: 'Scegli un modello',
      errorCompressConversation: 'Comprimi conversazione',
      errorCompressFailed: 'Impossibile comprimere la conversazione',
      errorOpenHermesFolder: 'Apri la cartella di Hermes',
      errorOpenHermesFolderFailed: 'Impossibile aprire la cartella di Hermes',
      errorUpdateApiKey: 'Aggiorna chiave API',
      errorSignInAgain: (provider: string) => `Accedi di nuovo a ${provider}`,
      errorSignInFreeTier: 'Accedi con un account Nous',
      errorOauthExpired: (provider: string) =>
        `La tua sessione di ${provider} è scaduta o è stata revocata. Accedi di nuovo per continuare a chattare.`,
      errorOpenLogs: 'Apri i log',
      errorOpenLogsFailed: 'Impossibile aprire la cartella dei log',
      errorOpenDesktopLogs: 'Apri i log di Desktop',
      errorCopyDiagnostics: 'Copia i dettagli dell’errore',
      errorSendDiagnostics: 'Invia diagnostica',
      filesChanged: count => (count === 1 ? '1 file modificato' : `${count} file modificati`),
      reviewChanges: 'Rivedi',
      readAloudFailed: 'Lettura ad alta voce non riuscita',
      preparingAudio: 'Preparazione audio...',
      stopReading: 'Interrompi lettura',
      readAloud: 'Leggi ad alta voce',
      copyFullResponse: 'Copia la risposta completa',
      readAloudFullResponseHint: 'Maiusc+clic: leggi la risposta completa',
      editMessage: 'Modifica messaggio',
      expandMessage: 'Espandi messaggio',
      scrollToBottom: 'Scorri verso il basso',
      stop: 'Interrompi',
      restorePrevious: 'Ripristina checkpoint precedente',
      restoreCheckpoint: 'Ripristina checkpoint',
      restoreFromHere: 'Ripristina il checkpoint ed esegui di nuovo da questo messaggio',
      restoreTitle: 'Ripristinare questo checkpoint?',
      restoreBody:
        'Tutto ciò che viene dopo questo messaggio viene eliminato dalla conversazione e il messaggio viene eseguito di nuovo da qui.',
      restoreConfirm: 'Ripristina ed esegui di nuovo',
      restoreNext: 'Ripristina checkpoint successivo',
      goForward: 'Avanti',
      sendEdited: 'Invia modifica',
      attachingFile: 'Allegamento file…'
    },
    approval: {
      gatewayDisconnected:
        'Hermes non è connesso in questo momento. Il comando continua ad attendere la tua risposta (finché non scade il tempo di approvazione). Riconnettiti e inviala di nuovo.',
      sendFailed: 'Impossibile inviare la tua risposta',
      reconnect: 'Riconnetti',
      timedOutSystemLine:
        'Tempo di approvazione scaduto: il comando non è stato eseguito. Chiedi a Hermes di riprovare o aumenta il limite in Impostazioni → Sicurezza → Tempo di approvazione.',
      openSafetySettings: 'Apri impostazioni di sicurezza',
      run: 'Esegui',
      command: 'Comando',
      moreOptions: 'Altre opzioni di approvazione',
      allowSession: 'Consenti per questa sessione',
      alwaysAllowMenu: 'Consenti sempre…',
      jumpToApproval: 'Approvazione necessaria',
      reject: 'Rifiuta',
      alwaysTitle: 'Consentire sempre questo comando?',
      alwaysDescription: pattern =>
        `Questo aggiunge il pattern “${pattern}” alla tua allowlist permanente (~/.hermes/config.yaml). Hermes non chiederà più conferma per comandi come questo, né in questa sessione né in quelle future.`,
      alwaysAllow: 'Consenti sempre'
    },
    clarify: {
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
      notDelivered:
        'Questa domanda non è arrivata all’app, quindi non si può rispondere qui. Premi Interrompi per terminare il turno e poi rispondi in chat.'
    },
    catalogInstall: {
      preparing: 'Preparazione dell’installazione…',
      install: 'Installa',
      advanced: 'Avanzate',
      skip: 'Salta',
      installing: 'Installazione…',
      installed: 'Installato',
      notInstalled: 'Non installato',
      failed: 'Non riuscito',
      showNames: 'mostra nomi',
      hideNames: 'nascondi nomi',
      skill: (name: string) => `skill ${name}`,
      kind: {
        plugin: 'plugin',
        skill: 'skill'
      },
      tier: {
        official: 'ufficiale',
        community: 'community'
      },
      targetProfile: (profile: string) => `Viene installato nel tuo profilo ${profile}`,
      sendFailed: 'Impossibile inviare la tua risposta. Riprova.',
      commitLabel: 'Commit',
      subdirLabel: 'Cartella',
      securityHeading: 'Sicurezza',
      scan: {
        passed: 'Analisi superata',
        warnings: 'L’analisi ha trovato avvisi',
        failed: 'Analisi non riuscita'
      },
      requirementsLabel: 'Richiede',
      credentialsHeading: 'Credenziali'
    },
    mcpSetup: {
      installTitle: 'Aggiungi server MCP',
      enableTitle: 'Attiva server MCP',
      authorizeTitle: 'Autorizza server MCP',
      installAction: 'Installa',
      enableAction: 'Attiva',
      authorizeAction: 'Autorizza',
      installed: server => `${server} installato`,
      enabled: server => `${server} attivato`,
      authorized: server => `${server} autorizzato`,
      failed: server => `Configurazione non riuscita per ${server}`,
      toolCount: count => (count === 1 ? '1 strumento' : `${count} strumenti`),
      envRequired: 'Compila prima le credenziali obbligatorie',
      sendFailed: 'Impossibile inviare la risposta di configurazione MCP',
      reloadFailed: 'Server salvato, ma il ricaricamento degli strumenti MCP non è riuscito — verranno caricati nella prossima sessione',
      gatewayDisconnected: 'Hermes non è connesso in questo momento. Riconnettiti e invialo di nuovo.'
    },
    tool: {
      copyCode: 'Copia codice',
      renderingImage: 'Rendering dell’immagine',
      copyOutput: 'Copia output',
      copyCommand: 'Copia comando',
      copyContent: 'Copia contenuto',
      copyUrl: 'Copia URL',
      copyResults: 'Copia risultati',
      copyQuery: 'Copia query',
      copyFile: 'Copia file',
      copyPath: 'Copia percorso',
      failedCalls: (count: number) =>
        `${count} ${count === 1 ? 'chiamata a strumento non riuscita' : 'chiamate a strumenti non riuscite'}`,
      skillActivity: {
        loading: 'Caricamento skill',
        loaded: 'Skill caricata',
        loadFailed: 'Impossibile caricare la skill',
        readingResource: 'Lettura risorsa della skill',
        readResource: 'Risorsa della skill letta',
        resourceFailed: 'Impossibile leggere la risorsa della skill',
        listing: 'Elenco delle skill',
        listed: 'Skill elencate',
        listFailed: 'Impossibile elencare le skill',
        unavailable: 'Risultato della skill non disponibile'
      },
      outputAlt: 'Output dello strumento',
      rawResponse: 'Risposta grezza',
      copyActivity: 'Copia attività',
      recoveredOne: 'Recuperato dopo 1 passo non riuscito',
      recoveredMany: count => `Recuperato dopo ${count} passi non riusciti`,
      failedOne: '1 passo non è riuscito',
      failedMany: count => `${count} passi non sono riusciti`,
      statusRunning: 'In esecuzione',
      statusError: 'Errore',
      statusRecovered: 'Recuperato',
      statusDone: 'Completato',
      resultUnavailable: 'Risultato non disponibile',
      resultInterrupted: 'Interrotto',
      memoryWriteNoted: 'Scrittura in memoria annotata',
      actions: {
        read: 'Ha letto',
        reading: 'Leggendo',
        opened: 'Ha aperto',
        opening: 'Aprendo',
        failedToOpen: 'Impossibile aprire',
        searched: 'Ha cercato',
        searching: 'Cercando',
        ran: 'Ha eseguito',
        running: 'Eseguendo',
        ranCode: 'Ha eseguito codice',
        runningCode: 'Eseguendo codice'
      },
      prefixes: {
        browser: 'Browser',
        web: 'Web'
      },
      titleTemplates: {
        actionCommand: (action, command) => `${action} ${command}`,
        actionQuoted: (action, value) => `${action} “${value}”`,
        actionTarget: (action, target) => `${action} ${target}`,
        prefixedDone: (prefix, action) => `${prefix} ${action}`,
        runningPrefixedTool: (prefix, action) => `Esecuzione di ${prefix.toLowerCase()} ${action.toLowerCase()}`,
        runningTool: action => `Esecuzione di ${action.toLowerCase()}`
      },
      titles: {
        browser_click: {
          done: 'Si è fatto clic su un elemento della pagina',
          pending: 'Fare clic su un elemento della pagina',
          pendingAction: 'Facendo clic'
        },
        browser_fill: {
          done: 'Campo del modulo compilato',
          pending: 'Compilare un campo del modulo',
          pendingAction: 'Completando'
        },
        browser_navigate: {
          done: 'Pagina aperta',
          pending: 'Aprire una pagina',
          pendingAction: 'Aprendo'
        },
        browser_snapshot: {
          done: 'Istantanea della pagina acquisita',
          pending: 'Acquisire un’istantanea della pagina',
          pendingAction: 'Acquisendo'
        },
        browser_take_screenshot: {
          done: 'Screenshot acquisito',
          pending: 'Acquisire uno screenshot',
          pendingAction: 'Acquisendo'
        },
        browser_type: {
          done: 'Testo scritto nella pagina',
          pending: 'Scrivere nella pagina',
          pendingAction: 'Scrivendo'
        },
        clarify: {
          done: 'È stata fatta una domanda',
          pending: 'Fare una domanda',
          pendingAction: 'Chiedendo'
        },
        cronjob: {
          done: 'Attività cron completata',
          pending: 'Pianificare un’attività cron',
          pendingAction: 'Pianificando'
        },
        edit_file: {
          done: 'File modificato',
          pending: 'Modificare un file',
          pendingAction: 'Modificando'
        },
        execute_code: {
          done: 'Codice eseguito',
          pending: 'Eseguire codice',
          pendingAction: 'Eseguendo'
        },
        image_generate: {
          done: 'Immagine generata',
          pending: 'Generare un’immagine',
          pendingAction: 'Generando'
        },
        list_files: {
          done: 'Elenco dei file ottenuto',
          pending: 'Elencare i file',
          pendingAction: 'Elencando'
        },
        memory: {
          done: 'Salvato in memoria',
          pending: 'Salvare in memoria',
          pendingAction: 'Salvando'
        },
        patch: {
          done: 'File patchato',
          pending: 'Applicare una patch al file',
          pendingAction: 'Applicando una patch'
        },
        read_file: {
          done: 'File letto',
          pending: 'Leggere un file',
          pendingAction: 'Leggendo'
        },
        search_files: {
          done: 'Ricerca dei file completata',
          pending: 'Cercare file',
          pendingAction: 'Cercando'
        },
        session_search_recall: {
          done: 'Ricerca nella cronologia delle sessioni completata',
          pending: 'Cercare nella cronologia delle sessioni',
          pendingAction: 'Cercando'
        },
        terminal: {
          done: 'Comando eseguito',
          pending: 'Eseguire un comando',
          pendingAction: 'Eseguendo'
        },
        todo: {
          done: 'Attività aggiornate',
          pending: 'Aggiornare le attività',
          pendingAction: 'Aggiornando'
        },
        vision_analyze: {
          done: 'Immagine analizzata',
          pending: 'Analizzare un’immagine',
          pendingAction: 'Analizzando'
        },
        web_extract: {
          done: 'Pagina web letta',
          pending: 'Leggere una pagina web',
          pendingAction: 'Leggendo'
        },
        web_search: {
          done: 'Ricerca web completata',
          pending: 'Cercare sul web',
          pendingAction: 'Cercando'
        },
        write_file: {
          done: 'File scritto',
          pending: 'Scrivere un file',
          pendingAction: 'Scrivendo'
        }
      }
    }
  },
  prompts: {
    gatewayDisconnected: 'Hermes non è connesso in questo momento. Riconnettiti e invialo di nuovo.',
    reconnect: 'Riconnetti',
    sudoSendFailed: 'Impossibile inviare la password sudo',
    secretSendFailed: 'Impossibile inviare il segreto',
    sudoTitle: 'Password di amministratore',
    sudoDesc:
      'Controlla il comando prima di inserire la tua password sudo. La password viene inviata all’agente che la esegue e viene salvata in cache per questa sessione.',
    sudoCommandUnavailable: 'Questo agente non ha indicato il comando. Annulla se non puoi verificarlo nella conversazione.',
    sudoInstallDesc:
      'Hermes ha bisogno della tua password sudo per installare i pacchetti di Bot Screen (TigerVNC + Xfce) sull’host del gateway. Viene inviata solo a quell’host.',
    sudoPlaceholder: 'password sudo',
    secretTitle: 'È richiesto un segreto',
    secretDesc: 'Hermes ha bisogno di una credenziale per continuare.',
    secretPlaceholder: 'valore segreto',
    vaultUnlockSendFailed: 'Impossibile inviare la password principale',
    vaultUnlockTitle: (name: string) => `Sblocca ${name}`,
    vaultUnlockDesc: (name: string) =>
      `L’agente vuole accedere a un sito con un accesso salvato in ${name}. Inserisci la tua password principale per sbloccarlo durante questa sessione: va direttamente a ${name} su questo computer e non viene mai salvata né mostrata all’agente.`,
    vaultUnlockPlaceholder: 'Password principale',
    vaultUnlockKeepLocked: 'Mantieni bloccato',
    vaultUnlockConfirm: 'Sblocca',
    vaultSaveSendFailed: 'Impossibile salvare l’accesso',
    vaultSaveTitle: (site: string) => `Salvare il tuo accesso a ${site}?`,
    vaultSaveDesc: (origin: string) =>
      `Hermes è arrivato a una pagina di accesso su ${origin} e non ha un accesso per essa. Inseriscilo una volta qui; viene cifrato su questo computer e compilato nella pagina senza che il modello veda mai la password.`,
    vaultSaveIdentifierLabel: 'Email o nome utente',
    vaultSaveIdentifierPlaceholder: 'tu@esempio.com',
    vaultSavePasswordPlaceholder: 'Password',
    vaultSaveFootnote: 'Gestisci gli accessi salvati in Impostazioni → Password e accessi.',
    vaultSaveDecline: 'Non salvare',
    vaultSaveConfirm: 'Salva e accedi',
    vaultCodeSendFailed: 'Impossibile inviare il codice',
    vaultCodeTitle: (site: string) => `Codice di verifica di ${site}`,
    vaultCodeDesc: (site: string) =>
      `${site} richiede un codice a uso singolo (SMS, email o app di autenticazione). Inseriscilo qui e Hermes lo scrive nella pagina; il modello non lo vede mai.`,
    vaultCodeLabel: 'Codice',
    vaultCodeFootnote:
      'Consiglio: salva la chiave dell’autenticatore con questo accesso in Impostazioni → Password e accessi e Hermes inserirà i codici per te.',
    vaultCodeSkip: 'Salta',
    vaultCodeConfirm: 'Inserisci codice'
  },
  desktop: {
    audioReadFailed: 'Impossibile leggere l’audio registrato',
    sessionUnavailable: 'Sessione non disponibile',
    createSessionFailed: 'Impossibile creare una nuova sessione',
    promptFailed: 'Prompt non riuscito',
    providerCredentialRequired: 'Aggiungi una credenziale del provider prima di inviare il tuo primo messaggio.',
    emptySlashCommand: 'comando slash vuoto',
    desktopCommands: 'Comandi del desktop',
    skillCommandsAvailable: count => `${count} comandi skill disponibili.`,
    warningLine: message => `avviso: ${message}`,
    yoloArmed: 'YOLO attivato per questa chat',
    yoloOff: 'YOLO disattivato',
    yoloSystem: active => `YOLO ${active ? 'attivato' : 'disattivato'} per questa sessione`,
    yoloTitle: 'YOLO',
    yoloToggleFailed: 'Impossibile cambiare lo stato di YOLO',
    profileStatus: current =>
      `Profilo: ${current}. Usa /profile <nome> o il selettore di "Nuova sessione" per avviare una chat in un altro profilo.`,
    unknownProfile: 'Profilo sconosciuto',
    noProfileNamed: (target, available) => `Non c’è nessun profilo chiamato "${target}". Disponibili: ${available}`,
    newChatsProfile: name => `Le nuove chat useranno il profilo ${name}.`,
    setProfileFailed: 'Impossibile impostare il profilo',
    sttDisabled: 'La trascrizione vocale è disabilitata nella configurazione.',
    stopFailed: 'Impossibile interrompere',
    regenerateFailed: 'Impossibile rigenerare',
    editFailed: 'Impossibile modificare',
    editTurnUnavailable: 'Questo turno non è più nella cronologia del server (potrebbe essere stato compresso).',
    resumeFailed: 'Impossibile riprendere',
    readOnlyTranscriptTitle: 'Aperto in modalità di sola lettura',
    readOnlyTranscriptBody:
      'Nessun backend connesso rivendica ancora questa vecchia chat, quindi è stata aperta come trascrizione di sola lettura. La cronologia è intatta; l’invio è disattivato finché un backend non la rivendica.',
    readOnlyTranscriptSendBlocked:
      'Questa chat è aperta come trascrizione di sola lettura: l’invio è disattivato.',
    resumeStrandedTitle: 'Impossibile caricare questa sessione',
    resumeStrandedBody:
      'Impossibile connettersi a questa sessione e i tentativi automatici sono esauriti. Verifica che il gateway sia in esecuzione e riprova.',
    poolSlotTimeoutBody:
      'Ci sono troppi bot in esecuzione contemporaneamente per il limite di questo computer. Aumenta il limite in Impostazioni → Avanzato oppure attendi che uno termini e riprova.',
    poolSlotTimeoutOpenSettings: 'Apri impostazioni avanzate',
    resumeRetry: 'Riprova',
    nothingToBranch: 'Non c’è niente da branchiare',
    branchNeedsChat: 'Avvia o riprendi una chat prima di creare un branch.',
    sessionBusy: 'Sessione occupata',
    branchStopCurrent: 'Interrompi il turno corrente prima di creare un branch da questa chat.',
    branchNoText: 'Questo messaggio non ha testo da cui creare un branch.',
    branchTitle: n => `Bozza: branch n. ${n}`,
    branchFailed: 'Impossibile creare il branch',
    deleteFailed: 'Impossibile eliminare',
    archived: 'Archiviato',
    archiveFailed: 'Impossibile archiviare',
    restored: 'Ripristinato',
    unarchiveFailed: 'Impossibile annullare l’archiviazione',
    cwdChangeFailed: 'Impossibile cambiare la directory di lavoro',
    cwdStagedTitle: 'Directory di lavoro preparata',
    cwdStagedMessage: 'Riavvia il backend desktop per applicare le modifiche di cwd a questa sessione attiva.',
    modelSwitchConfirmBody: 'Questa modifica del modello richiede conferma.',
    modelSwitchConfirmLabel: 'Cambia comunque',
    modelSwitchConfirmTitle: (model: string) => `Passare a ${model}?`,
    modelSwitchConfirmTitleFallback: 'Cambiare modello?',
    modelSwitchFailed: 'Impossibile cambiare modello',
    modelSwitchKeepLabel: 'Mantieni il modello attuale',
    modelSwitchStaleNotice: 'La selezione è cambiata: la modifica del modello non è stata applicata.',
    hydrationSyncing: (profile: string) => `Sincronizzazione di ${profile}\u2026`,
    sessionExported: 'Sessione esportata',
    sessionExportFailed: 'Impossibile esportare la sessione',
    imageSaved: 'Immagine salvata',
    downloadStarted: 'Download avviato',
    restartToUseSaveImage: 'Riavvia Hermes Desktop per usare Salva immagine.',
    restartToSaveImages: 'Riavvia Hermes Desktop per salvare le immagini',
    imageDownloadFailed: 'Download dell’immagine non riuscito',
    openImage: 'Apri immagine',
    downloadImage: 'Scarica immagine',
    savingImage: 'Salvataggio immagine',
    imagePreviewFailed: 'Anteprima dell’immagine non riuscita',
    imageAttach: 'Allega immagine',
    imageWriteFailed: 'Impossibile scrivere l’immagine su disco.',
    imageAttachFailed: 'Impossibile allegare l’immagine',
    pastedContent: 'Contenuto incollato',
    pasteAttachFailed: 'Impossibile allegare il testo incollato',
    attachImages: 'Allega immagini',
    clipboard: 'Appunti',
    noClipboardImage: 'Nessuna immagine trovata negli appunti',
    clipboardPasteFailed: 'Impossibile incollare dagli appunti',
    dropFiles: 'Rilascia i file',
    handoff: {
      pickPlatform: 'Scegli una destinazione',
      success: platform => `Trasferito a ${platform}. Puoi riprendere qui quando vuoi.`,
      systemNote: platform => `↻ Trasferito a ${platform}; puoi riprendere qui quando vuoi.`,
      failed: error => `Il trasferimento non è riuscito: ${error}`,
      timedOut:
        'Hermes non è riuscito a raggiungere la tua connessione di messaggistica. Avviala da Impostazioni → Messaggistica e riprova il trasferimento.',
      startMessaging: 'Avvia messaggistica'
    }
  },
  tips: {
    close: 'Non mostrare più questo consiglio',
    items: {
      'new-session': {
        title: 'Parti da zero',
        text: 'Una nuova chat ha il proprio contesto, terminale e directory di lavoro.'
      },
      skills: {
        title: 'Insegnagli una volta sola',
        text: 'Le skill sono cartelle di istruzioni che Hermes carica quando il lavoro le richiede.'
      },
      messaging: {
        title: 'Hermes lontano dal tuo desktop',
        text: 'Collega Telegram, Discord, Slack e altro: lo stesso agente, la stessa memoria.'
      },
      artifacts: {
        title: 'Tutto ciò che Hermes ha creato',
        text: 'Immagini, file e link di ogni sessione, indicizzati in un unico posto.'
      },
      cron: {
        title: 'Lavoro che si esegue da solo',
        text: 'Pianifica un prompt ogni ora, ogni notte o con un’espressione cron.'
      },
      'command-palette': {
        title: 'Una casella per tutto',
        text: 'Sessioni, configurazione, skill e comandi rispondono alla palette.'
      },
      profiles: {
        title: 'I profili sono indipendenti',
        text: 'Ognuno è un Hermes a sé: le sue chiavi, la sua memoria, le sue sessioni.'
      },
      'composer-mentions': {
        title: 'Allega e comanda',
        text: 'Scrivi @ per portare un file nella conversazione e / per eseguire un comando.'
      },
      'local-runtime-update': {
        title: 'È disponibile un aggiornamento del motore locale',
        text: 'Aggiorna il motore che esegue i tuoi modelli locali. Le richieste locali attive possono essere interrotte.',
        action: 'Aggiorna ora'
      },
      'local-setup': {
        title: 'Questo computer può eseguire modelli in locale',
        text: 'Il tuo hardware può servire un modello locale. Le chat restano sul tuo computer e non costano nulla.',
        action: 'Configura'
      },
      'right-pane': {
        title: 'Il pannello di lavoro',
        text: 'File, terminale, revisione e il browser integrato condividono il lato destro.'
      }
    }
  },
  errors: {
    genericFailure: 'Qualcosa è andato storto',
    boundaryTitle: 'Qualcosa si è rotto nell’interfaccia',
    boundaryDesc: 'La vista ha riscontrato un errore imprevisto. Le tue chat e la configurazione sono al sicuro.',
    boundaryDetails: 'Dettagli',
    sendDiagnostics: 'Invia diagnostica',
    reloadWindow: 'Ricarica la finestra',
    openLogs: 'Apri log'
  },
  ui: {
    search: {
      clear: 'Cancella ricerca'
    },
    logs: {
      bottom: 'Vai alla fine',
      search: 'Cerca nei log…',
      top: 'Vai all’inizio'
    },
    pagination: {
      label: 'paginazione',
      previous: 'Precedente',
      previousAria: 'Vai alla pagina precedente',
      next: 'Successiva',
      nextAria: 'Vai alla pagina successiva'
    },
    sidebar: {
      title: 'Barra laterale',
      description: 'Mostra la barra laterale mobile.',
      toggle: open => `${open ? 'Mostra' : 'Nascondi'} barra laterale`
    }
  }
} satisfies TranslationOverrides

export const it = defineLocale(itOverrides)
