import type { TranslationOverrides } from './define-locale'
import { itClarify } from './it_clarify'
import { itThemePicker } from './it_theme_picker'

export const itAssistant = {
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
    skillsLoadFailed: 'Impossibile caricare le skill',
    toolsetsRefreshFailed: 'Impossibile aggiornare i set di strumenti',
    skillEnabled: 'Skill attivata',
    skillDisabled: 'Skill disattivata',
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
    skillUpdated: 'Skill aggiornata',
    edit: 'Modifica',
    archive: 'Archivia',
    skillArchivedTitle: 'Skill archiviata',
    skillArchivedMessage: 'Puoi ripristinarla con hermes curator restore.',
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
      commandDetails: 'Dettagli del comando',
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
    clarify: itClarify,
    themePicker: itThemePicker,
    catalogInstall: {
      preparing: 'Preparazione dell’installazione…',
      install: 'Installa',
      advanced: 'Avanzate',
      skip: 'Salta',
      installing: 'Installazione…',
      installed: 'Installato',
      notInstalled: 'Non installato',
      alreadyInstalled: 'Già installato; lasciato così com’è',
      notEnabled: 'Installato ma non attivato',
      envVar: name => `variabile d’ambiente ${name}`,
      missingEnv: names => `Imposta ${names} per completare la configurazione`,
      requiresHermes: range => `Hermes ${range}`,
      serverNotConnected: (server, reason) =>
        `Il server MCP ${server} non è connesso${reason ? `: ${reason}` : ''}`,
      phase: {
        downloading: 'Scaricamento…',
        loading_tools: 'Caricamento degli strumenti…',
        python_packages: 'Installazione dei pacchetti Python…'
      },
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
  appTour: {
    sessions: {
      title: 'Le tue chat',
      text: 'Ogni conversazione si trova qui. Cerca, fissa o riapri una qualsiasi.'
    },
    newSession: {
      title: 'Ricomincia da capo',
      text: 'Una nuova sessione ha un proprio contesto. Usane una per attività.'
    },
    composer: {
      title: 'Chiedi qui',
      text: 'Dì cosa vuoi ottenere. Digita @ per portare un file.'
    },
    model: {
      title: 'Selettore del modello',
      text: 'Sceglie quale modello ti risponde.'
    },
    modelLocal: 'Questo computer può eseguirne uno in locale: Impostazioni → Provider → Modelli locali.',
    capabilities: {
      title: 'Capacità',
      text: 'Skill, strumenti e plugin che Hermes può usare. Aggiungine altri qui.'
    },
    messaging: {
      title: 'Messaggistica',
      text: 'Raggiungi Hermes da Telegram, Slack, Discord e altri.'
    },
    rightPane: {
      title: 'Il pannello di lavoro',
      text: 'Apre file, terminale, revisione e il browser integrato sulla destra.'
    }
  },
  externalOpenFailed: {
    title: 'Impossibile aprire questo link',
    message: 'Nessun browser è registrato per aprire questo indirizzo. Copia il link e aprilo manualmente.',
    copyUrl: 'Copia link',
    close: 'Chiudi',
    missing: {
      title: 'File non trovato',
      message: 'Questo file non esiste: potrebbe essere stato eliminato o spostato, oppure si trova su un’altra macchina.'
    }
  },
  skillDeepLink: {
    destinationChanged: 'La destinazione è cambiata. Chiudi questa finestra e apri di nuovo il link di installazione.',
    installDescription: 'Questa skill sarà disponibile nelle nuove sessioni. Installa solo da fonti affidabili.',
    installTitle: name => `Installare “${name}”?`,
    installTo: 'Installa in',
    installed: 'Installata',
    installing: 'Installazione…',
    installComplete: name => `“${name}” installata`,
    source: 'Sorgente',
    thisComputer: 'Questo computer'
  },
} satisfies TranslationOverrides
