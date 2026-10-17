import type { TranslationOverrides } from './define-locale'

export const itSettingsModels = {
  settings: {
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
      inheritMainEffort: 'eredita · sforzo del modello principale',
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
      followTask: task => `Segui ${task}`,
      inheritsFrom: task => `eredita ${task}`,
      tasks: {
        vision: {
          label: 'Visione',
          hint: 'Analisi delle immagini'
        },
        voice_chat: {
          label: 'Chat vocale',
          hint: 'Risposte parlate in modalità vocale'
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
        'product-default': 'Il modello predefinito per questa macchina, scelto dal suo produttore.'
      },
      downloadStatusRunning: 'Scaricamento',
      downloadPauseAction: 'Pausa',
      downloadPausedLabel: 'In pausa',
      downloadResumeAction: 'Riprendi',
      downloadEta: time => `~${time} rimanenti`,
      downloadEtaHours: (hours, minutes) => (minutes ? `${hours} h ${minutes} min` : `${hours} h`),
      downloadEtaMinutes: count => `${count} min`,
      downloadEtaSeconds: count => `${count} s`,
      downloadSpeed: rate => `${rate}`,
      downloadPauseFailed: model => `Impossibile mettere in pausa il download di ${model}`,
      downloadResumeFailed: model => `Impossibile riprendere il download di ${model}`,
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
          `${who} Riconnettiti da Impostazioni → Gateway per autorizzare di nuovo questo dispositivo.`,
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
      nousAuthDoneMessage: 'I backend del tuo abbonamento sono ora attivi.',
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
  }
} satisfies TranslationOverrides

export const itModelSections = {
  modelAssignment: {
    saveFailed: 'Hermes non ha salvato quella modifica al modello.',
    confirmTitle: 'Avviso sulla selezione del modello',
    confirmDetail: 'Conferma solo se accetti questo compromesso.',
    confirmAction: 'Conferma',
    declined: 'Cambio del modello annullato: hai rifiutato l’avviso del livello con addestramento sui dati.'
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
    emptyDesc: 'Quando un turno delega lavoro, gli agenti figli mostrano qui i loro progressi in streaming.',
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
} satisfies TranslationOverrides
