import { defineFieldCopy } from '@/app/settings/field-copy'

import type { TranslationOverrides } from './define-locale'

export const itSettings = {
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
    closeSettings: 'Chiudi impostazioni',
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
      keysSettings: 'Impostazioni',
      mcp: 'MCP',
      plugins: 'Plugin',
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
        runtime: 'runtime'
      },
      installModal: {
        installFromGit: 'Installa da Git',
        reviewRepository: 'Esamina il repository',
        repoPlaceholder: 'https://github.com/proprietario/repo',
        title: 'Installa plugin',
        description: 'Controlla cosa contiene questo repository prima di installare qualcosa.',
        repoLabel: 'Repository',
        includesHeading: 'Questo pacchetto include',
        agentLabel: 'Plugin dell’agente',
        desktopLabel: 'Interfaccia desktop',
        profileLabel: 'Installa per il profilo',
        agentTargetLocal: (profile, dir) => `Si installa nel backend ${profile} (${dir})`,
        agentTargetRemote: profile => `Si installa nel backend ${profile} connesso`,
        installsOnly: profile => `Si installa per ${profile}.`,
        installsAndEnables: profile => `Si installa per ${profile} e si attiva subito.`,
        reviewedByNous: 'Verificato da Nous.',
        reviewedPin: sha => (sha ? `Pin verificato ${sha}` : 'Pin verificato'),
        unverified: 'Non verificato',
        verified: 'Verificato',
        versionLabel: 'Versione',
        latestCommit: 'Ultimo commit',
        nothingToInstall: 'Questo pacchetto non ha niente da installare qui.',
        advanced: 'Avanzate',
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
        securityIntro:
          'Installa solo da fonti attendibili; controlla il repository qui sotto se vuoi vedere cosa verrà aggiunto.',
        sourceHeading: 'Codice sorgente',
        viewRepository: 'Visualizza repository',
        viewPluginFiles: 'Visualizza i file del plugin',
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
        'Dì “accedi a GitHub” e l’agente lo farà per te. La prima volta che incontra una pagina di accesso, ti chiederà i dati subito lì; dopo, funzionerà semplicemente. Le password vengono cifrate su questo computer e inserite direttamente nella pagina: il modello non le vede mai.',
      count: n => `${n} element${n === 1 ? 'o' : 'i'} salvat${n === 1 ? 'o' : 'i'}`,
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
      originPlaceholderCheckout: 'https://negozio.example.com',
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
        count: n => `${n} mascotte.`,
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
        streaming: 'Trascrizione live',
        echoTranscripts: 'Eco delle trascrizioni',
        provider: 'Provider di voce in testo',
        local: {
          model: 'Modello locale di trascrizione',
          language: 'Lingua della trascrizione'
        },
        openai: {
          model: 'Modello STT di OpenAI',
          streamingModel: 'Modello di trascrizione live di OpenAI'
        },
        xai: {
          model: 'Modello STT xAI'
        },
        deepinfra: {
          model: 'Modello STT DeepInfra'
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
        streaming:
          'Mostra il testo mentre parli (OpenAI, xAI, ElevenLabs). In caso di qualsiasi errore torna alla registrazione.',
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
      managedBody: 'Questa installazione è gestita dal sistema, quindi Hermes non può rimuoversi da solo.',
      dataKept: path =>
        `La tua configurazione, le chat e i segreti si trovano in ${path}. Rimuovere l’app non li elimina.`,
      openAppsSettings: 'Apri impostazioni App',
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
      updates: 'Aggiornamenti',
      channel: {
        title: 'Canale di aggiornamento',
        description:
          'Le installazioni stabili ricevono solo release pubblicate. Con l’opzione «ogni commit» segui il branch main man mano che le modifiche arrivano.',
        detailsLabel: 'Canale',
        stable: 'Release stabili',
        main: 'Ogni commit',
        change: 'Cambia',
        failed: 'Impossibile cambiare il canale di aggiornamento',
        latestRelease: v => `Sei sull’ultima release stabile (${v}).`,
        branch: b => `Branch: ${b}`,
        rollbackTitle: v => `Torna alla release stabile ${v}`,
        rollbackBody: v =>
          `Questa build è più recente della release stabile ${v}. Installarla riporta Hermes a quella release.`,
        rollbackAction: v => `Torna a ${v}`,
        stableConfirmTitle: 'Passare alle release stabili?',
        stableConfirmBody:
          'Hermes passerà all’ultima release pubblicata, che può essere più vecchia del codice che usi ora. I dati scritti da codice più recente potrebbero non essere leggibili correttamente: fai prima un backup.',
        stableConfirm: 'Passa alla stabile',
        allChanges: 'Vedi tutte le modifiche'
      }
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
      systemDefault: 'Predefinito di sistema',
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
      showOptions: 'Mostra opzioni',
      alwaysExternalLinksTitle: 'Apri sempre i link nel browser esterno',
      alwaysExternalLinksDesc:
        'Apre ogni link su cui fai clic nel browser di sistema invece che nel browser integrato. “Apri nel browser integrato” nel menu contestuale continua a funzionare.',
      voiceShortcutHintTitle: 'Scorciatoia di registrazione vocale',
      voiceShortcutHintDesc:
        'Imposta la scorciatoia di registrazione vocale in Impostazioni → Scorciatoie da tastiera (“Avvia / interrompi conversazione vocale”). Il valore di configurazione voice.record_key si applica solo a CLI e TUI.'
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
        'Locale per impostazione predefinita. Usa remoto quando questa app deve controllare un backend di Hermes altrove. Le connessioni al gateway sono a livello di macchina; i profili vengono scoperti dai gateway a cui ti connetti.',
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
  }
} satisfies TranslationOverrides
