import { defineFieldCopy } from '@/app/settings/field-copy'

import { defineLocale, type TranslationOverrides } from './define-locale'
import { introDe } from './intro-de'

export const deOverrides = {
  intro: introDe,
  connectors: {
    title: 'Verbinden Sie Ihre Apps',
    connect: 'Verbinden',
    skip: 'Nicht jetzt',
    cancel: 'Warten beenden',
    retry: 'Erneut versuchen',
    grant: 'Neu verbinden',
    connected: 'Verbunden',
    checking: 'Ihre Apps werden geprüft…',
    notConnected: 'Verbindung fehlgeschlagen',
    skipped: 'Übersprungen',
    disabled: 'Nicht verfügbar',
    failed: 'Verbindung fehlgeschlagen',
    needsAuth: 'Zugriff abgelaufen',
    opening: 'Anmeldung wird geöffnet…',
    waiting: 'Schließen Sie die Verbindung im Browser ab…',
    timeout: 'Freigabe steht noch aus.',
    refresh: 'Status aktualisieren',
    connectError: 'Freigabe konnte nicht gestartet werden. Versuchen Sie es erneut.',
    connectErrorFor: app => `Die Freigabe für ${app} konnte nicht gestartet werden.`,
    unavailable: 'Verbindungen sind für diese Session nicht verfügbar.',
    ownerMissing: 'Öffnen Sie diese Konversation erneut, um ihre Verbindungen zu verwalten.',
    search: 'App finden',
    empty: 'Keine passende App',
    disclaimer: 'Das Verbinden ist freiwillig. Geben Sie nur den Apps Zugriff, die Hermes verwenden soll.',
    execution: 'Verbindungs-Tools',
    setup: server => `${server} einrichten`,
    openInBrowser: 'Im Browser öffnen',
    setupCancel: 'Abbrechen',
    authorizedToolsUnavailable: 'Autorisiert. Tools nicht verfügbar.',
    required: 'Erforderlich'
  },
  connectorsPage: {
    title: 'Konnektoren',
    searchPlaceholder: (count: number) => `${count} Apps durchsuchen`,
    filterCategory: 'Kategorie',
    categoryAll: 'Alle Kategorien',
    uncategorised: 'Ohne Kategorie',
    residencyLocal: 'Auf diesem Gerät',
    segment: {
      all: 'Alle',
      available: 'Verfügbar',
      connected: 'Verbunden',
      off: 'Ausgeschaltet'
    },
    group: {
      connected: 'Verbunden',
      connectedNote: 'Fehlerhafte Verbindungen zuerst.',
      available: 'Verfügbar',
      off: 'Ausgeschaltet',
      offNote: 'Anmeldungen bleiben erhalten.'
    },
    card: {
      kindManaged: 'Verwaltet',
      kindCatalog: 'MCP · Katalog',
      kindCustom: 'MCP · Benutzerdefiniert',
      kindPlugin: (plugin: string) => `MCP · Plugin ${plugin}`,
      inCatalog: 'Im Hermes-Katalog',
      hostedTwin: 'Verwaltete Version verfügbar',
      alsoLocal: 'Läuft auch auf diesem Gerät',
      open: (name: string) => `${name} öffnen`,
      turnServerOn: (name: string) => `${name} einschalten`,
      turnServerOff: (name: string) => `${name} ausschalten`,
      state: {
        accessExpired: 'Zugriff abgelaufen',
        available: 'Verfügbar',
        connected: 'Verbunden',
        connecting: 'Verbindet',
        connectionUnknown: 'Status unbekannt',
        couldNotConnect: 'Verbindung fehlgeschlagen',
        offByYourOrganisation: 'Von Ihrer Organisation deaktiviert',
        offForYou: 'Für Sie deaktiviert',
        serverConnecting: 'Verbindet…',
        serverError: 'Fehler',
        serverNeedsAuth: 'Authentifizierung erforderlich',
        serverOff: 'Aus',
        serverOn: 'An',
        serverOnUnused: 'An, ungenutzt'
      },
      fact: {
        tools: (count: number) => `${count} Tool${count === 1 ? '' : 's'}`,
        toolsOff: (count: number) => `${count} Tool${count === 1 ? '' : 's'} aus`,
        toolsOn: (count: number) => `${count} Tool${count === 1 ? '' : 's'} an`,
        toolsSomeOn: (total: number, on: number) => `${total} Tools, ${on} an`
      },
      verb: {
        authenticate: 'Authentifizieren',
        connect: 'Verbinden',
        install: 'Installieren',
        openLogs: 'Logs öffnen',
        reconnect: 'Neu verbinden',
        stopWaiting: 'Warten beenden',
        tryAgain: 'Erneut versuchen',
        turnBackOn: 'Wieder einschalten'
      },
      reason: {
        finishSignIn: 'Schließen Sie die Anmeldung in Ihrem Browser ab.',
        reconnect: 'Verbinden Sie sich neu, damit diese App weiter funktioniert.',
        serverError: 'Der Server hat die Verbindung abgelehnt.',
        serverNeedsAuth: 'Melden Sie sich an, damit dieser Server antworten kann.'
      }
    },
    page: {
      loading: 'Katalog und Server auf diesem Computer werden gelesen',
      emptyTitle: 'Noch keine Apps. Fügen Sie einen Server auf diesem Computer hinzu, um loszulegen.',
      noMatchTitle: 'Keine passenden Apps',
      noMatchBody: 'Keine Treffer. Verweisen Sie Hermes auf Ihren eigenen MCP-Server, um ihn hinzuzufügen.',
      clearSearch: 'Suche löschen',
      hostedFailedTitle: 'Die gehosteten Apps sind nicht erreichbar.',
      hostedFailedBody:
        'Die Server auf diesem Computer sind nicht betroffen und laufen weiter. Nichts wurde ausgeschaltet.',
      retry: 'Erneut versuchen',
      matchesElsewhere: (count: number) => `${count} weitere${count === 1 ? 'r' : ''} Treffer in anderen Gruppen.`,
      showAllMatches: 'Alle Treffer anzeigen',
      segmentNoMatch: (segment: string) => `Kein Treffer in ${segment}, daher werden alle Treffer angezeigt.`,
      freeTierNote: 'Verbindungen bleiben auf diesem Computer, bis Sie sich anmelden.',
      signInLine: 'Melden Sie sich bei Nous an, um verwaltete Apps zu nutzen.',
      signIn: 'Anmelden',
      managedUnavailable: 'Verwaltete Apps sind für dieses Konto noch nicht verfügbar.',
      writeFailed: 'Diese Änderung wurde nicht gespeichert.',
      refreshFailed: 'Die Tool-Liste wurde nicht aktualisiert.',
      disconnectNoAccount:
        'Hermes hat hier kein Konto zum Trennen. Aktualisieren Sie die Seite und versuchen Sie es erneut.',
      disconnectRefused:
        'Nous kann diese Anmeldung gerade nicht entfernen. Schalten Sie die App stattdessen mit dem Schalter aus oder versuchen Sie es später erneut.'
    },
    add: {
      action: 'Eigenen hinzufügen',
      title: 'Mit einem benutzerdefinierten MCP verbinden',
      hint: 'ein neuer Eintrag in mcp.json auf diesem Gerät',
      pasteLabel: 'Befehl oder Snippet einfügen',
      pastePlaceholder: 'npx -y @modelcontextprotocol/server-filesystem /pfad/zum/ordner',
      pasteNoMatch: 'Hier ist kein Server erkennbar. Füllen Sie stattdessen die Felder unten aus.',
      name: 'Name',
      nameTaken: 'Dieser Name wird bereits verwendet.',
      type: 'Typ',
      typeStdio: 'STDIO',
      typeHttp: 'Streamable HTTP',
      command: 'Startbefehl',
      args: 'Argumente',
      addArg: '+ Argument hinzufügen',
      envVars: 'Umgebungsvariablen',
      addEnvVar: '+ Umgebungsvariable hinzufügen',
      passthrough: 'Durchreichen von Umgebungsvariablen',
      addPassthrough: '+ Variable hinzufügen',
      cwd: 'Arbeitsverzeichnis',
      url: 'URL',
      headers: 'Header',
      addHeader: '+ Header hinzufügen',
      auth: 'Authentifizierung',
      authNone: 'Keine',
      authOauth: 'OAuth',
      authBearer: 'Bearer-Token',
      keyPlaceholder: 'SCHLÜSSEL',
      valuePlaceholder: 'Wert',
      removeRow: 'Diese Zeile entfernen',
      editJson: 'mcp.json bearbeiten',
      saveFailed: 'Dieser Server wurde nicht gespeichert.'
    },
    dialog: {
      disconnect: 'Trennen',
      disconnectTitle: (name: string) => `${name} trennen?`,
      disconnectBody: 'Hermes handelt nicht mehr über dieses Konto. Sie können sich jederzeit wieder verbinden.',
      menuRefreshTools: 'Tools aktualisieren',
      moreActions: 'Weitere Aktionen',
      removeServerTitle: (name: string) => `${name} entfernen?`,
      removeServerBody: 'Der Eintrag wird aus mcp.json auf diesem Computer entfernt. Sonst wird nichts gelöscht.',
      appSwitch: (name: string) => `Hermes darf ${name} verwenden`,
      waysTitle: (name: string) => `Wo ${name} läuft`,
      wayNotConnected: (name: string) => `Noch nicht verbunden. Melden Sie sich im Browser bei ${name} an.`,
      wayHosted: 'Verwaltet',
      bothOn: (name: string) => `Beide sind an, daher sieht Hermes jedes ${name}-Tool doppelt.`,
      turnOffLocal: 'Lokalen Server ausschalten',
      providedByPlugin: (plugin: string) => `Bereitgestellt vom Plugin ${plugin}`,
      openPlugins: 'Tab „Plugins“ öffnen',
      nousLine: 'Nous-Apps folgen Ihrem Konto, nicht dem Profil.',
      rulesReadOnly: 'Die Regeln können gerade nicht geändert werden.',
      rulesAppOff: (name: string) => `Schalten Sie ${name} ein, um die Tools zu ändern.`,
      rulesSignIn: 'Melden Sie sich an, um festzulegen, was Hermes hier darf.',
      orgNote: (count: number) => `Ihre Organisation hat ${count} Tool${count === 1 ? '' : 's'} deaktiviert.`,
      orgLink: 'Konnektor-Verwaltung öffnen',
      connectEnded: 'Die Anmeldung wurde nicht abgeschlossen.',
      connectOpenAgain: 'Link erneut öffnen',
      tokensPerCall: 'Tokens pro Aufruf',
      usesPerMonth: 'Nutzungen in 30 Tagen',
      advanced: 'Erweitert',
      advancedHint: 'der mcp.json-Eintrag und die Logs'
    },
    tools: {
      title: 'Tools',
      notInstalledBody: 'Installieren Sie ihn auf diesem Gerät, um seine Tools zu sehen.',
      summaryTitle: (name: string) => `Was Hermes mit ${name} tun darf`,
      summaryPreviewTitle: (name: string) => `Was Hermes mit ${name} tun könnte, sobald Sie verbunden sind`,
      summaryCount: (count: number) => `${count} Tool${count === 1 ? '' : 's'}`,
      summaryAllTools: 'Alle Tools',
      summaryOther: 'Sonstige',
      allToolsSwitch: 'Alle Tools ein- oder ausschalten',
      summaryAllOn: 'alle an',
      summarySomeOn: (on: number, total: number) => `${on} von ${total} an`,
      summaryOff: 'aus',
      showAllTools: (count: number) => `Alle ${count} Tool${count === 1 ? '' : 's'} anzeigen`,
      showSummary: 'Zusammenfassung anzeigen',
      facetSwitch: (facet: string) => `${facet}-Tools ein- oder ausschalten`,
      moreHints: (count: number) => `+${count}`,
      staleSignIn: 'Melden Sie sich an, um die aktuelle Tool-Liste zu laden.',
      searchCountPlaceholder: (count: number) => `${count} Tools durchsuchen`,
      toolList: (name: string) => `${name}-Tools`,
      categorySelect: (count: number) => `${count} Kategorie${count === 1 ? '' : 'n'}`,
      showDeprecated: (count: number) => `${count} veraltete anzeigen`,
      hideDeprecated: (count: number) => `${count} veraltete ausblenden`,
      quickReadOnly: 'Nur lesen',
      quickNoDestructive: 'Destruktive ausschalten',
      quickEverythingOn: 'Alles an',
      lockedHint: 'von Ihrer Organisation deaktiviert',
      turnToolOn: (tool: string) => `${tool} einschalten`,
      turnToolOff: (tool: string) => `${tool} ausschalten`,
      showDetails: (tool: string) => `Anzeigen, was ${tool} tut`,
      hideDetails: (tool: string) => `Ausblenden, was ${tool} tut`,
      noMatch: 'Kein Tool passt zu diesen Filtern.',
      loading: 'Tool-Liste wird gelesen',
      unavailableLine: 'Tool-Liste nicht verfügbar.',
      needsAuthTitle: (name: string) => `Melden Sie sich bei ${name} an, um die Tools zu laden.`,
      needsAuthBody: 'Die Anmeldung bleibt auf diesem Computer. Nichts verlässt ihn.',
      retry: 'Erneut versuchen',
      goneTitle: (name: string) => `${name} ist nicht mehr im Katalog.`,
      goneBody:
        'Hermes kann ihn nicht mehr aufrufen. Die Zeile bleibt, bis Sie sie entfernen, damit nichts verschwindet.',
      remove: 'Entfernen',
      offTitle: (name: string) => `${name} ist aus.`,
      offBody: 'Schalten Sie ihn mit dem Schalter oben ein, um seine Tools zu laden.',
      signedOutTitle: 'Melden Sie sich bei Nous an, um die Tool-Liste zu laden.',
      signedOutBody: 'Ihre Server auf diesem Computer sind nicht betroffen.',
      conflictTitle: 'Jemand hat diese Regel geändert, während Sie sie bearbeitet haben.',
      conflictBody: (theyOff: number, theyOn: number) => {
        const they = [
          theyOff > 0 ? `${theyOff} Tool${theyOff === 1 ? '' : 's'} ausgeschaltet, die bei Ihnen an sind` : '',
          theyOn > 0 ? `${theyOn} Tool${theyOn === 1 ? '' : 's'} angelassen, die Sie ausgeschaltet haben` : ''
        ].filter(Boolean)

        return `${they.length > 0 ? `Die andere Person hat ${they.join(' und ')}. ` : ''}Ihre Änderungen bleiben auf dem Bildschirm; nichts wurde geschrieben.`
      },
      conflictReload: 'Deren Version neu laden',
      conflictSave: 'Deren Version überschreiben',
      saveFailed: 'Diese Tool-Regeln wurden nicht gespeichert.',
      footerDirty: (off: number, backOn: number) =>
        `${off} Tool${off === 1 ? '' : 's'} aus, ${backOn === 0 ? 'keines' : backOn} wieder an`,
      discard: 'Verwerfen',
      save: 'Änderungen speichern',
      saving: 'Wird gespeichert...'
    },
    vocabulary: {
      facetRead: {
        label: 'Lesen',
        long: 'Liest Daten aus dieser App. Ändert nichts.'
      },
      facetWrite: {
        label: 'Schreiben',
        long: 'Erstellt oder ändert etwas in dieser App.'
      },
      facetDestructive: {
        label: 'Destruktiv',
        long: 'Kann etwas in dieser App endgültig entfernen.'
      },
      facetUnclassified: {
        label: 'Unbekannte Wirkung',
        long: 'Die App hat nie angegeben, was dieses Tool tut.'
      },
      hintReadOnly: {
        label: 'Nur lesen',
        long: 'Das Tool gibt an, dass es nur liest.'
      },
      hintCreate: {
        label: 'Erstellt',
        long: 'Erstellt etwas Neues.'
      },
      hintUpdate: {
        label: 'Aktualisiert',
        long: 'Ändert etwas, das bereits existiert.'
      },
      hintDelete: {
        label: 'Löscht',
        long: 'Entfernt etwas.'
      },
      hintDestructive: {
        label: 'Destruktiv',
        long: 'Die Änderung lässt sich hier nicht rückgängig machen.'
      },
      hintIdempotent: {
        label: 'Wiederholbar',
        long: 'Zweimal ausgeführt bewirkt es dasselbe wie einmal.'
      },
      hintOpenWorld: {
        label: 'Extern',
        long: 'Greift auf etwas außerhalb dieser App zu.'
      }
    }
  },
  sessionImport: {
    title: 'Von einer anderen App fortfahren',
    subtitle: 'Holen Sie eine Konversation in Hermes und machen Sie dort weiter, wo Sie aufgehört haben.',
    action: 'Session importieren',
    readingFrom: 'Lesen von',
    connectedComputer: 'dem verbundenen Computer',
    destination: 'Importieren in',
    all: 'Alle',
    search: 'Geladene Sessions durchsuchen',
    scanning: 'Konversationen werden gesucht',
    scanError: 'Sessions konnten nicht gefunden werden',
    scanHelp:
      'Überprüfen Sie Ihre Backend-Verbindung und versuchen Sie es dann erneut. Ältere Backends benötigen eventuell ein Update.',
    empty: 'Keine Konversationen gefunden',
    emptyHelp: 'Claude Code- und Codex-Sessions auf diesem Backend werden hier angezeigt.',
    noMatches: 'Keine passenden Konversationen',
    searchHelp: 'Versuchen Sie es mit einem anderen Titel oder Ordner, oder laden Sie weitere Sessions.',
    skipped: 'Einige Protokolle waren leer, nicht lesbar oder zu groß für die Vorschau.',
    more: 'Weitere Sessions laden',
    messages: 'Nachrichten',
    choose: 'Eine Konversation, die sich lohnt',
    chooseHelp: 'Wählen Sie eine Session, um ihren Verlauf zu lesen, bevor Sie sie in Hermes übernehmen.',
    previewLoading: 'Vorschau wird geöffnet',
    previewError: 'Vorschau nicht verfügbar',
    previewHelp:
      'Die Quelle wurde möglicherweise verschoben oder geändert. Aktualisieren Sie die Liste und versuchen Sie es erneut.',
    previewLimit: 'Vorschau für bessere Lesbarkeit gekürzt. Die vollständige Konversation wird importiert.',
    you: 'Sie',
    snapshot: 'Diese Konversation ist bereits in Hermes. Öffnen Sie Ihre vorhandene Kopie, um weiterzumachen.',
    copyNotice:
      'Kopiert den Konversationstext. Quelldateien bleiben unverändert. Tool-Ausgaben und Überlegungen werden nicht übernommen.',
    importing: 'Importieren…',
    open: 'In Hermes öffnen',
    continue: 'In Hermes fortfahren',
    importError: 'Diese Konversation konnte nicht importiert werden.'
  },
  common: {
    apply: 'Übernehmen',
    back: 'Zurück',
    save: 'Speichern',
    saving: 'Speichern…',
    cancel: 'Abbrechen',
    change: 'Ändern',
    choose: 'Wählen',
    clear: 'Leeren',
    close: 'Schließen',
    collapse: 'Einklappen',
    confirm: 'Bestätigen',
    connect: 'Verbinden',
    connecting: 'Verbindung wird hergestellt…',
    continue: 'Weiter',
    bots: 'Bots',
    copied: 'Kopiert',
    copy: 'Kopieren',
    copyFailed: 'Kopieren fehlgeschlagen',
    delete: 'Löschen',
    docs: 'Doku',
    done: 'Fertig',
    error: 'Fehler',
    expand: 'Aufklappen',
    failed: 'Fehlgeschlagen',
    formatJson: 'JSON formatieren',
    free: 'Kostenlos',
    loading: 'Lädt…',
    notSet: 'Nicht gesetzt',
    refresh: 'Aktualisieren',
    remove: 'Entfernen',
    replace: 'Ersetzen',
    retry: 'Erneut versuchen',
    run: 'Ausführen',
    send: 'Senden',
    set: 'Setzen',
    skip: 'Überspringen',
    update: 'Aktualisieren',
    tryHint: term => `Versuchen Sie „${term}“`,
    on: 'An',
    off: 'Aus'
  },
  fileMenu: {
    revealFinder: 'Im Finder anzeigen',
    revealExplorer: 'Im Datei-Explorer anzeigen',
    revealFileManager: 'Enthaltenden Ordner öffnen',
    revealInSidebar: 'In Dateibaum anzeigen',
    copyPath: 'Pfad kopieren',
    copyRelativePath: 'Relativen Pfad kopieren',
    download: 'Herunterladen',
    downloadSaved: 'Gespeichert',
    downloadFailed: 'Download fehlgeschlagen',
    rename: 'Umbenennen…',
    delete: 'Löschen',
    renameTitle: 'Umbenennen',
    renameLabel: 'Neuer Name',
    deleteTitle: name => `„${name}“ löschen?`,
    deleteBody: 'Es wird in den Papierkorb verschoben – von dort können Sie es wiederherstellen.',
    pathCopied: 'Pfad kopiert',
    revealMissing: 'Dieser Ordner befindet sich nicht auf diesem Computer',
    revealUnavailable:
      'Dieser Pfad befindet sich nicht auf diesem Computer, sondern auf dem Backend-Rechner. Verwenden Sie „Im Dateibaum anzeigen“.'
  },
  boot: {
    ready: 'Hermes Desktop ist bereit',
    desktopBootFailedWithMessage: message => `Desktop-Start fehlgeschlagen: ${message}`,
    steps: {
      connectingGateway: 'Live-Desktop-Gateway wird verbunden',
      loadingSettings: 'Hermes-Einstellungen werden geladen',
      loadingSessions: 'Letzte Sessions werden geladen',
      retryingRemoteBackend: 'Wird mit dem Remote-Hermes-Backend neu verbunden…',
      startingDesktopConnection: 'Desktop-Verbindung wird gestartet',
      startingHermesDesktop: 'Hermes Desktop wird gestartet…'
    },
    errors: {
      backgroundExited: 'Der Hermes-Hintergrundprozess wurde beendet.',
      backgroundExitedDuringStartup: 'Der Hermes-Hintergrundprozess wurde während des Starts beendet.',
      backendStopped: 'Backend gestoppt',
      restartHermes: 'Hermes neu starten',
      openLogs: 'Logs öffnen',
      desktopBootFailed: 'Desktop-Start fehlgeschlagen',
      gatewayConnectionLost: 'Verbindung zum Gateway verloren',
      gatewayConnectionLostDetail:
        'Im Hintergrund wird weiterhin versucht, die Verbindung herzustellen. Sie können weiterlesen und weiterschreiben – öffnen Sie die Gateway-Einstellungen, falls das anhält.',
      reconnectNow: 'Jetzt neu verbinden',
      connectionSettings: 'Verbindungseinstellungen',
      gatewaySignInRequired: 'Gateway-Sign-in erforderlich',
      gatewaySignInRequiredDetail:
        'Melden Sie sich erneut an, um die Verbindung wiederherzustellen. Ihre Chats und Einstellungen sind sicher.',
      signInAgain: 'Erneut anmelden',
      ipcBridgeUnavailable: 'Der Desktop-IPC-Bridge ist nicht verfügbar.'
    },
    causes: {
      exitedEarly: 'Der Hintergrunddienst von Hermes hat direkt nach dem Start aufgehört.',
      timedOut: 'Der Hintergrunddienst von Hermes hat nicht rechtzeitig geantwortet.',
      permission: 'Hermes konnte nicht in seinen Datenordner schreiben (Berechtigungsproblem).',
      diskFull: 'Die Festplatte ist voll, deshalb konnte Hermes nicht starten.',
      portInUse: 'Ein anderes Programm verwendet den Netzwerkport, den Hermes braucht.',
      installMissing:
        'Ein Teil der Hermes-Installation fehlt. Wählen Sie „Installation reparieren“, um sie wiederherzustellen.'
    },
    failure: {
      title: 'Hermes konnte nicht gestartet werden',
      description:
        'Das Hintergrund-Gateway ist nicht gestartet. Probieren Sie einen der Wiederherstellungsschritte unten. Keiner davon löscht Ihre Chats oder Einstellungen.',
      details: 'Details',
      remoteTitle: 'Remote-Gateway-Sign-in erforderlich',
      remoteDescription:
        'Ihre Remote-Gateway-Session ist abgelaufen. Melden Sie sich erneut an, um die Verbindung wiederherzustellen. Keiner dieser Schritte löscht Ihre Chats oder Einstellungen.',
      retry: 'Erneut versuchen',
      repairInstall: 'Installation reparieren',
      useLocalGateway: 'Lokales Gateway verwenden',
      gatewaySettings: 'Gateway-Einstellungen',
      back: 'Zurück',
      openLogs: 'Logs öffnen',
      repairHint:
        'Die Reparatur führt den Installer erneut aus und kann auf einem frischen Computer ein paar Minuten dauern.',
      remoteSignInHint: signInLabel =>
        `Meldet Sie von der gespeicherten Remote-Browser-Session ab und öffnet dann ${signInLabel}. Verwenden Sie das lokale Gateway, um stattdessen zum integrierten Backend zu wechseln.`,
      signOutAndSignIn: 'Abmelden & anmelden',
      remoteFailureHint:
        'Überprüfen Sie die Gateway-URL und die Anmeldung in den Gateway-Einstellungen, oder wechseln Sie zum lokalen Gateway.',
      cloudDownTitle: 'Nous Cloud Agent ist down',
      cloudDownDescription:
        'Der von Nous verwaltete Cloud-Agent, mit dem sich dieses Gateway verbindet, meldet einen Serverfehler. Er kann von hier aus nicht neu gestartet werden – prüfen Sie seinen Status, wechseln Sie zum lokalen Gateway oder wenden Sie sich an den Support.',
      cloudDownHint:
        'Die Schaltflächen unten öffnen das Nous Portal (Instanzstatus und Steuerung) und unseren Discord für Support.',
      cloudDownCheckPortal: 'Portal-Status prüfen',
      cloudDownDiscord: 'Hilfe auf Discord holen',
      hideRecentLogs: 'Neueste Logs ausblenden',
      showRecentLogs: 'Neueste Logs anzeigen',
      signedInTitle: 'Angemeldet',
      signedInMessage: 'Wird mit dem Remote-Gateway neu verbunden…',
      signInIncompleteTitle: 'Sign-in unvollständig',
      signInIncompleteMessage: 'Das Anmeldefenster wurde geschlossen, bevor die Authentifizierung abgeschlossen war.',
      signInFailed: 'Sign-in fehlgeschlagen',
      signInToRemoteGateway: 'Beim Remote-Gateway anmelden',
      signInWithProvider: provider => `Mit ${provider} anmelden`,
      identityProvider: 'Ihr Identity-Provider'
    }
  },
  notifications: {
    region: 'Benachrichtigungen',
    hide: 'Ausblenden',
    show: 'Anzeigen',
    more: count => `${count} weitere ${count === 1 ? 'Benachrichtigung' : 'Benachrichtigungen'}`,
    clearAll: 'Alle löschen',
    dismiss: 'Benachrichtigung schließen',
    details: 'Details',
    copyDetail: 'Detail kopieren',
    copyDetailFailed: 'Notification-Detail konnte nicht kopiert werden',
    backendOutOfDateTitle: 'Backend veraltet',
    backendOutOfDateMessage:
      'Ihr Hermes-Backend ist älter als dieser Desktop-Build und funktioniert möglicherweise nicht richtig. Aktualisieren Sie, um beide abzugleichen.',
    desktopOutOfDateTitle: 'Hermes-App veraltet',
    desktopOutOfDateMessage:
      'Diese Hermes-App ist älter als das verbundene Backend und funktioniert möglicherweise nicht richtig. Aktualisieren Sie die App, um beide abzugleichen.',
    updateDesktopApp: 'App aktualisieren',
    installMethodUnsupportedTitle: 'Nicht unterstützte Installationsmethode',
    updateHermes: 'Hermes aktualisieren',
    updateReadyTitle: 'Update bereit',
    updateReadyMessage: count => `${count} neue Änderung${count === 1 ? '' : 'en'} verfügbar.`,
    updateReadyMessageUnknown: 'Ein neues Update ist verfügbar.',
    seeWhatsNew: 'Neues ansehen',
    mcp: {
      needsAuthTitle: 'MCP-Server braucht erneute Authentifizierung',
      needsAuthMessage: name => `${name} MCP braucht erneute Authentifizierung.`,
      errorTitle: 'MCP-Server nicht erreichbar',
      errorMessage: name => `${name} MCP hat den Health-Check nicht bestanden.`,
      signIn: 'Anmelden',
      view: 'Anzeigen',
      disable: 'Deaktivieren',
      disabledMessage: name =>
        `${name} MCP deaktiviert. Sie können es jederzeit unter Fähigkeiten → MCP wieder aktivieren.`,
      disableFailed: name => `${name} MCP konnte nicht deaktiviert werden.`
    },
    errors: {
      elevenLabsNeedsKey: 'ElevenLabs-STT braucht ELEVENLABS_API_KEY.',
      elevenLabsRejectedKey: 'ElevenLabs hat den API-Key abgelehnt (401).',
      diskFull: 'Festplatte voll – geben Sie Speicherplatz frei und versuchen Sie es dann erneut.',
      storageFailure:
        'Hermes konnte nicht in seinen Datenordner speichern. Öffnen Sie die Wartung, um das Problem zu prüfen und zu beheben.',
      gatewayAuthFailed: 'Gateway-Authentifizierung fehlgeschlagen – überprüfen Sie Ihren API_SERVER_KEY.',
      methodNotAllowed:
        'Das Desktop-Backend hat diese Anfrage abgelehnt (405 Method Not Allowed). Starten Sie Hermes Desktop neu.',
      microphonePermission: 'Die Mikrofonberechtigung wurde verweigert.',
      openaiRejectedApiKey:
        'OpenAI hat Ihren API-Key abgelehnt. Aktualisieren Sie ihn unter Einstellungen → Schlüssel und versuchen Sie es erneut.',
      openaiTtsNeedsKey: 'OpenAI-TTS braucht VOICE_TOOLS_OPENAI_KEY oder OPENAI_API_KEY.',
      codeSkewRestartRequired:
        'Dieses Backend läuft nach einem Update mit altem Code. Starten Sie es neu, um den neuen Code zu laden.',
      rpcOutOfSync: 'App und Backend laufen in unterschiedlichen Versionen. Aktualisieren Sie beide.',
      restartHermesFailed: 'Hermes konnte nicht neu gestartet werden'
    },
    actions: {
      restartHermes: 'Hermes neu starten',
      openKeys: 'Schlüssel öffnen',
      openGateways: 'Gateways öffnen',
      openMaintenance: 'Wartung öffnen'
    },
    voice: {
      configureSpeechToText: 'Richten Sie Speech-to-Text ein, um den Sprachmodus zu verwenden.',
      couldNotStartSession: 'Sprachsession konnte nicht gestartet werden',
      microphoneAccessDenied: 'Mikrofonzugriff verweigert.',
      microphoneConstraintsUnsupported: 'Mikrofon-Einschränkungen werden von diesem Gerät nicht unterstützt.',
      microphoneFailed: 'Mikrofon fehlgeschlagen',
      microphoneInUse: 'Das Mikrofon wird bereits von einer anderen App verwendet.',
      microphonePermissionDenied: 'Die Mikrofonberechtigung wurde verweigert.',
      microphoneStartFailed: 'Mikrofonaufnahme konnte nicht gestartet werden.',
      microphoneUnsupported: 'Diese Laufzeitumgebung unterstützt keine Mikrofonaufnahme.',
      noMicrophone: 'Es wurde kein Mikrofon gefunden.',
      noSpeechDetected: 'Keine Sprache erkannt',
      playbackFailed: 'Sprachwiedergabe fehlgeschlagen',
      recordingFailed: 'Sprachaufnahme fehlgeschlagen',
      sayStopToEnd: phrase => `Sagen Sie „${phrase}“, um den Sprachchat zu beenden.`,
      transcriptionFailed: 'Sprachtranskription fehlgeschlagen',
      transcriptionUnavailable: 'Sprachtranskription ist noch nicht verfügbar.',
      tryRecordingAgain: 'Versuchen Sie die Aufnahme erneut.',
      unavailable: 'Sprache nicht verfügbar',
      liveEnded: 'Live-Sprach-Session beendet',
      liveEndedConnectionLost: 'Die Live-Sprach-Session hat die Verbindung verloren.',
      liveEndedClosed: 'Die Live-Sprach-Session wurde vom Dienst geschlossen.',
      liveError: 'Live-Sprache',
      liveDelegationFailed: 'Anfrage konnte nicht an Hermes übergeben werden',
      liveUnavailable: reason =>
        `GPT-Live-Sprachchat ist nicht verfügbar: ${reason}. Stattdessen wird Sprache-zu-Text verwendet.`
    },
    native: {
      approvalTitle: 'Genehmigung erforderlich',
      approvalTitleNamed: session => `Freigabe erforderlich — ${session}`,
      approveAction: 'Genehmigen',
      rejectAction: 'Ablehnen',
      inputTitle: 'Eingabe erforderlich',
      inputTitleNamed: session => `Eingabe erforderlich — ${session}`,
      inputBody: 'Hermes wartet auf Ihre Antwort.',
      turnDoneTitle: 'Hermes fertig',
      turnDoneBody: '',
      turnErrorTitle: 'Turn fehlgeschlagen',
      backgroundDoneTitle: 'Hintergrundaufgabe abgeschlossen',
      backgroundFailedTitle: 'Hintergrundaufgabe fehlgeschlagen',
      creditsTitle: 'Credits'
    }
  },
  remoteDisplayBanner: {
    message: reason =>
      `Software-Rendering aktiv — Remote-Display erkannt (${reason}). GPU-Beschleunigung ist deaktiviert, um Flackern zu verhindern.`
  },
  billingBlock: {
    titleNous: 'Keine Nous-Credits mehr',
    titleProvider: provider => `Keine Credits mehr — ${provider}`,
    fallbackMessage: 'Auf Ihrem Konto sind keine Credits mehr übrig. Fügen Sie Credits hinzu, um fortzufahren.',
    openBilling: 'Billing öffnen',
    addCredits: 'Credits hinzufügen',
    dismiss: 'Schließen'
  },
  sendDiagnostics: {
    title: 'Diagnosedaten an Nous senden',
    privacyNotice:
      'Damit laden Sie ein Debug-Paket in den internen Nous-Speicher hoch (kein öffentliches Paste). Es enthält Systeminfos (Betriebssystem, Versionen, Provider, welche API-Keys konfiguriert sind – niemals die Keys selbst) sowie vollständige Agent-, Gateway- und Desktop-Logs (bis zu 512 KB je Datei), die sehr wahrscheinlich Gesprächsinhalte, Tool-Ausgaben und Dateipfade enthalten. Geheimnisse werden vor dem Upload geschwärzt. Das Paket ist nur für Nous-Mitarbeitende und freigeschaltete Discord-Moderatoren einsehbar und wird nach 14 Tagen automatisch gelöscht.',
    upload: 'Hochladen',
    uploading: 'Wird hochgeladen…',
    cancel: 'Abbrechen',
    close: 'Schließen',
    copyLink: 'Link kopieren',
    uploadIdFallback: id => `Kein Link zurückgegeben — zitiere die Upload-ID ${id} im Support`,
    doneTitle: 'Diagnosedaten gesendet',
    doneDescription:
      'Ihr Paket wurde privat hochgeladen. Teilen Sie den Link unten in Ihrem Support-Thread, damit das Team Ihre Logs sehen kann.',
    failedTitle: 'Hochladen fehlgeschlagen',
    failedHint:
      'Sie können auch `hermes debug share --nous` im Terminal ausführen oder `hermes debug share --local`, um den Bericht ohne Hochladen auszugeben.',
    handoffLead: 'Diskussion hier fortsetzen:',
    links: {
      github: 'GitHub Issues',
      portal: 'Nous-Portal-Support',
      discord: 'Discord'
    }
  },
  titlebar: {
    hideSidebar: 'Sidebar ausblenden',
    showSidebar: 'Sidebar einblenden',
    search: 'Suchen',
    searchTitle: 'Sessions, Ansichten und Aktionen durchsuchen',
    swapSidebarSides: 'Sidebar-Seiten tauschen',
    hideRightSidebar: 'Rechte Sidebar ausblenden',
    showRightSidebar: 'Rechte Sidebar einblenden',
    unreadSessions: count => (count === 1 ? '1 ungelesene Session' : `${count} ungelesene Sessions`),
    muteHaptics: 'Haptik stummschalten',
    unmuteHaptics: 'Haptik einschalten',
    openSettings: 'Einstellungen öffnen',
    openStarmap: 'Speicher-Graph öffnen',
    enterHud: 'HUD-Modus',
    exitHud: 'HUD-Modus beenden',
    resetHudLayout: 'HUD-Größe und -Position zurücksetzen',
    layoutEditor: 'Layout-Editor',
    layoutEditorTitle: mod => `Layout-Editor — ${mod}-Klick setzt das Layout zurück`
  },
  keybinds: {
    title: 'Tastaturkürzel',
    subtitle: open => `Klicken Sie auf ein Kürzel, um es neu zu belegen · ${open} öffnet dieses Panel erneut.`,
    search: 'Kürzel suchen…',
    rebind: 'Neu belegen',
    reset: 'Auf Standard zurücksetzen',
    resetAll: 'Alle zurücksetzen',
    pressKey: 'Taste drücken…',
    set: 'setzen',
    conflictWith: label => `Auch belegt mit „${label}“`,
    categories: {
      composer: 'Composer',
      profiles: 'Profile',
      session: 'Session',
      navigation: 'Navigation',
      view: 'Ansicht'
    },
    actions: {
      'keybinds.openPanel': 'Tastaturkürzel öffnen',
      'nav.commandPalette': 'Befehlspalette öffnen',
      'nav.commandCenter': 'Befehlszentrum öffnen',
      'nav.settings': 'Einstellungen öffnen',
      'nav.profiles': 'Profile öffnen',
      'nav.capabilities': 'Skills öffnen',
      'nav.messaging': 'Messaging öffnen',
      'nav.artifacts': 'Artefakte öffnen',
      'nav.cron': 'Geplante Jobs öffnen',
      'nav.agents': 'Agenten öffnen',
      'session.new': 'Neue Session',
      'session.newTab': 'Neuer Session-Tab',
      'session.newWindow': 'Neues Fenster',
      'session.next': 'Nächste Session',
      'session.prev': 'Vorherige Session',
      'session.slot.1': 'Zu letzter Session 1 wechseln',
      'session.slot.2': 'Zu letzter Session 2 wechseln',
      'session.slot.3': 'Zu letzter Session 3 wechseln',
      'session.slot.4': 'Zu letzter Session 4 wechseln',
      'session.slot.5': 'Zu letzter Session 5 wechseln',
      'session.slot.6': 'Zu letzter Session 6 wechseln',
      'session.slot.7': 'Zu letzter Session 7 wechseln',
      'session.slot.8': 'Zu letzter Session 8 wechseln',
      'session.slot.9': 'Zu letzter Session 9 wechseln',
      'session.focusSearch': 'Sessions durchsuchen',
      'session.togglePin': 'Aktuelle Session anheften / lösen',
      'session.archive': 'Aktuelle Session archivieren',
      'workspace.newWorktree': 'Neues Worktree',
      'workspace.openFolder': 'Ordner als Projekt öffnen',
      'composer.focus': 'Composer fokussieren',
      'composer.modelPicker': 'Modellauswahl öffnen',
      'composer.voice': 'Sprachkonversation starten / stoppen',
      'view.toggleSidebar': 'Session-Sidebar umschalten',
      'view.cycleSidebarGrouping': 'Session-Gruppierung wechseln',
      'view.toggleRightSidebar': 'Dateibrowser umschalten',
      'view.toggleReview': 'Review-Bereich umschalten',
      'view.toggleStatusbar': 'Statusleiste umschalten',
      'view.toggleTabStrip': 'Tabs umschalten',
      'view.toggleProfileRail': 'Profil-Leiste ein-/ausblenden',
      'view.toggleSimpleMode': 'Einfachen Modus umschalten',
      'view.showFiles': 'Dateibrowser anzeigen',
      'view.showBrowser': 'Browser öffnen',
      'view.toggleHud': 'HUD-Modus umschalten',
      'hud.snapToPointer': 'HUD zum Zeiger bewegen (global, während HUD offen ist)',
      'view.showTerminal': 'Terminal umschalten',
      'view.newTerminal': 'Neues Terminal',
      'view.nextTerminal': 'Nächstes Terminal',
      'view.prevTerminal': 'Vorheriges Terminal',
      'view.closeTerminal': 'Terminal schließen',
      'view.selectionToComposer': 'Auswahl an Composer senden',
      'view.terminalCopy': 'Terminal-Auswahl kopieren',
      'view.terminalPaste': 'In Terminal einfügen',
      'view.closeTab': 'Tab schließen',
      'view.reopenTab': 'Geschlossenen Tab wieder öffnen',
      'view.flipPanes': 'Sidebar-Seiten tauschen',
      'view.findInPage': 'Auf Seite suchen',
      'view.findNext': 'Nächsten Treffer suchen',
      'view.findPrevious': 'Vorherigen Treffer suchen',
      'appearance.toggleMode': 'Hell / dunkel umschalten',
      'profile.default': 'Zu Standardprofil wechseln',
      'profile.switch.1': 'Zu Profil 1 wechseln',
      'profile.switch.2': 'Zu Profil 2 wechseln',
      'profile.switch.3': 'Zu Profil 3 wechseln',
      'profile.switch.4': 'Zu Profil 4 wechseln',
      'profile.switch.5': 'Zu Profil 5 wechseln',
      'profile.switch.6': 'Zu Profil 6 wechseln',
      'profile.switch.7': 'Zu Profil 7 wechseln',
      'profile.switch.8': 'Zu Profil 8 wechseln',
      'profile.switch.9': 'Zu Profil 9 wechseln',
      'profile.switch.10': 'Zu Profil 10 wechseln',
      'profile.switch.11': 'Zu Profil 11 wechseln',
      'profile.switch.12': 'Zu Profil 12 wechseln',
      'profile.switch.13': 'Zu Profil 13 wechseln',
      'profile.switch.14': 'Zu Profil 14 wechseln',
      'profile.switch.15': 'Zu Profil 15 wechseln',
      'profile.switch.16': 'Zu Profil 16 wechseln',
      'profile.switch.17': 'Zu Profil 17 wechseln',
      'profile.switch.18': 'Zu Profil 18 wechseln',
      'profile.next': 'Nächstes Profil',
      'profile.prev': 'Vorheriges Profil',
      'profile.toggleAll': 'Alle-Profile-Ansicht umschalten',
      'profile.create': 'Profil erstellen',
      'composer.send': 'Nachricht senden',
      'composer.newline': 'Neue Zeile einfügen',
      'composer.steer': 'Laufenden Turn steuern',
      'composer.queue': 'Nachricht in Warteschlange',
      'composer.sendQueued': 'Nächsten eingereihten Turn senden',
      'composer.mention': 'Dateien, Ordner, URLs referenzieren',
      'composer.slash': 'Slash-Befehlspalette',
      'composer.help': 'Schnellhilfe',
      'composer.history': 'Popover / Verlauf durchblättern',
      'composer.cancel': 'Popover schließen · Lauf abbrechen'
    }
  },
  findInPage: {
    next: 'Nächster Treffer',
    previous: 'Vorheriger Treffer'
  },
  language: {
    label: 'Sprache',
    description: 'Wählen Sie die Sprache der Desktop-Oberfläche.',
    saving: 'Sprache wird gespeichert…',
    saveError: 'Sprachupdate fehlgeschlagen',
    switchTo: 'Sprache wechseln',
    searchPlaceholder: 'Sprachen suchen…',
    noResults: 'Keine Sprachen gefunden'
  },
  settings: {
    subpages: {
      appearanceTheme: 'Design',
      appearanceTypography: 'Typografie',
      appearanceWindowLayout: 'Fenster & Layout',
      appearanceChatDisplay: 'Chat-Anzeige',
      appearancePet: 'Begleiter',
      appearanceGeneral: 'Allgemein',
      modelMain: 'Hauptmodell',
      modelAuxiliary: 'Hilfsmodelle',
      modelMoa: 'Mixture of Agents',
      modelFallbacks: 'Fallback-Modelle',
      chatBehavior: 'Verhalten',
      chatAttachments: 'Anhänge',
      workspaceProjects: 'Projekte & Erkennung',
      workspaceShell: 'Shell-Umgebung',
      workspaceFiles: 'Dateien & Ausführung',
      safetyApprovals: 'Freigaben',
      safetyPrivacy: 'Datenschutz & Netzwerk',
      safetyCheckpoints: 'Checkpoints',
      browserProfile: 'Browserprofil',
      browserNetwork: 'Lokale & private URLs',
      memoryPersistent: 'Dauerhaftes Gedächtnis',
      memoryContext: 'Kontext & Komprimierung',
      voiceConversation: 'Sprachunterhaltung',
      voiceTranscription: 'Sprache zu Text',
      voiceSpeech: 'Text zu Sprache',
      advancedRuntime: 'Agent-Limits',
      advancedTools: 'Tool-Zugriff',
      advancedTerminal: 'Terminal-Backend',
      advancedOutput: 'Ausgabelimits',
      advancedDelegation: 'Subagenten',
      advancedDesktop: 'Desktop & Start',
      gatewayConnection: 'Dieses Fenster',
      gatewayDevices: 'Gespeicherte Verbindungen',
      gatewayManagedUpdates: 'Remote-Updates',
      gatewayManagedUpdatesUnavailable:
        'Remote-Updates erfordern eine Desktop-Version mit Unterstützung für verwaltete SSH-Updates.',
      gatewayManagedUpdatesEmpty:
        'Fügen Sie unter „Gespeicherte Verbindungen“ eine SSH-Verbindung hinzu, um deren Updates hier zu verwalten.',
      keyboardShortcuts: 'Tastenbelegung',
      hudGesture: 'HUD-Geste',
      screenCapture: 'Bildschirmaufnahme',
      notificationAlerts: 'Desktop-Hinweise',
      notificationSounds: 'Töne',
      archivedSessions: 'Archiv & Aufbewahrung',
      defaultDirectory: 'Standard-Projektordner',
      vaultCredentials: 'Gespeicherte Zugangsdaten',
      vaultSources: 'Passwortmanager',
      appUpdates: 'Version & Updates',
      uninstall: 'Deinstallieren',
      billingOverview: 'Übersicht',
      billingPlans: 'Tarife'
    },
    closeSettings: 'Einstellungen schließen',
    exportConfig: 'Konfiguration exportieren',
    importConfig: 'Konfiguration importieren',
    resetToDefaults: 'Auf Standard zurücksetzen',
    resetConfirm: 'Alle Einstellungen auf Hermes-Standard zurücksetzen?',
    exportFailed: 'Export fehlgeschlagen',
    resetFailed: 'Zurücksetzen fehlgeschlagen',
    nav: {
      providers: 'Anbieter',
      providerAccounts: 'Konten',
      providerApiKeys: 'API-Schlüssel',
      providerCustomEndpoints: 'Benutzerdefinierte Endpunkte',
      providerLocalModels: 'Lokale Modelle',
      gateway: 'Gateways',
      apiKeys: 'Tools & Schlüssel',
      keybinds: 'Tastaturkürzel',
      keysTools: 'Tools',
      keysSettings: 'Einstellungen',
      mcp: 'MCP',
      archivedChats: 'Archivierte Chats',
      sessions: 'Sessions',
      about: 'Über',
      billing: 'Abrechnung',
      notifications: 'Benachrichtigungen',
      vault: 'Passwörter & Logins'
    },
    plugins: {
      title: 'Desktop-Plugins',
      blurb: 'Gebündelt oder im Ordner „Desktop-Plugins“ abgelegt. Deaktivieren, um live zu entladen.',
      count: n => `${n} installiert`,
      openFolder: 'Ordner für Desktop-Plugins öffnen',
      rescan: 'Erneut scannen',
      reveal: 'Im Dateimanager anzeigen',
      enable: 'Aktivieren',
      disable: 'Deaktivieren',
      failed: 'fehlgeschlagen',
      empty: 'Noch keine Desktop-Plugins installiert.',
      kinds: {
        bundled: 'gebündelt',
        disk: 'auf Datenträger',
        runtime: 'Laufzeit'
      },
      agentHalfMissing: 'Agent-Hälfte fehlt hier',
      agentHalfMissingTip:
        'Das ist die Desktop-Hälfte eines gebündelten Plugins, aber seine Agent-Hälfte ist auf dem aktuell verbundenen Backend/Profil nicht installiert. Installieren Sie sie unter Fähigkeiten → Plugins.',
      installModal: {
        installFromGit: 'Von Git installieren',
        reviewRepository: 'Repository prüfen',
        repoPlaceholder: 'https://github.com/owner/repo',
        title: 'Plugin installieren',
        description: 'Prüfen Sie, was dieses Repository enthält, bevor Sie etwas installieren.',
        repoLabel: 'Repository',
        includesHeading: 'Dieses Paket enthält',
        agentLabel: 'Agent-Plugin',
        desktopLabel: 'Desktop-UI',
        profileLabel: 'Für Profil installieren',
        agentTargetLocal: (profile, dir) => `Wird in das Backend ${profile} installiert (${dir})`,
        agentTargetRemote: profile => `Installiert in das verbundene ${profile}-Backend`,
        catalogPinned: (name, sha) =>
          `Hermes-Katalog-Eintrag „${name}" — die Agent-Komponente wird am geprüften Pin installiert${sha ? ` ${sha}` : ''}, nicht an der Spitze des Branches.`,
        reviewedHeading: 'Geprüfter Katalog-Eintrag',
        reviewedIntro:
          'Dieser Eintrag wurde an seinem gepinnten Commit von einem Menschen geprüft. Sie können den genauen Code trotzdem unten ansehen.',
        toolsConnected: n => (n === 1 ? '1 Tool verbunden' : `${n} Tools verbunden`),
        skillsReady: names => (names.length === 1 ? `Skill ${names[0]} bereit` : `${names.length} Skills bereit`),
        nextChat: 'weitere Tools in Ihrem nächsten Chat verfügbar',
        serverNotConnected: (server, reason) =>
          `MCP-Server ${server} ist nicht verbunden${reason ? `: ${reason}` : '.'}`,
        missingEnvAction: 'Einrichten',
        alreadyInstalled: name => `${name} ist bereits installiert.`,
        desktopTarget: 'Installiert in den lokalen Desktop-Plugins-Ordner dieser App',
        desktopTargetFromPackage: 'Aus dem Paket oben in diese App geladen — gleich für jedes Profil',
        desktopOnlyNote: 'Nur-Desktop-Pakete installieren kein Backend-Agent-Plugin.',
        insecureWarning:
          'Diese URL verwendet ein unsicheres oder lokales Schema. Für Produktionsinstallationen bevorzuge https:// oder git@.',
        securityHeading: 'Vor der Installation',
        securityIntro:
          'Installieren Sie nur aus Quellen, denen Sie vertrauen – prüfen Sie das Repository unten, wenn Sie sehen möchten, was hinzugefügt wird.',
        sourceHeading: 'Quellcode',
        viewRepository: 'Repository ansehen',
        viewPluginFiles: 'Plugin-Dateien ansehen',
        gitCloneLabel: 'Git-Clone-URL',
        enableAgent: 'Agent-Plugin nach der Installation aktivieren',
        forceReinstall: 'Neuinstallation erzwingen (ersetzen, falls bereits installiert)',
        pinToCommit: 'An Commit binden (optional)',
        pinToCommitPlaceholder: 'Vollständiger 40-stelliger Commit-SHA',
        pinToCommitHint:
          'Alle, die diesen SHA installieren, bekommen denselben Code; das Plugin lehnt danach Updates ab, bis es neu gepinnt wird. Leer lassen für den neuesten Commit.',
        pinToCommitInvalid:
          'Muss ein vollständiger 40-stelliger Commit-SHA sein (Branches und Tags werden nicht akzeptiert).',
        install: 'Installieren',
        installing: 'Wird installiert…',
        probing: 'Repository wird untersucht…',
        probeUnavailable: 'Die Plugin-Untersuchung ist in dieser Umgebung nicht verfügbar.',
        desktopUnavailable: 'Die Desktop-Plugin-Installation ist in dieser Umgebung nicht verfügbar.',
        selectComponent: 'Wählen Sie mindestens eine Komponente zur Installation aus.',
        agentSuccess: name => `Agent-Plugin ${name} installiert`,
        desktopSuccess: name => `Desktop-Plugin ${name} installiert`,
        agentFailed: 'Installation des Agent-Plugins fehlgeschlagen',
        installUncertain:
          'Hermes wartet nicht mehr auf das Installationsergebnis, aber das Plugin wird möglicherweise noch installiert. Schließe dieses Fenster und aktualisiere die Pluginliste, bevor du die Installation erneut startest.',
        desktopFailed: 'Installation des Desktop-Plugins fehlgeschlagen',
        missingEnv: (name, vars) =>
          `${name} ist installiert, benötigt aber einen Schlüssel, um zu funktionieren: ${vars}. Fügen Sie ihn jetzt hinzu, sonst schlagen die Tools des Plugins fehl.`
      }
    },
    vault: {
      title: 'Passwörter & Logins',
      blurb:
        'Sagen Sie „melde mich bei GitHub an“, und der Agent meldet Sie an. Beim ersten Mal auf einer Anmeldeseite fragt er Sie direkt dort nach dem Login; danach läuft es einfach. Passwörter sind auf diesem Rechner verschlüsselt und werden direkt in die Seite eingetragen – das Modell sieht sie nie.',
      count: n => `${n} gespeichert`,
      loadFailed: 'Gespeicherte Einträge konnten nicht geladen werden',
      empty: 'Noch nichts gespeichert',
      emptyDesc:
        'Sie müssen hier nichts eintragen. Bitten Sie den Agenten, sich bei einer Seite anzumelden – er fragt Sie dann einmal direkt dort nach dem Login. Über „Hinzufügen“ können Sie einen Eintrag auch vorab anlegen.',
      add: 'Hinzufügen',
      addTitle: 'Login, Karte oder Adresse hinzufügen',
      addDescription: 'Verschlüsselt auf diesem Rechner gespeichert. Der Agent sieht das Passwort nie.',
      added: 'Gespeichert.',
      adding: 'Wird gespeichert…',
      addConfirm: 'Speichern',
      kindField: 'Art',
      kinds: {
        login: 'Login',
        payment: 'Zahlungskarte',
        address: 'Adresse'
      },
      labelField: 'Bezeichnung',
      labelPlaceholder: 'z. B. GitHub Firma',
      labelRequired: 'Eine Bezeichnung ist erforderlich.',
      originField: 'Ursprung der Seite',
      originPlaceholder: 'https://github.com',
      originPlaceholderCheckout: 'https://shop.example.com',
      originInvalid: 'Geben Sie eine gültige URL wie https://example.com ein.',
      identifierTypeField: 'Art der Kennung',
      identifierTypes: {
        email: 'E-Mail',
        phone: 'Telefon',
        username: 'Benutzername'
      },
      identifierField: 'Kennung',
      identifierShown: identifier => identifier,
      passwordField: 'Passwort',
      loginFieldsRequired: 'Kennung und Passwort sind erforderlich.',
      cardNumberField: 'Kartennummer',
      cardNameField: 'Name auf der Karte',
      expMonthField: 'Ablaufmonat',
      expYearField: 'Ablaufjahr',
      cvcField: 'CVC',
      postalField: 'Postleitzahl',
      addressLine1Field: 'Adresszeile 1',
      addressLine2Field: 'Adresszeile 2',
      cityField: 'Stadt',
      stateField: 'Bundesland / Region',
      countryField: 'Land',
      optional: '(optional)',
      createdOn: date => `Hinzugefügt am ${date}`,
      deleteAction: 'Gespeicherten Eintrag entfernen',
      otpField: 'Authentifizierungsschlüssel',
      otpPlaceholder: 'Base32-Geheimnis oder otpauth://-Link',
      otpHint:
        'Der „Einrichtungsschlüssel", den die Seite beim Aktivieren von 2FA anzeigt. Ist er gespeichert, erzeugt Hermes die Codes selbst.',
      twoFactorBadge: '2FA automatisch',
      deleteTitle: 'Diesen Eintrag löschen?',
      deleteDescription: label => `„${label}" wird entfernt. Das kann nicht rückgängig gemacht werden.`,
      deleteConfirm: 'Löschen',
      sources: {
        title: 'Passwortmanager',
        blurb:
          'Installierte Passwortmanager werden automatisch erkannt. Der Agent bittet Sie, einen zu entsperren, wenn er zum ersten Mal einen Login daraus braucht (einmal pro Session); nur ein Session-Token bleibt im Speicher, und der Agent sieht weder Ihr Master-Passwort noch einen Login.',
        toggleFailed: 'Passwortmanager konnte nicht geändert werden',
        notInstalled: name =>
          `Nicht erkannt. Installieren Sie das ${name}-Kommandozeilenwerkzeug und melden Sie sich dort an; Hermes erkennt es automatisch.`,
        disabledDesc: 'Erkannt, aber für Hermes ausgeschaltet.',
        lockedDesc:
          'Erkannt. Der Agent bittet Sie, ihn zu entsperren, wenn er einen Login braucht – oder entsperren Sie ihn jetzt.',
        unlockedDesc:
          'Für diese Session entsperrt. Sperrt automatisch nach 30 Minuten Inaktivität oder wenn Hermes geschlossen wird.',
        statusLocked: 'Gesperrt',
        statusNotDetected: 'Nicht erkannt',
        statusOff: 'Aus',
        statusUnlocked: 'Entsperrt',
        unlock: 'Entsperren',
        unlocking: 'Wird entsperrt…',
        lock: 'Sperren',
        unlocked: name => `${name} ist für diese Session entsperrt.`,
        unlockTitle: name => `${name} entsperren`,
        unlockDescription:
          'Geben Sie Ihr Master-Passwort ein. Es geht an den Passwortmanager auf diesem Rechner und wird danach verworfen – es wird nie gespeichert, protokolliert oder dem Agenten gezeigt.',
        masterPasswordPlaceholder: 'Master-Passwort'
      }
    },
    notifications: {
      title: 'Benachrichtigungen',
      intro: 'OS-Benachrichtigungen (keine In-App-Toasts). Pro Gerät.',
      enableAll: 'Benachrichtigungen aktivieren',
      enableAllDesc: 'Aus schaltet jede Benachrichtigung unten stumm.',
      focusedHint: 'Abschluss-Alerts feuern nur, während Hermes im Hintergrund ist.',
      kinds: {
        approval: {
          label: 'Genehmigung nötig',
          description: 'Ein Befehl wartet darauf, dass Sie ihn genehmigen oder ablehnen.'
        },
        input: {
          label: 'Eingabe nötig',
          description: 'Hermes hat eine Frage gestellt oder braucht ein Passwort oder Geheimnis.'
        },
        turnDone: {
          label: 'Antwort bereit',
          description: 'Ein Turn wurde beendet, während Hermes im Hintergrund war.'
        },
        turnError: {
          label: 'Turn fehlgeschlagen',
          description: 'Fehler bei Background-Turns.'
        },
        backgroundDone: {
          label: 'Hintergrund-Task fertig',
          description: 'Ein Terminal-Befehl im Hintergrund wurde abgeschlossen.'
        },
        credits: {
          label: 'Credit-Alerts',
          description: 'Der Credit-Zugriff wird pausiert oder wiederhergestellt.'
        },
        plugin: {
          label: 'Plugin-Benachrichtigungen',
          description: 'Ein Desktop-Plugin hat eine Benachrichtigung gesendet, während Hermes im Hintergrund war.'
        }
      },
      test: 'Testbenachrichtigung senden',
      testTitle: 'Hermes',
      testBody: 'Benachrichtigungen funktionieren.',
      testSent:
        'Test gesendet. Wenn nichts erscheint, überprüfen Sie die Benachrichtigungsberechtigungen Ihres Betriebssystems und Fokus/Nicht stören.',
      testUnsupported: 'Dieses System unterstützt keine nativen Benachrichtigungen.',
      completionSoundTitle: 'Abschluss-Sound',
      completionSoundDesc:
        'Wird abgespielt, wenn ein Agent-Turn endet. Wählen Sie eine Vorlage aus und hören Sie sie hier an.',
      completionSoundPreview: 'Vorschau'
    },
    sections: {
      model: 'Modell',
      chat: 'Chat',
      appearance: 'Darstellung',
      workspace: 'Arbeitsbereich',
      safety: 'Sicherheit',
      memory: 'Speicher & Kontext',
      voice: 'Sprache',
      advanced: 'Erweitert'
    },
    searchPlaceholder: {
      about: 'Über Hermes Desktop',
      config: 'Einstellungen durchsuchen…',
      gateway: 'Gateway-Verbindung…',
      keys: 'API-Schlüssel durchsuchen…',
      mcp: 'MCP-Server durchsuchen…',
      sessions: 'Archivierte Sessions durchsuchen…'
    },
    modeOptions: {
      light: {
        label: 'Hell',
        description: 'Helle Desktop-Oberflächen'
      },
      dark: {
        label: 'Dunkel',
        description: 'Arbeitsbereich mit geringer Blendung'
      },
      system: {
        label: 'System',
        description: 'Der Darstellung des Systems folgen'
      }
    },
    appearance: {
      title: 'Darstellung',
      intro: 'Nur für Desktop. Modus ist die Helligkeit; Theme ist Farbpalette und Chat-Design.',
      colorMode: 'Farbmodus',
      colorModeDesc: 'Wählen Sie einen festen Modus oder lassen Sie Hermes Ihrer Systemeinstellung folgen.',
      toolViewTitle: 'Tool-Aufruf-Anzeige',
      toolViewDesc: 'Produkt versteckt rohe Tool-Payloads; Technisch zeigt vollständige Ein-/Ausgabe.',
      hideCodeDiffsTitle: 'Code-Diffs ausblenden',
      hideCodeDiffsDesc:
        'Dateiänderungen als Tool-Zeilen mit Anzahl hinzugefügter/entfernter Zeilen anzeigen, ohne den Code.',
      hideThreadTimelineTitle: 'Zeitleistenbalken ausblenden',
      hideThreadTimelineDesc: 'Blendet die Navigationsbalken am rechten Rand jeder Unterhaltung aus.',
      reasoningCollapsedTitle: 'Gedanken standardmäßig einklappen',
      reasoningCollapsedDesc: 'Gestreamte Gedankengänge verfügbar halten, ohne sie aufzuklappen, bis Sie sie öffnen.',
      uiScaleTitle: 'UI-Skalierung',
      uiScaleDesc: (percent: number) =>
        `Skaliert Text und Bedienelemente in der gesamten App. Cmd/Ctrl mit +, - und 0 funktioniert ebenfalls. Aktuell: ${percent}%.`,
      sessionDensityTitle: 'Dichte der Session-Liste',
      sessionDensityDesc: 'Wählen Sie, wie viel Kontext unter den Session-Titeln in der Seitenleiste erscheint.',
      sessionDensityCompact: 'Kompakt',
      sessionDensityComfortable: 'Komfortabel',
      sessionDensityDetailed: 'Detailreich',
      tabStripTitle: 'Tab-Leiste',
      tabStripDesc:
        'Zeigt Tabs über einer Zone. Blendet sie automatisch aus, wenn eine Zone nur einen einzelnen Bereich enthält.',
      tabStripAuto: 'Automatisch',
      tabStripAlways: 'Immer',
      tabStripNever: 'Nie',
      appActionsTitle: 'App-Aktionen',
      appActionsDesc:
        'Wo Einstellungen, Layout und HUD in der Titelleiste sitzen. Rechts lässt Platz für Tabs auf der linken Seite.',
      appActionsLeft: 'Links',
      appActionsRight: 'Rechts',
      terminalFontTitle: 'Terminalschrift',
      terminalFontDesc:
        'Wählen Sie eine installierte Schrift für Desktop-Terminals. Nerd Fonts rendern Powerlevel10k- und Shell-Icons; lassen Sie das Feld leer, um das gebündelte JetBrains Mono zu verwenden.',
      terminalFontPlaceholder: 'MesloLGS NF oder ein CSS-Font-Stack',
      terminalFontPreview: 'Glyph-Vorschau',
      terminalFontReset: 'Standard verwenden',
      chatFontTitle: 'Chat-Schrift',
      chatFontDesc:
        'Wählen Sie eine installierte Schrift für den Chat und den Rest der App. Praktisch für Lesbarkeitsschriften wie OpenDyslexic; lassen Sie das Feld leer, um die Schrift des Themes zu verwenden.',
      chatFontPlaceholder: 'OpenDyslexic oder ein CSS-Font-Stack',
      chatFontPreview: 'Vorschau',
      chatFontSample: 'Franz jagt im komplett verwahrlosten Taxi quer durch Bayern. 0123456789',
      chatFontReset: 'Theme-Schrift verwenden',
      translucencyTitle: 'Fenster-Transluzenz',
      translucencyDesc:
        'Sehen Sie Ihren Desktop durch das ganze Fenster hindurch, einschließlich Text. Für hell und dunkel separat abgestimmt.',
      translucencyGlassDesc:
        'Mattglas: Der Desktop scheint als weicher Blur durch, während der Text scharf bleibt. Für hell und dunkel separat abgestimmt.',
      translucencyModeClear: 'Klar',
      translucencyModeGlass: 'Glas',
      translucencyTintTitle: 'Tönung',
      translucencyFadeTitle: 'Verblassen',
      translucencyFrostTitle: 'Frost',
      translucencyFrost: {
        'under-window': 'Tief',
        popover: 'Weich',
        titlebar: 'Hell',
        header: 'Glanz'
      },
      translucencyScopeTitle: 'Bereich',
      translucencyScope: {
        window: 'Ganzes Fenster',
        sidebar: 'Nur Seitenleiste'
      },
      backdropTitle: 'Chat-Hintergrund',
      backdropDesc: 'Das zarte Statuenbild hinter der Konversation.',
      userBubbleTitle: 'Nachrichten-Blase',
      userBubbleDesc:
        'Wie durchsichtig Ihre eigenen Nachrichten sind. Bei 0 deckend; bei 100 bleibt nur die Kontur übrig.',
      textDirectionTitle: 'Textrichtung',
      textDirectionDesc:
        'Legt die Schreibrichtung von Chatnachrichten und Eingabefeld fest. Auto richtet sich nach dem ersten Buchstaben jedes Absatzes; wählen Sie eine Richtung, wenn gemischter Text falsch ausgerichtet ist. Code bleibt immer linksläufig.',
      textDirection: { auto: 'Auto', rtl: 'Rechts nach links', ltr: 'Links nach rechts' },
      introSplashTitle: 'Intro-Splash',
      introSplashDesc: 'Das Wortzeichen und der Prompt, die bei einem leeren Chat angezeigt werden.',
      reactionsTitle: 'Nachrichten-Reaktionen',
      reactionsDesc:
        'Emoji-Tapbacks im iMessage-Stil – reagieren Sie auf Nachrichten, und Hermes kann auf Ihre reagieren.',
      tipsTitle: 'In-App-Tipps',
      tipsDesc:
        'Eine kleine Blase, die auf einen Teil der App zeigt und gelegentlich im Leerlauf sowie von Hermes erscheint, wenn es hilft. Beim Schließen wird sie für immer ausgeblendet.',
      tipsReset: (count: number) => `${count} geschlossene ${count === 1 ? 'Blase' : 'Blasen'} zurückholen`,
      toursTitle: 'Geführte Touren',
      toursDesc:
        'Lassen Sie sich von Hermes durch die App führen – der Bildschirm wird abgedunkelt und jeder Schritt hervorgehoben.',
      composerPopoutTitle: 'Schwebender Composer',
      composerPopoutDesc:
        'Erlaubt, den Composer aus seiner Ablage herauszuziehen. Schalten Sie das aus, um ihn unten fixiert zu halten.',
      vibeHeartsTitle: 'Vibe-Herzen',
      vibeHeartsDesc:
        'Schwebende Herzen, wenn Sie danke, ilu, guter Bot sagen oder ein Herz senden. Unabhängig von den Nachrichten-Reaktionen oben.',
      embedsTitle: 'Inline-Embeds',
      embedsDesc:
        'Reichhaltige Vorschauen werden von Drittanbieter-Sites geladen (YouTube, X, …). „Fragen“ zeigt einen Platzhalter, bis Sie jede einzelne erlauben; „Immer“ lädt sie automatisch; „Aus“ belässt einfache Links.',
      embedsAsk: 'Fragen',
      embedsAlways: 'Immer',
      embedsOff: 'Aus',
      embedsReset: (count: number) => `${count} erlaubte ${count === 1 ? 'Aktivität' : 'Aktivitäten'} zurücksetzen`,
      resumeLastSessionTitle: 'Letzten Chat beim Start wieder öffnen',
      resumeLastSessionDesc:
        'Wenn aktiviert, öffnet die App beim Kaltstart Ihren letzten Chat wieder. Schalten Sie es aus, um immer mit einem frischen Chat zu starten.',
      product: 'Produkt',
      productDesc: 'Menschlich verständliche Tool-Aktivität mit knappen Zusammenfassungen.',
      technical: 'Technisch',
      technicalDesc: 'Rohe Tool-Argumente/-Ergebnisse und Low-Level-Details einbeziehen.',
      themeTitle: 'Theme',
      themeDesc: 'Nur Desktop-Paletten. Der gewählte Modus wird oben drauf angewendet.',
      themeSearchPlaceholder: 'Ihre Themes oder den VS Code Marketplace durchsuchen…',
      themeProfileNote: profile => `Für das Profil ${profile} gespeichert — jedes Profil behält sein eigenes Theme.`,
      installTitle: 'Aus VS Code installieren',
      installDesc:
        'Fügen Sie eine Marketplace-Erweiterungs-ID ein (z. B. dracula-theme.theme-dracula), um ihr Farbschema in eine Desktop-Palette umzuwandeln.',
      installPlaceholder: 'publisher.extension',
      installButton: 'Installieren',
      installing: 'Wird installiert…',
      installError: 'Dieses Theme konnte nicht installiert werden.',
      installed: name => `„${name}“ installiert.`,
      removeTheme: 'Theme entfernen',
      importedBadge: 'Importiert',
      pet: {
        title: 'Haustier',
        intro:
          'Adoptieren Sie ein animiertes Petdex-Maskottchen, das über der App schwebt und darauf reagiert, was Hermes gerade tut – es rennt, während Tools laufen, feiert bei Erfolg und schmollt bei Fehlern.',
        restartHint:
          'Haustiere erfordern einen kurzen Neustart – die laufende App wurde gestartet, bevor diese Funktion hinzugefügt wurde. Schließen Sie Hermes, öffnen Sie es erneut und kehren Sie dann hierher zurück.',
        scaleTitle: 'Größe',
        scaleDesc: 'Ändert die Größe des schwebenden Maskottchens. Wirkt überall sofort.',
        roamTitle: 'Herumstreifen',
        roamDesc: 'Das Haustier wandert im Leerlauf selbstständig durch das Fenster.',
        chooseTitle: 'Haustier auswählen',
        chooseDesc: 'Auswählen installiert eines (falls nötig) und macht es aktiv.',
        searchPlaceholder: 'Haustiere suchen…',
        unreachable:
          'Die Petdex-Galerie konnte nicht erreicht werden. Prüfen Sie Ihre Verbindung und öffnen Sie diese Seite erneut.',
        noMatch: query => `Keine Haustiere passen zu „${query}“.`,
        installedTag: 'installiert',
        generatedTag: 'Generiert',
        countCapped: (cap, total) => `${cap} von ${total} angezeigt – tippen Sie, um die Auswahl einzugrenzen.`,
        count: n => `${n} Haustier${n === 1 ? '' : 'er'}.`,
        uninstall: name => `${name} deinstallieren`,
        delete: name => `${name} löschen`,
        deleteTitle: name => `${name} löschen?`,
        deleteBody: 'Das löscht das Haustier endgültig — es kann nicht neu installiert werden.',
        deleteConfirm: 'Löschen',
        rename: name => `${name} umbenennen`,
        renameTitle: 'Haustier umbenennen',
        renamePlaceholder: 'Geben Sie Ihrem Haustier einen Namen',
        renameSave: 'Speichern',
        exportPet: name => `${name} exportieren`,
        adoptFailed: slug => `„${slug}“ konnte nicht adoptiert werden`,
        uninstallFailed: slug => `„${slug}“ konnte nicht deinstalliert werden`,
        renameFailed: slug => `„${slug}“ konnte nicht umbenannt werden`,
        exportFailed: slug => `„${slug}“ konnte nicht exportiert werden`,
        noneAvailable: 'Aktuell sind keine Haustiere verfügbar, die eingeschaltet werden können.',
        turnOnFailed: 'Das Haustier konnte nicht eingeschaltet werden.',
        turnOffFailed: 'Das Haustier konnte nicht ausgeschaltet werden.'
      }
    },
    fieldLabels: defineFieldCopy({
      model: 'Standardmodell',
      modelContextLength: 'Kontextfenster',
      fallbackProviders: 'Fallback-Modelle',
      toolsets: 'Aktivierte Toolsets',
      timezone: 'Zeitzone',
      display: {
        personality: 'Persönlichkeit',
        showReasoning: 'Denkblöcke'
      },
      desktop: {
        repoScanEnabled: 'Automatische Repository-Erkennung',
        repoScanRoots: 'Repository-Erkennungs-Wurzeln',
        repoScanExcludePaths: 'Ausgeschlossene Repository-Pfade'
      },
      agent: {
        maxTurns: 'Maximale Agent-Schritte',
        imageInputMode: 'Bildanhänge',
        apiMaxRetries: 'API-Wiederholungen',
        serviceTier: 'Service-Stufe',
        toolUseEnforcement: 'Tool-Nutzungs-Durchsetzung'
      },
      terminal: {
        cwd: 'Arbeitsverzeichnis',
        backend: 'Ausführungs-Backend',
        timeout: 'Befehls-Timeout',
        persistentShell: 'Persistente Shell',
        envPassthrough: 'Umgebungsvariablen-Durchreichung',
        dockerImage: 'Docker-Image',
        singularityImage: 'Singularity-Image',
        modalImage: 'Modal-Image',
        daytonaImage: 'Daytona-Image'
      },
      fileReadMaxChars: 'Datei-Lesegrenze',
      toolOutput: {
        maxBytes: 'Terminal-Ausgabelimit',
        maxLines: 'Datei-Seitenlimit',
        maxLineLength: 'Zeilenlängenlimit'
      },
      codeExecution: {
        mode: 'Code-Ausführungsmodus'
      },
      approvals: {
        mode: 'Genehmigungsmodus',
        timeout: 'Genehmigungs-Timeout',
        mcpReloadConfirm: 'MCP-Neuladen bestätigen'
      },
      commandAllowlist: 'Befehls-Whitelist',
      security: {
        redactSecrets: 'Geheimnisse schwärzen',
        allowPrivateUrls: 'Private URLs erlauben'
      },
      browser: {
        allowPrivateUrls: 'Private Browser-URLs',
        autoLocalForPrivateUrls: 'Lokaler Browser für private URLs',
        useRealProfile: 'Mein echtes Browser-Profil verwenden'
      },
      checkpoints: {
        enabled: 'Datei-Checkpoints',
        maxSnapshots: 'Checkpoint-Limit'
      },
      voice: {
        maxRecordingSeconds: 'Maximale Aufnahmelänge',
        autoTts: 'Antworten vorlesen',
        voiceChatMode: 'Sprachchat-Modus',
        gptLive: {
          voice: 'GPT-Live-Stimme',
          instructions: 'GPT-Live-Persona'
        }
      },
      stt: {
        enabled: 'Spracherkennung',
        echoTranscripts: 'Transkripte wiedergeben',
        provider: 'Spracherkennungs-Anbieter',
        local: {
          model: 'Lokales Transkriptionsmodell',
          language: 'Transkriptionssprache'
        },
        openai: {
          model: 'OpenAI-STT-Modell'
        },
        groq: {
          model: 'Groq-STT-Modell'
        },
        mistral: {
          model: 'Mistral-STT-Modell'
        },
        elevenlabs: {
          modelId: 'ElevenLabs-STT-Modell',
          languageCode: 'ElevenLabs-Sprache',
          tagAudioEvents: 'Audio-Ereignisse markieren',
          diarize: 'Sprecher-Diarisierung'
        }
      },
      tts: {
        provider: 'Text-zu-Sprache-Anbieter',
        edge: {
          voice: 'Edge-Stimme'
        },
        openai: {
          model: 'OpenAI-TTS-Modell',
          voice: 'OpenAI-Stimme'
        },
        elevenlabs: {
          voiceId: 'ElevenLabs-Stimme',
          modelId: 'ElevenLabs-Modell'
        },
        xai: {
          voiceId: 'xAI (Grok) Stimme',
          language: 'xAI-Sprache',
          speed: 'xAI-Wiedergabegeschwindigkeit',
          autoSpeechTags: 'xAI automatische Sprach-Tags',
          optimizeStreamingLatency: 'xAI Streaming-Latenz-Optimierung',
          sampleRate: 'xAI-Abtastrate',
          bitRate: 'xAI-Bitrate'
        },
        minimax: {
          model: 'MiniMax-TTS-Modell',
          voiceId: 'MiniMax-Stimme'
        },
        mistral: {
          model: 'Mistral-TTS-Modell',
          voiceId: 'Mistral-Stimme'
        },
        gemini: {
          model: 'Gemini-TTS-Modell',
          voice: 'Gemini-Stimme'
        },
        neutts: {
          model: 'NeuTTS-Modell',
          device: 'NeuTTS-Gerät'
        },
        kittentts: {
          model: 'KittenTTS-Modell',
          voice: 'KittenTTS-Stimme'
        },
        piper: {
          voice: 'Piper-Stimme'
        },
        deepinfra: {
          model: 'DeepInfra-TTS-Modell',
          voice: 'DeepInfra-Stimme'
        }
      },
      memory: {
        memoryEnabled: 'Persistentes Gedächtnis',
        userProfileEnabled: 'Benutzerprofil',
        memoryCharLimit: 'Gedächtnis-Budget',
        userCharLimit: 'Profil-Budget',
        provider: 'Gedächtnis-Anbieter'
      },
      context: {
        engine: 'Kontext-Engine'
      },
      compression: {
        enabled: 'Auto-Kompression',
        threshold: 'Kompression-Schwelle',
        codexGpt55Autoraise: 'Automatische Codex-Komprimierungsanhebung',
        targetRatio: 'Kompression-Ziel',
        protectLastN: 'Geschützte letzte Nachrichten'
      },
      auxiliary: {
        compression: {
          timeout: 'Timeout des Komprimierungsmodells (s)'
        }
      },
      delegation: {
        model: 'Subagent-Modell',
        provider: 'Subagent-Anbieter',
        maxIterations: 'Subagent-Rundenlimit',
        maxConcurrentChildren: 'Parallele Subagenten',
        childTimeoutSeconds: 'Subagent-Timeout',
        reasoningEffort: 'Subagent-Denkanstrengung'
      },
      updates: {
        nonInteractiveLocalChanges: 'Lokale Änderungen bei In-App-Update'
      }
    }),
    fieldDescriptions: defineFieldCopy({
      model: 'Wird für neue Chats verwendet, sofern Sie im Composer kein anderes Modell wählen.',
      modelContextLength: 'Auf 0 lassen, um das erkannte Kontextfenster des gewählten Modells zu verwenden.',
      fallbackProviders: 'Backup-Anbieter:Modell-Einträge, die versucht werden, wenn das Standardmodell fehlschlägt.',
      display: {
        personality: 'Standard-Assistentenstil für neue Sessions.',
        showReasoning: 'Denkabschnitte anzeigen, wenn das Backend sie liefert.'
      },
      desktop: {
        repoScanEnabled: 'Lokale Ordner nach Git-Repositories durchsuchen, die in Projekten angezeigt werden.',
        repoScanRoots: 'Zu durchsuchende Ordner. Leer lassen, um Ihr Home-Verzeichnis zu durchsuchen.',
        repoScanExcludePaths: 'Ordner und deren Unterordner, die bei der Repository-Erkennung übersprungen werden.'
      },
      timezone: 'IANA-Zeitzonenkennung. Leer verwendet die Systemzeitzone.',
      browser: {
        useRealProfile:
          'Lokales Browsen nutzt Ihre echten Anmeldungen. Hermes kopiert das Profil Ihres Standardbrowsers (Cookies, Anmeldungen, Einstellungen) in einen verwalteten Schnappschuss und steuert ihn mit seinem gebündelten Chromium – Ihr Live-Profil wird nie direkt geöffnet, und die Kopie wird bei jedem Lauf daraus aktualisiert. Erlaubt dem Agenten außerdem, auf Anfrage eine lokale Session mit echtem Profil zu öffnen, selbst wenn ein Cloud-Browser-Backend konfiguriert ist. Nur Chromium-Browser (Chrome, Edge, Brave, Brave Origin, Chromium) werden unterstützt; ein Nicht-Chromium-Standard schlägt mit einer klaren Meldung fehl. Standardmäßig aus.'
      },
      agent: {
        imageInputMode: 'Steuert, wie Bildanhänge an das Modell gesendet werden.',
        maxTurns: 'Obergrenze für Tool-Aufruf-Runden, bevor Hermes einen Lauf stoppt.'
      },
      terminal: {
        cwd: 'Standard-Projektordner für Tool- und Terminal-Arbeit.',
        persistentShell: 'Shell-Zustand zwischen Befehlen beibehalten, wenn das Backend es unterstützt.',
        envPassthrough: 'Umgebungsvariablen, die in die Tool-Ausführung durchgereicht werden.',
        dockerImage: 'Container-Image, das verwendet wird, wenn das Ausführungs-Backend Docker ist.',
        singularityImage: 'Image, das verwendet wird, wenn das Ausführungs-Backend Singularity ist.',
        modalImage: 'Image, das verwendet wird, wenn das Ausführungs-Backend Modal ist.',
        daytonaImage: 'Image, das verwendet wird, wenn das Ausführungs-Backend Daytona ist.'
      },
      codeExecution: {
        mode: 'Wie streng die Code-Ausführung auf das aktuelle Projekt begrenzt ist.'
      },
      fileReadMaxChars: 'Maximale Zeichenzahl, die Hermes aus einer Dateianfrage lesen kann.',
      approvals: {
        mode: 'Wie Hermes Befehle behandelt, die eine explizite Genehmigung benötigen.',
        timeout: 'Wie lange Genehmigungsaufforderungen warten, bevor sie ablaufen.'
      },
      security: {
        redactSecrets: 'Erkannte Geheimnisse nach Möglichkeit aus modellsichtbarem Inhalt ausblenden.'
      },
      checkpoints: {
        enabled: 'Rollback-Schnappschüsse vor Dateibearbeitungen erstellen.'
      },
      memory: {
        memoryEnabled: 'Dauerhafte Erinnerungen speichern, die zukünftigen Sessions helfen können.',
        userProfileEnabled: 'Ein kompaktes Profil der Benutzerpräferenzen pflegen.'
      },
      context: {
        engine: 'Strategie zur Verwaltung langer Gespräche nahe der Kontextgrenze.'
      },
      compression: {
        enabled: 'Älteren Kontext zusammenfassen, wenn Gespräche groß werden.',
        codexGpt55Autoraise: 'Komprimierung bei unterstützten ChatGPT-Codex-OAuth-Modellen auf 85 % anheben.'
      },
      auxiliary: {
        compression: {
          timeout:
            'Sekunden, die pro Aufruf auf das Hilfsmodell für Komprimierung gewartet wird (Standard 120). Für langsame lokale Modelle erhöhen.'
        }
      },
      voice: {
        autoTts: 'Assistentenantworten automatisch vorlesen.',
        voiceChatMode:
          'chained: Sprache zu Text → Hermes → Text zu Sprache mit den Anbietern unten. gpt-live: Ein Vollduplex-Sprachmodell von OpenAI (gpt-live-1) hört zu und spricht und übergibt jede echte Anfrage an Hermes – das von Ihnen gewählte Modell antwortet mit allen Tools. Erfordert einen OpenAI-API-Schlüssel; die Sprachschicht kostet 0,05 $ pro Minute.',
        gptLive: {
          voice: 'Stimme für den GPT-Live-Modus. Eigene Stimm-IDs werden akzeptiert.',
          instructions:
            'Zusätzliche Sätze für die Live-Sprachpersona (Ton, Tempo, Sprache). Hermes behält seinen eigenen System-Prompt.'
        }
      },
      tts: {
        xai: {
          voiceId: 'xAI-Stimm-ID (z. B. eve) oder eine benutzerdefinierte Stimm-ID.',
          language: 'Sprachcode (z. B. en, pt-BR) oder „auto“ für automatische Erkennung.',
          speed: 'Wiedergabegeschwindigkeit. 0,7 = langsamer, 1,0 = normal, 1,5 = schneller.',
          autoSpeechTags:
            'Ein LLM expressive Audio-Tags ([laughing], [sighs]) vor der Synthese in das Skript einfügen lassen.',
          optimizeStreamingLatency: 'Latenz- vs. Qualitäts-Abwägung. 0 = beste Qualität, 2 = niedrigste Latenz.',
          sampleRate: 'Audio-Abtastrate in Hz. Höher = bessere Qualität, größere Dateien.',
          bitRate: 'MP3-Bitrate in bps. Gilt nur, wenn der Codec mp3 ist.'
        },
        neutts: {
          device: 'Lokales Inferenzgerät für NeuTTS.'
        }
      },
      stt: {
        enabled: 'Lokale oder anbieterbasierte Sprachtranskription aktivieren.',
        echoTranscripts: 'Das rohe 🎙️-Transkript von Sprachnachrichten zurück in den Chat posten.',
        elevenlabs: {
          languageCode: 'Optionaler ISO-639-3-Sprachcode. Leer lässt ElevenLabs automatisch erkennen.'
        }
      },
      updates: {
        nonInteractiveLocalChanges:
          'Wenn Hermes sich aus der App selbst aktualisiert (ohne Terminal-Aufforderung), lokale Quellcode-Änderungen behalten (stash) oder verwerfen (discard). Terminal-Updates fragen immer nach.'
      }
    }),
    uninstallSection: {
      dangerZone: 'Gefahrenzone',
      checkingInstalled: 'Installierte Komponenten werden geprüft…',
      uninstallHermes: 'Hermes deinstallieren',
      chooseHowMuch:
        'Wählen Sie, wie viel entfernt werden soll. Die App wird zum Abschluss geschlossen; Sie können das Installationsprogramm jederzeit erneut öffnen, um zurückzukehren.',
      confirmUninstall: 'Deinstallation bestätigen',
      confirmBody: what => `Dadurch wird Folgendes entfernt: ${what}. Dies kann nicht rückgängig gemacht werden.`,
      appLabel: 'App:',
      couldNotStart: 'Die Deinstallation konnte nicht gestartet werden.',
      uninstalling: 'Wird deinstalliert…',
      yesUninstall: 'Ja, deinstallieren',
      options: {
        gui: {
          title: 'Nur die Chat-Oberfläche deinstallieren',
          description:
            'Entfernt diese Desktop-App. Der Hermes-Agent, Ihre Konfiguration und Ihre Chats bleiben erhalten.',
          consequence: 'die Desktop-Chat-Oberfläche (diese App und ihre Daten)'
        },
        lite: {
          title: 'Oberfläche + Agent deinstallieren, Daten behalten',
          description:
            'Entfernt die App und den Hermes-Agent, behält aber Konfiguration, Chats und Geheimnisse für eine spätere Neuinstallation.',
          consequence:
            'die Chat-Oberfläche und den Hermes-Agent (Konfiguration, Chats und Geheimnisse bleiben erhalten)'
        },
        full: {
          title: 'Alles deinstallieren',
          description:
            'Entfernt die App, den Agent und alle Benutzerdaten – Konfiguration, Chats, geplante Jobs, Geheimnisse, Logs.',
          consequence:
            'ALLES – die Chat-Oberfläche, den Hermes-Agent sowie Ihre gesamte Konfiguration, Chats, Geheimnisse und Logs'
        }
      }
    },
    poolLimits: {
      warmBotBackendsAria: 'Bot-Backends vorwärmen',
      warmBotBackendsTitle: 'Bot-Backends vorwärmen',
      backendIdleTimeoutAria: 'Leerlauf-Timeout des Backends in Millisekunden',
      backendIdleTimeoutTitle: 'Leerlauf-Timeout des Backends'
    },
    customEndpoints: {
      active: 'Aktiv',
      apiKeySet: 'API-Schlüssel gesetzt',
      use: 'Verwenden',
      editTitle: 'Endpunkt bearbeiten',
      addTitle: 'Endpunkt hinzufügen',
      fields: {
        name: 'Name',
        providerId: 'Anbieter-ID',
        endpointUrl: 'Endpunkt-URL',
        defaultModel: 'Standardmodell',
        context: 'Kontext',
        apiKey: 'API-Schlüssel',
        apiKeyNewPlaceholder: 'Leer lassen, um den aktuellen Schlüssel zu behalten',
        apiKeyPlaceholder: 'Optional',
        useNewChats: 'Für neue Chats verwenden',
        discoverModels: 'Modelle ermitteln'
      },
      test: 'Testen',
      save: 'Speichern',
      newEndpoint: 'Neuer Endpunkt',
      apiMode: 'API-Modus',
      autoDetect: 'Automatisch erkennen',
      couldNotLoad: 'Benutzerdefinierte Endpunkte konnten nicht geladen werden',
      endpointSaved: 'Benutzerdefinierter Endpunkt gespeichert.',
      saveFailed: 'Speichern fehlgeschlagen',
      endpointReachable: 'Der Endpunkt ist erreichbar.',
      endpointReachableTransport: transport => `Der Endpunkt ist erreichbar (${transport}-Route bedient).`,
      endpointReachableModels: (reachable, count) => `${reachable} ${count} Modell${count === 1 ? '' : 'e'} gefunden.`,
      endpointValidationFailed: 'Die Validierung des Endpunkts ist fehlgeschlagen.',
      validationFailed: 'Validierung fehlgeschlagen',
      activationFailed: 'Aktivierung fehlgeschlagen',
      deleteConfirm: name => `${name} löschen?`,
      deleteFailed: 'Löschen fehlgeschlagen',
      title: 'Eigene Endpunkte',
      deleteEndpoint: 'Endpunkt löschen',
      emptyDescription: 'Fügen Sie unten einen OpenAI-kompatiblen Endpunkt hinzu.',
      emptyTitle: 'Keine eigenen Endpunkte',
      namePlaceholder: 'Axet Proxy',
      contextPlaceholder: 'Automatisch'
    },
    computerUse: {
      accessibility: 'Bedienungshilfen',
      screenRecording: 'Bildschirmaufnahme',
      driverHealth: 'Treiberstatus'
    },
    about: {
      updates: 'Updates'
    },
    config: {
      minimizeToTrayTitle: 'In den Infobereich minimieren',
      minimizeToTrayDesc:
        'Beim Minimieren von Fenstern oder Schließen des Hauptfensters werden diese im Infobereich (Menüleiste unter macOS) ausgeblendet und Hermes läuft weiter. Beenden Sie über „Hermes beenden“ im Infobereich-Menü oder mit Cmd+Q. Standardmäßig aus; gilt nur für dieses Gerät.',
      minimizeToTrayUnavailable:
        'Der Infobereich ist nicht verfügbar. Fenster werden normal minimiert und geschlossen. Schalten Sie die Option aus und wieder ein, um es erneut zu versuchen.',
      none: 'Keine',
      noneParen: '(keine)',
      builtinOnly: 'Nur eingebaut',
      notSet: 'Nicht festgelegt',
      commaSeparated: 'durch Komma getrennte Werte',
      searchPlaceholder: 'Suchen…',
      noResults: 'Keine Ergebnisse gefunden',
      systemDefault: 'Systemstandard',
      loading: 'Hermes-Konfiguration wird geladen...',
      emptyTitle: 'Nichts zu konfigurieren',
      emptyDesc: 'Dieser Bereich hat keine einstellbaren Optionen.',
      failedLoad: 'Einstellungen konnten nicht geladen werden',
      autosaveFailed: 'Autospeichern fehlgeschlagen',
      imported: 'Konfiguration importiert',
      invalidJson: 'Ungültige Konfigurations-JSON',
      toolsetsWipeConfirm:
        'Alle aktivierten Toolsets entfernen? Das deaktiviert Speicher, Terminal, Websuche, Delegation und die meisten anderen Tools, bis Sie sie wieder aktivieren.',
      keepAwakeTitle: 'Computer wach halten',
      keepAwakeDesc:
        'Verhindert, dass dieser Rechner in den Ruhezustand wechselt, damit Läufe über Nacht oder länger weiterlaufen. Der Bildschirm kann trotzdem abdunkeln.',
      disableF12Title: 'F12-DevTools deaktivieren',
      disableF12Desc:
        'Verhindert, dass F12 die Entwicklertools öffnet. Strg+Umschalt+I (bzw. Cmd+Opt+I auf dem Mac) funktioniert weiterhin.',
      attachmentSizeTitle: 'Maximale Vorschau-/Bildladegröße',
      attachmentSizeDesc:
        'Wie groß eine lokale Datei sein darf, die Desktop für Vorschauen und Bildanhänge lädt, in MB. Standard ist 16. Remote-Anhänge ohne Bild verwenden ein eigenes Limit von 256 MB. Ein sehr hoher Wert lädt die gesamte Datei in den Speicher, was die App einfrieren oder abstürzen lassen kann.',
      attachmentSizeUnit: 'MB',
      attachmentSizeLabel: 'Maximale Vorschau-/Bildladegröße in Megabyte',
      showOptions: 'Optionen anzeigen'
    },
    hudModifier: {
      title: 'Tippen, um das HUD aufzurufen',
      description:
        'Tippen Sie kurz auf ⌘ + Option (Mac) bzw. Strg + Alt (Windows/Linux), um das HUD aus jeder App nach vorn zu holen. Standardmäßig aus; gilt nur für dieses Gerät.',
      permission:
        'Erlauben Sie Hermes unter Systemeinstellungen → Datenschutz & Sicherheit → Eingabeüberwachung und versuchen Sie es erneut. Diese Geste zeichnet keine Tastenanschläge auf und nimmt Ihren Bildschirm nicht auf.',
      unavailable:
        'Das Hilfsprogramm für die HUD-Geste konnte nicht starten oder wurde unerwartet beendet. Versuchen Sie es erneut oder starten Sie Hermes neu. Das bestehende HUD-Tastenkürzel funktioniert innerhalb von Hermes weiterhin.',
      missingHelper:
        'In dieser Hermes-Installation fehlt das Hilfsprogramm für die HUD-Geste. Aktualisieren oder installieren Sie Hermes neu und versuchen Sie es erneut.',
      unsupportedSession:
        'Diese Desktop-Session unterstützt keine globalen Modifikator-Taps. Linux erfordert X11; Wayland wird nicht unterstützt.'
    },
    screenshot: {
      enabledTitle: 'Screenshot-Kurzbefehl',
      enabledDesc:
        'Drücken Sie in einer beliebigen App beide Befehlstasten gleichzeitig, um deren vorderstes Fenster aufzunehmen und an Ihren aktuellen Hermes-Entwurf anzuhängen. Es wird nie automatisch gesendet. Standardmäßig aus; gilt nur für diesen Mac. Fensterinhalte können vertraulich sein – prüfen Sie den Anhang vor dem Senden.',
      statusTitle: 'Status des Screenshot-Kurzbefehls',
      checking: 'Screenshot-Kurzbefehl wird geprüft…',
      disabled: 'Der Screenshot-Kurzbefehl ist aus.',
      starting: 'Der Kurzbefehl-Listener wird gestartet. Er ist noch nicht bereit.',
      ready:
        'Der Kurzbefehl ist bereit. Screenshots werden an Ihren aktuellen Entwurf angehängt, ohne gesendet zu werden.',
      inputPermission:
        'Mit der Berechtigung „Eingabeüberwachung“ kann Hermes beide Befehlstasten erkennen, während eine andere App aktiv ist. Erlauben Sie Hermes unter Systemeinstellungen → Datenschutz & Sicherheit → Eingabeüberwachung, kehren Sie dann hierher zurück und versuchen Sie es erneut.',
      screenPermission:
        'Mit der Berechtigung „Bildschirmaufnahme“ kann Hermes das vorderste App-Fenster aufnehmen, wenn Sie diesen Kurzbefehl verwenden. Erlauben Sie Hermes unter Systemeinstellungen → Datenschutz & Sicherheit → Bildschirmaufnahme, kehren Sie dann hierher zurück und versuchen Sie es erneut. Starten Sie Hermes neu, wenn macOS dazu auffordert.',
      openSettings: 'Systemeinstellungen öffnen',
      retry: 'Erneut versuchen',
      unavailable: 'Der Screenshot-Kurzbefehl ist nicht verfügbar. Versuchen Sie es erneut oder schalten Sie ihn aus.',
      errorTitle: 'Fehler beim Screenshot-Kurzbefehl',
      loadFailed:
        'Der Status des Kurzbefehls konnte nicht gelesen werden. Versuchen Sie es erneut, um die aktuelle Einstellung zu prüfen.',
      saveFailed:
        'Die Änderung am Kurzbefehl konnte nicht bestätigt werden. Versuchen Sie es erneut, um die aktuelle Einstellung zu prüfen.',
      permissionFailed:
        'Die Systemeinstellungen konnten nicht geöffnet werden. Öffnen Sie „Datenschutz & Sicherheit“ manuell und versuchen Sie es erneut.',
      captureFailed: 'Das vorderste Fenster konnte nicht aufgenommen werden. Es wurde nichts angehängt oder gesendet.',
      contextChanged:
        'Der aktuelle Entwurf hat sich während der Aufnahme geändert. Der Screenshot wurde weder angehängt noch gesendet.'
    },
    quickEntry: {
      enabledTitle: 'Schnelleingabe',
      enabledDesc:
        'Öffnen Sie mit einem globalen Tastaturkürzel von überall einen kleinen Eingabebereich und senden Sie einen Prompt, ohne Hermes zu öffnen.',
      shortcutTitle: 'Tastaturkürzel der Schnelleingabe',
      shortcutDesc: 'Benötigt mindestens eine Zusatztaste, z. B. CommandOrControl+Shift+Leertaste.',
      active: 'Das Tastaturkürzel ist aktiv.',
      takenBy: 'Eine andere App verwendet dieses Tastaturkürzel bereits — wählen Sie ein anderes.',
      invalidShortcut: 'Kein gültiges Tastaturkürzel. Fügen Sie mindestens eine Zusatztaste hinzu.'
    },
    credentials: {
      pasteKey: 'Schlüssel einfügen',
      pasteLabelKey: label => `Schlüssel für ${label} einfügen`,
      optional: 'Optional',
      enterValueFirst: 'Geben Sie zuerst einen Wert ein.',
      couldNotSave: 'Die Anmeldedaten konnten nicht gespeichert werden.',
      remove: 'Entfernen',
      getKey: 'Einen Schlüssel erhalten',
      saving: 'Wird gespeichert'
    },
    envActions: {
      actions: 'Aktionen',
      manageInKeys: 'Unter API-Schlüssel verwalten',
      docs: 'Doku',
      hideValue: 'Wert ausblenden',
      revealValue: 'Wert anzeigen',
      replace: 'Ersetzen',
      set: 'Setzen',
      clear: 'Leeren'
    },
    connections: {
      title: 'Registrierte Gateways',
      intro:
        'Verwalten Sie dieses Gerät und jedes Hermes Gateway, das es über Remote-, SSH- oder Cloud-Verbindungen erreichen kann.',
      stagedNote:
        'Wechseln Sie Gateways über Sessions. Profile, Chats, Nachrichten und Cron-Jobs bleiben bei ihrem Gateway; Arbeit auf anderen Gateways läuft weiter.',
      launchModeTitle: 'Beim Start zu Sessions auf dem zuletzt verwendeten Gateway zurückkehren',
      launchModeDesc: 'Wenn deaktiviert, öffnet Sessions auf dem primären Gateway.',
      searchPlaceholder: 'Gateways durchsuchen…',
      noSearchResults: 'Keine Gateways entsprechen Ihrer Suche.',
      loadFailed: 'Verbindungen konnten nicht geladen werden',
      currentPill: 'Aktuell',
      primaryPill: 'Primär',
      managedPill: 'Von der App verwaltet',
      addConnection: 'Verbindung hinzufügen',
      editConnection: 'Bearbeiten',
      removeConnection: 'Entfernen',
      removeConfirmTitle: 'Diese Verbindung entfernen?',
      removeConfirmDesc: (label: string) =>
        `„${label}“ wird aus dieser App entfernt. Die Instanz selbst bleibt unberührt – Sie können sie jederzeit wieder hinzufügen.`,
      makePrimary: 'Als primär festlegen',
      testConnection: 'Testen',
      testOk: 'Erreichbar',
      testFailed: 'Verbindungstest fehlgeschlagen',
      saveFailed: 'Die Verbindung konnte nicht gespeichert werden',
      removeFailed: 'Die Verbindung konnte nicht entfernt werden',
      updateAll: 'Alle Instanzen aktualisieren',
      updateAllRunning: 'Alle Instanzen werden aktualisiert…',
      updateAllDone: 'Updates versendet',
      updateAllFailed: 'Update-Verteilung fehlgeschlagen',
      updateSkippedCloud: 'Wird von Hermes Cloud verwaltet',
      kindLocal: 'Lokal',
      kindRemote: 'Remote-Gateway',
      kindCloud: 'Hermes Cloud',
      kindSsh: 'SSH',
      kindLocalDesc: 'Die Hermes-Laufzeitumgebung, die von dieser App verwaltet wird.',
      kindRemoteDesc: 'Ein Hermes Gateway, das über HTTP(S) erreichbar ist – LAN, Tailscale oder das Internet.',
      kindCloudDesc: 'Eine gehostete Instanz, die über Ihr Hermes-Cloud-Konto gefunden wurde.',
      kindSshDesc: 'Eine Hermes-Installation, die über SSH erreicht wird.',
      labelTitle: 'Name',
      labelDesc:
        'Pflichtfeld. Wird überall angezeigt, wo diese Instanz erscheint; muss eindeutig sein (z. B. „Homelab“, „Arbeitslaptop“).',
      labelPlaceholder: 'Homelab',
      urlTitle: 'Gateway-URL',
      sshHostTitle: 'SSH-Host',
      headersTitle: 'Zusätzliche Gateway-Header',
      headersDesc:
        'Wird mit jeder HTTP- und WebSocket-Anfrage an dieses Gateway gesendet – für Zugriffs-Proxys wie Cloudflare Access (CF-Access-Client-Id / CF-Access-Client-Secret). Werte werden verschlüsselt gespeichert. Header, die Hermes verwaltet (Authorization, Cookie, Host…), werden ignoriert.',
      headerValuePlaceholder: 'Wert',
      headerValueSaved: 'Gespeichert – leer lassen, um zu behalten',
      headerAdd: 'Header hinzufügen',
      headerRemove: 'Entfernen',
      duplicateLocal: 'Diese App verwaltet bereits eine lokale Verbindung – es kann nur eine geben.',
      duplicateUrl: (label: string) => `Eine Verbindung zu dieser Gateway-URL existiert bereits („${label}“).`,
      duplicateSsh: (label: string) => `Eine Verbindung zu diesem SSH-Host existiert bereits („${label}“).`,
      sameBackendHint: (label: string) => `Gleiches Backend wie „${label}“`,
      localAddHint: 'Lokal ist nicht verfügbar: Die verwaltete lokale Verbindung existiert bereits (es gibt nur eine).',
      cloudAddHint:
        'Tipp: Die Anmeldung unter Hermes Cloud oben erkennt Ihre Agents automatisch – verwenden Sie dieses Formular nur, um eine bekannte Instanz-URL manuell zu registrieren.',
      save: 'Verbindung speichern',
      saving: 'Wird gespeichert…',
      cancel: 'Abbrechen',
      empty: 'Noch keine Verbindungen registriert.'
    },
    managedUpdates: {
      title: 'Verwaltete Updates',
      intro:
        'Aktualisiert von Desktop verwaltete SSH-Installationen transaktional: Sessions werden beendet, das Remote-Repository wird aktualisiert und jedes Profil wird mit einer zugehörigen Quittung wiederhergestellt.',
      sshConnection: 'Desktop-verwaltete SSH-Installation',
      update: 'Aktualisieren',
      updating: 'Wird aktualisiert…',
      progress:
        'Sessions werden beendet, die Remote-Installation wird aktualisiert und Profile werden wiederhergestellt…',
      updated: 'Aktualisiert',
      partial: 'Aktualisiert – Wiederherstellung fehlgeschlagen',
      refused: 'Verweigert',
      failed: 'Update fehlgeschlagen',
      alreadyRunning: 'Update läuft bereits',
      receipt: (id: string, outcome: string) => `Quittung ${id} · ${outcome}`,
      receiptVersions: (pre: string, post: string) => `${pre} → ${post}`,
      scopesRestored: (profiles: string) => `Wiederhergestellte Profile: ${profiles}`,
      scopeNotRestored: (profile: string, error: string) => `Profil „${profile}“ nicht wiederhergestellt: ${error}`
    },
    gateway: {
      loading: 'Gateway-Einstellungen werden geladen…',
      unavailableTitle: 'Gateway-Einstellungen nicht verfügbar',
      unavailableDesc: 'Die Desktop-IPC-Brücke stellt keine Gateway-Einstellungen bereit.',
      title: 'Gateway-Verbindung',
      envOverride: 'ENV-Überschreibung',
      intro:
        'Standardmäßig lokal. Verwenden Sie Remote, wenn diese App ein Hermes-Backend an einem anderen Ort steuern soll. Gateway-Verbindungen gelten pro Gerät; Profile werden von den Gateways ermittelt, mit denen Sie sich verbinden.',
      envOverrideTitle: 'Umgebungsvariablen steuern diese Desktop-Session.',
      envOverrideDesc:
        'Entfernen Sie HERMES_DESKTOP_REMOTE_URL und HERMES_DESKTOP_REMOTE_TOKEN, um die unten gespeicherte Einstellung zu verwenden.',
      modeTitle: 'Verbindungsmodus',
      localTitle: 'Lokales Gateway',
      localDesc: 'Startet ein privates Hermes-Backend auf localhost. Das ist der Standard und funktioniert offline.',
      remoteTitle: 'Remote-Gateway',
      remoteDesc: 'Verbindet diese Desktop-Shell mit einem entfernten Hermes-Backend.',
      remoteAuthHint:
        'Gehostete Gateways verwenden OAuth oder Benutzername und Passwort; selbst gehostete können ein Session-Token verwenden.',
      cloudTitle: 'Hermes Cloud',
      cloudDesc:
        'Melden Sie sich einmal bei Hermes Cloud an und wählen Sie aus den Agents in Ihrem Konto – ohne eine URL einzufügen.',
      cloudSignInTitle: 'Hermes Cloud',
      cloudSignIn: 'Bei Hermes Cloud anmelden',
      cloudSignedIn: 'Bei Hermes Cloud angemeldet',
      cloudNeedsSignIn: 'Melden Sie sich bei Hermes Cloud an, um die Agents in Ihrem Konto zu finden.',
      cloudSignedInDesc:
        'Sie sind angemeldet. Wählen Sie unten einen Agent; die Session wird automatisch aktualisiert.',
      cloudAgentsTitle: 'Ihre Agents',
      cloudOrgPickerTitle: 'Organisation auswählen',
      cloudOrgSelect: 'Auswählen',
      cloudOrgChange: 'Organisation ändern',
      cloudOrgRole: role => `Rolle: ${role}`,
      cloudLoadingAgents: 'Ihre Agents werden geladen…',
      cloudNoAgents: {
        before: 'Keine Agents in diesem Konto gefunden. Legen Sie einen im ',
        linkText: 'Nous-Portal',
        after: ' an und aktualisieren Sie dann.'
      },
      cloudRefresh: 'Aktualisieren',
      cloudConnect: 'Verbinden',
      cloudSavedTitle: 'Gespeicherte Cloud-Gateways',
      cloudSavedDesc:
        'Verwenden Sie ein gespeichertes Gateway, ohne Ihren Standard zu ändern. Melden Sie sich unten an, um Instanzen hinzuzufügen. Namen und Anmeldung verwalten Sie in der Liste der gespeicherten Verbindungen.',
      cloudUseSaved: 'Gateway verwenden',
      cloudActive: 'In diesem Fenster aktiv',
      cloudConnecting: 'Verbindung wird hergestellt…',
      cloudDiscoverFailed: 'Ihre Hermes-Cloud-Agents konnten nicht geladen werden',
      cloudConnectFailed: 'Keine Verbindung zu diesem Agent möglich',
      cloudSignInFailed: 'Anmeldung bei Hermes Cloud fehlgeschlagen',
      cloudSignedOutTitle: 'Von Hermes Cloud abgemeldet',
      cloudSignedOutMessage: 'Die Hermes-Cloud-Session wurde geleert.',
      cloudConnectedTitle: 'Verbunden',
      cloudConnectedPill: 'Verbunden',
      cloudConnectedTo: name => `Mit ${name} verbunden.`,
      cloudAgentProvisioning: 'Provisionierung…',
      cloudStatusLabel: status => `Status: ${status}`,
      remoteUrlTitle: 'Remote-URL',
      remoteUrlDesc: 'Basis-URL für das Remote-Dashboard-Backend. Pfad-Präfixe werden unterstützt, z. B. /hermes.',
      probing: 'Authentifizierungsmethode dieses Gateways wird geprüft…',
      probeError:
        'Dieses Gateway ist noch nicht erreichbar. Prüfen Sie die URL – die Authentifizierungsmethode erscheint, sobald es antwortet.',
      signedIn: 'Angemeldet',
      signIn: 'Anmelden',
      signOut: 'Abmelden',
      signInWith: provider => `Mit ${provider} anmelden`,
      authTitle: 'Authentifizierung',
      authSignedInPassword:
        'Dieses Gateway verwendet Benutzername und Passwort. Sie sind angemeldet; die Session wird automatisch aktualisiert.',
      authSignedInOauth:
        'Dieses Gateway verwendet OAuth. Sie sind angemeldet; die Session wird automatisch aktualisiert.',
      authNeedsPassword:
        'Dieses Gateway verwendet Benutzername und Passwort. Melden Sie sich an, um diese Desktop-App zu autorisieren.',
      authNeedsOauth: provider =>
        `Dieses Gateway verwendet OAuth. Melden Sie sich mit ${provider} an, um diese Desktop-App zu autorisieren.`,
      tokenTitle: 'Session-Token',
      tokenDesc:
        'Das Dashboard-Session-Token für REST- und WebSocket-Zugriff. Leer lassen, um das gespeicherte Token zu behalten.',
      existingToken: value => `Bestehendes Token ${value}`,
      savedToken: 'gespeichert',
      pasteSessionToken: 'Session-Token einfügen',
      plainTextConfirmTitle: 'Gateway-Token im Klartext speichern?',
      plainTextConfirmDesc:
        'Auf diesem Gerät wurde kein Schlüsselbund-Dienst des Betriebssystems gefunden. Das Token würde daher unverschlüsselt in der Verbindungseinstellungsdatei der App gespeichert und wäre für jeden Prozess dieses Benutzers lesbar. Installieren oder aktivieren Sie GNOME Keyring oder KWallet für verschlüsselte Speicherung.',
      plainTextConfirmAction: 'Als Klartext speichern',
      plainTextStoredTitle: 'Token im Klartext gespeichert',
      plainTextStoredDesc:
        'Sichere Speicherung ist nicht verfügbar, daher wird das gespeicherte Token unverschlüsselt in der Verbindungseinstellungsdatei der App auf diesem Gerät gespeichert. Installieren oder aktivieren Sie GNOME Keyring oder KWallet, um es zu verschlüsseln.',
      keychainEncryptionTitle: 'Gespeicherte Geheimnisse mit dem OS-Keychain verschlüsseln',
      keychainEncryptionDesc:
        'Standardmäßig aus. Wenn aktiv, werden Gateway-Tokens und Anmeldedaten mit dem Schlüsselbund Ihres Systems verschlüsselt (Schlüsselbundverwaltung, GNOME Keyring oder Windows DPAPI) – Ihr System fragt möglicherweise nach einer Erlaubnis oder einem Passwort. Wenn aus, werden sie als Klartextdateien gespeichert, die nur Ihr Benutzerkonto lesen kann.',
      keychainEncryptionFailed: 'Geheimnis-Verschlüsselung konnte nicht geändert werden',
      testRemote: 'Remote testen',
      saveForRestart: 'Für den nächsten Neustart speichern',
      saveAndReconnect: 'Speichern und neu verbinden',
      diagnostics: 'Diagnose',
      diagnosticsDesc: 'Zeigt desktop.log in Ihrem Dateimanager an – nützlich, wenn das Gateway nicht startet.',
      openLogs: 'Protokolle öffnen',
      incompleteTitle: 'Remote-Gateway unvollständig',
      incompleteSignIn: 'Geben Sie eine Remote-URL ein und melden Sie sich an, bevor Sie zu Remote wechseln.',
      incompleteToken: 'Geben Sie eine Remote-URL und ein Session-Token ein, bevor Sie zu Remote wechseln.',
      incompleteSignInTest: 'Geben Sie eine Remote-URL ein und melden Sie sich an, bevor Sie testen.',
      incompleteTokenTest: 'Geben Sie eine Remote-URL und ein Session-Token ein, bevor Sie testen.',
      enterUrlFirst: 'Geben Sie zuerst eine Remote-URL ein.',
      restartingTitle: 'Gateway-Verbindung wird neu gestartet',
      savedTitle: 'Gateway-Einstellungen gespeichert',
      restartingMessage:
        'Hermes Desktop stellt mit den gespeicherten Einstellungen die Verbindung wieder her — die Shell bleibt offen.',
      savedMessage: 'Für den nächsten Neustart gespeichert.',
      connectedTo: (baseUrl, version) => `Verbunden mit ${baseUrl}${version ? ` · Hermes ${version}` : ''}`,
      reachableTitle: 'Remote-Gateway erreichbar',
      signedOutTitle: 'Abgemeldet',
      signedOutMessage: 'Die Remote-Gateway-Session wurde geleert.',
      failedLoad: 'Gateway-Einstellungen konnten nicht geladen werden',
      signInFailed: 'Anmeldung fehlgeschlagen',
      signOutFailed: 'Abmeldung fehlgeschlagen',
      testFailed: 'Remote-Gateway-Test fehlgeschlagen',
      applyFailed: 'Gateway-Einstellungen konnten nicht angewendet werden',
      saveFailed: 'Gateway-Einstellungen konnten nicht gespeichert werden',
      sshTitle: 'Über SSH verbinden',
      sshDesc:
        'Hermes wird per SSH auf dem Remote-Gerät gestartet und in diese App getunnelt – Sie müssen nichts selbst starten oder freigeben. Erfordert funktionierenden, schlüsselbasierten SSH-Zugriff auf den Host.',
      sshTrustHint: 'Der erste präsentierte Host-Key wird vertraut und gepinnt; spätere Änderungen schlagen fehl.',
      sshHostTitle: 'Host',
      sshHostDesc: 'user@host oder ein Host-Alias aus ~/.ssh/config.',
      sshHostPick: 'Host auswählen…',
      sshHostPickTitle: 'Host',
      sshHostPickDesc: 'Ein Host-Alias aus ~/.ssh/config oder Custom zum manuellen Eingeben.',
      sshHostCustom: 'Custom (manuell eingeben)…',
      sshUserTitle: 'Benutzer',
      sshUserDesc: 'Leer = ~/.ssh/config oder Ihr aktueller Benutzer.',
      sshUserPlaceholder: 'aus ~/.ssh/config',
      sshPortTitle: 'Port',
      sshPortDesc: 'Leer = 22 oder der Port aus ~/.ssh/config.',
      sshKeyTitle: 'Identitätsdatei',
      sshKeyDesc: 'Pfad zum privaten Schlüssel. Leer = ssh-agent oder ~/.ssh/config.',
      sshHermesPathTitle: 'Hermes-Pfad (optional)',
      sshHermesPathDesc: 'Vollständiger Pfad zum Remote-Hermes-Binary. Leer = automatisch erkennen.',
      sshHermesPathPlaceholder: 'automatisch erkennen',
      sshTestConnection: 'SSH testen',
      sshConnect: 'Verbinden',
      sshButtonsHint: 'Speichern wird beim nächsten Start angewendet. Verbinden verbindet sofort neu.',
      sshReachable: (host, platform) => `Erreichbar: ${host} (${platform}) — Hermes gefunden`,
      sshIncompleteHost: 'Geben Sie einen SSH-Host ein, bevor Sie sich verbinden.',
      sshErrUnreachable: 'Dieser Host ist über SSH nicht erreichbar. Prüfen Sie Host, Port und Ihr Netzwerk.',
      sshErrAuth:
        'SSH-Authentifizierung fehlgeschlagen. Laden Sie Ihren Schlüssel in den ssh-agent (ssh-add) oder setzen Sie eine IdentityFile in ~/.ssh/config – Hermes führt ssh nicht interaktiv aus.',
      sshErrHostKey:
        'Der Host-Key hat sich seit Ihrer letzten Verbindung GEÄNDERT. Prüfen Sie, ob das erwartet ist, führen Sie dann ssh-keygen -R <host> aus und verbinden Sie sich erneut.',
      sshErrNotInstalled:
        'Hermes ist auf dem Remote-Host nicht installiert. Installieren Sie es dort (curl -fsSL https://hermes-agent.nousresearch.com/install.sh | sh) oder legen Sie den Hermes-Pfad fest.',
      sshErrPlatform:
        'Nicht unterstützte Remote-Plattform. Der Desktop-SSH-Modus von Hermes unterstützt Linux-, macOS- und Windows-Remote-Hosts.',
      sshErrTimeout: 'SSH-Verbindung ist ausgelaufen. Der Host ist möglicherweise nicht erreichbar oder schläft.',
      sshErrUpdateRequired: 'Aktualisieren Sie Hermes auf dem Remote-Host, bevor Sie sich mit Desktop-SSH verbinden.',
      sshErrUnknown: 'SSH-Verbindung fehlgeschlagen.'
    },
    keys: {
      loading: 'API-Schlüssel und Anmeldedaten werden geladen…',
      failedLoad: 'API-Schlüssel konnten nicht geladen werden',
      empty: 'In dieser Kategorie ist noch nichts konfiguriert.'
    },
    search: {
      placeholder: 'Alle Einstellungen durchsuchen…',
      pill: 'Suchen'
    },
    profileScope: {
      appliesTo: 'Gilt für',
      editsProfile: profile => `Änderungen auf dieser Seite gelten für das Profil „${profile}“.`
    },
    mcp: {
      loading: 'MCP-Server werden geladen…',
      invalidJson: 'Ungültiges MCP-JSON',
      saveFailed: 'Speichern fehlgeschlagen',
      removeFailed: 'Entfernen fehlgeschlagen',
      reloadFailed: 'MCP-Neuladen fehlgeschlagen',
      savedTitle: 'MCP-Server gespeichert',
      savedMessage: name => `${name} wird nach MCP-Neuladen angewendet.`,
      disabled: 'deaktiviert',
      name: 'Name',
      serverJson: 'Server-JSON',
      remove: 'Entfernen',
      test: 'Verbindung testen',
      catalogLoading: 'MCP-Katalog wird geladen…',
      catalogInstallFailed: name => `${name} konnte nicht installiert werden`,
      catalogEnvRequired: 'Füllen Sie die erforderlichen Werte aus, bevor Sie installieren.',
      capabilitySummary: (tools, prompts, resources) =>
        `${[`${tools} Tools`, ...(prompts ? [`${prompts} Prompts`] : []), ...(resources ? [`${resources} Ressourcen`] : [])].join(', ')} aktiviert`,
      costTokens: tokens => `~${tokens} Tok/Aufruf`,
      usage30d: uses => `${uses} Nutzungen/30d`,
      statusConnecting: 'Verbindung wird hergestellt…',
      statusNeedsAuth: 'Authentifizierung nötig',
      statusError: 'Fehler',
      statusOff: 'Aus',
      allServers: 'Alle Server',
      authenticatedTitle: 'Authentifiziert',
      authenticatedMessage: (server, count) => `${server}: ${count} Tools`,
      authenticate: 'Authentifizieren',
      noOutput: 'Noch keine Ausgabe.',
      deepLinkTitle: 'MCP-Server hinzufügen?',
      deepLinkDescription:
        'Ein Link möchte diesen MCP-Server zu Hermes hinzufügen. Prüfen Sie die genaue Konfiguration unten – sie stammt vom Link, nicht von Hermes.',
      deepLinkStdioWarning:
        'Dieser Server führt mit dem unten angezeigten Befehl einen lokalen Prozess auf Ihrem Rechner aus. Fahren Sie nur fort, wenn Sie seiner Quelle vertrauen.',
      deepLinkConfirm: 'Server hinzufügen',
      deepLinkNameInvalid: 'Namen verwenden 1–64 Buchstaben, Ziffern, Punkte, Striche oder Unterstriche.',
      deepLinkNameConflict: name =>
        `Ein Server namens ${name} existiert bereits — wählen Sie einen anderen Namen oder brechen Sie ab.`,
      deepLinkErrorTitle: 'MCP-Installationslink abgelehnt',
      deepLinkErrorName: 'Der Servername des Links fehlt oder ist ungültig.',
      deepLinkErrorConfig: 'Die Config des Links ist kein gültiges base64-kodiertes JSON.',
      deepLinkErrorShape: 'Die Config muss ein JSON-Objekt mit einem String-Feld `url` oder `command` sein.',
      deepLinkErrorUrl: 'Nur http:// und https:// Server-URLs sind erlaubt.',
      deepLinkErrorTooLarge: 'Die Config-Payload überschreitet das 32-KB-Limit.'
    },
    model: {
      setupProviderFallback: 'Anbieter',
      setUpProvider: name => `${name} einrichten`,
      staleAuxBefore: (count, names) =>
        `${count} Hilfsaufgabe${count === 1 ? '' : 'n'} (${names}) ${count === 1 ? 'läuft' : 'laufen'} noch auf `,
      staleAuxAfter: ', nicht auf Ihrem Hauptmodell.',
      staleAuxOtherProviders: 'anderen Anbietern',
      moaEnabled: 'Aktiviert',
      moaSetDefault: 'Als Standard festlegen',
      moaNewPresetPlaceholder: 'neue Vorlage',
      moaAddPreset: 'Vorlage hinzufügen',
      customModel: 'Eigenes Modell…',
      customModelPlaceholder: 'Modell-ID',
      chooseFromList: 'Aus Liste wählen',
      moaDefault: 'Standard:',
      moaReferenceToggle: (enabled, index) => `Referenz ${index} ${enabled ? 'deaktivieren' : 'aktivieren'}`,
      moaReferenceTitle: index => `Referenz ${index}`,
      moaAddReference: 'Referenzmodell hinzufügen',
      loading: 'Modellkonfiguration wird geladen…',
      appliesDesc:
        'Gilt für neue Sessions. Verwenden Sie die Modellauswahl im Composer, um im aktiven Chat schnell zu wechseln.',
      provider: 'Anbieter',
      model: 'Modell',
      applying: 'Wird angewendet…',
      defaultsLabel: 'Voreinstellungen',
      reasoning: 'Denken',
      reasoningOff: 'Aus',
      defaultsFailed: 'Voreinstellungen des Modells konnten nicht gespeichert werden',
      loadFailed: 'Modelle konnten nicht geladen werden',
      restartRequired:
        'Dieses Backend führt nach einem Update alten Code aus. Starten Sie es neu, um den neuen Code zu laden.',
      restartBackend: 'Backend neu starten',
      restartingBackend: 'Backend wird neu gestartet…',
      restartFailed: 'Das Backend konnte nicht neu gestartet werden',
      auxiliaryTitle: 'Hilfsmodelle',
      resetAllToMain: 'Alle auf Hauptmodell zurücksetzen',
      auxiliaryDesc:
        'Hilfsaufgaben laufen standardmäßig auf dem Hauptmodell. Weise einer Aufgabe ein eigenes Modell zu, um das zu überschreiben.',
      setToMain: 'Auf Hauptmodell setzen',
      change: 'Ändern',
      autoUseMain: 'automatisch · Hauptmodell verwenden',
      inheritMainEffort: 'übernehmen · Aufwand des Hauptmodells',
      providerDefault: '(Anbietervorgabe)',
      fallbackAdd: 'Fallback hinzufügen',
      fallbackEmpty: 'Keine Fallback-Modelle — es wird das Standardmodell verwendet, außer es schlägt fehl.',
      notInCatalog:
        'ist nicht in der Modellliste dieses Anbieters enthalten — Aufrufe können auf ein Backup ausweichen.',
      moaTitle: 'Mixture of Agents',
      moaPreset: 'Voreinstellung',
      moaDescription:
        'Konfigurieren Sie benannte Vorlagen, die als Modelle unter dem Anbieter „Mixture of Agents“ erscheinen. Der Aggregator ist das handelnde Modell – er führt jeden Schritt der Tool-Schleife aus, und fast alle Kosten des Laufs werden seinem Anbieter berechnet. Referenzen beraten standardmäßig nur einmal pro Benutzer-Turn.',
      moaAggregator: 'Aggregator',
      moaAggregatorBilled: 'handelndes Modell · wird für den Lauf berechnet',
      moaReferenceHint: 'berät standardmäßig einmal pro Turn',
      tasks: {
        vision: {
          label: 'Sehen',
          hint: 'Bildanalyse'
        },
        compression: {
          label: 'Kompression',
          hint: 'Kontext-Verdichtung'
        },
        skills_hub: {
          label: 'Skills-Hub',
          hint: 'Skill-Suche'
        },
        approval: {
          label: 'Freigabe',
          hint: 'Intelligente Auto-Freigabe'
        },
        mcp: {
          label: 'MCP',
          hint: 'MCP-Tool-Routing'
        },
        title_generation: {
          label: 'Titel-Generierung',
          hint: 'Session-Titel'
        },
        review: {
          label: 'Review',
          hint: '/review Bewertungs-Subagent'
        },
        triage_specifier: {
          label: 'Triage-Spezifizierer',
          hint: 'Kanban-Spezifikation ausarbeiten'
        },
        kanban_decomposer: {
          label: 'Kanban-Zerleger',
          hint: 'Aufgaben zerlegen'
        },
        profile_describer: {
          label: 'Profil-Beschreiber',
          hint: 'Automatische Profilbeschreibungen'
        },
        curator: {
          label: 'Kurator',
          hint: 'Skill-Nutzungs-Review'
        }
      }
    },
    localModels: {
      connectionChanged: 'Verbindung für lokale Modelle geändert',
      title: 'Lokale Modelle',
      runtimeTitle: 'Lokale Laufzeit',
      runtimeReady: backend => `Bereit · ${backend}`,
      serverRunning: 'Läuft',
      runtimeInstalled: 'llama.cpp-Laufzeit installiert',
      runtimeInstalledDetail: (tag, backend) =>
        `Build ${tag}, ${backend}-Backend. Hermes startet und verwaltet den Server für Sie.`,
      installTitle: 'Lokale Laufzeit installieren',
      installDetail:
        'Lädt die llama.cpp-Inferenz-Engine herunter (einige hundert MB). Heruntergeladene Modelle laufen komplett auf diesem Rechner – kein Konto, nichts verlässt Ihren Computer.',
      installAction: 'Laufzeit installieren',
      installing: 'Laufzeit wird installiert…',
      installFailed: 'Laufzeit-Installation fehlgeschlagen',
      hardwareTitle: 'Dieser Rechner',
      hardwareLoading: 'Ihre Hardware wird geprüft…',
      vram: label => `${label} GPU-Speicher`,
      ram: label => `${label} RAM`,
      unifiedMemory: 'Kombinierter Speicher',
      modelsTitle: 'Modelle',
      recommended: 'Empfohlen',
      recommendedReason: {
        'best-quality-resident':
          'Das Modell mit der höchsten Qualität, das komplett auf Ihrer GPU mit voller Geschwindigkeit läuft. Die Auswahl wägt Qualität gegen die erwartete Geschwindigkeit auf dieser Hardware ab.',
        'speed-gated-quality':
          'Ein besseres Modell würde auf diesen Rechner passen, aber bei seiner Speicherbandbreite zu langsam reagieren — das ist das beste Modell, das schnell bleibt.',
        'fastest-resident':
          'Kein Modell erreicht volle Geschwindigkeit auf dieser Hardware; dieses kommt am nächsten und läuft komplett im GPU-Speicher.'
      },
      noRecommendationTitle: 'Keine automatische Empfehlung für diesen Rechner',
      noRecommendationDetail:
        'Die automatische Einrichtung braucht ein kuratiertes Modell, das vollständig in den Grafikspeicher oder den gemeinsamen Speicher passt. Sie können unten trotzdem ein Modell wählen oder weitere Modelle durchsuchen.',
      noRecommendationAction: 'Modelle durchsuchen',
      downloaded: 'Heruntergeladen',
      downloadAction: size => `Download · ${size}`,
      downloadProgress: (done, total) => `${done} von ${total} werden heruntergeladen`,
      downloadDoneToast: model => `${model} ist bereit.`,
      installDoneToast: 'Lokale Laufzeit installiert und bereit.',
      quickstartTitle: 'Ein Modell auf diesem Rechner ausführen',
      quickstartDetail: (model, size) =>
        `Ein Klick richtet alles ein: die lokale Engine, ${model} (${size} Download) und Ihre Voreinstellung für neue Chats. Nichts verlässt diesen Computer.`,
      quickstartDetailReady: model =>
        `Ein Klick macht ${model} zu Ihrer Voreinstellung für neue Chats. Alles läuft auf diesem Rechner.`,
      quickstartAction: 'Für mich einrichten',
      quickstartConfigure: 'Konfigurieren…',
      quickstartDoneToast: model => `${model} ist eingerichtet — neue Chats laufen auf diesem Rechner.`,
      quickstartFailed: 'Einrichtung des lokalen Modells fehlgeschlagen',
      quickstartStageEngine: 'Engine',
      quickstartStageModel: 'Modell',
      quickstartStageFinish: 'Fertig',
      useAction: 'Verwenden',
      activePill: 'Voreinstellung',
      updateTitle: 'Engine-Update verfügbar',
      updateDetail: (next, current) =>
        `Eine neuere llama.cpp-Version (${next}) ist bereit zur Installation – Sie verwenden ${current}. Während des Downloads laufen die Modelle weiter.`,
      updateAction: 'Engine aktualisieren',
      updating: 'Engine wird aktualisiert…',
      upToDateTitle: 'Engine aktuell',
      upToDateDetail: (tag, backend) =>
        `llama.cpp ${tag} (${backend}) wird ausgeführt — der neueste Build, den Hermes mitliefert.`,
      activeDetail: 'Neue Chats verwenden dieses Modell – es wird geladen, wenn Sie Ihre erste Nachricht senden',
      activeNotLoaded: 'Wird bei Ihrer ersten Nachricht geladen',
      loadedPill: 'Im Speicher',
      placementResident: 'komplett auf GPU',
      placementSpilled: 'teils im RAM',
      placementResidentTip: 'Läuft komplett im GPU-Speicher bei diesem Kontextfenster — volle Geschwindigkeit.',
      placementSpilledTip:
        'Ein Teil dieses Modells läuft aus dem Arbeitsspeicher — es funktioniert, aber langsamer. Ein kompakterer Build oder ein kleinerer Kontext würde komplett passen.',
      loadingPill: 'Wird geladen…',
      ejectTip: 'GPU-Speicher freigeben (wird bei der nächsten Nachricht wieder geladen)',
      ejected: 'Modell entladen — GPU-Speicher freigegeben.',
      ejectFailed: 'Das Modell konnte nicht entladen werden',
      stopServer: 'Ausschalten',
      startServer: 'Einschalten',
      runtimeRunningDetail:
        'Der lokale Server läuft. Wenn Sie ihn ausschalten, wird der gesamte GPU-Speicher freigegeben, und neue Chats können keine lokalen Modelle verwenden, bis Sie ihn wieder einschalten.',
      serverStopped: 'Lokaler Server gestoppt — GPU-Speicher freigegeben.',
      serverStarted: 'Lokaler Server läuft.',
      serverStopFailed: 'Der lokale Server konnte nicht gestoppt werden',
      serverStartFailed: 'Der lokale Server konnte nicht gestartet werden',
      activating: 'Wird gestartet…',
      activateFailed: model => `Konnte nicht zu ${model} wechseln`,
      activateDoneToast: model => `Neue Chats verwenden ${model}.`,
      downloadFailed: model => `Download von ${model} fehlgeschlagen`,
      pillFitsGpu: 'Passt auf Ihre GPU',
      pillUsesRam: 'Verwendet Arbeitsspeicher',
      pillTooBig: 'Zu groß für diesen Rechner',
      browseTitle: 'Weitere Modelle finden',
      browseHint:
        'Durchsuchen Sie ganz Hugging Face. Hier heruntergeladene Modelle werden automatisch an Ihren Rechner angepasst, aber nicht von uns getestet.',
      browsePlaceholder: 'Modelle nach Name oder Autor suchen…',
      browseSearching: 'Hugging Face wird durchsucht',
      browseListing: 'Modelldateien werden gelesen',
      browseShowFiles: 'Dateien anzeigen',
      browseRefresh: 'Aktualisieren',
      browseDownloads: 'Downloads',
      browseLikes: 'Likes',
      browseGated: 'erfordert Hugging-Face-Anmeldung',
      browseNoGguf: 'Keine kompatiblen Modelldateien gefunden.',
      browseFitUnknown: 'Passform unbekannt',
      browseAlreadyDownloaded: 'Bereits heruntergeladen.',
      addedByYou: 'Von Ihnen hinzugefügt',
      browseDownloadStarted: '{name} wird heruntergeladen',
      browseDownloadAria: '{name} herunterladen',
      sideloadButton: 'Modelldatei hinzufügen',
      sideloadTitle: 'Eine GGUF-Modelldatei wählen',
      sideloadDone: '{name} hinzugefügt.',
      sideloadAlreadyPresent: 'Bereits in Ihrer Bibliothek.',
      pillFullContext: max => `Voller ${max}-Kontext`,
      pillFullContextTip: 'Läuft von Anfang an mit dem kompletten Kontextfenster des Modells',
      pillUpTo: max => `Bis zu ${max} Kontext`,
      pillGrowsTip: 'Wächst automatisch, wenn Ihr Gespräch mehr Platz braucht',
      pillVision: 'Sieht Bilder',
      deleteAction: 'Modell löschen',
      deleteConfirm: model => `${model} von der Festplatte löschen?`,
      deleted: model => `${model} gelöscht.`,
      deleteFailed: 'Löschen fehlgeschlagen'
    },
    billing: {
      perMonth: (amount: string) => `${amount}/Monat`,
      creditsPerMonth: (amount: string) => `${amount} Credits/Monat`,
      usageLabel: (label: string) => `${label}-Nutzung`,
      freeTier: {
        signIn: 'Anmelden',
        title: 'Sie nutzen den kostenlosen Nous-Tarif',
        message: 'Melden Sie sich mit einem Nous-Konto an, um weitere Modelle und Tools freizuschalten.',
        caption:
          'Läuft mit nous/welcome, Konnektoren inklusive. Nach der Anmeldung bleiben Ihre Konnektoren erhalten, und Sie erhalten die kontopflichtigen Tools sowie alle weiteren Modelle.',
        name: 'Nous · kostenloser Tarif',
        footnote:
          'Der kostenlose Tarif hat kein Guthaben und nichts zu bezahlen. Zahlung und Nutzung werden angezeigt, sobald Sie sich mit einem Nous-Konto anmelden.',
        plan: 'Kostenloser Tarif',
        model: 'Modell',
        connectors: 'Konnektoren',
        included: 'Inklusive'
      },
      amountValidation: {
        reloadTo: 'Aufladen auf',
        greaterThanThreshold: 'Der Aufladebetrag muss größer als der Schwellenwert sein.',
        decimal: (label: string) => `${label}: Geben Sie einen Dollarbetrag mit höchstens 2 Nachkommastellen ein.`,
        positive: (label: string) => `${label}: Der Betrag muss größer als 0 $ sein.`,
        minimum: (label: string, amount: string) => `${label}: Minimum ist ${amount}.`,
        maximum: (label: string, amount: string) => `${label}: Maximum ist ${amount}.`
      },
      stepUp: {
        openVerification: 'Verifizierungsseite öffnen',
        dismiss: 'Schließen',
        waiting: 'Warten auf Verifizierungslink…',
        verify: 'Zum Fortfahren verifizieren',
        deniedTitle: 'Die Verifizierung wurde nicht genehmigt',
        deniedBody: 'Die Verifizierung wurde abgeschlossen, ohne Remote-Ausgaben für dieses Terminal zu erlauben.',
        successTitle: 'Verifizierung abgeschlossen',
        successBody: 'Remote-Ausgaben sind für dieses Terminal erlaubt.'
      },
      charge: {
        added: (amount?: string) => (amount ? `${amount} $ hinzugefügt.` : 'Credits hinzugefügt.'),
        failedTitle: 'Abbuchung fehlgeschlagen',
        unconfirmedTitle: 'Ergebnis der Abbuchung unbestätigt',
        unconfirmedBody: (message: string) =>
          `${message} Das Ergebnis Ihrer letzten Abbuchung ist unbestätigt - prüfen Sie Guthaben/Verlauf, bevor Sie es erneut versuchen.`,
        checkTitle: 'Abbuchung konnte nicht geprüft werden',
        checkBody: 'Die Abbuchung konnte nicht geprüft werden.',
        untrackedTitle: 'Abbuchung konnte nicht verfolgt werden',
        untrackedBody: 'Der Abrechnungsdienst hat die Anfrage angenommen, aber keine Abbuchungs-ID zurückgegeben.',
        timeoutTitle: 'Nach 5 Minuten noch in Bearbeitung',
        timeoutBody:
          'Die Abbuchung kann noch abgeschlossen werden. Prüfen Sie das Portal, bevor Sie es erneut versuchen.',
        authenticationRequired:
          'Ihre Bank verlangt eine Verifizierung (3DS). Schließen Sie sie im Portal ab, um diesen Kauf abzuschließen.',
        expired: 'Ihre Karte ist abgelaufen. Aktualisieren Sie sie im Portal.',
        declined: 'Ihre Karte wurde abgelehnt. Versuchen Sie im Portal eine andere Karte.',
        failedBody: (reason: string) => `Die Abbuchung ist nicht durchgegangen (${reason}).`
      },
      title: 'Abrechnung',
      preview: 'Vorschau',
      summary: {
        balance: 'Guthaben',
        plan: 'Tarif',
        autoRefill: 'Automatisch aufladen'
      },
      sections: {
        invoices: 'Rechnungen',
        plan: 'Tarif',
        paymentAndCredits: 'Zahlung & Credits',
        usage: 'Nutzung'
      },
      usage: {
        title: 'Nutzung'
      },
      buyCredits: {
        customAmount: 'Eigener Credit-Betrag',
        title: 'Jetzt Credits kaufen',
        buyButton: 'Kaufen',
        processing: 'Wird verarbeitet… Abrechnung wird geprüft',
        added: (amount: string) => `${amount} hinzugefügt. Guthaben wird aktualisiert.`,
        retry: 'Erneut versuchen',
        openPortal: 'Portal öffnen'
      },
      plan: {
        title: 'Tarife',
        changePlan: 'Tarif ändern',
        viewPlans: 'Tarife ansehen',
        backAria: 'Zurück zur Abrechnung',
        current: 'Aktueller Tarif',
        scheduled: 'Geplant',
        empty: 'Derzeit sind keine Tarife zum Wechseln verfügbar.',
        undo: 'Rückgängig',
        undoing: 'Wird rückgängig gemacht…',
        downgrade: 'Herabstufen',
        confirmDowngrade: 'Herabstufung bestätigen',
        tryAgain: 'Erneut versuchen',
        checkingChange: 'Änderung wird geprüft…',
        cannotChange: 'Diese Änderung ist hier nicht möglich.',
        alreadyOn: (name: string) => `Sie nutzen bereits ${name} – nichts zu ändern.`,
        notScheduleable: 'Diese Änderung kann hier nicht geplant werden.',
        scheduling: 'Wird geplant…',
        cancel: 'Abbrechen',
        effectScheduled: (targetName: string, effectiveAt: string, creditsDelta?: string) =>
          `Wechsel zu ${targetName} – wirksam ${effectiveAt}. Jetzt keine Abbuchung; bis dahin behalten Sie Ihren aktuellen Tarif.${creditsDelta ? ` Änderung der monatlichen Credits: ${creditsDelta}.` : ''}`
      },
      autoReload: {
        threshold: 'Schwellenwert',
        thresholdAria: 'Schwellenwert für automatisches Aufladen',
        reloadTo: 'Aufladen auf',
        reloadToAria: 'Zielbetrag für automatisches Aufladen',
        turnOffConfirm: 'Automatisches Aufladen deaktivieren?',
        turnOff: 'Ausschalten',
        disable: 'Deaktivieren',
        updated: 'Automatisches Aufladen aktualisiert.',
        turnedOff: 'Automatisches Aufladen deaktiviert.',
        manage: 'Verwalten',
        save: 'Speichern',
        saving: 'Wird gespeichert…',
        cancel: 'Abbrechen'
      },
      state: {
        notice: {
          loggedOut: {
            title: 'Nous-Konto verbinden',
            message: 'Melden Sie sich mit Ihrem Nous-Konto an, um hier Guthaben, Tarif und Nutzung zu sehen.',
            action: 'Anmelden'
          },
          openPortal: 'Portal öffnen ↗',
          noCard: {
            title: 'Keine Zahlungsmethode hinterlegt',
            message:
              'Der Kauf von Credits und das automatische Aufladen bleiben deaktiviert, bis eine Karte hinterlegt ist. Fügen Sie im Portal eine hinzu.',
            action: 'Karte hinzufügen ↗'
          }
        },
        paymentMethod: {
          title: 'Zahlungsmethode',
          description: 'Verwalten Sie die Karte für Aufladungen und Abo-Verlängerungen.',
          addAction: 'Zahlungsmethode hinzufügen',
          updateAction: 'Aktualisieren',
          provenance: {
            autoRefill: 'Karte für automatisches Aufladen',
            customerDefault: 'Standardkarte des Kunden',
            subPin: 'Abo-Karte',
            suffix: (label: string) => ` - ${label}`
          }
        },
        buyCredits: {
          description: 'Eine einmalige Abbuchung von Ihrer Karte, die noch heute Ihrem Guthaben gutgeschrieben wird.'
        },
        autoRefill: {
          title: 'Bei niedrigem Guthaben aufladen',
          genericDescription: 'Hält Ihr Guthaben aufgeladen, wenn es unter Ihren Schwellenwert fällt.',
          offPill: 'Aus',
          enabledPill: 'Aktiviert',
          notAvailablePill: '—',
          manageCaption: 'Automatisches Aufladen im Portal verwalten.',
          turnOnCaption: 'Automatisches Aufladen im Portal aktivieren',
          chargesDescription: (reloadTo: string, threshold: string) =>
            `Bucht automatisch ${reloadTo} ab, wenn Ihr Guthaben unter ${threshold} fällt.`,
          distinctCardCaption: (cardLabel: string) =>
            `Automatisches Aufladen belastet ${cardLabel} – im Portal abgleichen`,
          distinctCardFallback: 'eine andere Karte',
          reconcileAction: 'Abgleichen ↗'
        },
        usage: {
          subscriptionCredits: {
            title: 'Abo-Credits',
            barLabel: 'Verbleibende Abo-Credits',
            captionResets: (date: string) => `Wird zurückgesetzt: ${date}`,
            valueOf: (remaining: string, monthly: string) => `${remaining} von ${monthly} übrig`,
            valueOver: (remaining: string, monthly: string, over: string) =>
              `${remaining} von ${monthly} übrig · ${over} darüber`
          },
          topupCredits: {
            title: 'Aufgeladene Credits',
            caption: 'Verfallen nicht'
          },
          monthlyCap: {
            title: 'Monatliches Ausgabenlimit',
            barLabel: 'Genutztes monatliches Ausgabenlimit',
            captionDefault: 'Standardobergrenze',
            captionSpending: 'Monatliche Remote-Ausgaben',
            valueUsed: (spent: string, limit: string) => `${spent} von ${limit} genutzt`
          }
        },
        planCard: {
          freeTier: 'Kostenlos',
          chooseAction: 'Auswählen ↗',
          adjustPlanAction: 'Tarif anpassen ↗',
          unavailableCaption: 'Abo-Details sind nicht verfügbar; das Portal kann weiterhin geöffnet werden.',
          downgradeCaption: (tierName: string, when: string) => `Wechselt am ${when} zu ${tierName}.`,
          cancellationCaption: (when: string) => `Endet am ${when}.`,
          renewsCaption: (date: string) => `Verlängert sich ${date}`,
          noSubscriptionCaption: 'Kein aktives Abo – kostenpflichtige Modelle verbrauchen aufgeladene Credits.'
        }
      },
      errors: {
        consentRequired: {
          title: 'Kartenbestätigung erforderlich',
          message: 'Bestätigen Sie diese Karte im Portal für Terminal-Abbuchungen'
        },
        insufficientScope: {
          title: 'Remote-Ausgaben müssen genehmigt werden',
          message:
            'Dafür müssen Remote-Ausgaben erlaubt sein. Starten Sie eine Aufladung, um sie zu erlauben, und versuchen Sie es erneut.'
        },
        remoteSpendingRevoked: {
          title: 'Remote-Ausgaben wurden gestoppt',
          messageByAdmin: 'Ein Administrator hat Remote-Ausgaben für dieses Terminal gestoppt.',
          messageBySelf: 'Sie haben Remote-Ausgaben für dieses Terminal gestoppt.'
        },
        remoteSpendingReconnect: (who: string) =>
          `${who} Verbinden Sie sich unter Einstellungen -> Gateway erneut, um dieses Gerät neu zu autorisieren.`,
        sessionRevoked: {
          title: 'Session abgemeldet',
          message: 'Ihre Session wurde abgemeldet. Melden Sie sich unter Einstellungen → Gateway erneut an.'
        },
        cliBillingDisabled: {
          title: 'Remote-Ausgaben sind deaktiviert',
          message:
            'Remote-Ausgaben sind für dieses Konto deaktiviert – ein Abrechnungsadministrator kann sie auf der Hermes-Agent-Seite des Portals aktivieren.'
        },
        roleRequired: {
          title: 'Administratorrolle erforderlich',
          message:
            'Zum Aufladen ist ein Org-Administrator oder -Inhaber nötig. Fragen Sie einen Administrator oder verwalten Sie es im Portal.'
        },
        idempotencyConflict: {
          title: 'Neue Aufladung starten',
          message:
            '🔴 Dieser Abbuchungsschlüssel wurde bereits für einen anderen Betrag verwendet. Starten Sie eine neue Aufladung.'
        },
        noPaymentMethod: {
          title: 'Keine gespeicherte Karte',
          message:
            '💳 Noch keine gespeicherte Karte für Terminal-Abbuchungen. Richten Sie im Portal eine ein ' +
            '(einmalige Credit-Käufe speichern keine wiederverwendbare Karte).'
        },
        orgAccessDenied: {
          title: 'Org-Zugriff verweigert',
          message: 'Dieses Token ist an keine Organisation gebunden, die Sie verwalten können'
        },
        monthlyCapExceeded: {
          title: 'Monatliches Ausgabenlimit erreicht',
          messageReached: '🔴 Monatliches Ausgabenlimit erreicht.',
          messageHeadroom: (remaining: string) =>
            `🔴 Monatliches Ausgabenlimit erreicht – ${remaining} $ Spielraum übrig.`
        },
        rateLimited: {
          title: 'Zu viele Abbuchungen im Moment',
          message: (mins: number) =>
            mins > 0
              ? `🟡 Zu viele Abbuchungen im Moment (erneut versuchen in ~${mins} Min.). Das ist kein Zahlungsfehler.`
              : '🟡 Zu viele Abbuchungen im Moment. Das ist kein Zahlungsfehler.'
        },
        stripeUnavailable: {
          title: 'Stripe hat Probleme',
          message: (mins: number) =>
            mins > 0
              ? `Stripe hat Probleme – erneut versuchen in ~${mins} Min.`
              : 'Stripe hat Probleme – versuchen Sie es gleich erneut'
        },
        upgradeCapExceeded: {
          title: 'Tägliches Limit für Tarifwechsel erreicht',
          message: 'Tägliches Limit für Tarifwechsel erreicht – versuchen Sie es morgen erneut'
        },
        endpointUnavailable: {
          title: 'Abrechnungs-Endpunkt nicht verfügbar',
          message:
            'Der Abrechnungs-Endpunkt hat eine Nicht-JSON-Antwort zurückgegeben (er ist in diesem Deployment möglicherweise nicht verfügbar).'
        },
        timeout: {
          title: 'Zeitüberschreitung bei der Abrechnungsanfrage',
          message: 'Zeitüberschreitung bei der Abrechnungsanfrage.'
        },
        transport: {
          title: 'Abrechnungsverbindung fehlgeschlagen',
          message: 'Die Abrechnungsanfrage ist fehlgeschlagen, bevor sie das Gateway erreicht hat.'
        },
        default: {
          title: 'Abrechnungsanfrage fehlgeschlagen',
          message: 'Die Abrechnungsanfrage ist fehlgeschlagen.'
        }
      }
    },
    providers: {
      connectAccount: 'Ein Konto verbinden',
      haveApiKey: 'Haben Sie stattdessen einen API-Key?',
      intro:
        'Melden Sie sich mit einem Abo an – kein API-Key zum Kopieren. Hermes übernimmt die Browser-Anmeldung für Sie, direkt hier in der App.',
      connected: 'Verbunden',
      collapse: 'Einklappen',
      connectAnother: 'Weiteren Provider verbinden',
      otherProviders: 'Weitere Provider',
      disconnect: 'Trennen',
      disconnectInTerminal: 'Trennen (führt den Entfernungsbefehl im Terminal aus)',
      removeConfirm: provider => `${provider} entfernen?`,
      removeExternalGeneric: provider => `${provider} wird von einer eigenen CLI verwaltet – entfernen Sie ihn dort.`,
      removeKeyManaged: provider =>
        `${provider} ist über einen API-Key konfiguriert. Entfernen Sie ihn unter API-Keys.`,
      removeTerminalConfirm: (provider, command) =>
        `${provider} trennen? Dadurch wird "${command}" im Terminal ausgeführt, um die Zugangsdaten zu löschen.`,
      removeTerminalRunning: provider => `${provider}-Trennung wird im Terminal ausgeführt…`,
      removedTitle: 'Konto entfernt',
      removedMessage: provider => `${provider} wurde entfernt.`,
      failedRemove: provider => `${provider} konnte nicht entfernt werden`,
      noProviderKeys: 'Keine Provider-API-Keys verfügbar.',
      searchKeys: 'Provider suchen…',
      noKeysMatch: 'Keine Provider entsprechen Ihrer Suche.',
      localEndpoint: {
        title: 'Lokaler / eigener Endpoint',
        description:
          'Verbinden Sie Hermes mit einem beliebigen OpenAI-kompatiblen Endpunkt (Zyphra, vLLM, llama.cpp, Ollama usw.).'
      },
      loading: 'Provider werden geladen…'
    },
    sessions: {
      loading: 'Archivierte Sessions werden geladen…',
      archivedTitle: 'Archivierte Sessions',
      archivedIntro:
        'Archivierte Chats sind in der Seitenleiste ausgeblendet, behalten aber alle Nachrichten. Alt/⌥+Umschalt/⇧-Klick auf einen Chat in der Seitenleiste archiviert ihn.',
      emptyArchivedTitle: 'Nichts archiviert',
      emptyArchivedDesc: 'Archivieren Sie einen Chat, um ihn hier auszublenden.',
      unarchive: 'Archivierung aufheben',
      deletePermanently: 'Endgültig löschen',
      messages: count => `${count} ${count === 1 ? 'Nachricht' : 'Nachrichten'}`,
      restored: 'Wiederhergestellt',
      deleteConfirm: title => `"${title}" endgültig löschen? Das kann nicht rückgängig gemacht werden.`,
      autoArchiveTitle: 'Veraltete Chats automatisch archivieren',
      autoArchiveDesc:
        'Archiviert automatisch Chats, die Sie eine Weile nicht verwendet haben. Angeheftete Chats werden nie archiviert, und nichts wird gelöscht – archivierte Chats landen nur hier.',
      autoArchiveDaysLabel: 'Archivieren nach',
      autoArchiveDaysUnit: 'Tagen Inaktivität',
      autoArchiveFailed: 'Auto-Archivierung konnte nicht aktualisiert werden',
      defaultDirTitle: 'Standard-Projektordner',
      defaultDirDesc:
        'Neue Sessions starten in diesem Ordner, sofern Sie keinen anderen wählen. Nicht festlegen, um Ihr Home-Verzeichnis zu verwenden.',
      defaultDirUpdated:
        'Standard-Projektordner aktualisiert – starten Sie einen neuen Chat (Strg/⌘+N), damit er wirksam wird',
      defaultsTo: label => `Standardmäßig ${label}.`,
      change: 'Ändern',
      choose: 'Auswählen',
      clear: 'Löschen',
      notSet: 'Nicht festgelegt',
      failedLoad: 'Archivierte Sessions konnten nicht geladen werden',
      unarchiveFailed: 'Archivierung aufheben fehlgeschlagen',
      deleteFailed: 'Löschen fehlgeschlagen',
      updateDirFailed: 'Standardordner konnte nicht aktualisiert werden',
      clearDirFailed: 'Standardordner konnte nicht geleert werden'
    },
    toolsets: {
      loadingConfig: 'Konfiguration wird geladen',
      savedTitle: 'Zugangsdaten gespeichert',
      savedMessage: key => `${key} aktualisiert.`,
      removedTitle: 'Zugangsdaten entfernt',
      removedMessage: key => `${key} entfernt.`,
      failedSave: key => `${key} konnte nicht gespeichert werden`,
      failedRemove: key => `${key} konnte nicht entfernt werden`,
      failedReveal: key => `${key} konnte nicht angezeigt werden`,
      removeConfirm: key => `${key} aus der .env entfernen?`,
      set: 'Festgelegt',
      notSet: 'Nicht festgelegt',
      selectedTitle: 'Provider ausgewählt',
      selectedMessage: provider => `${provider} ist jetzt aktiv.`,
      failedSelect: provider => `${provider} konnte nicht ausgewählt werden`,
      failedLoad: 'Tool-Konfiguration konnte nicht geladen werden',
      noProviderOptions:
        'Dieses Toolset hat keine Provider-Optionen – aktivieren Sie es, und es funktioniert mit Ihrem aktuellen Setup.',
      noProviders: 'Für dieses Toolset sind gerade keine Provider verfügbar.',
      ready: 'Bereit',
      needsSignIn: 'Anmeldung erforderlich',
      needsSetup: 'Setup erforderlich',
      activeBackend: 'Aktiv',
      activeBackendHint: 'Das ist Ihr aktives Backend',
      useBackend: 'Dieses Backend verwenden',
      nousIncluded: 'In einem Nous-Abo enthalten – melden Sie sich im Nous Portal an, um es zu aktivieren.',
      nousAuthNeededTitle: 'Im Nous Portal anmelden',
      nousAuthNeededMessage: provider =>
        `${provider} ist gespeichert, wird aber erst aktiviert, wenn Sie sich im Nous Portal anmelden.`,
      nousAuthSignIn: 'Anmelden',
      nousAuthDoneTitle: 'Nous Portal verbunden',
      nousAuthDoneMessage: 'Ihre Abo-Backends sind jetzt aktiv.',
      nousAuthFailed: 'Die Nous-Portal-Anmeldung wurde nicht abgeschlossen',
      nousAuthFailedMessage: 'Versuchen Sie es erneut.',
      nousAuthTryAgain: 'Erneut versuchen',
      noApiKeyRequired: 'Kein API-Key erforderlich.',
      postSetupHint: step =>
        `Dieses Backend braucht eine einmalige Installation (${step}). Läuft auf diesem Rechner – kann ein paar Minuten dauern.`,
      postSetupInstalledHint: 'Installiert. Führen Sie das Setup nur erneut aus, wenn etwas nicht funktioniert.',
      postSetupRun: 'Setup ausführen',
      postSetupRerun: 'Setup erneut ausführen',
      postSetupInstalled: 'Installiert',
      postSetupRunning: 'Wird installiert…',
      postSetupStarting: 'Wird gestartet…',
      postSetupCompleteTitle: 'Setup abgeschlossen',
      postSetupCompleteMessage: step => `${step} installiert.`,
      postSetupErrorTitle: 'Setup mit Fehlern abgeschlossen',
      postSetupErrorMessage: step => `Prüfen Sie das ${step}-Protokoll.`,
      postSetupOpenLogs: 'Logs öffnen',
      postSetupRunAgain: 'Erneut ausführen',
      postSetupFailed: step => `Das ${step}-Setup konnte nicht ausgeführt werden`,
      webSearchActive: backend => `Suche: ${backend}`,
      webExtractActive: backend => `Extrahieren: ${backend}`,
      webCapabilityUnset: 'nicht festgelegt',
      webUseForSearch: 'Für die Suche verwenden',
      webUseForExtract: 'Für Extrahieren verwenden',
      webUsedForSearch: 'Such-Backend',
      webUsedForExtract: 'Extraktions-Backend',
      webCapabilitySelectedMessage: (provider, capability) =>
        `${provider} übernimmt jetzt ${capability === 'search' ? 'die Websuche' : 'das Extrahieren von Webinhalten'}.`,
      failedSelectCapability: provider => `${provider} konnte nicht festgelegt werden`,
      loadingModels: 'Modellkatalog wird geladen…',
      modelSectionTitle: 'Modell',
      modelCount: count => `${count} Modell${count === 1 ? '' : 'e'}`,
      modelInUse: 'In Verwendung',
      modelDefault: 'Standard',
      modelInactiveHint: 'Wählen Sie zuerst dieses Backend aus, um sein Modell zu ändern.',
      modelSelectedTitle: 'Modell ausgewählt',
      modelSelectedMessage: model => `${model} gilt für neue Sessions.`,
      failedSelectModel: model => `${model} konnte nicht ausgewählt werden`,
      terminalBackend: {
        sectionTitle: 'Ausführungs-Backend',
        loading: 'Ausführungs-Backends werden geprüft…',
        failedLoad: 'Terminal-Backends konnten nicht geladen werden',
        ready: 'Bereit',
        needsSetup: 'Setup erforderlich',
        unavailable: 'Nicht verfügbar',
        inUse: 'In Verwendung',
        selectedTitle: 'Backend ausgewählt',
        selectedMessage: backend => `Terminal-Befehle laufen jetzt über ${backend}. Gilt für neue Sessions.`,
        failedSelect: backend => `${backend} konnte nicht ausgewählt werden`,
        needsSetupHint:
          'Sie können dieses Backend jetzt auswählen – Befehle schlagen fehl, bis das Setup abgeschlossen ist.',
        needsSetupConfirmTitle: (backend: string) => `${backend} trotzdem auswählen?`,
        needsSetupConfirmDescription: (detail: string) =>
          `${detail} Sessions, die nach dieser Änderung starten, haben keine Terminal- oder Datei-Tools, bis die Einrichtung abgeschlossen ist.`,
        needsSetupConfirmDescriptionGeneric:
          'Dieses Backend ist noch nicht eingerichtet. Sessions, die nach dieser Änderung starten, haben keine Terminal- oder Datei-Tools, bis die Einrichtung abgeschlossen ist.',
        needsSetupConfirmAction: 'Trotzdem auswählen',
        unavailableTitle: 'Terminalbefehle sind nicht verfügbar',
        unavailableMessage: backend =>
          `Hermes kann gerade keine Shell-Befehle ausführen: ${backend} ist nicht bereit. Wechseln Sie zu Lokal oder schließen Sie die Einrichtung von ${backend} ab und versuchen Sie es erneut.`,
        openBackendSettings: 'Terminal-Einstellungen öffnen',
        useLocal: 'Lokal verwenden',
        switchedToLocal: 'Terminalbefehle laufen jetzt lokal. Gilt für neue Sessions.'
      },
      browserRealProfile: {
        label: 'Mein echtes Browser-Profil verwenden',
        description:
          'Kopiert die Anmeldungen und Cookies Ihres Standardbrowsers in einen verwalteten Schnappschuss, mit dem der Agent browst. Ihr Live-Profil wird nie direkt geöffnet. Gilt für neue Sessions.',
        enabledTitle: 'Echtes Profil-Browsen aktiv',
        enabledMessage: 'Neue Sessions browsen mit einem Schnappschuss Ihres Standard-Browserprofils.',
        disabledTitle: 'Echtes Profil-Browsen deaktiviert',
        disabledMessage: 'Der Profil-Schnappschuss wird gelöscht; neue Sessions verwenden einen sauberen Browser.',
        failedSave: 'Die Echtes-Profil-Einstellung konnte nicht gespeichert werden',
        prompt: {
          title: 'Auf Ihren Websites angemeldet bleiben',
          body: 'Lassen Sie Hermes mit einem Schnappschuss Ihres Standard-Browserprofils browsen, damit Websites bereits angemeldet öffnen.',
          bulletSnapshot: 'Cookies und Anmeldungen werden in einen verwalteten Schnappschuss kopiert.',
          bulletLiveProfile: 'Ihr Live-Browserprofil wird nie direkt geöffnet.',
          bulletLocal: 'Nichts verlässt diesen Computer.',
          dontShowAgain: 'Nicht mehr anzeigen',
          notNow: 'Jetzt nicht',
          enable: 'Mein Profil verwenden'
        }
      }
    }
  },
  skills: {
    tabSkills: 'Skills',
    tabToolsets: 'Tools',
    configuringProfile: 'Konfiguriert:',
    all: 'Alle',
    searchSkills: 'Skills durchsuchen...',
    searchToolsets: 'Tools durchsuchen...',
    refresh: 'Skills aktualisieren',
    refreshing: 'Skills werden aktualisiert',
    loading: 'Fähigkeiten werden geladen...',
    noSkillsTitle: 'Keine Skills gefunden',
    noSkillsDesc: 'Versuchen Sie eine breitere Suche oder eine andere Kategorie.',
    noToolsetsTitle: 'Keine Toolsets gefunden',
    noToolsetsDesc: 'Versuchen Sie eine breitere Suchanfrage.',
    noDescription: 'Keine Beschreibung.',
    configured: 'Konfiguriert',
    needsKeys: 'Benötigt Keys',
    visionModelHint:
      'Vision verwendet Ihre Hilfsmodell-Konfiguration – das bildfähige Modell wird dort gewählt, nicht hier pro Provider.',
    visionModelLink: 'Vision-Modell in Einstellungen → Modelle wählen',
    toolsetsEnabled: (enabled, total) => `${enabled}/${total} Toolsets aktiviert`,
    configureToolset: label => `${label} konfigurieren`,
    toggleToolset: (label, enabled) => `Toolset ${label} ${enabled ? 'einschalten' : 'ausschalten'}`,
    skillsLoadFailed: 'Skills konnten nicht geladen werden',
    toolsetsRefreshFailed: 'Toolsets konnten nicht aktualisiert werden',
    skillEnabled: 'Skill aktiviert',
    skillDisabled: 'Skill deaktiviert',
    toolsetEnabled: 'Toolset aktiviert',
    toolsetDisabled: 'Toolset deaktiviert',
    appliesToNewSessions: name => `${name} gilt für neue Sessions.`,
    failedToUpdate: name => `${name} konnte nicht aktualisiert werden`,
    sortMostUsed: 'Am häufigsten genutzt',
    sortAlpha: 'A–Z',
    sortMostUsedDesc: '↓ Am häufigsten genutzt',
    sortLeastUsedAsc: '↑ Am seltensten genutzt',
    enableAll: 'Alle aktivieren',
    disableAll: 'Alle deaktivieren',
    disableUnused: 'Unbenutzte deaktivieren',
    bulkUpdated: count => `${count} ${count === 1 ? 'Element' : 'Elemente'} für neue Sessions aktualisiert.`,
    bulkNoChange: 'Nichts zu ändern.',
    usageCount: count => `${count}× verwendet`,
    provenance: {
      agent: 'Gelernt',
      bundled: 'Integriert',
      hub: 'Hub'
    },
    emptyNoneFound: noun => `Kein ${noun} gefunden`,
    emptyNothingMatches: query => `Nichts passt zu “${query}”.`,
    emptyNoneAvailable: noun => `Noch kein ${noun} verfügbar.`,
    changesApplyNewSessions: 'Änderungen gelten für neue Sessions.',
    skillUpdated: 'Skill aktualisiert',
    edit: 'Bearbeiten',
    archive: 'Archivieren',
    skillArchivedTitle: 'Skill archiviert',
    skillArchivedMessage: 'Wiederherstellbar über „hermes curator restore“.',
    tabPlugins: 'Plugins',
    plugins: {
      agentTitle: 'Agent-Plugins',
      agentBlurb:
        'Erweitert den Agenten für das gewählte Profil — Tools, Hooks, Anbieter. Wirkt nach einem Gateway-Neustart.',
      pageBlurb: 'Ein Plugin kann diese App, den Agent oder beides erweitern – jede Hälfte hat einen eigenen Schalter.',
      halfDesktop: 'Desktop',
      halfDesktopHint: 'diese App, gleich für jedes Profil',
      halfAgent: 'Agent',
      halfAgentIn: profile => `Agent in ${profile}`,
      defaultProfile: 'Hermes (Standard)',
      kindAgent: 'Agent',
      kindDesktop: 'Desktop',
      kindBoth: 'Agent + Desktop',
      installAgentHere: 'Hier installieren',
      installAgentHereTip: profile =>
        `Die Desktop-Hälfte ist in dieser App geladen, aber die Agent-Hälfte ist in ${profile} nicht installiert. Installieren Sie sie dort.`,
      installAgentHereNoOrigin:
        'Die Agent-Hälfte ist in diesem Profil nicht installiert, und dieses Paket wurde von Hand hineinkopiert (kein Katalogeintrag und kein Git-Remote), daher kann es nicht von hier aus installiert werden. Kopieren Sie seinen Ordner ins Profil oder installieren Sie es neu aus Git.',
      desktopHalfPending: 'wird kopiert…',
      desktopHalfPendingTip:
        'Dieses Paket enthält eine Desktop-Hälfte, die noch nicht in die App kopiert wurde. Verwenden Sie „Erneut scannen“ oder starten Sie die App neu.',
      desktopHalfRemote: 'nicht verfügbar (Remote-Backend)',
      desktopHalfRemoteTip:
        'Die Desktop-Hälfte dieses Pakets liegt auf der Festplatte des Remote-Backends, die diese App nicht lesen kann. Um sie hier zu nutzen, führen Sie „Aus Git installieren“ mit der Repository-URL des Pakets und aktiviertem Desktop-Ziel aus – dadurch wird die Desktop-Hälfte auf diesen Rechner geklont.',
      emptyAll: 'Noch keine Plugins.',
      empty: 'Für dieses Profil sind keine Agent-Plugins installiert.',
      emptyHint: 'Durchsuchen Sie unten den Katalog und installieren Sie ein geprüftes Plugin mit einem Klick.',
      loadFailed: 'Agent-Plugins konnten nicht geladen werden',
      toggleFailed: name => `${name} konnte nicht umgeschaltet werden`,
      legacyBackend:
        'Dieses Backend ist älter als schlüsseladressierte Plugin-Schalter — aktualisieren Sie Hermes, um es hier zu verwalten.',
      portableBadge: 'tragbar',
      serverStates: {
        connected: 'verbunden',
        app_not_running: 'App läuft nicht',
        endpoint_unavailable: 'Endpunkt nicht verfügbar',
        no_interactive_session: 'keine interaktive Session',
        version_too_old: 'Version zu alt',
        missing_app: 'App fehlt',
        unknown: 'Status unbekannt'
      },
      catalogTitle: 'Plugin-Katalog',
      catalogBrowse: 'Durchsuchen',
      catalogHide: 'Katalog-Browser ausblenden',
      catalogHint:
        'Klicken Sie bei einem Plugin auf „+ Zu diesem Agenten hinzufügen“ – geprüfte Einträge werden an ihrem gepinnten Commit in das gewählte Profil installiert. Gebündelte Agent+Desktop-Plugins bieten beide Hälften an.',
      alreadyInstalled: name => `${name} ist in diesem Profil bereits installiert.`,
      catalogProvenance: sha => `Aus dem Hermes-Katalog installiert${sha ? ` am Pin ${sha}` : ''}.`,
      pinnedProvenance: sha =>
        `An Commit ${sha} gepinnt. Updates werden abgelehnt, bis es mit einem neuen Pin neu installiert wird.`,
      pinnedBadge: sha => `gepinnt @ ${sha}`,
      tierOfficial: 'offiziell',
      tierCommunity: 'Community',
      updateToPin: sha => `Auf ${sha} aktualisieren`,
      updateFailed: name => `${name} konnte nicht aktualisiert werden`,
      updated: name =>
        `${name} wurde auf den aktuellen Katalog-Pin aktualisiert. Starten Sie das Gateway neu, damit es wirksam wird.`,
      updateConsentTitle: (name: string) => `${name} fordert mehr an`,
      updateConsentBody: (name: string, sha: string) =>
        `Der neue Katalog-Pin von ${name} (${sha}) fügt Oberflächen hinzu, die die installierte Version nicht hat. Wenden Sie ihn nur an, wenn Sie ihnen vertrauen:`,
      updateConsentConfirm: 'Update anwenden',
      uninstall: 'Deinstallieren',
      uninstallTip: (name: string, profile: string) => `${name} aus ${profile} deinstallieren`,
      uninstallConfirmTitle: (name: string) => `${name} deinstallieren?`,
      uninstallConfirmBody: (name: string, profile: string) =>
        `Dadurch werden die Dateien des Plugins aus dem Profil ${profile} gelöscht. Eine mitgelieferte Desktop-Hälfte wird ebenfalls entfernt. Sie können es jederzeit aus dem Katalog oder aus Git neu installieren.`,
      uninstallFailed: (name: string) => `${name} konnte nicht deinstalliert werden`,
      uninstalled: (name: string) => `${name} deinstalliert. Starten Sie das Gateway neu, um es zu entladen.`,
      uninstallDesktopTip: (name: string) => `${name} aus dieser App deinstallieren`,
      uninstallDesktopConfirmBody: (name: string) =>
        `Dadurch wird ${name} aus dem Ordner desktop-plugins auf diesem Computer gelöscht und sofort entladen. Sie können es jederzeit aus Git neu installieren oder den Ordner zurücklegen.`,
      uninstalledDesktop: (name: string) => `${name} deinstalliert.`,
      deepLinkErrorTitle: 'Plugin-Installationslink abgelehnt',
      deepLinkCatalogInvalidName: 'Der Katalogname im Link fehlt oder ist ungültig.',
      deepLinkCatalogUnknown: (name: string) =>
        `\u201E${name}\u201C ist nicht im Hermes-Plugin-Katalog. Es wurde nichts installiert.`,
      deepLinkCatalogUnavailable:
        'Der Hermes-Plugin-Katalog konnte nicht geladen werden. Prüfen Sie Ihre Verbindung und öffnen Sie den Link erneut.',
      settingsToggle: (name: string) => `Einstellungen: ${name}`,
      settingsForm: {
        save: 'Einstellungen speichern',
        saved: (name: string) => `Einstellungen von ${name} gespeichert.`,
        saveFailed: (name: string) => `Einstellungen von ${name} konnten nicht gespeichert werden`,
        optional: '(optional)',
        secretSet: '•••••••• (gesetzt)',
        secretStoredAs: (env: string) =>
          `Wird in der .env des Profils als ${env} gespeichert, nie in config.yaml; leer lassen, um den aktuellen Wert zu behalten.`
      }
    },
    officialCatalog: 'Verfügbar zum Installieren',
    officialPill: 'Offiziell',
    hub: {
      searchPlaceholder: 'Skill-Hub durchsuchen',
      search: 'Suchen',
      searching: 'Wird gesucht...',
      connectingHubs: 'Verbindung zu Skill-Hubs wird hergestellt…',
      connectedHubs: 'Verbundene Hubs:',
      featured: 'Empfohlene Skills',
      landingHint:
        'Durchsuchen Sie den Hub nach installierbaren Skills aus dem offiziellen Index, von GitHub und aus Community-Quellen.',
      noResults: 'Im Hub wurden keine passenden Skills gefunden.',
      resultCount: (count, ms) => `${count} Result${count === 1 ? '' : 'ate'}${ms !== null ? ` in ${ms}ms` : ''}`,
      timedOut: sources => `Zeitüberschreitung: ${sources}`,
      installed: 'Installiert',
      install: 'Installieren',
      installing: 'Wird installiert...',
      uninstall: 'Deinstallieren',
      uninstalling: 'Wird deinstalliert...',
      updateAll: 'Installierte aktualisieren',
      updating: 'Wird aktualisiert...',
      preview: 'Vorschau',
      scan: 'Scannen',
      scanning: 'Wird gescannt...',
      close: 'Schließen',
      files: 'Dateien',
      noReadme: 'Dieser Skill hat keine SKILL.md-Vorschau.',
      trust: {
        builtin: 'integriert',
        trusted: 'vertraut',
        community: 'Community'
      },
      verdictSafe: 'Sicher',
      verdictCaution: 'Vorsicht',
      verdictDangerous: 'Gefährlich',
      policyAllow: 'Installation erlaubt',
      policyAsk: 'Vor der Installation prüfen',
      policyBlock: 'Installation durch Richtlinie blockiert',
      findings: count => `${count} Befund${count === 1 ? '' : 'e'}`,
      noFindings: 'Keine Sicherheitsbefunde.',
      installStarted: name => `${name} wird installiert...`,
      uninstallStarted: name => `${name} wird deinstalliert...`,
      updateStarted: 'Installierte Skills werden aktualisiert...',
      actionFailed: 'Skill-Aktion fehlgeschlagen',
      installBlockedTitle: name => `${name} konnte nicht installiert werden`,
      installBlockedMessage: (findings, unverified) =>
        `Der Sicherheitsscan hat ${findings > 0 ? `${findings} Punkt${findings === 1 ? '' : 'e'}` : 'riskante Muster'} zur Prüfung markiert${unverified ? ' und der Skill stammt aus einer unbestätigten Quelle' : ''}. Lesen Sie den Scan, bevor Sie dem Autor vertrauen.`,
      viewScan: 'Scan ansehen',
      openLog: 'Log öffnen',
      actionLog: 'Aktionsprotokoll',
      alreadyInstalled: (name: string) => `"${name}" ist bereits installiert`,
      pickerTitle: 'Skills Hub',
      pickerBrowse: 'Den ganzen Hub durchstöbern',
      pickerHide: 'Hub-Durchsucher ausblenden',
      pickerHint:
        'Klicken Sie bei einem beliebigen Skill auf „+ Zu diesem Agenten hinzufügen“ – er wird installiert und erscheint in der Liste oben.',
      loadFailed: 'Skill-Hub konnte nicht geladen werden',
      previewFailed: 'Skill-Vorschau fehlgeschlagen',
      scanFailed: 'Sicherheitsscan fehlgeschlagen',
      searchFailed: 'Hub-Suche fehlgeschlagen'
    }
  },
  starmap: {
    title: 'Speichergraph',
    subtitle: (nodes, clusters) => `${nodes} Skills in ${clusters} Kategorien`,
    close: 'Speichergraph schließen',
    refresh: 'Aktualisieren',
    memory: 'Speicher',
    filterAll: 'Alle',
    filterUsed: 'Verwendet',
    filterLearned: 'Gelernt',
    viewGraph: 'Graph',
    loadFailed: 'Speichergraph konnte nicht geladen werden',
    loading: 'Wird geladen…',
    emptyTitle: 'Noch nichts gelernt',
    emptyDesc: 'Sobald Hermes Skills und Erinnerungen zu Ihrer Arbeit aufbaut, erscheinen sie hier.',
    share: 'Map teilen',
    shareHint:
      'Kopieren Sie den Code, um diese Map zu teilen, oder fügen Sie einen ein, um sie zu laden. Er enthält nur das Layout, nicht Ihren Speicher oder Skill-Text.',
    shareTitle: 'Map importieren / exportieren',
    sharePlaceholder: 'Einen Map-Code einfügen…',
    copy: 'Map-Code kopieren',
    copied: 'Kopiert!',
    importMap: 'Eine Map importieren',
    importBtn: 'Laden',
    importEmpty: 'Fügen Sie einen Map-Code ein, um ihn zu laden.',
    importSuccess: nodes => `Eine Map mit ${nodes} ${nodes === 1 ? 'Knoten' : 'Knoten'} geladen.`,
    importedBadge: 'importierte Map',
    resetToMine: 'Zurück zu meiner Map'
  },
  agents: {
    extendedTranscript: 'Extended Transcript',
    transcriptTruncated: 'Zeigt die letzten 16 KiB',
    transcriptUnavailable: 'Live-Transcript nicht verfügbar',
    close: 'Agents schließen',
    title: 'Spawn-Baum',
    subtitle: 'Live-Subagent-Aktivität für die aktuelle Runde.',
    emptyTitle: 'Keine Live-Subagents',
    emptyDesc: 'Sobald eine Runde Arbeit delegiert, streamen die Unter-Agents ihren Fortschritt hierher.',
    running: 'Läuft',
    failed: 'Fehlgeschlagen',
    done: 'Fertig',
    streaming: 'Streaming',
    files: 'Dateien',
    moreFiles: count => `+${count} weitere Dateien`,
    moreAgents: count => `+${count} weitere Agents`,
    queued: 'In der Warteschlange',
    waitingActivity: 'Warten auf Aktivität',
    steer: 'Steuern',
    steerPlaceholder: 'Anweisungen für diesen Subagent',
    steerQueued: 'Für den nächsten Checkpoint eingereiht',
    stopRequested: 'Stopp angefordert',
    requestRejected: 'Der Subagent hat die Anfrage nicht angenommen',
    delegation: index => `Delegation ${index}`,
    workers: count => `${count} Worker`,
    workersActive: count => `${count} aktiv`,
    agentsCount: count => `${count} ${count === 1 ? 'Agent' : 'Agents'}`,
    activeCount: count => `${count} aktiv`,
    failedCount: count => `${count} fehlgeschlagen`,
    toolsCount: count => `${count} Tools`,
    filesCount: count => `${count} Dateien`,
    updatedAgo: age => `vor ${age} aktualisiert`,
    ageNow: 'jetzt',
    ageSeconds: seconds => `vor ${seconds}s`,
    ageMinutes: minutes => `vor ${minutes}m`,
    ageHours: hours => `vor ${hours}h`,
    ageDays: days => `vor ${days}d`,
    durationSeconds: seconds => `${seconds}s`,
    durationMinutes: (minutes, seconds) => `${minutes}m ${seconds}s`,
    tokens: value => `${value} Tok`
  },
  commandCenter: {
    close: 'Command Center schließen',
    paletteTitle: 'Befehlspalette',
    back: 'Zurück',
    searchPlaceholder: 'Sessions, Ansichten und Aktionen durchsuchen',
    goTo: 'Wechseln zu',
    goToSession: 'Zur Session gehen',
    branches: 'Branches',
    projects: 'Projekte',
    openFolder: 'Ordner als Projekt öffnen…',
    openFolderAt: path => `Ordner als Projekt öffnen — ${path}`,
    newSessionInProject: project => `Neue Session in ${project}`,
    commands: 'Befehle',
    startInBranch: branch => `Neue Konversation in ${branch}`,
    commandCenter: 'Command Center',
    appearance: 'Darstellung',
    settings: 'Einstellungen',
    changeTheme: 'Design wechseln',
    changeColorMode: 'Farbmodus wechseln…',
    pets: {
      title: 'Pets',
      placeholder: 'Pets suchen…',
      loading: 'Petdex-Galerie wird geladen…',
      error: 'Die Petdex-Galerie konnte nicht erreicht werden.',
      staleBackend: 'Starten Sie Hermes neu, um Pets zu verwenden – das Backend ist älter als diese Funktion.',
      empty: 'Keine passenden Pets.',
      turnOff: 'Ausschalten',
      turnOn: 'Einschalten',
      installed: 'Installiert',
      generatedTag: 'Generiert',
      adoptFailed: 'Dieses Pet konnte nicht adoptiert werden.',
      toggleFailed: enabled => `Das Pet konnte nicht ${enabled ? 'eingeschaltet' : 'ausgeschaltet'} werden.`,
      noneAvailable: 'Keine Pets verfügbar — wählen Sie weiter unten eines zum Installieren aus.'
    },
    generatePet: {
      title: 'Ein Pet generieren',
      placeholder: 'Beschreiben Sie ein Pet zum Generieren…',
      promptHint: 'Geben Sie eine Beschreibung ein und drücken Sie dann Enter, um vier Designs zu entwerfen.',
      readyHint: 'Drücken Sie Enter, um aus Ihrer Beschreibung vier Designs zu entwerfen.',
      generate: 'Generieren',
      generating: 'Wird generiert…',
      retry: 'Erneut versuchen',
      hatch: 'Schlüpfen',
      spawning: 'Wird erzeugt…',
      hatching: 'Ihr Pet schlüpft…',
      hatchingSub: 'Wir erwecken es zum Leben…',
      hatched: 'Es ist geschlüpft!',
      hatchRow: (_state, done, total) => `Frame ${done} von ${total} wird gezeichnet…`,
      hatchComposing: 'Es wird zusammengesetzt…',
      hatchSaving: 'Fast geschafft…',
      namePlaceholder: 'Geben Sie Ihrem Pet einen Namen',
      staleBackend: 'Aktualisieren Sie Hermes, um Pets zu generieren.',
      backgroundHint: 'Sie können dieses Fenster schließen – Hermes benachrichtigt Sie, wenn es fertig ist.',
      slowProviderHint: 'Das kann mehrere Minuten dauern',
      remix: 'Remixen',
      remixConfirmTitle: 'Dieses Design remixen?',
      remixConfirmBody:
        'Das erzeugt einen neuen Satz an Entwürfen und verwendet dieses als Ausgangspunkt. Das kann mehrere Minuten dauern.',
      genericError: 'Generierung fehlgeschlagen – versuchen Sie es erneut oder wählen Sie einen Vorschlag.',
      referenceImageTooLarge: 'Das Referenzbild ist zu groß. Verwenden Sie eines unter 16 MB.',
      referenceImageInvalid: 'Das Referenzbild konnte nicht gelesen werden. Probieren Sie ein PNG, JPG, WebP oder GIF.',
      adopt: 'Adoptieren',
      startOver: 'Neu beginnen'
    },
    installTheme: {
      title: 'Design installieren…',
      pageTitle: 'Design installieren',
      placeholder: 'VS Code Marketplace durchsuchen...',
      loading: 'Marketplace wird durchsucht...',
      error: 'Der Marketplace konnte nicht erreicht werden.',
      empty: 'Keine passenden Designs.',
      install: 'Installieren',
      installing: 'Wird installiert...',
      installed: 'Installiert',
      installs: count => `${count} Installationen`
    },
    settingsFields: 'Einstellungsfelder',
    mcpServers: 'MCP-Server',
    archivedChats: 'Archivierte Chats',
    sections: {
      maintenance: 'Wartung',
      sessions: 'Sessions',
      system: 'System',
      usage: 'Nutzung'
    },
    sectionDescriptions: {
      maintenance: 'Diagnose, Backups, Curator und Memory-Daten',
      sessions: 'Sessions durchsuchen und verwalten',
      system: 'Status, Logs und Systemaktionen',
      usage: 'Token-, Kosten- und Skill-Aktivität im Zeitverlauf'
    },
    nav: {
      newChat: {
        title: 'Neue Session',
        detail: 'Eine neue Session starten'
      },
      settings: {
        title: 'Einstellungen',
        detail: 'Hermes Desktop konfigurieren'
      },
      capabilities: {
        title: 'Fähigkeiten',
        detail: 'Skills, Tools, MCP-Server und Plugins'
      },
      messaging: {
        title: 'Messaging',
        detail: 'Telegram, Slack, Discord und mehr einrichten'
      },
      artifacts: {
        title: 'Artefakte',
        detail: 'Generierte Ausgaben durchstöbern'
      }
    },
    sectionEntries: {
      sessions: {
        title: 'Sessions-Bereich',
        detail: 'Sessions durchsuchen, anpinnen und verwalten'
      },
      system: {
        title: 'System-Bereich',
        detail: 'Gateway-Status, Logs, Neustart/Update'
      },
      usage: {
        title: 'Nutzungs-Bereich',
        detail: 'Token-, Kosten- und Skill-Aktivität'
      }
    },
    providerNavigate: 'Navigieren',
    providerSessions: 'Sessions',
    refresh: 'Aktualisieren',
    refreshing: 'Wird aktualisiert...',
    noResults: 'Keine passenden Ergebnisse gefunden.',
    pinSession: 'Session anpinnen',
    unpinSession: 'Session lösen',
    exportSession: 'Session exportieren',
    deleteSession: 'Session löschen',
    noSessions: 'Noch keine Sessions.',
    gatewayRunning: 'Messaging-Gateway läuft',
    gatewayStopped: 'Messaging-Gateway gestoppt',
    hermesActiveSessions: (version, count) => `Hermes ${version} · Aktive Sessions ${count}`,
    restartGateway: 'Gateway neu starten',
    openBrowser: 'Browser öffnen',
    gatewayRestartFailed: 'Gateway-Neustart fehlgeschlagen.',
    sharedGatewayRestartTitle: 'Gemeinsames Gateway neu starten?',
    sharedGatewayRestartDescription: bots => `Alle Bots auf diesem Gerät verbinden sich neu: ${bots}`,
    sharedGatewayRestartConfirm: 'Alle neu starten',
    sharedGatewayRestarted: count => `Gemeinsames Gateway neu gestartet (${count} ${count === 1 ? 'Bot' : 'Bots'})`,
    updateHermes: 'Hermes aktualisieren',
    reloadWindow: 'Fenster neu laden',
    actionRunning: 'läuft',
    actionDone: 'fertig',
    actionFailed: 'fehlgeschlagen',
    actionStartedWaiting: 'Aktion gestartet, wartet auf Status…',
    loadingStatus: 'Status wird geladen...',
    recentLogs: 'Letzte Logs',
    noLogs: 'Noch keine Logs geladen.',
    days: count => `${count} T`,
    statSessions: 'Sessions',
    statApiCalls: 'API-Aufrufe',
    statTokens: 'Token ein/aus',
    statCost: 'Gesch. Kosten',
    actualCost: cost => `tatsächlich ${cost}`,
    loadingUsage: 'Nutzung wird geladen...',
    noUsage: period => `In den letzten ${period} Tagen keine Nutzung.`,
    retry: 'Erneut versuchen',
    dailyTokens: 'Tägliche Token',
    input: 'Eingang',
    output: 'Ausgang',
    noDailyActivity: 'Keine Tagesaktivität.',
    topModels: 'Top-Modelle',
    noModelUsage: 'Noch keine Model-Nutzung.',
    topSkills: 'Top-Skills',
    noSkillActivity: 'Noch keine Skill-Aktivität.',
    actions: count => `${count} Aktionen`,
    logFile: 'Logdatei',
    logLevel: 'Stufe',
    logSearchPlaceholder: 'Logzeilen filtern...',
    maintenance: {
      runOps: 'Diagnose',
      doctor: 'Doctor ausführen',
      doctorDesc: 'Installation, Konfiguration und Provider auf Fehler prüfen',
      securityAudit: 'Sicherheitsaudit',
      securityAuditDesc: 'Konfiguration und Skills auf riskante Einstellungen scannen',
      backup: 'Backup erstellen',
      backupDesc: 'Konfiguration, Memories, Skills und Sessions als ZIP packen',
      debugShare: 'Debug-Freigabe',
      debugShareDesc:
        'Einen geschwärzten Bericht + Logs hochladen, teilbare Links erhalten (wird nach 6h automatisch gelöscht)',
      debugShareRunning: 'Debug-Bericht wird hochgeladen...',
      debugShareLinks: 'Freigabelinks',
      debugShareFailed: 'Debug-Freigabe fehlgeschlagen',
      copyLink: 'Link kopieren',
      linkCopied: 'Link kopiert',
      curator: 'Skill-Curator',
      curatorDesc: 'Hintergrundprüfung, die veraltete, von Agenten erstellte Skills archiviert',
      curatorPaused: 'Pausiert',
      curatorActive: 'Aktiv',
      curatorDisabled: 'Deaktiviert',
      curatorLastRun: when => `Zuletzt ausgeführt ${when}`,
      curatorNeverRan: 'Nie ausgeführt',
      pause: 'Pausieren',
      resume: 'Fortsetzen',
      runNow: 'Jetzt ausführen',
      memoryData: 'Memory-Daten',
      memoryDataDesc: 'Eingebaute Memory-Dateien, die in jede Session eingefügt werden',
      memoryProvider: name => `Aktiver Provider: ${name}`,
      builtinMemory: 'eingebaut',
      memoryFile: 'Agent-Memory (MEMORY.md)',
      userFile: 'Benutzerprofil (USER.md)',
      bytes: size => size,
      empty: 'leer',
      resetMemory: 'Memory zurücksetzen',
      resetUser: 'Profil zurücksetzen',
      resetAll: 'Beide zurücksetzen',
      resetConfirm: target => `${target} löschen? Das kann nicht rückgängig gemacht werden.`,
      resetDone: files => `${files} gelöscht.`,
      resetFailed: 'Memory-Zurücksetzen fehlgeschlagen',
      actionStarted: name => `${name} gestartet — Log wird verfolgt...`,
      actionFailed: name => `${name} konnte nicht gestartet werden`,
      running: 'Läuft...',
      viewLog: 'Aktionslog'
    }
  },
  messaging: {
    search: 'Messaging durchsuchen...',
    loading: 'Messaging-Plattformen werden geladen...',
    loadFailed: 'Messaging-Plattformen konnten nicht geladen werden',
    states: {
      connected: 'Verbunden',
      connecting: 'Verbindung wird hergestellt',
      disabled: 'Deaktiviert',
      fatal: 'Fehler',
      gateway_stopped: 'Messaging-Gateway gestoppt',
      not_configured: 'Einrichtung nötig',
      pending_restart: 'Neustart nötig',
      retrying: 'Neuer Versuch',
      startup_failed: 'Start fehlgeschlagen'
    },
    unknown: 'Unbekannt',
    hintPendingRestart: 'Starten Sie das Gateway über die Statusleiste neu, um diese Änderung zu übernehmen.',
    sharedListenerUrl: 'Erreichbar über den gemeinsamen Gateway-Listener unter',
    hintGatewayStopped: 'Starten Sie das Gateway über die Statusleiste, um die Verbindung herzustellen.',
    credentialsSet: 'Zugangsdaten gesetzt',
    needsSetup: 'Einrichtung nötig',
    gatewayStopped: 'Messaging-Gateway gestoppt',
    getCredentials: 'Zugangsdaten abrufen',
    openSetupGuide: 'Einrichtungsanleitung öffnen',
    required: 'Erforderlich',
    recommended: 'Empfohlen',
    advanced: count => `Erweitert (${count})`,
    noTokenNeeded:
      'Für diese Plattform benötigen Sie hier kein Token. Folgen Sie der Einrichtungsanleitung oben und aktivieren Sie sie darunter.',
    enabled: 'Aktiviert',
    disabled: 'Deaktiviert',
    unsavedChanges: 'Ungespeicherte Änderungen',
    saving: 'Wird gespeichert…',
    saveChanges: 'Änderungen speichern',
    saved: 'Gespeichert',
    replaceValue: 'Aktuellen Wert ersetzen',
    openDocs: 'Dokumentation öffnen',
    clearField: key => `${key} löschen`,
    enableAria: name => `${name} aktivieren`,
    disableAria: name => `${name} deaktivieren`,
    platformEnabled: name => `${name} aktiviert`,
    platformDisabled: name => `${name} deaktiviert`,
    restartToApply: 'Diese Änderung wird nach einem Gateway-Neustart wirksam.',
    setupSaved: name => `${name}-Einrichtung gespeichert`,
    restartToReconnect: 'Neue Zugangsdaten werden nach einem Gateway-Neustart wirksam.',
    appliedLive: 'Auf das laufende Gateway angewendet.',
    connectingLive: 'Das laufende Gateway verbindet sich mit den neuen Zugangsdaten.',
    keyCleared: key => `${key} gelöscht`,
    setupUpdated: name => `${name}-Einrichtung wurde aktualisiert.`,
    failedUpdate: name => `Konnte ${name} nicht aktualisieren`,
    failedSave: name => `Konnte ${name} nicht speichern`,
    failedClear: key => `Konnte ${key} nicht löschen`,
    pendingRequests: count => `Wartende Anfragen (${count})`,
    pendingAria: count => `${count} wartende ${count === 1 ? 'Koppelung' : 'Koppelungen'}`,
    approvedUsers: count => `Freigegebene Benutzer (${count})`,
    approve: 'Freigeben',
    approving: 'Freigeben...',
    revoke: 'Entziehen',
    revoking: 'Wird entzogen...',
    revokeAria: name => `Entziehe ${name}`,
    revokeTitle: 'Zugriff entziehen',
    revokeDesc: (name: string) =>
      `${name} verliert den Zugriff und wird bei seiner nächsten Message nicht mehr erkannt.`,
    approvedUser: name => `${name} freigegeben`,
    approvedHint: 'Sie werden bei ihrer nächsten Message automatisch erkannt.',
    revokedUser: name => `Zugriff für ${name} entzogen`,
    failedApprove: name => `Konnte ${name} nicht freigeben`,
    failedRevoke: name => `Konnte ${name} nicht entziehen`,
    pairingLockedOut:
      'Zu viele fehlgeschlagene Freigaben – diese Plattform ist gesperrt. Versuchen Sie es später erneut.',
    waitingSince: minutes => (minutes < 1 ? 'gerade eben' : `${minutes} Min. her`),
    restartNeeded: 'Gespeichert. Starten Sie das Messaging-Gateway neu, damit die neuen Einstellungen wirksam werden.',
    restartNow: 'Jetzt neu starten',
    restarting: 'Wird neu gestartet…',
    restartFailedManual:
      'Gateway-Neustart fehlgeschlagen – starten Sie es manuell neu und prüfen Sie die Gateway-Logs.',
    restartFailedManualDetail:
      'Versuchen Sie den Neustart erneut; wenn er weiterhin fehlschlägt, öffnen Sie die Logs und senden Sie Diagnosedaten.',
    restartAgain: 'Erneut neu starten',
    openLogs: 'Logs öffnen',
    telegramQr: {
      title: 'Wählen Sie, wie Sie Ihren Telegram-Bot verbinden',
      subtitle:
        'Beide Optionen verbinden einen Bot, den Sie kontrollieren, und speichern seine Zugangsdaten nur in dieser Hermes-Installation.',
      quickSetup: 'Schnelleinrichtung',
      recommended: 'Empfohlen',
      quickHelp:
        'Scannen Sie einen QR-Code und bestätigen Sie in Telegram. Hermes legt den Bot an und erkennt Ihre Telegram-Benutzer-ID automatisch.',
      createWithQr: 'Mit QR-Code anlegen',
      starting: 'Wird gestartet…',
      replaceWarning:
        'Es sind bereits Telegram-Zugangsdaten konfiguriert. Eine neue QR-Einrichtung oder ein Bot-Token ersetzt beim Speichern den aktuellen Bot.',
      scanHint: 'Scannen Sie mit der Telegram-App auf Ihrem Handy oder öffnen Sie den Link auf diesem Computer.',
      waiting: 'Warten auf Telegram…',
      expiresIn: remaining => `Läuft ab in ${remaining}`,
      expired: 'Abgelaufen',
      openTelegram: 'Telegram öffnen',
      ready: 'Bot angelegt',
      allowedUsers: 'Erlaubte Benutzer',
      ownerDetected: 'Eigentümer erkannt',
      addAtLeastOne: 'Fügen Sie mindestens eine Telegram-Benutzer-ID hinzu.',
      userIdPlaceholder: 'Telegram-Benutzer-ID',
      add: 'Hinzufügen',
      numericOnly: 'Erlaubte Telegram-Benutzer-IDs müssen numerisch sein.',
      saveAndRestart: 'Speichern und neu starten',
      applying: 'Wird gespeichert…',
      pairingExpired: 'Telegram-Kopplung abgelaufen. Starten Sie eine neue QR-Einrichtung, um es erneut zu versuchen.',
      stillWaiting: detail => `Noch keine Antwort von Telegram. Neuer Versuch nach: ${detail}`,
      savedRestarting: 'Telegram gespeichert; Gateway wird neu gestartet…',
      savedRestartFailed: detail => `Telegram gespeichert; Gateway-Neustart fehlgeschlagen${detail}`
    },
    fieldCopy: {
      TELEGRAM_BOT_TOKEN: {
        label: 'Bot-Token',
        help: 'Erstellen Sie einen Bot mit @BotFather und fügen Sie das Token ein, das Sie von ihm erhalten.',
        placeholder: 'Telegram-Bot-Token einfügen'
      },
      TELEGRAM_ALLOWED_USERS: {
        label: 'Erlaubte Telegram-Benutzer-IDs',
        help: 'Empfohlen. Numerische IDs von @userinfobot, durch Kommas getrennt. Ohne diese können Ihnen beliebige Benutzer Direktnachrichten senden.'
      },
      TELEGRAM_PROXY: {
        label: 'Proxy-URL',
        help: 'Nur auf Netzwerken nötig, wo Telegram blockiert ist.'
      },
      DISCORD_BOT_TOKEN: {
        label: 'Bot-Token',
        help: 'Erstellen Sie im Discord Developer Portal eine Anwendung, fügen Sie einen Bot hinzu und fügen Sie sein Token ein.'
      },
      DISCORD_ALLOWED_USERS: {
        label: 'Erlaubte Discord-Benutzer-IDs',
        help: 'Empfohlen. Discord-Benutzer-IDs, durch Kommas getrennt.'
      },
      DISCORD_REPLY_TO_MODE: {
        label: 'Antwortstil',
        help: 'first, all oder off.'
      },
      DISCORD_ALLOW_ALL_USERS: {
        label: 'Alle Discord-Benutzer zulassen',
        help: 'Nur für die Entwicklung. Bei true kann jeder Benutzer dem Bot per DM schreiben, ohne Allowlist.'
      },
      DISCORD_HOME_CHANNEL: {
        label: 'Home-Kanal-ID',
        help: 'Kanal, in den der Bot proaktive Messages sendet (Cron-Ausgabe, Erinnerungen).'
      },
      DISCORD_HOME_CHANNEL_NAME: {
        label: 'Name des Home-Kanals',
        help: 'Anzeigename des Home-Kanals in Logs und Status-Ausgabe.'
      },
      BLUEBUBBLES_ALLOW_ALL_USERS: {
        label: 'Alle iMessage-Benutzer zulassen',
        help: 'Bei true wird die BlueBubbles-Allowlist übersprungen.'
      },
      MATTERMOST_ALLOW_ALL_USERS: {
        label: 'Alle Mattermost-Benutzer zulassen'
      },
      MATTERMOST_HOME_CHANNEL: {
        label: 'Home-Kanal'
      },
      QQ_ALLOW_ALL_USERS: {
        label: 'Alle QQ-Benutzer zulassen'
      },
      QQBOT_HOME_CHANNEL: {
        label: 'QQ-Home-Kanal',
        help: 'Standard-Kanal oder -Gruppe für die Cron-Zustellung.'
      },
      QQBOT_HOME_CHANNEL_NAME: {
        label: 'Name des QQ-Home-Kanals'
      },
      SLACK_BOT_TOKEN: {
        label: 'Slack-Bot-Token',
        help: 'Verwenden Sie das Bot-Token aus OAuth & Permissions, nachdem Sie Ihre Slack-App installiert haben.',
        placeholder: 'Slack-Bot-Token einfügen'
      },
      SLACK_APP_TOKEN: {
        label: 'Slack-App-Token',
        help: 'Verwenden Sie das App-Level-Token, das für den Socket Mode erforderlich ist.',
        placeholder: 'Slack-App-Token einfügen'
      },
      SLACK_ALLOWED_USERS: {
        label: 'Erlaubte Slack-Benutzer-IDs',
        help: 'Empfohlen. Slack-Benutzer-IDs, durch Kommas getrennt.'
      },
      MATTERMOST_URL: {
        label: 'Server-URL',
        placeholder: 'https://mattermost.example.com'
      },
      MATTERMOST_TOKEN: {
        label: 'Bot-Token'
      },
      MATTERMOST_ALLOWED_USERS: {
        label: 'Erlaubte Benutzer-IDs',
        help: 'Empfohlen. Mattermost-Benutzer-IDs, durch Kommas getrennt.'
      },
      MATRIX_HOMESERVER: {
        label: 'Homeserver-URL',
        placeholder: 'https://matrix.org'
      },
      MATRIX_ACCESS_TOKEN: {
        label: 'Zugriffs-Token'
      },
      MATRIX_USER_ID: {
        label: 'Bot-Benutzer-ID',
        placeholder: '@hermes:example.org'
      },
      MATRIX_ALLOWED_USERS: {
        label: 'Erlaubte Matrix-Benutzer-IDs',
        help: 'Empfohlen. Benutzer-IDs im Format @benutzer:server, durch Kommas getrennt.'
      },
      SIGNAL_HTTP_URL: {
        label: 'Signal-Bridge-URL',
        placeholder: 'http://127.0.0.1:8080',
        help: 'URL einer laufenden signal-cli-REST-Bridge.'
      },
      SIGNAL_ACCOUNT: {
        label: 'Telefonnummer',
        help: 'Die Nummer, die bei Ihrer signal-cli-Bridge registriert ist.'
      },
      SIGNAL_ALLOWED_USERS: {
        label: 'Erlaubte Signal-Benutzer',
        help: 'Empfohlen. Signal-Kennungen, durch Kommas getrennt.'
      },
      WHATSAPP_ENABLED: {
        label: 'WhatsApp-Bridge aktivieren',
        help: 'Wird automatisch über den Schalter unten gesetzt. Lassen Sie es unverändert, sofern Sie es nicht ausdrücklich benötigen.'
      },
      WHATSAPP_MODE: {
        label: 'Bridge-Modus'
      },
      WHATSAPP_ALLOWED_USERS: {
        label: 'Erlaubte WhatsApp-Benutzer',
        help: 'Empfohlen. Telefonnummern oder WhatsApp-IDs, durch Kommas getrennt.'
      }
    },
    platformIntro: {}
  },
  webhooks: {
    search: 'Webhooks suchen …',
    loading: 'Webhooks werden geladen …',
    loadFailed: 'Webhooks konnten nicht geladen werden',
    subscriptions: (count: number) => `Abonnements (${count})`,
    hint: 'Änderungen an Abonnements werden heiß nachgeladen, sobald der Empfänger läuft. Deaktivierte Abonnements lehnen eingehende Events ab.',
    empty: 'Noch keine Webhook-Abonnements vorhanden.',
    disabledTitle: 'Webhook-Empfänger deaktiviert',
    disabledBody:
      'Webhooks sind eine eigene Gateway-Plattform. Aktivieren Sie sie hier, um eingehende HTTP-Events anzunehmen; Chat-Kanäle werden nur benötigt, wenn ein Abonnement an Telegram, Discord, Slack oder einen anderen Kanal zustellt.',
    enable: 'Webhooks aktivieren',
    enabling: 'Wird aktiviert …',
    enabled: (name: string) => `Aktiviert: „${name}“`,
    disabled: (name: string) => `Deaktiviert: „${name}“`,
    enableRow: 'Aktivieren',
    disableRow: 'Deaktivieren',
    delete: 'Löschen',
    deleting: 'Wird gelöscht …',
    deleted: 'Webhook gelöscht',
    deleteTitle: 'Webhook löschen',
    deleteDescPrefix: 'Dadurch wird ',
    deleteDescSuffix: ' endgültig entfernt. Das kann nicht rückgängig gemacht werden.',
    deleteFailed: (name: string) => `Löschen von „${name}“ fehlgeschlagen`,
    toggleFailed: (name, enabled) =>
      `Umschalten von „${name}“ ${enabled ? 'einschalten' : 'ausschalten'} fehlgeschlagen`,
    newSubscription: 'Neues Abonnement',
    restarting: 'Gateway wird neu gestartet …',
    restartNeeded:
      'Webhooks sind aktiviert, aber das Gateway braucht vor dem Onlinegehen des Empfängers noch einen Neustart.',
    restartGateway: 'Gateway neu starten',
    restartingGateway: 'Wird neu gestartet …',
    restartFailed: (detail: string) => `Gateway-Neustart fehlgeschlagen${detail}`,
    enabledRestarting: 'Webhooks aktiviert; Gateway wird neu gestartet …',
    all: '(alle)',
    deliverOnly: 'nur zustellen',
    createdTitle: 'Abonnement erstellt',
    createdSecretHint: 'Kopieren Sie das Secret jetzt – es wird nur einmal angezeigt.',
    webhookUrl: 'Webhook-URL',
    secretOnce: 'Secret (nur einmal angezeigt)',
    done: 'Fertig',
    fieldName: 'Name',
    fieldNamePlaceholder: 'z. B. github-push',
    fieldDescription: 'Beschreibung',
    fieldDescriptionPlaceholder: 'Was dieser Webhook tut (optional)',
    fieldEvents: 'Events',
    fieldEventsPlaceholder: 'durch Kommas getrennt, für alle leer lassen',
    fieldSkills: 'Skills',
    fieldSkillsPlaceholder: 'durch Kommas getrennte Skill-Namen (optional)',
    fieldDeliver: 'Zustellen an',
    fieldDeliverOnly: 'Nur die Payload zustellen',
    fieldPrompt: 'Prompt',
    fieldPromptPlaceholder: 'Anweisungen für den Agenten, wenn dieser Webhook ausgelöst wird (optional)',
    nameRequired: 'Name erforderlich',
    create: 'Erstellen',
    creating: 'Wird erstellt …',
    created: 'Erstellt',
    createFailed: (detail: string) => `Erstellen fehlgeschlagen: ${detail}`,
    copy: 'Kopieren',
    deliverOptions: {
      log: 'Log',
      telegram: 'Telegram',
      discord: 'Discord',
      slack: 'Slack',
      email: 'E-Mail',
      github_comment: 'GitHub-Kommentar'
    }
  },
  profiles: {
    close: 'Profile schließen',
    nameHint:
      'Kleinbuchstaben, Ziffern, Bindestriche und Unterstriche. Muss mit einem Buchstaben oder einer Ziffer beginnen.',
    title: 'Profile',
    count: count => `${count} ${count === 1 ? 'Profil' : 'Profile'}`,
    search: 'Profile durchsuchen...',
    loading: 'Profile werden geladen...',
    newProfile: 'Neues Profil',
    importProfile: 'Profil importieren…',
    exportProfile: 'Profil exportieren…',
    imported: 'Profil importiert',
    exported: 'Profil exportiert',
    failedImport: 'Profil konnte nicht importiert werden',
    failedExport: 'Profil konnte nicht exportiert werden',
    allProfiles: 'Alle Profile',
    showAllProfiles: 'Alle Profile anzeigen',
    switchToProfile: name => `Zu ${name} wechseln`,
    switchToConnection: name => `Zu ${name} wechseln`,
    switchConnectionFailed: name => `Keine Verbindung zu ${name} möglich`,
    manageProfiles: 'Profile verwalten…',
    connectGateway: 'Gateways verwalten…',
    fleet: {
      allOnGateway: 'Alle Profile auf diesem Gateway',
      gateway: gateway => `Profile auf ${gateway}`,
      gatewayUnreachable: gateway => `${gateway} · nicht erreichbar`,
      onGateway: (name, gateway) => `${name} · ${gateway}`,
      switchTo: (name, gateway) => `Zu ${name} auf ${gateway} wechseln`,
      deleteOn: gateway => ` auf ${gateway}`,
      localDevice:
        'Dieses Gerät (lokales Backend — installiert Hermes, falls es fehlt, sonst öffnet es eine neue Sitzung)',
      switchDeviceTitle: 'Zu diesem Gerät wechseln?',
      switchDeviceDesc:
        'Das öffnet eine neue Sitzung auf diesem Computer. Das aktuelle Gespräch bleibt auf dem anderen Gateway.',
      switchDeviceConfirm: 'Wechseln',
      installDeviceTitle: 'Zu diesem Gerät wechseln?',
      installDeviceDesc:
        'Hermes wird lokal installiert und danach eine neue Sitzung auf diesem Computer geöffnet. Ohne Bestätigung startet keine Installation.',
      installDeviceConfirm: 'Lokal installieren',
      connectExistingInstead: 'Stattdessen vorhandenes verbinden'
    },
    status: {
      unread: (count: number) => (count === 1 ? '1 ungelesene Sitzung' : `${count} ungelesene Sitzungen`),
      needsInput: (count: number) =>
        count === 1 ? '1 Sitzung wartet auf Ihre Antwort' : `${count} Sitzungen warten auf Ihre Antwort`,
      working: (count: number) => (count === 1 ? '1 laufende Sitzung' : `${count} laufende Sitzungen`)
    },
    remoteOverride: {
      menuItem: 'Mit Remote-Host verbinden…',
      badge: (host: string) => `Läuft auf ${host}`,
      title: (profile: string) => `${profile} mit einem Remote-Host verbinden`,
      description:
        'Sessions in diesem Profil laufen auf dem von Ihnen festgelegten Remote-Hermes statt auf diesem Computer.',
      urlLabel: 'Remote-Adresse',
      urlPlaceholder: 'https://hermes.example.com',
      urlInvalid: 'Geben Sie eine vollständige Adresse ein, die mit http:// oder https:// beginnt',
      tokenLabel: 'Access-Token',
      tokenPlaceholder: 'Remote-Session-Token einfügen',
      tokenSavedHint: 'Ein Token ist bereits gespeichert. Leer lassen, um es zu behalten.',
      plainTextOptIn:
        'Dieser Computer hat keine sichere Schlüssel-Ablage, daher würde das Token unverschlüsselt auf der Platte gespeichert. Trotzdem speichern?',
      collisionWarning: (label: string) =>
        `Ein Gateway mit dem Namen “${label}” existiert bereits in den Einstellungen. Diese Profil-Verbindung ist davon getrennt und ändert es nicht.`,
      confirmTitle: 'Dieses Profil mit einem Remote-Host verbinden?',
      confirmNote: (profile: string, host: string) =>
        `Neue Chats in ${profile} laufen auf ${host}. Befehle werden dort ausgeführt und Dateien dort gelesen, nicht auf diesem Rechner. Verbinden Sie sich nur mit einem Host, dem Sie vertrauen.`,
      confirmBack: 'Zurück',
      connect: 'Verbinden',
      connecting: 'Verbindung wird hergestellt…',
      disconnect: 'Remote-Verbindung entfernen',
      savedTitle: 'Profil verbunden',
      savedMessage: (profile: string, host: string) => `${profile} läuft jetzt auf ${host}`,
      removedTitle: 'Remote-Verbindung entfernt',
      removedMessage: (profile: string) => `${profile} läuft jetzt auf diesem Computer`,
      removeFailed: 'Die Remote-Verbindung konnte nicht entfernt werden',
      authFailedTitle: 'Remote-Host hat das gespeicherte Token abgelehnt',
      authFailedMessage: (profile: string, host: string) =>
        `${host} hat das für ${profile} gespeicherte Token abgelehnt. Es könnte auf der Remote-Seite geändert worden sein.`,
      updateToken: 'Neues Token eingeben…'
    },
    actions: 'Aktionen',
    color: 'Farbe…',
    colorFor: 'Farbe',
    openInNewWindow: 'In neuem Fenster öffnen',
    setAsDefault: 'Als Standard festlegen',
    defaultProfile: 'Standardprofil',
    defaultSet: (name: string) => `${name} ist jetzt der Standard`,
    defaultDescription:
      'Wird beim Öffnen von Hermes und für neue Chats verwendet. Bestehende Sessions bleiben in ihren Profilen.',
    failedSetDefault: 'Das Standardprofil konnte nicht festgelegt werden',
    setColor: color => `Farbe ${color} setzen`,
    autoColor: 'Auto',
    noProfiles: 'Noch keine Profile.',
    selectPrompt: 'Wählen Sie ein Profil aus, um seine Details zu sehen.',
    refresh: 'Profile aktualisieren',
    refreshing: 'Profile werden aktualisiert',
    default: 'default',
    skills: count => `${count} ${count === 1 ? 'Skill' : 'Skills'}`,
    env: 'env',
    defaultBadge: 'Standard',
    rename: 'Umbenennen',
    renameMenu: 'Umbenennen…',
    exportMenu: 'Exportieren…',
    editSoul: 'SOUL.md bearbeiten…',
    copySetup: 'Setup kopieren',
    copying: 'Wird kopiert...',
    modelLabel: 'Modell',
    skillsLabel: 'Skills',
    notSet: 'Nicht gesetzt',
    soulDesc: 'Die System-Prompt- und Personen-Anweisungen, die in diesem Profil fest eingebacken sind.',
    soulOptional: 'optional',
    soulPlaceholder: mode =>
      `Die System-Prompt- bzw. Personen-Anweisung für dieses Profil.\nLeer lassen, um den ${mode}-Standard zu behalten.`,
    soulPlaceholderCloned: 'geklont',
    soulPlaceholderEmpty: 'leer',
    unsavedChanges: 'Ungespeicherte Änderungen',
    loadingSoul: 'SOUL.md wird geladen...',
    emptySoul: 'Leere SOUL.md — beginnen Sie, die Persona zu schreiben…',
    saving: 'Wird gespeichert...',
    saveSoul: 'SOUL.md speichern',
    deleteTitle: 'Profil löschen?',
    deleteDescPrefix: 'Das löscht ',
    deleteDescMid: ' und entfernt sein ',
    deleteDescSuffix: ' Verzeichnis. Das kann nicht rückgängig gemacht werden.',
    deleting: 'Wird gelöscht...',
    createDesc: 'Profile sind unabhängige Hermes-Umgebungen: eigene Config, eigene Skills und eigene SOUL.md.',
    nameLabel: 'Name',
    cloneFrom: 'Klonen von',
    cloneFromNone: 'Keine (leer)',
    cloneFromDesc: 'Kopiert Config, Skills und SOUL.md aus dem gewählten Quellprofil.',
    cloneFromDefault: 'Vom Standardprofil klonen',
    cloneFromDefaultDesc: 'Config, Skills und SOUL.md aus Ihrem Standardprofil kopieren.',
    invalidName: hint => `Ungültiger Name. ${hint}`,
    nameRequired: 'Name ist erforderlich.',
    creating: 'Wird erstellt...',
    createAction: 'Profil erstellen',
    renameTitle: 'Profil umbenennen',
    renameDescPrefix: 'Durch das Umbenennen werden das Profilverzeichnis und alle Wrapper-Scripts in ',
    renameDescSuffix: ' aktualisiert.',
    displayNameTitle: 'Diesen Agent benennen',
    displayNameDesc:
      'Legt einen Anzeigenamen fest, der in der ganzen App angezeigt wird. Die interne Profil-ID bleibt "default".',
    displayNameLabel: 'Anzeigename',
    newNameLabel: 'Neuer Name',
    renaming: 'Wird umbenannt...',
    created: 'Profil erstellt',
    renamed: 'Profil umbenannt',
    deleted: 'Profil gelöscht',
    setupCopied: 'Setup-Befehl kopiert',
    soulSaved: 'SOUL.md gespeichert',
    failedLoad: 'Profile konnten nicht geladen werden',
    failedDelete: 'Profil konnte nicht gelöscht werden',
    failedCopy: 'Setup-Befehl konnte nicht kopiert werden',
    failedLoadSoul: 'SOUL.md konnte nicht geladen werden',
    failedSaveSoul: 'SOUL.md konnte nicht gespeichert werden',
    failedCreate: 'Profil konnte nicht erstellt werden',
    failedRename: 'Profil konnte nicht umbenannt werden'
  },
  modelAssignment: {
    saveFailed: 'Hermes hat diese Modelländerung nicht gespeichert.',
    confirmTitle: 'Warnung zur Modellauswahl',
    confirmDetail: 'Bestätigen Sie nur, wenn Sie diesen Kompromiss akzeptieren.',
    confirmAction: 'Bestätigen',
    declined: 'Modelländerung abgebrochen – Sie haben die Warnung zur Datentrainings-Stufe abgelehnt.'
  },
  cron: {
    close: 'Cron schließen',
    title: 'Geplante Jobs',
    count: count => `${count} ${count === 1 ? 'Job' : 'Jobs'}`,
    search: 'Cron-Jobs durchsuchen...',
    loading: 'Cron-Jobs werden geladen...',
    states: {
      enabled: 'aktiviert',
      scheduled: 'geplant',
      running: 'läuft',
      paused: 'pausiert',
      disabled: 'deaktiviert',
      error: 'Fehler',
      completed: 'abgeschlossen'
    },
    lastRunFailed: 'Letzter Lauf fehlgeschlagen:',
    editJob: 'Job bearbeiten',
    runAgain: 'Erneut ausführen',
    deliveryLabels: {
      local: 'Dieser Desktop',
      telegram: 'Telegram',
      discord: 'Discord',
      slack: 'Slack',
      email: 'E-Mail'
    },
    scheduleLabels: {
      daily: 'Täglich',
      weekdays: 'Werktags',
      weekly: 'Wöchentlich',
      monthly: 'Monatlich',
      hourly: 'Stündlich',
      'every-15-minutes': 'Alle 15 Minuten',
      custom: 'Benutzerdefiniert'
    },
    scheduleHints: {
      daily: 'Jeden Tag um 9:00 Uhr',
      weekdays: 'Montag bis Freitag um 9:00 Uhr',
      weekly: 'Jeden Montag um 9:00 Uhr',
      monthly: 'Am ersten Tag jedes Monats um 9:00 Uhr',
      hourly: 'Zu jeder vollen Stunde',
      'every-15-minutes': 'Alle 15 Minuten',
      custom: 'Cron-Syntax oder natürliche Sprache'
    },
    days: {
      '0': 'Sonntag',
      '1': 'Montag',
      '2': 'Dienstag',
      '3': 'Mittwoch',
      '4': 'Donnerstag',
      '5': 'Freitag',
      '6': 'Samstag',
      '7': 'Sonntag'
    },
    dayFallback: value => `Tag ${value}`,
    everyDayAt: time => `Jeden Tag um ${time}`,
    weekdaysAt: time => `Werktags um ${time}`,
    everyDayOfWeekAt: (day, time) => `Jeden ${day} um ${time}`,
    monthlyOnDayAt: (dayOfMonth, time) => `Monatlich am ${dayOfMonth}. um ${time}`,
    topOfHour: 'Zu jeder vollen Stunde',
    everyHourAt: minute => `Jede Stunde um :${minute}`,
    newCron: 'Neuer Cron',
    emptyDescNew:
      'Planen Sie einen Prompt, der nach einem Cron-Ausdruck ausgeführt wird. Hermes führt ihn aus und liefert die Ergebnisse an das von Ihnen gewählte Ziel.',
    emptyDescSearch: 'Versuchen Sie eine breitere Suchanfrage.',
    emptyTitleNew: 'Noch keine geplanten Jobs',
    emptyTitleSearch: 'Keine Treffer',
    last: 'Zuletzt:',
    next: 'Als Nächstes:',
    overdueSince: 'Überfällig seit:',
    noRuns: 'Noch keine Ausführungen',
    manage: 'Verwalten',
    showRuns: 'Ausführungen anzeigen',
    hideRuns: 'Ausführungen ausblenden',
    runHistory: 'Ausführungsverlauf',
    actionsTitle: 'Cron-Job-Aktionen',
    resume: 'Cron fortsetzen',
    pause: 'Cron pausieren',
    resumeTitle: 'Fortsetzen',
    pauseTitle: 'Pausieren',
    triggerNow: 'Jetzt auslösen',
    edit: 'Cron bearbeiten',
    deleteTitle: 'Cron-Job löschen?',
    deleteDescPrefix: 'Das entfernt ',
    deleteDescSuffix: ' dauerhaft. Er wird sofort aufhören auszulösen.',
    deleting: 'Wird gelöscht...',
    resumed: 'Cron fortgesetzt',
    paused: 'Cron pausiert',
    triggered: 'Cron ausgelöst',
    deleted: 'Cron gelöscht',
    created: 'Cron erstellt',
    updated: 'Cron aktualisiert',
    failedLoad: 'Cron-Jobs konnten nicht geladen werden',
    failedUpdate: 'Cron-Job konnte nicht aktualisiert werden',
    failedTrigger: 'Cron-Job konnte nicht ausgelöst werden',
    failedDelete: 'Cron-Job konnte nicht gelöscht werden',
    failedSave: 'Cron-Job konnte nicht gespeichert werden',
    editTitle: 'Cron-Job bearbeiten',
    createTitle: 'Neuer Cron-Job',
    editDesc: 'Aktualisieren Sie den Zeitplan, den Prompt oder das Zustellziel. Änderungen gelten beim nächsten Lauf.',
    createDesc:
      'Planen Sie einen Prompt, der automatisch ausgeführt wird. Nutzen Sie die Cron-Syntax oder einen natürlichen Ausdruck wie „every 15 minutes“.',
    nameLabel: 'Name',
    namePlaceholder: 'Morgenübersicht',
    promptLabel: 'Prompt',
    promptPlaceholder: 'Fass meine ungelesenen Slack-Kanäle zusammen und schick mir die Top 5 per E-Mail...',
    frequencyLabel: 'Häufigkeit',
    deliverLabel: 'Zustellen an',
    deliverNeedsHomeChannel: 'zuerst einen Home-Channel festlegen',
    modelLabel: 'Modell',
    modelDefault: 'Standard (globales Modell)',
    customScheduleLabel: 'Benutzerdefinierter Zeitplan',
    customPlaceholder: '0 9 * * * oder weekdays at 9am',
    customHint: 'Cron-Ausdruck oder Ausdrücke wie „every hour“ oder „weekdays at 9am“.',
    optional: 'Optional',
    promptRequired: 'Prompt ist erforderlich.',
    promptScheduleRequired: 'Prompt und Zeitplan sind erforderlich.',
    scheduleRequired: 'Zeitplan ist erforderlich.',
    scriptOnlyEditHint: 'Nur-Skript-Job (ohne AI-Prompt). Job-ID:',
    saveChanges: 'Änderungen speichern',
    createAction: 'Cron erstellen',
    tabs: {
      jobs: 'Jobs',
      blueprints: 'Blueprints'
    },
    blueprints: {
      tab: 'Blueprints',
      startFrom: 'Starten von',
      custom: 'Benutzerdefiniert',
      subtitle: 'Fertige Automatisierungen',
      dialogDesc: 'Füllen Sie die Details aus und planen Sie den Job ein.',
      scheduleIt: 'Einplanen',
      scheduling: 'Wird eingeplant...',
      scheduled: 'Blueprint eingeplant',
      loading: 'Blueprints werden geladen...',
      failedLoad: 'Blueprints konnten nicht geladen werden',
      emptyTitle: 'Keine Blueprints verfügbar',
      emptyDesc: 'Auf diesem Backend sind keine Automatisierungs-Blueprints verfügbar.'
    }
  },
  artifacts: {
    search: 'Nach Artifacts suchen...',
    refresh: 'Artifacts aktualisieren',
    refreshing: 'Artifacts werden aktualisiert',
    indexing: 'Artifacts aus den letzten Sessions werden indiziert',
    tabAll: 'Alle',
    tabImages: 'Bilder',
    tabFiles: 'Dateien',
    tabLinks: 'Links',
    noArtifactsTitle: 'Keine Artifacts gefunden',
    noArtifactsDesc: 'Generierte Bilder und Datei-Ausgaben erscheinen hier, sobald Sessions sie erstellen.',
    failedLoad: 'Artifacts konnten nicht geladen werden',
    openFailed: 'Öffnen fehlgeschlagen',
    itemsImage: 'Bilder',
    itemsLink: 'Links',
    itemsFile: 'Dateien',
    itemsGeneric: 'Elemente',
    zero: '0',
    rangeOf: (start, end, total) => `${start}-${end} von ${total}`,
    goToPage: (itemLabel, page) => `Zu ${itemLabel} Seite ${page}`,
    colTitleLink: 'Link-Titel',
    colTitleFile: 'Name',
    colTitleDefault: 'Titel / Name',
    colLocationLink: 'URL',
    colLocationFile: 'Pfad',
    colLocationDefault: 'Speicherort',
    colSession: 'Session',
    kindImage: 'Bild',
    kindFile: 'Datei',
    kindLink: 'Link',
    chat: 'Chat',
    copyUrl: 'URL kopieren',
    copyPath: 'Pfad kopieren'
  },
  artifactCard: {
    kind: {
      code: 'Code',
      html: 'Interaktive Seite',
      svg: 'Grafik'
    },
    generating: lines => `Wird generiert… ${lines} Zeilen`,
    versionBadge: count => `${count} Versionen`,
    open: 'Öffnen'
  },
  artifactPreview: {
    versionOf: (current, total) => `v${current} von ${total}`,
    olderVersion: 'Ältere Version',
    newerVersion: 'Neuere Version',
    latest: 'Neueste',
    copyContent: 'Inhalt kopieren',
    download: 'Herunterladen',
    openInBrowser: 'Im Browser öffnen',
    openInBrowserFailed: 'Konnte nicht im Browser geöffnet werden',
    missingTitle: 'Artifact nicht verfügbar',
    missingBody: 'Dieses Artifact ist nicht mehr in der lokalen Registry.'
  },
  sidebar: {
    filter: {
      grouping: 'Gruppierung',
      ordering: 'Sortierung',
      show: 'Anzeigen',
      filters: 'Filter',
      status: 'Status',
      pullRequest: 'Pull Request',
      profile: 'Profil',
      project: 'Projekt',
      archived: 'Archiviert',
      resetToDefaults: 'Auf Standard zurücksetzen',
      expandAll: 'Alle ausklappen',
      collapseAll: 'Alle einklappen',
      inboxStyle: 'Posteingangsstil',
      updated: 'Aktualisiert',
      created: 'Erstellt',
      tokens: 'Tokens',
      cost: 'Kosten',
      manual: 'Manuell',
      preview: 'Vorschau',
      pr: 'PR',
      needsInput: 'Eingabe nötig',
      working: 'In Arbeit',
      unread: 'Ungelesen',
      draft: 'Entwurf',
      idle: 'Inaktiv',
      open: 'Offen',
      merged: 'Gemergt',
      closed: 'Geschlossen',
      noPR: 'Kein PR'
    },
    gatewayGroups: {
      grouping: 'Gateway & Profil',
      rename: 'Gruppe umbenennen',
      aliasLabel: 'Anzeigename',
      aliasHint: 'Nur der Anzeigename; Gateway- und Profilname bleiben unverändert.',
      resetName: 'Name zurücksetzen',
      moveUp: 'Nach oben',
      moveDown: 'Nach unten',
      reorder: 'Gruppe neu anordnen',
      actions: 'Gruppenaktionen'
    },
    profileRail: 'Profil-Leiste',
    nav: {
      'new-session': 'Neue Session',
      capabilities: 'Fähigkeiten',
      messaging: 'Messaging',
      artifacts: 'Artefakte',
      cron: 'Geplante Jobs'
    },
    searchAria: 'Sessions durchsuchen',
    searchPlaceholder: 'Sessions durchsuchen…',
    clearSearch: 'Suche löschen',
    noMatch: query => `Keine Sessions passen zu “${query}”.`,
    results: 'Ergebnisse',
    pinned: 'Angepinnt',
    sessions: 'Sessions',
    terminal: 'Terminal',
    files: 'Dateien',
    review: 'Review',
    logs: 'Logs',
    cronJobs: 'Cron-Jobs',
    groupAriaGrouped: 'Sessions als einzelne Liste anzeigen',
    groupAriaUngrouped: 'Sessions nach Workspace gruppieren',
    showProjects: 'Projekte anzeigen',
    showSessions: 'Sessions anzeigen',
    groupTitleGrouped: 'Sessions aufteilen',
    groupTitleUngrouped: 'Nach Workspace gruppieren',
    allPinned: 'Alles hier ist angeheftet. Lösen Sie einen Chat, damit er unter den letzten angezeigt wird.',
    shiftClickHint: 'Shift-Klick auf einen Chat zum Anpinnen',
    noWorkspace: 'Kein Workspace',
    projectEmpty: 'Noch keine Sessions',
    projectLoadFailed: 'Sessions konnten nicht geladen werden',
    noSessions: 'Noch keine Sessions',
    storageCorrupt: {
      title: 'Die Session-Datenbank ist beschädigt',
      body: (profiles: string) =>
        `Hermes kann nicht den gesamten Session-Verlauf für ${profiles} lesen. Chats, die in dieser Liste fehlen, wurden nicht gelöscht; die Datei, in der sie gespeichert sind, ist beschädigt.`,
      action:
        'Beenden Sie Hermes für dieses Profil und prüfen Sie die Datei dann, ohne sie zu ändern, oder stellen Sie einen Snapshot wieder her:',
      guide: 'Wiederherstellungsanleitung'
    },
    noFilterMatches: 'Keine Sessions passen zu diesen Filtern',
    projects: {
      showAllSessions: 'Alle Sessions anzeigen',
      sectionLabel: 'Projekte',
      home: 'Start',
      autoDiscovered: 'Automatisch erkannt',
      newButton: 'Neues Projekt',
      createTitle: 'Neues Projekt',
      createDesc: 'Geben Sie dem Workspace einen Namen und fügen Sie einen oder mehrere Ordner hinzu.',
      renameTitle: 'Projekt umbenennen',
      addFolderTitle: 'Ordner hinzufügen',
      namePlaceholder: 'z. B. Skunkworks',
      foldersLabel: 'Ordner',
      ideaLabel: 'Idee',
      ideaPlaceholder: 'Worum geht es in diesem Projekt? (gespeichert in IDEA.md)',
      ideaGenerate: 'Idee generieren',
      ideaGenerating: 'Wird generiert…',
      ideaShuffle: 'Vorlagen mischen',
      noFolders: 'Noch keine Ordner hinzugefügt.',
      addFolder: 'Ordner hinzufügen',
      primaryBadge: 'primär',
      removeFolder: 'Entfernen',
      create: 'Erstellen',
      menu: 'Aktionen',
      menuRename: 'Umbenennen…',
      menuAppearance: 'Aussehen',
      noColor: 'Keine Farbe',
      menuAddFolder: 'Ordner hinzufügen',
      menuSetActive: 'Als aktiv festlegen',
      menuDelete: 'Löschen',
      moveToProject: 'In Projekt verschieben',
      movedTo: name => `In ${name} verschoben`,
      moveFailed: 'Session konnte nicht verschoben werden',
      moveNoFolder: 'Dieses Projekt hat keinen Ordner, in den verschoben werden kann',
      moveNoProjects: 'Keine anderen Projekte',
      reveal: 'Im Ordner anzeigen',
      copyPath: 'Pfad kopieren',
      removeFromSidebar: 'Aus der Sidebar ausblenden',
      createFailed: 'Projekt konnte nicht erstellt werden',
      staleBackend:
        'Aktualisieren Sie das Hermes-Backend, um Projekte zu erstellen – Ihr Backend ist älter als diese Desktop-App (Einstellungen → Updates → Backend).',
      deleteConfirm:
        'Das entfernt das gespeicherte Projekt aus Hermes. Dateien, Git-Repos und Worktrees bleiben unberührt.',
      startWork: 'Neuer Worktree',
      newWorktreeTitle: 'Neuer Worktree',
      newWorktreeDesc: 'Benennen Sie den Branch für diesen Worktree.',
      branchPlaceholder: 'z. B. my-feature',
      branchOff: () => ({ after: '', before: 'abzweigen von ' }),
      baseBranchPlaceholder: 'Branches durchsuchen…',
      baseBranchNone: 'Keine Branches gefunden',
      startWorkFailed: 'Worktree konnte nicht erstellt werden',
      worktreeStaleBackend:
        'Aktualisieren Sie das Hermes-Backend, um Worktrees über diese Remote-Verbindung zu erstellen – es ist älter als die Git-Worktree-API.',
      worktreeProjectLabel: 'Projekt',
      worktreeProjectPlaceholder: 'Projekte durchsuchen…',
      worktreeProjectNone: 'Keine Projekte mit Ordner',
      convertBranch: 'Einen Branch konvertieren…',
      convertBranchTitle: 'Einen Branch konvertieren',
      convertBranchDesc: 'Ausgecheckte Branches öffnen oder einen Worktree für einen freien Branch erstellen.',
      convertBranchPlaceholder: 'Branches durchsuchen…',
      convertBranchInstead: 'Einen bestehenden Branch konvertieren',
      branchOpenExisting: 'öffnen',
      branchSwitchHome: 'Start wechseln',
      branchCreateWorktree: 'neuer Worktree',
      branchTrackRemote: 'Remote verfolgen',
      branchesLoading: 'Branches werden geladen…',
      noBranches: 'Keine Branches gefunden',
      removeWorktree: 'Worktree entfernen',
      removeWorktreeFailed: 'Worktree konnte nicht entfernt werden (nicht committete Änderungen?)',
      removeWorktreeConfirm:
        'Aus Git entfernen (löscht das Worktree-Verzeichnis; der Branch bleibt), oder einfach die Lane aus der Sidebar ausblenden und den Worktree auf der Festplatte belassen.',
      removeWorktreeDirty:
        'Dieser Worktree hat nicht committete Änderungen. Kraftvoll entfernen (verwirft diese Änderungen), oder einfach die Lane ausblenden und ihn auf der Festplatte behalten.',
      forceRemove: 'Kraftvoll entfernen',
      enter: label => `${label} öffnen`,
      reorder: label => `${label} neu anordnen`,
      toggle: (label, open) => `${open ? 'Anzeigen' : 'Ausblenden'} ${label} Sessions`,
      showAllCount: (count: number) => `Alle ${count} Sessions anzeigen`,
      back: 'Alle Projekte'
    },
    newSessionIn: label => `Neue Session in ${label}`,
    showMoreIn: (count, label) => `Noch ${count} mehr in ${label} anzeigen`,
    loading: 'Wird geladen…',
    loadMore: 'Mehr laden',
    loadCount: step => `${step} mehr laden`,
    messageCount: count => `${count} ${count === 1 ? 'Nachricht' : 'Nachrichten'}`,
    toolCallCount: count => `${count} ${count === 1 ? 'Tool-Aufruf' : 'Tool-Aufrufe'}`,
    row: {
      pin: 'Anpinnen',
      unpin: 'Lösen',
      markUnread: 'Als ungelesen markieren',
      markRead: 'Als gelesen markieren',
      unreadFailed: 'Ungelesen-Status konnte nicht aktualisiert werden',
      copyId: 'ID kopieren',
      export: 'Exportieren',
      branchFrom: 'Branch',
      rename: 'Umbenennen…',
      archive: 'Archivieren',
      unarchive: 'Archivierung aufheben',
      newWindow: 'Neues Fenster',
      openInTerminal: 'Im Terminal öffnen',
      hideTabBar: 'Tab-Leiste ausblenden',
      openInNewTab: 'In neuem Tab öffnen',
      openInSplit: 'Im Split öffnen',
      copyIdFailed: 'Session-ID konnte nicht kopiert werden',
      sessionActions: 'Session-Aktionen',
      sessionRunning: 'Session läuft',
      needsInput: 'Braucht Ihre Eingabe',
      waitingForAnswer: 'Wartet auf Ihre Antwort',
      finishedUnread: 'Abgeschlossen – ungelesen',
      backgroundRunning: 'Hintergrundaufgabe läuft',
      draftSession: 'Entwurf – noch nichts gesendet',
      handoffOrigin: platform => `Übergeben von ${platform}`,
      ownedByProfile: profile => `Profil: ${profile}`,
      renamed: 'Umbenannt',
      renameFailed: 'Umbenennen fehlgeschlagen',
      renameTitle: 'Session umbenennen',
      renameDesc: 'Leer lassen, um zu löschen.',
      untitledPlaceholder: 'Unbenannte Session',
      deleteTitle: 'Session löschen?',
      deleteDesc: title => `Das löscht “${title}” dauerhaft. Das kann nicht rückgängig gemacht werden.`,
      deleting: 'Wird gelöscht…',
      deleted: 'Session gelöscht',
      untitledChat: id => `Chat ${id}`,
      messageCount: count => `${count} ${count === 1 ? 'Nachricht' : 'Nachrichten'}`,
      todoProgress: 'Aufgaben abgeschlossen',
      ageNow: 'jetzt',
      ageDay: 'T',
      ageHour: 'h',
      ageMin: 'Min'
    },
    dateDivider: {
      today: 'Heute früher',
      yesterday: 'Gestern',
      thisWeek: 'Diese Woche früher',
      lastWeek: 'Letzte Woche',
      thisMonth: 'Diesen Monat früher'
    },
    statusDivider: {
      working: 'In Arbeit',
      done: 'Erledigt'
    },
    markAllRead: 'Alle als gelesen markieren'
  },
  composer: {
    message: 'Nachricht',
    wakingProfile: profile => `Wecke ${profile}…`,
    placeholderStarting: 'Hermes wird gestartet…',
    placeholderReconnecting: 'Verbindung zu Hermes wird wiederhergestellt…',
    placeholderFollowUp: 'Folge senden',
    newSessionPlaceholders: [
      'Was bauen wir?',
      'Geben Sie Hermes eine Aufgabe',
      'Was ist Ihnen wichtig?',
      'Beschreiben Sie, was Sie brauchen',
      'Was sollen wir angehen?',
      'Fragen Sie irgendetwas',
      'Beginnen Sie mit einem Ziel'
    ],
    followUpPlaceholders: [
      'Folge senden',
      'Mehr Kontext hinzufügen',
      'Anfrage verfeinern',
      'Was kommt als Nächstes?',
      'Weiter so',
      'Noch weiter',
      'Anpassen oder fortfahren'
    ],
    startVoice: 'Sprachkonversation starten',
    openDirective: 'Öffnen',
    queueMessage: 'Nachricht einreihen',
    steer: 'Laufenden Lauf steuern',
    stop: 'Stopp',
    send: 'Senden',
    speaking: 'Spricht',
    transcribing: 'Transkribiert',
    thinking: 'Denkt',
    muted: 'Stummgeschaltet',
    listening: 'Hört zu',
    muteMic: 'Mikrofon stummschalten',
    unmuteMic: 'Mikrofon einschalten',
    stopListening: 'Zuhören stoppen und senden',
    stopShort: 'Stopp',
    endConversation: 'Sprachkonversation beenden',
    endShort: 'Ende',
    stopDictation: 'Diktat stoppen',
    transcribingDictation: 'Transkribiert Diktat',
    voiceControls: 'Sprache',
    voiceEngine: 'Sprachchat-Engine',
    voiceEngineChained: 'Sprache-zu-Text + Hermes-Stimme',
    voiceEngineLive: 'GPT-Live (Vollduplex, delegiert an Hermes)',
    voiceEngineLiveNeedsKey: 'Benötigt einen OpenAI-API-Schlüssel',
    voiceEngineChangeFailed: 'Sprachchat-Engine konnte nicht geändert werden',
    voiceEngineChainedShort: 'Sprache-zu-Text',
    voiceEngineLiveShort: 'GPT-Live',
    voiceDictation: 'Sprachdiktat',
    speakReplies: 'Antworten vorlesen',
    stopSpeakingReplies: 'Antworten nicht mehr vorlesen',
    wakeWord: (phrase: string) => `Aktivierungswort „${phrase}“`,
    wakeWordListening: phrase => `Aufwachwort: „${phrase}“ — hört zu`,
    wakeWordOff: phrase => `Aufwachwort: „${phrase}“ — aus`,
    wakeWordPausedVoice: phrase => `Aufwachwort: „${phrase}“ — während Sprachchat pausiert`,
    lookupLoading: 'Sucht…',
    lookupNoMatches: 'Keine Treffer.',
    lookupTry: 'Versuchen Sie',
    lookupOr: 'oder',
    commonCommands: 'Häufige Befehle',
    hotkeys: 'Tastenkürzel',
    helpFooter: 'öffnet das volle Panel · Backspace schließt',
    commandDescs: {
      '/help': 'Desktop-Slash-Befehle anzeigen',
      '/clear': 'neue Session starten',
      '/resume': 'Gespeicherte Session fortsetzen',
      '/details': 'Transkript-Detailgrad steuern',
      '/copy': 'Auswahl oder letzte Assistenten-Nachricht kopieren',
      '/quit': 'hermes beenden',
      '/start': 'Start-Pings der Plattform ohne Antwort bestätigen',
      '/new': 'Neuen Desktop-Chat starten',
      '/topic': 'Telegram-DM-Themen-Sessions aktivieren oder prüfen',
      '/save': 'Aktuelles Transkript als JSON speichern',
      '/retry': 'Letzte Nachricht wiederholen (erneut an den Agent senden)',
      '/prompt': 'Nächsten Prompt in $EDITOR (Markdown) verfassen und dann senden',
      '/undo': 'N Benutzer-Turns zurückgehen und neu prompten (Standard 1)',
      '/title': 'Aktuelle Session umbenennen',
      '/handoff': 'Diese Session an eine Messaging-Plattform übergeben',
      '/branch': 'Letzte Nachricht in einen neuen Chat abzweigen',
      '/worktree': 'Isolierte Git-Worktrees anzeigen, auflisten, erstellen oder bereinigen',
      '/compress': 'Kontext dieser Unterhaltung komprimieren',
      '/rollback':
        'Dateisystem-Checkpoints auflisten oder wiederherstellen (Wiederherstellungen behalten Ihre manuellen Änderungen; --all überschreibt sie)',
      '/export': 'Ein Profil (Konfiguration, Skills, Theme) als teilbares Archiv exportieren',
      '/import': 'Ein geteiltes Profilarchiv als neues Profil importieren',
      '/stop': 'Aktiven Turn und Hintergrundprozesse stoppen',
      '/pause': "Neue Arbeit global pausieren (Notstopp); '/pause off' setzt fort",
      '/bg': 'Einen Prompt in einer separaten Hintergrund-Session ausführen',
      '/btw': 'Eine Nebenfrage zu dieser Unterhaltung stellen, ohne sie zu unterbrechen',
      '/agents': 'Aktive Agents und laufende Aufgaben anzeigen',
      '/journey': 'Den Gedächtnisgraphen öffnen – Skills und Erinnerungen im Zeitverlauf',
      '/queue':
        'Einen Prompt für den nächsten Turn einreihen oder eingereihte Prompts auflisten/bearbeiten/entfernen/verschieben/leeren',
      '/steer': 'Nach dem nächsten Tool-Aufruf eine Nachricht einfügen, ohne zu unterbrechen',
      '/goal': 'Ein dauerhaftes Ziel festlegen, an dem Hermes über mehrere Turns arbeitet, bis es erreicht ist',
      '/heartbeat': 'Einen wiederkehrenden Prompt festlegen, der bei Leerlauf in diese Session zurückkehrt',
      '/refine': 'Diese Unterhaltung jetzt prüfen und Erkenntnisse in Gedächtnis/Skills speichern',
      '/review': 'Einen unabhängigen Subagent starten, der die gerade besprochene Arbeit prüft (PR, Code, Doku)',
      '/loop': 'Einen Prompt in dieser Session in regelmäßigen Abständen erneut ausführen',
      '/plan': 'Einen Markdown-Umsetzungsplan in .hermes/plans/ schreiben, ohne etwas auszuführen',
      '/moa': 'Einen Prompt mit der Standard-Mixture-of-Agents-Vorlage ausführen und dann Ihr Modell wiederherstellen',
      '/subgoal': 'Zusätzliche Kriterien zum aktiven Ziel hinzufügen oder verwalten',
      '/status': 'Status der aktuellen Session anzeigen',
      '/egress': 'Status des Docker-Egress-Proxys anzeigen',
      '/context':
        'Detaillierte Ansicht des Kontextfensters mit Nutzungsanzeige, Aufschlüsselung nach Kategorie, Komprimierungsstatistik und Durchsatz anzeigen',
      '/whoami': 'Ihren Zugriff auf Slash-Befehle anzeigen (Admin / Benutzer)',
      '/profile': 'Aktives Hermes-Profil wechseln',
      '/codex-runtime': 'Codex-App-Server-Runtime für OpenAI/Codex-Modelle umschalten',
      '/personality': 'Eine vordefinierte Persönlichkeit festlegen',
      '/battery': 'Farbcodierte Akkuanzeige in der Statusleiste umschalten',
      '/timestamps': '[HH:MM]-Zeitstempel bei Nachrichten und /history umschalten',
      '/diff': 'Git-Änderungen im Arbeitsverzeichnis anzeigen',
      '/focus': 'Fokusansicht umschalten – nur Ihren Prompt und die finale Antwort anzeigen',
      '/yolo': 'YOLO umschalten – gefährliche Befehle automatisch genehmigen',
      '/approvals': 'Dauerhaften Genehmigungsmodus für gefährliche Befehle anzeigen oder festlegen',
      '/reasoning': 'Reasoning-Aufwand oder -Anzeige [<level> [--global]|show|hide|full|clamp]',
      '/skin': 'Desktop-Theme wechseln oder zum nächsten weiterschalten',
      '/wake': 'Desktop-Aktivierungswort-Listener steuern [on|off|status]',
      '/tools': 'Tools verwalten: /tools [list|disable|enable] [name...]',
      '/memory': 'Ausstehende Gedächtnis-Schreibvorgänge prüfen / Genehmigungssperre umschalten',
      '/bundles': 'Skill-Bundles auflisten (Aliasse /<name> für mehrere Skills)',
      '/pet': 'Ein Petdex-Maskottchen umschalten oder adoptieren (/pet, /pet list, /pet boba)',
      '/hatch': 'Ein neues Haustier erzeugen (öffnet den Generator)',
      '/learn':
        'Einen wiederverwendbaren Skill aus allem lernen, was Sie beschreiben (Ordner, URLs, dieser Chat, Notizen)',
      '/init': 'AGENTS.md-Projektanweisungen aus einem Repo-Scan erzeugen oder aktualisieren',
      '/suggestions': 'Vorgeschlagene Automatisierungen prüfen (annehmen/verwerfen)',
      '/blueprint': 'Eine Automatisierung aus einer Blueprint-Vorlage einrichten',
      '/browser': 'Browser-CDP-Verbindung verwalten [connect|disconnect|status] (nur lokales Gateway)',
      '/palette': 'Die unscharfe Befehlspalette öffnen (auch Strg+P)',
      '/usage': 'Token-Nutzung und Ratenlimits anzeigen; `reset` löst ein angespartes Codex-Limit-Reset ein',
      '/subscription': 'Ihren Nous-Tarif ansehen und im Browser ändern',
      '/topup': 'Ihr Nous-Guthaben anzeigen und die Abrechnung im Portal verwalten',
      '/platform': 'Eine fehlerhafte Gateway-Plattform pausieren, fortsetzen oder auflisten',
      '/version': 'Hermes-Agent-Version anzeigen',
      '/debug': 'Debug-Bericht (Systeminfos + Logs) hochladen und teilbare Links erhalten',
      '/model': 'Modell für diese Session wechseln'
    },
    hotkeyDescs: {
      'composer.mention': 'Dateien, Ordner, URLs, git referenzieren',
      'composer.slash': 'Slash-Befehlspalette',
      'composer.help': 'diese Schnellhilfe (Löschen zum Schließen)',
      'composer.sendNewline': 'senden · Shift+Enter für neue Zeile',
      'composer.sendQueued': 'nächsten eingereihten Turn senden',
      'keybinds.openPanel': 'alle Tastaturkürzel',
      'composer.cancel': 'Popover schließen · Lauf abbrechen',
      'composer.history': 'Popover / Verlauf durchblättern'
    },
    attachUrlTitle: 'URL anhängen',
    attachUrlDesc: 'Hermes ruft die Seite ab und fügt sie als Kontext für diesen Turn hinzu.',
    urlPlaceholder: 'https://example.com/post',
    urlHintPre: 'Geben Sie die vollständige URL an, z. B. ',
    attach: 'Anhängen',
    queued: count => `${count} eingereiht`,
    queuedPaused: count => `${count} eingereiht — pausiert`,
    attachmentOnly: 'Nur-Anhang-Turn',
    emptyTurn: 'Leerer Turn',
    hiddenQueued: 'Einrichtungshinweis',
    attachments: count => `${count} Anhang${count === 1 ? '' : 'e'}`,
    editingInComposer: 'Bearbeitet im Composer',
    editingQueuedInComposer: 'Bearbeitet eingereihten Turn im Composer',
    restoredDraftNotice: 'Ihre nicht gesendete Nachricht wurde wiederhergestellt',
    restoredDraftUndo: 'Rückgängig',
    queueEdit: 'Bearbeiten',
    queueSendNext: 'Weiter',
    queueSteer: 'Steuern — laufenden Turn jetzt umleiten',
    queueSend: 'Senden',
    queueDelete: 'Löschen',
    queueResume: 'Fortsetzen',
    queueResumeTip: 'Durch Stopp pausiert — fortsetzen, um die eingereihten Turns zu senden',
    queueStuckTitle: 'Eingereihte Nachricht nicht gesendet',
    queueStuckBody:
      'Ein eingereihter Turn konnte nicht gesendet werden. Er ist noch in der Warteschlange — versuchen Sie, ihn erneut zu senden.',
    previewUnavailable: 'Vorschau nicht verfügbar',
    previewLabel: label => `Vorschau ${label}`,
    couldNotPreview: label => `Vorschau von ${label} fehlgeschlagen`,
    removeAttachment: label => `${label} entfernen`,
    dictating: 'Diktiert',
    preparingAudio: 'Bereitet Audio vor',
    speakingResponse: 'Spricht Antwort',
    readingAloud: 'Liest vor',
    themeSuggestions: 'Desktop-Theme-Vorschläge',
    noMatchingThemes: 'Keine passenden Themes.',
    themeTryPre: 'Versuchen Sie ',
    themeTryPost: '.',
    attachLabel: 'Anhängen',
    files: 'Dateien…',
    folder: 'Ordner…',
    images: 'Bilder…',
    pasteImage: 'Bild einfügen',
    url: 'URL…',
    promptSnippets: 'Prompt-Schnipsel…',
    tipPre: 'Tipp: Geben Sie ',
    tipPost: ' ein, um Dateien inline zu referenzieren.',
    snippetsTitle: 'Prompt-Schnipsel',
    snippetsDesc: 'Wählen Sie einen Start-Prompt, um ihn in den Composer einzufügen.',
    dropFiles: 'Dateien zum Anhängen ablegen',
    dropSession: 'Ablegen, um diesen Chat zu verlinken',
    mcpSuggestions: {
      label: server => `${server} hinzufügen`,
      tip: keyword => `Vorgeschlagen, weil Sie „${keyword}“ erwähnt haben – zum Verbinden klicken`,
      connecting: server => `${server} wird verbunden…`,
      cancelTip: 'Klicken zum Abbrechen',
      added: server => `${server} hinzugefügt`,
      addedTip: 'Verbunden — seine Tools sind in diesem Chat bereit',
      connectFailed: server => `${server} konnte nicht verbunden werden`
    },
    skillSuggestions: {
      label: skill => `Fähigkeit verwenden: ${skill}`,
      tip: skill => `Sie haben „${skill}“ erwähnt – klicken, um mit diesem Skill zu beginnen`,
      done: skill => `/skill hinzugefügt: ${skill}`,
      doneTip: 'Die Fähigkeit wird beim Senden geladen'
    },
    githubSuggestions: {
      label: 'GitHub einrichten',
      tip: 'GitHub funktioniert hier über die gh-CLI-Skills – klicken, um Ihr Konto zu verbinden',
      done: '/github-auth hinzugefügt',
      doneTip: 'Senden Sie die Nachricht, und der Agent führt Sie durch die GitHub-Anmeldung'
    },
    repairSuggestions: {
      label: server => `${server} erneut verbinden`,
      tip: server => `Ein ${server}-Aufruf ist gerade mit einem Verbindungsfehler fehlgeschlagen`,
      working: server => `${server} wird erneut verbunden…`,
      workingTip: 'Klicken zum Abbrechen',
      done: server => `${server} erneut verbunden`,
      doneTip: 'Frische Anmeldedaten sind in diesem Chat aktiv',
      failed: server => `${server} konnte nicht erneut verbunden werden`
    },
    cronSuggestions: {
      label: 'Dies planen',
      tip: phrase => `„${phrase}“ klingt wiederkehrend — führen Sie es stattdessen nach Zeitplan aus`,
      prefix: 'Als geplanten Job einrichten:',
      done: 'Für Planung markiert',
      doneTip: 'Senden Sie die Nachricht, und der Agent erstellt den Job'
    },
    snippets: {
      codeReview: {
        label: 'Code-Review',
        description: 'Prüft die aktuelle Änderung auf Regressionen, übersehene Randfälle und fehlende Tests.',
        text: 'Bitte prüfe dies auf Bugs, Regressionen und fehlende Tests.'
      },
      implementationPlan: {
        label: 'Implementierungsplan',
        description: 'Skizziert einen Ansatz, bevor Code angefasst wird, damit der Diff fokussiert bleibt.',
        text: 'Bitte erstelle einen prägnanten Implementierungsplan, bevor du Code änderst.'
      },
      explainThis: {
        label: 'Erkläre dies',
        description: 'Erklärt, wie der ausgewählte Code funktioniert, und verlinkt die wichtigsten Dateien.',
        text: 'Bitte erkläre, wie das funktioniert, und zeige mir die Schlüsseldateien.'
      }
    }
  },
  statusStack: {
    hideStack: 'Statusstapel ausblenden',
    showStack: 'Statusstapel anzeigen',
    agents: 'Agents',
    background: count => `${count} Hintergrund`,
    goalActive: 'Ziel aktiv',
    goalBlocked: 'Ziel blockiert',
    goalDone: 'Ziel erledigt',
    goalPaused: 'Ziel pausiert',
    goalWaiting: 'Ziel wartet',
    subagents: count => `${count} Subagent${count === 1 ? '' : 'en'}`,
    todos: (done, total) => `Aufgaben ${done}/${total}`,
    running: 'Läuft',
    stop: 'Stopp',
    dismiss: 'Verwerfen',
    exit: code => `exit ${code}`,
    control: {
      goalActiveTurns: (turn, maxTurns) => `Runde ${turn}/${maxTurns}`,
      goalDoneTurns: turns => `${turns} Runde${turns === 1 ? '' : 'n'}`,
      goalTurn: turn => `Runde ${turn}`,
      goalActions: 'Ziel-Aktionen',
      viewDetails: 'Details ansehen',
      addCriterion: 'Kriterium hinzufügen',
      addCriterionDialogTitle: 'Kriterium hinzufügen',
      addCriterionPlaceholder: 'Kriterien-Text eingeben …',
      criterionLabel: 'Kriterium',
      pauseGoal: 'Ziel pausieren',
      resumeGoal: 'Ziel fortsetzen',
      resumeNow: 'Jetzt fortsetzen',
      clearGoal: 'Ziel löschen',
      clearGoalConfirmTitle: 'Ziel löschen?',
      clearGoalConfirmBody: 'Möchten Sie das aktive Ziel wirklich löschen? Das kann nicht rückgängig gemacht werden.',
      copyCriterion: index => `Kriterium ${index} kopieren`,
      removeCriterion: index => `Kriterium ${index} entfernen`,
      removeCriterionConfirmTitle: index => `Kriterium ${index} entfernen?`,
      removeCriterionConfirmBody: index => `Möchten Sie Kriterium ${index} wirklich entfernen?`,
      clearCriteria: 'Alle Kriterien löschen',
      clearCriteriaConfirmTitle: 'Alle Kriterien löschen?',
      clearCriteriaConfirmBody: 'Möchten Sie wirklich alle Kriterien von diesem Ziel entfernen?',
      criteriaHeader: count => `Kriterien · ${count}`,
      noCriteria: 'Keine Kriterien',
      goalDetailsTitle: 'Ziel-Details',
      objectiveLabel: 'Zielvorgabe',
      contractOutcome: 'Ergebnis',
      contractVerification: 'Verifikation',
      contractConstraints: 'Einschränkungen',
      contractBoundaries: 'Grenzen',
      contractStopWhen: 'Stopp, wenn',
      waitBarrierTitle: 'Wartebedingung',
      waitUntil: target => `Warten bis ${target}`,
      waitSession: target => `Warten auf Session ${target}`,
      waitPid: pid => `Warten auf Prozess ${pid}`,
      qualityGatesTitle: 'Quality Gates',
      gateCommand: 'Befehl',
      gateAttempts: (attempts, max) => `${attempts}/${max} Versuche`,
      gateTimeout: seconds => `${seconds}s Timeout`,
      gateLastExit: code => (code === null ? 'Ausstehend' : `Exit-Code: ${code}`),
      loopActive: 'Loop aktiv',
      loopPaused: 'Loop pausiert',
      loopDeferred: 'Loop aufgeschoben',
      loopFinished: 'Loop beendet',
      loopRuns: runs => `${runs} Lauf${runs === 1 ? '' : 'läufe'}`,
      loopRunCount: (current, total) => `Lauf ${current}/${total}`,
      loopNext: time => `nächstes ${time}`,
      loopEverySeconds: seconds => `alle ${seconds}s`,
      loopEveryMinutes: minutes => `alle ${minutes}m`,
      loopEveryHours: hours => `alle ${hours}h`,
      loopSelfPaced: 'eigenes Tempo',
      loopActions: 'Loop-Aktionen',
      pauseLoop: 'Loop pausieren',
      resumeLoop: 'Loop fortsetzen',
      stopLoop: 'Loop stoppen',
      stopLoopConfirmTitle: 'Loop stoppen?',
      stopLoopConfirmBody: 'Möchten Sie diesen Loop wirklich stoppen?',
      dismissLoop: 'Loop verwerfen',
      loopPromptLabel: 'Prompt',
      loopCadenceLabel: 'Takt',
      loopUntilLabel: 'Bis-Bedingung',
      loopDeferredNotice: 'Ein aktives Ziel steuert derzeit die Session.',
      loopAwaitingResponse: 'Warten auf Antwort',
      heartbeatActive: 'Heartbeat aktiv',
      heartbeatPaused: 'Heartbeat pausiert',
      heartbeatEveryMinutes: minutes => `alle ${minutes}m`,
      heartbeatEveryHours: hours => `alle ${hours}h`,
      heartbeatEverySeconds: seconds => `alle ${seconds}s`,
      heartbeatNext: time => `nächstes ${time}`,
      heartbeatDueWaitingForIdle: 'fällig — wartet auf Ruhezustand',
      heartbeatActions: 'Heartbeat-Aktionen',
      pauseHeartbeat: 'Heartbeat pausieren',
      resumeHeartbeat: 'Heartbeat fortsetzen',
      clearHeartbeat: 'Heartbeat löschen',
      clearHeartbeatConfirmTitle: 'Heartbeat löschen?',
      clearHeartbeatConfirmBody: 'Möchten Sie diesen Heartbeat wirklich löschen?',
      heartbeatFiredCount: count => `${count} Mal ausgelöst${count === 1 ? '' : ''}`,
      actionFailed: msg => `Aktion fehlgeschlagen: ${msg}`,
      actionSucceeded: 'Aktion erfolgreich',
      copySuccess: 'Kriterium in die Zwischenablage kopiert',
      copyFailure: 'Kriterium konnte nicht in die Zwischenablage kopiert werden',
      continuationFailed: 'Ziel-Fortsetzung konnte nicht übermittelt werden',
      continuationQueued: 'Ziel fortgesetzt — Fortsetzung in der Warteschlange, bis die aktuelle Runde endet',
      continuationBusy: 'Ziel fortgesetzt — Session ist beschäftigt; stoppe zuerst die aktuelle Antwort (Stopp-Button oder Esc), um fortzufahren',
      controlUnavailable: msg => `Session-Steuerung nicht verfügbar: ${msg}`,
      dismissError: 'Fehler verwerfen',
      add: 'Hinzufügen'
    },
    coding: {
      title: 'Arbeitsverzeichnis',
      noBranch: 'Kein Branch',
      detached: 'losgelöst',
      clean: 'Sauber',
      changed: count => `${count} geändert`,
      ahead: count => `${count} voraus`,
      behind: count => `${count} zurück`,
      review: 'Überprüfen',
      close: 'Schließen',
      openChanges: 'Änderungen öffnen',
      openFile: 'Datei öffnen',
      stage: 'Stagen',
      unstage: 'Unstagen',
      stageAll: 'Alles stagen',
      viewAsTree: 'Als Baum ansehen',
      viewAsList: 'Als Liste ansehen',
      revert: 'Zurücksetzen',
      revertAll: 'Alles zurücksetzen',
      revertConfirm:
        'Änderungen an dieser Datei verwerfen und sie in den committeten Zustand zurücksetzen? Das kann nicht rückgängig gemacht werden.',
      revertAllConfirm:
        'Alle Änderungen verwerfen und alle Dateien in den committeten Zustand zurücksetzen? Das kann nicht rückgängig gemacht werden.',
      staged: 'Gestaged',
      noChanges: 'Keine Änderungen',
      notRepo: 'Kein git-Repository',
      noDiff: 'Kein Diff zum Anzeigen',
      scopeUncommitted: 'Nicht committet',
      scopeBranch: 'Branch',
      scopeLastTurn: 'Letzte Runde',
      commit: 'Commit',
      commitAndPush: 'Commit & Push',
      commitPlaceholder: shortcut => `Nachricht (${shortcut} zum Committen)`,
      generateCommitMessage: 'Commit-Message generieren',
      stopGenerating: 'Generieren stoppen',
      createPr: 'PR erstellen',
      openPr: 'PR öffnen',
      ghMissing: 'Installieren Sie die GitHub CLI (gh) und melden Sie sich an, um PRs zu öffnen',
      agentShip: 'Hermes bitten, einen PR zu öffnen',
      agentShipUnavailable: 'Der Chat, der diese Änderungen besitzt, ist nicht auf dem Bildschirm.',
      agentShipPrompt:
        'Überprüfe die aktuellen Änderungen, committe sie mit einer klaren Conventional-Commit-Message, pushe den Branch und öffne einen Pull Request.',
      newBranch: 'Neuer Branch',
      branchOffFrom: base => `Neuer Branch von ${base}`,
      switchTo: branch => `Zu ${branch} wechseln`,
      switchFailed: branch => `Zu ${branch} konnte nicht gewechselt werden`,
      worktrees: 'Worktrees'
    }
  },
  updates: {
    discontinuedTitle: 'Dieser Hermes-Build wird nicht mehr unterstützt',
    discontinuedBody: 'Dieser Hermes-Build wird nicht mehr unterstützt und funktioniert möglicherweise nicht mehr — deinstallieren Sie ihn. Ihre Daten bleiben auf dem Datenträger.',
    channels: { stable: 'Stabil', canary: 'Canary' },
    appName: 'Hermes',
    availableBodyRelease: tag => `Version ${tag} ist bereit zur Installation.`,
    releaseAvailable: tag => `Version ${tag} ist verfügbar.`,
    checkingShort: 'Wird geprüft…',
    availableBodyAppInstaller: 'Eine neue Hermes-Version ist bereit. Hermes wird geschlossen, Windows schließt das Update ab und Hermes startet automatisch neu.',
    applyingBodyAppInstaller: 'Hermes wird geschlossen und Windows schließt das Update ab. Danach startet Hermes automatisch neu.',
    applyingCloseAppInstaller: 'Dieses Fenster schließt sich, Windows schließt das Update ab und Hermes startet automatisch neu.',
    checkUnknownTitleAppInstaller: 'Update-Check fehlgeschlagen',
    checkUnknownBodyAppInstaller: 'Windows konnte gerade nicht nach Updates suchen. Updates werden auch beim Neustart von Hermes automatisch installiert.',
    versionDetailsTitle: 'Versionsdetails',
    versionDetailsBody: 'Diese Installation wird außerhalb der App verwaltet. Aktualisieren Sie sie auf dieselbe Weise, wie Sie sie installiert haben.',
    versionDetailsVersion: 'Version',
    versionDetailsCommit: 'Commit',
    versionDetailsBuildOrigin: 'Build-Ursprung',
    versionDetailsDistribution: 'Distribution',
    versionDetailsDistributionDesktop: 'Desktop-App',
    versionDetailsDistributionDesktopMsix: 'Desktop-App (MSIX)',
    versionDetailsDistributionDesktopInstaller: 'Desktop-App (Installer)',
    versionDetailsDistributionSourceInstaller: 'Quellcode (Installationsskript)',
    versionDetailsDistributionSourceInstallerDesktop: 'Quellcode (Installationsskript) + hermes desktop',
    versionDetailsDistributionSource: 'Quellcode',
    versionDetailsDistributionSourceDesktop: 'Quellcode + hermes desktop',
    versionDetailsDistributionStore: 'Microsoft Store',
    versionDetailsRuntime: 'Laufzeit',
    versionDetailsRuntimeEmbedded: 'Eingebettete Laufzeit',
    versionDetailsRuntimeExternal: 'Extern (nutzt die Laufzeit des Computers)',
    versionDetailsInstallId: 'Installations-ID',
    versionDetailsUncommittedChanges: 'nicht committete Änderungen',
    version: value => `Version ${value}`,
    versionUnavailable: 'Version nicht verfügbar',
    bundleOutOfSync: 'App-Build ist veraltet',
    bundleOutOfSyncDesc:
        'Die Hermes-Laufzeit wurde aktualisiert, die Desktop-App selbst ist aber noch ein älterer Build – neue Oberflächenfunktionen (wie der Bot-Modus) fehlen, bis sie aktualisiert wird. Führen Sie das Update unten aus, um die App neu zu bauen. Falls das die Warnung nicht behebt, installieren Sie den neuesten Desktop-Installer neu.',
    bundleOutOfSyncAction: 'Installer herunterladen',
    bundleSwapPending: 'Neustart zum Abschließen des Updates',
    bundleSwapPendingDesc:
        'Die aktualisierte App ist bereits installiert — Hermes muss nur noch neu gestartet werden, um sie zu laden. Chats und Einstellungen bleiben unberührt.',
    bundleSwapPendingAction: 'Hermes neu starten',
    checkNow: 'Jetzt prüfen',
    seeWhatsNew: 'Neuigkeiten ansehen',
    releaseNotes: 'Versionshinweise',
    onLatest: 'Sie verwenden die neueste Version.',
    installing: 'Ein Update wird derzeit installiert.',
    cantReach: 'Der Update-Server konnte nicht erreicht werden.',
    tapCheck: 'Tippen Sie auf „Jetzt prüfen“, um nach Updates zu suchen.',
    updateReady: count => `Ein neues Update ist bereit (${count} Änderung${count === 1 ? '' : 'en'} enthalten).`,
    updateReadyUnknown: 'Ein neues Update ist bereit.',
    lastChecked: age => `Zuletzt geprüft ${age}`,
    justNowSuffix: ' · gerade eben',
    never: 'nie',
    justNow: 'gerade eben',
    minAgo: count => `${count} Min. zuvor`,
    hoursAgo: count => `${count} Std. zuvor`,
    daysAgo: count => `${count} Tage zuvor`,
    stages: {
      idle: 'Wird vorbereitet…',
      prepare: 'Wird vorbereitet…',
      fetch: 'Wird heruntergeladen…',
      pull: 'Fast fertig…',
      pydeps: 'Wird abgeschlossen…',
      update: 'Aktualisiert Hermes…',
      rebuild: 'Baut die Desktop-App neu…',
      restart: 'Startet Hermes neu…',
      done: 'Update abgeschlossen',
      manual: 'Über Ihr Terminal aktualisieren',
      guiSkew: 'Desktop-App aktualisieren',
      error: 'Update pausiert'
    },
    checking: 'Nach Updates wird gesucht…',
    checkFailedTitle: 'Update-Check fehlgeschlagen',
    tryAgain: 'Erneut versuchen',
    notAvailableTitle: 'Kein Update verfügbar',
    unsupportedMessage: 'Diese Hermes-Version kann sich nicht aus der App heraus aktualisieren.',
    connectionRetry: 'Prüfen Sie Ihre Verbindung und versuchen Sie es erneut.',
    gitUnusable: 'Hermes konnte Git auf diesem Computer nicht ausführen und daher nicht nach Updates suchen.',
    connectionSettings: 'Verbindungseinstellungen',
    openDownloadPage: 'Download-Seite öffnen',
    latestBody: 'Sie verwenden die neueste Version.',
    latestBodyBackend: 'Das Backend läuft mit der neuesten Version.',
    allSetTitle: 'Alles bereit',
    availableTitle: 'Neues Update verfügbar',
    availableBody: 'Eine neue Hermes-Version ist bereit zur Installation.',
    availableTitleBackend: 'Backend-Update verfügbar',
    availableBodyBackend: 'Eine neuere Version des verbundenen Hermes-Backends ist bereit zur Installation.',
    availableBodyNoChangelog:
      'Eine neuere Version ist bereit. Release-Notizen sind für diesen Installationstyp nicht verfügbar.',
    updateNow: 'Jetzt aktualisieren',
    maybeLater: 'Später',
    moreChanges: count => `+ ${count} weitere Änderung${count === 1 ? '' : 'en'} enthalten.`,
    manualTitle: 'Über Ihr Terminal aktualisieren',
    manualUnavailableTitle: 'Aktualisierung hier nicht möglich',
    manualBody:
      'Sie haben Hermes über die Befehlszeile installiert, daher laufen Updates auch dort. Fügen Sie dies in Ihr Terminal ein:',
    manualPickedUp: 'Hermes übernimmt die neue Version beim nächsten Start.',
    manualBodyBackend: 'Das Hermes-Backend wird außerhalb dieser App verwaltet. Führen Sie dies auf dem Server aus, der es hostet:',
    manualPickedUpBackend: 'Das Backend lädt die neue Version, sobald das Update abgeschlossen ist.',
    guiSkewTitle: 'Desktop-App aktualisieren',
    guiSkewBody:
      'Das Backend wurde aktualisiert, aber dieses Desktop-App-Paket nicht. Aktualisieren oder installieren Sie die Hermes-Desktop-App neu (Ihr AppImage / .deb / .rpm), um beide abzugleichen.',
    copy: 'Kopieren',
    copied: 'Kopiert',
    done: 'Fertig',
    applyingBody:
      'Der Hermes-Updater übernimmt in einem eigenen Fenster und öffnet Hermes automatisch wieder, wenn er fertig ist. Bitte öffnen Sie Hermes während des Updates nicht selbst erneut.',
    applyingBodyBackend:
      'Das Remote-Backend wendet das Update an und startet neu. Hermes verbindet sich automatisch wieder, wenn es zurück ist.',
    applyingClose: 'Dieses Fenster schließt sich während des Updates, dann öffnet sich Hermes von selbst wieder.',
    errorTitle: 'Update nicht abgeschlossen',
    errorBody: 'Keine Sorge – es ging nichts verloren. Sie können es jetzt erneut versuchen.',
    blockerTitle: 'Lokale Vorschauen schließen, um Hermes zu aktualisieren?',
    blockerBody:
      'Hermes muss diese lokalen Vorschauen vor dem Update stoppen. Ihre Dateien werden dabei weder geändert noch gelöscht.',
    foreignBlockerTitle: 'Andere Prozesse schließen, um Hermes zu aktualisieren',
    foreignBlockerBody:
      'Hermes kann diese Prozesse nicht sicher automatisch beenden. Schließen Sie die App, das Terminal oder den Dienst, zu dem sie gehören, und versuchen Sie das Update erneut.',
    mixedBlockerBody:
      'Hermes kann die unten aufgeführten lokalen Vorschauen schließen. Andere Prozesse müssen manuell geschlossen werden, bevor das Update fortgesetzt werden kann.',
    closePreviewsAndUpdate: 'Vorschauen schließen und aktualisieren',
    closePreviewsAndCheckAgain: 'Vorschauen schließen und erneut prüfen',
    localPreview: 'Lokale Vorschau',
    portLabel: port => `Port ${port}`,
    pidLabel: pid => `PID ${pid}`,
    technicalDetails: 'Technische Details',
    notNow: 'Nicht jetzt',
    clientAlsoBehindTitle: 'Desktop-App ist veraltet',
    clientAlsoBehindMessage:
      'Das Backend ist aktuell, aber diese Desktop-App läuft noch auf einer älteren Version. Aktualisieren Sie sie, um die neuesten Fixes zu erhalten.',
    clientAlsoBehindAction: 'Desktop-App aktualisieren',
    everythingDispatched: 'Update ausgelöst',
    everythingSkipped: 'Übersprungen',
    everythingRowFailed: 'Update fehlgeschlagen',
    everythingFanoutFailedTitle: 'Andere Instanzen konnten nicht aktualisiert werden',
    changeLogNew: 'Neuigkeiten',
    changeLogFixed: 'Behoben',
    changeLogFaster: 'Schneller',
    changeLogImproved: 'Verbessert',
    changeLogOther: 'Weitere Verbesserungen',
    changeLogFallbackLabel: 'In diesem Update',
    changeLogFallbackItem: 'Verbesserungen und Fehlerbehebungen',
    applyStatus: {
      preparing: 'Aktualisiert Backend…',
      pulling: 'Backend wird aktualisiert…',
      restarting: 'Backend startet neu, um das Update zu laden…',
      notAvailable: 'Kein Update für dieses Backend verfügbar.',
      failed: 'Backend-Update fehlgeschlagen.',
      noReturn:
        'Das Backend kam nicht wieder online. Das Update wurde möglicherweise nicht abgeschlossen — prüfen Sie den Backend-Host.'
    }
  },
  handoffTour: {
    profileTitle: 'Ihre erste Aufgabe läuft im Standardprofil',
    profileText:
      'Diese Leiste wechselt die Profile. Das jetzt hervorgehobene ist „Standard", wo die Aufgaben-Session lebt. Das andere ist das Einrichtungsprofil, wo der Willkommens-Chat lebt.',
    sessionsTitle: 'Jedes Profil führt seine eigenen Sessions',
    sessionsText:
      'Diese Liste gehört zum Standardprofil. „Neue Session“ startet eine im jeweils gewählten Profil. Wechseln Sie Profile über die Leiste, und die Liste ändert sich mit.',
    stayTitle: 'Hermes ist einen Klick entfernt',
    stayText:
      'Wechseln Sie ins Einrichtungsprofil und öffnen Sie „Willkommen bei Hermes“, wenn Sie Hilfe brauchen. Es bleibt dort.'
  },
  guidedGreeting: {
    line: 'Hallo und willkommen. Ich bin Hermes. Geben Sie mir zwei Minuten, um alles für Sie einzurichten, dann setzen wir mich auf etwas an, das Sie wirklich erledigt haben möchten.\\n\\nAber zuerst: Wie soll ich Sie nennen?',
    nameSuggestion: name => `(Ich kann Sie auch einfach ${name} nennen, wenn Ihnen das lieber ist.)`
  },
  install: {
    stageStates: {
      pending: 'Ausstehend',
      running: 'Am Installieren',
      succeeded: 'Fertig',
      skipped: 'Übersprungen',
      failed: 'Fehlgeschlagen'
    },
    oneTimeTitle: 'Hermes braucht eine einmalige Installation',
    unsupportedDesc: platform =>
      `Die automatische Installation beim ersten Start ist auf ${platform} noch nicht verfügbar. Öffnen Sie ein Terminal, führen Sie den Befehl unten aus und starten Sie die App dann neu. Bei späteren Starts wird dieser Schritt übersprungen.`,
    installCommand: 'Installationsbefehl',
    copyCommand: 'Befehl kopieren',
    viewDocs: 'Installations-Doku ansehen',
    installTo: 'Wird installiert nach',
    retryAfterRun: "Ich hab's ausgeführt – erneut versuchen",
    setupChoiceTitle: 'Hermes Desktop einrichten',
    setupChoiceDesc:
      'Verbinden Sie diese App mit einem Hermes Gateway, das Sie bereits betreiben, oder installieren Sie Hermes lokal auf diesem Computer.',
    connectExistingTitle: 'Mit bestehendem Hermes verbinden',
    connectExistingShort: 'Bestehendes verbinden',
    connectExistingDesc:
      'Ein Remote-Backend mit Session-Token oder Browser-Anmeldung verwenden. Es wird keine lokale Installation gestartet.',
    installLocalTitle: 'Hermes lokal installieren',
    installLocalDesc:
      'Hermes herunterladen, seine Python-Umgebung erstellen und das Backend auf diesem Computer ausführen.',
    localStartUnavailable:
      'Die lokale Installation konnte nicht gestartet werden. Starten Sie Hermes Desktop neu und versuchen Sie es erneut.',
    remoteSetupTitle: 'Mit bestehendem Hermes verbinden',
    remoteSetupDesc:
      'Geben Sie die URL Ihres Gateways ein. Hermes Desktop erkennt, ob ein Token oder eine Browser-Anmeldung nötig ist.',
    remoteUrlTitle: 'Gateway-URL',
    remoteUrlDesc: 'Verwenden Sie die Basis-URL des Hermes Gateways, bei Remote-Gateways einschließlich https://.',
    remoteUrlPlaceholder: 'https://gateway.example.com/hermes',
    probing: 'Gateway-Authentifizierung wird erkannt...',
    probeError: 'Dieses Hermes Gateway konnte nicht erreicht werden.',
    probeErrorDetails: 'Details',
    identityProvider: 'Ihr Identity-Provider',
    authTitle: 'Authentifizierung',
    authNeedsOauth: provider => `Melden Sie sich mit ${provider} an, bevor Sie dieses Gateway testen.`,
    authSignedIn: 'Browser-Sign-in abgeschlossen.',
    connected: 'Verbunden',
    signIn: 'Anmelden',
    signInWith: provider => `Mit ${provider} anmelden`,
    enterUrlFirst: 'Geben Sie zuerst eine Gateway-URL ein.',
    signInIncomplete: 'Das Anmeldefenster wurde geschlossen, bevor die Authentifizierung abgeschlossen war.',
    tokenTitle: 'Session-Token',
    tokenDesc: 'Fügen Sie das Session-Token aus der .env-Datei des Remote-Gateways ein.',
    pasteSessionToken: 'Session-Token einfügen',
    incompleteSignInTest: 'Melden Sie sich an, bevor Sie dieses OAuth-geschützte Gateway testen.',
    incompleteTokenTest: 'Geben Sie ein Session-Token ein, bevor Sie dieses Gateway testen.',
    testConnection: 'Verbindung testen',
    testSucceeded: (baseUrl, version) => `Verbunden mit ${baseUrl}${version ? ` (${version})` : ''}.`,
    applyRemote: 'Übernehmen und neu verbinden',
    backToSetup: 'Zurück',
    failedTitle: 'Installation fehlgeschlagen',
    settingUpTitle: 'Hermes Agent wird eingerichtet',
    finishingTitle: 'Wird abgeschlossen',
    failedDesc:
      'Einer der Installationsschritte ist fehlgeschlagen. Unter Windows kann das passieren, wenn eine andere Hermes-CLI- oder Desktop-Instanz läuft. Beenden Sie alle laufenden Hermes-Instanzen und versuchen Sie es dann erneut. Das vollständige Protokoll finden Sie in den Details unten oder im Desktop-Log.',
    activeDesc:
      'Das ist eine einmalige Einrichtung. Der Hermes-Installer lädt Abhängigkeiten herunter und konfiguriert Ihren Computer. Bei späteren Starts wird dieser Schritt übersprungen.',
    progress: (completed, total) => `${completed} von ${total} Schritten fertig`,
    currentStage: stage => ` – gerade: ${stage}`,
    fetchingManifest: 'Installer-Manifest wird geholt...',
    error: 'Fehler',
    hideOutput: 'Installer-Ausgabe ausblenden',
    showOutput: 'Installer-Ausgabe anzeigen',
    lines: count => `${count} Zeile${count === 1 ? '' : 'n'}`,
    noOutput: 'Noch keine Ausgabe.',
    cancelling: 'Wird abgebrochen...',
    cancelInstall: 'Installation abbrechen',
    transcriptSaved: 'Vollständiges Protokoll gespeichert unter',
    copiedOutput: 'Kopiert!',
    copyOutput: 'Ausgabe kopieren',
    reloadRetry: 'Neu laden und erneut versuchen',
    openLogs: 'Logs öffnen'
  },
  onboarding: {
    headerTitle: 'Hermes Agent für Sie einrichten',
    headerDesc:
      'Verbinden Sie einen Modell-Anbieter, um mit dem Chatten zu beginnen. Die meisten Optionen brauchen nur einen Klick.',
    preparingInstall:
      'Hermes schließt die Installation ab. Das dauert beim ersten Start normalerweise unter einer Minute.',
    starting: 'Hermes wird gestartet…',
    lookingUpProviders: 'Anbieter werden gesucht...',
    collapse: 'Einklappen',
    otherProviders: 'Andere Anbieter',
    haveApiKey: 'Ich habe bereits einen API-Key',
    chooseLater: 'Ich wähle später einen Anbieter',
    recommended: 'Empfohlen',
    connected: 'Verbunden',
    featuredPitch: 'Ein Abo, 300+ Frontier-Modelle – die empfohlene Art, Hermes zu nutzen',
    fireworksPitch: 'Direkte Model-API – Fireworks-gehostete Frontier-Modelle',
    localModelsTitle: 'Modelle lokal ausführen',
    localModelsPitch: 'Kein Konto nötig – laden Sie ein Modell herunter und führen Sie es auf diesem Rechner aus',
    openRouterPitch: 'Ein Key, hunderte Modelle – ein solider Standard',
    apiKeyOptions: {
      fireworks: {
        short: 'direkte Model-API',
        description: 'Direkter Zugriff auf Modelle, die von Fireworks AI gehostet werden.'
      },
      openrouter: {
        short: 'ein Key, viele Modelle',
        description: 'Hostet hunderte Modelle hinter einem einzigen Key. Guter Standard für neue Installationen.'
      },
      openai: {
        short: 'GPT-Klasse-Modelle',
        description: 'Direkter Zugriff auf OpenAI-Modelle.'
      },
      gemini: {
        short: 'Gemini-Modelle',
        description: 'Direkter Zugriff auf Google-Gemini-Modelle.'
      },
      xai: {
        short: 'Grok-Modelle',
        description: 'Direkter Zugriff auf xAI-Grok-Modelle.'
      },
      local: {
        short: 'selbst gehostet',
        description:
          'Verbinden Sie Hermes mit einem lokalen oder selbst gehosteten OpenAI-kompatiblen Endpunkt (vLLM, llama.cpp, Ollama usw.).'
      }
    },
    backToSignIn: 'Zurück zur Anmeldung',
    getKey: 'Einen Key holen',
    replaceCurrent: 'Aktuellen Wert ersetzen',
    pasteApiKey: 'API-Key einfügen',
    localApiKeyPlaceholder: 'API-Key (optional – nur falls Ihr Endpunkt einen benötigt)',
    localModelNamePlaceholder: 'Modellname (z. B. command-a-plus-05-2026)',
    couldNotSave: 'Anmeldedaten konnten nicht gespeichert werden.',
    connecting: 'Verbinden',
    update: 'Aktualisieren',
    flowSubtitles: {
      pkce: 'Öffnet Ihren Browser zur Anmeldung und fährt dann hier fort',
      device_code: 'Öffnet eine Verifizierungsseite in Ihrem Browser – Hermes verbindet sich automatisch',
      external: 'Melden Sie sich einmal in Ihrem Terminal an und kehren Sie dann zum Chatten zurück'
    },
    startingSignIn: provider => `Anmeldung für ${provider} wird gestartet...`,
    verifyingCode: provider => `Ihr Code wird mit ${provider} überprüft…`,
    connectedProvider: provider => `${provider} verbunden`,
    connectedPicking: provider => `${provider} verbunden. Standardmodell wird ausgewählt...`,
    signInFailed: 'Anmeldung fehlgeschlagen. Versuchen Sie es erneut.',
    signInExpired:
      'Die Anmeldung ist beim Warten auf die Autorisierung abgelaufen. Meist bedeutet das, dass die Anmeldeseite im geöffneten Tab hängen geblieben ist (serverseitiges Problem) – schließen Sie die Anmeldung dort ab und versuchen Sie es dann erneut. Wenn es weiterhin fehlschlägt, verwenden Sie stattdessen einen API-Key oder die CLI als Alternative.',
    signInDidNotFinish: provider =>
      `Die Anmeldung bei ${provider} wurde nicht abgeschlossen. Prüfen Sie Ihre Internetverbindung und versuchen Sie es erneut, oder wählen Sie einen anderen Anbieter.`,
    tryAgain: 'Erneut versuchen',
    useApiKeyInstead: 'API-Key verwenden',
    errorDetails: 'Details',
    pickDifferentProvider: 'Einen anderen Anbieter wählen',
    signInWith: provider => `Mit ${provider} anmelden`,
    openedBrowser: provider => `Wir haben ${provider} in Ihrem Browser geöffnet.`,
    authorizeThere: 'Autorisieren Sie Hermes dort.',
    copyAuthCode: 'Kopieren Sie den Autorisierungscode und fügen Sie ihn unten ein.',
    pasteAuthCode: 'Autorisierungscode einfügen',
    reopenAuthPage: 'Autorisierungsseite erneut öffnen',
    autoBrowser: provider =>
      `Wir haben ${provider} in Ihrem Browser geöffnet. Autorisieren Sie Hermes dort, und Sie werden automatisch verbunden – nichts zu kopieren oder einzufügen.`,
    reopenSignInPage: 'Anmeldeseite erneut öffnen',
    waitingAuthorize: 'Warten auf Ihre Autorisierung…',
    externalPending: provider =>
      `${provider} meldet sich über seine eigene CLI an. Führen Sie diesen Befehl in einem Terminal aus, kehren Sie dann zurück und wählen Sie „Ich habe mich angemeldet“:`,
    signedIn: 'Ich habe mich angemeldet',
    deviceCodeOpened: provider => `Wir haben ${provider} in Ihrem Browser geöffnet. Geben Sie dort diesen Code ein:`,
    reopenVerification: 'Verifikationsseite erneut öffnen',
    copy: 'Kopieren',
    defaultModel: 'Standardmodell',
    freeTier: 'Free-Tier',
    pro: 'Pro',
    free: 'Kostenlos',
    price: (input, output) => `${input} rein / ${output} raus pro Mtok`,
    change: 'Ändern',
    startChatting: 'Loslegen',
    docs: provider => `${provider}-Doku`
  },
  freeTier: {
    providerRowTitle: 'Nous · Gratis-Tarif',
    providerRowPitch: 'Melden Sie sich mit einem Nous-Konto an, um mehr Modelle und Tools freizuschalten.',
    readyTitle: 'Hermes ist bereit.',
    readyCaption: 'Kostenlos · Verbindungen inklusive',
    begin: 'Loslegen',
    signInInstead: 'Stattdessen mit einem Nous-Konto anmelden',
    otherProviders: 'Andere Anbieter',
    stripTitle: 'Kostenlose Nous-Inferenz und Verbindungen sind jetzt verfügbar.',
    stripBody: 'Öffnen Sie die Modellauswahl, um sie auszuprobieren, oder melden Sie sich mit einem Nous-Konto an.',
    openModelPicker: 'Modellauswahl öffnen',
    dismiss: 'Ausblenden',
    providerName: 'Nous',
    statusLabel: model => `Nous · ${model}`,
    signIn: 'Anmelden',
    signInHeading: 'Melden Sie sich mit einem Nous-Konto an, um mehr Modelle und Tools freizuschalten.',
    settingUp: 'Kostenlose Inferenz wird eingerichtet…',
    codeBody: 'Geben Sie diesen Code in Ihrem Browser ein, um die Anmeldung abzuschließen.',
    copyLink: 'Link kopieren',
    doNotShare: 'Diesen Code nicht weitergeben.',
    waiting: 'Warten auf Anmeldung…',
    finishingHeading: 'Anmeldung wird abgeschlossen…',
    finishingBody: 'Im Browser bestätigt. Ihre Konto-Tokens werden abgerufen.',
    signedInAs: email => `Angemeldet als ${email}`,
    signedIn: 'Angemeldet.',
    completedBody: 'Ihr Konto hat jetzt Zugriff auf Inferenz und Tools.',
    defaultModel: 'Standardmodell',
    change: 'Ändern',
    done: 'Fertig',
    notNow: 'Nicht jetzt',
    tryAgain: 'Erneut versuchen',
    startAgain: 'Neu starten',
    didNotComplete: 'Anmeldung nicht abgeschlossen',
    rejectedBody: 'Die Anmeldung wurde im Browser abgelehnt. Sie bleiben im kostenlosen Tarif.',
    supersededBody: 'Ein neuerer Anmeldecode hat diesen ersetzt.',
    timedOutHeading: 'Anmeldung abgelaufen',
    timedOutBody: 'Der Code wurde nicht rechtzeitig verwendet. Sie bleiben im kostenlosen Tarif.',
    retiredBody:
      'Diese Gratis-Tarif-Identität wurde bereits verwendet oder ist abgelaufen; beim nächsten Start wird eine neue eingerichtet.',
    errorBody: 'Die Anmeldung wurde nicht abgeschlossen; starten Sie sie erneut.',
    busyHeading: 'Fast geschafft',
    busyBody: wait =>
      `Hermes konnte Ihre Anmeldung nicht abschließen, weil der Nous-Dienst ausgelastet ist. Versuchen Sie es in ${wait} erneut. Ihre Session bleibt so lange erhalten.`,
    unreachableBody:
      'Hermes konnte den Nous-Dienst nicht erreichen, um Ihre Anmeldung abzuschließen. Prüfen Sie Ihre Internetverbindung und versuchen Sie es erneut. Ihre Session bleibt erhalten.',
    alreadySignedInHeading: 'Bereits angemeldet.',
    alreadySignedInBody: 'Dieses Hermes ist bereits mit einem Nous-Konto angemeldet.',
    setupFailed: {
      gateClosed:
        'Diese Hermes-Version kann ohne Nous-Konto nicht starten. Melden Sie sich an oder legen Sie eines an – kostenlos und in einer Minute erledigt.',
      paused:
        'Chatten ohne Anmeldung ist vorübergehend pausiert. Hermes prüft weiter. Die Anmeldung ist kostenlos, und Sie können sofort weitermachen.',
      rateLimited: wait =>
        `Gerade starten sehr viele Leute, deshalb versucht Hermes es in ${wait} erneut. Die Anmeldung ist kostenlos und überspringt das Warten.`,
      unreachable:
        'Hermes konnte den Nous-Dienst nicht erreichen. Prüfen Sie Ihre Internetverbindung und tippen Sie dann auf „Erneut versuchen“. Oder verbinden Sie vorerst einen anderen Anbieter.',
      serverError:
        'Beim Nous-Dienst ist ein Fehler aufgetreten. Tippen Sie gleich auf „Erneut versuchen“ oder verbinden Sie vorerst einen anderen Anbieter.',
      powRequired:
        'Der Nous-Server verlangt einen Proof of Work, den Ihr Agent noch nicht unterstützt. Melden Sie sich an oder legen Sie ein kostenloses Nous-Konto an, um fortzufahren.',
      locked:
        'Diese Session kann ohne Anmeldung nicht fortgesetzt werden. Melden Sie sich an oder legen Sie ein kostenloses Nous-Konto an, um weiterzumachen.',
      generic:
        'Hermes konnte den kostenlosen Zugang ohne Anmeldung nicht einrichten. Die Anmeldung ist kostenlos — oder verbinden Sie einen anderen Anbieter.',
      signInBelow: 'Die Anmeldung ist kostenlos. Wählen Sie unten Nous.',
      tryAgain: 'Erneut versuchen',
      retrying: 'Wird erneut versucht…'
    }
  },
  modelPicker: {
    title: 'Modell wechseln',
    current: 'aktuell:',
    unknown: '(unbekannt)',
    search: 'Anbieter und Modelle filtern...',
    noModels: 'Keine Modelle gefunden.',
    addProvider: 'Anbieter hinzufügen',
    loadFailed: 'Modelle konnten nicht geladen werden',
    loadingIntoMemory: 'Wird in den Speicher geladen',
    downloading: 'Wird heruntergeladen',
    localDownloadsHeading: 'Lokal',
    noAuthenticatedProviders: 'Keine authentifizierten Anbieter.',
    pro: 'Pro',
    proNeedsSubscription: 'Pro-Modelle benötigen ein bezahltes Nous-Abo.',
    free: 'Kostenlos',
    freeTier: 'Kostenlose Stufe',
    priceTitle: 'Eingabe-/Ausgabepreis pro Million Tokens',
    wasPrice: 'war',
    customModel: 'Eigenes Modell',
    addCustomModelAction: 'Eigenes Modell hinzufügen…',
    customModelPlaceholder: 'Modell-ID eingeben, z. B. openai/gpt-5'
  },
  modelVisibility: {
    title: 'Modelle',
    search: 'Modelle suchen',
    noAuthenticatedProviders: 'Keine authentifizierten Anbieter.',
    addProvider: 'Anbieter hinzufügen…',
    addCustomModel: 'Eigenes Modell hinzufügen',
    removeCustomModel: 'Eigenes Modell entfernen'
  },
  shell: {
    windowControls: 'Fenster-Bedienelemente',
    paneControls: 'Panele-Bedienelemente',
    appControls: 'App-Bedienelemente',
    modelMenu: {
      search: 'Modelle durchsuchen',
      noModels: 'Keine Modelle gefunden',
      editModels: 'Modelle bearbeiten…',
      followDefault: 'Standard aus den Einstellungen verwenden',
      refreshModels: 'Modelle aktualisieren',
      fast: 'Schnell'
    },
    modelOptions: {
      noOptions: 'Keine Optionen für dieses Modell',
      options: 'Optionen',
      thinking: 'Denken',
      fast: 'Schnell',
      effort: 'Aufwand',
      minimal: 'Minimal',
      low: 'Niedrig',
      medium: 'Mittel',
      high: 'Hoch',
      xhigh: 'Extra hoch',
      max: 'Max',
      ultra: 'Ultra',
      sendsOnRoute: (level: string) => `sendet ${level} auf dieser Route`,
      updateFailed: 'Aktualisierung der Modelloptie schlug fehl',
      fastFailed: 'Aktualisierung des Schnell-Modus schlug fehl'
    },
    gatewayMenu: {
      gateway: 'Gateway',
      connected: 'Verbunden',
      connecting: 'Wird verbunden',
      offline: 'Offline',
      inferenceReady: 'Inference bereit',
      inferenceNotReady: 'Inference nicht bereit',
      checkingInference: 'Inference wird geprüft',
      disconnected: 'Getrennt',
      reconnectGateway: 'Gateway neu verbinden',
      openSystem: 'System-Panel öffnen',
      connection: label => `Verbindung: ${label}`,
      recentActivity: 'Letzte Aktivität',
      viewAllLogs: 'Alle Logs ansehen →',
      messagingPlatforms: 'Messaging-Plattformen'
    },
    approvalMode: {
      title: 'Genehmigungsmodus',
      ariaLabel: mode => `Genehmigungsmodus: ${mode}`,
      manual: 'Manuell',
      manualDescription: 'Nachfragen bei Aktionen, die eine Genehmigung erfordern',
      smart: 'Smart',
      smartDescription: 'Aktionen automatisch beurteilen und bei Bedarf nachfragen',
      off: 'Aus',
      offDescription: 'Ohne Genehmigungs-Abfragen ausführen'
    },
    statusbar: {
      unknown: 'unbekannt',
      restart: 'Neustart',
      update: 'Update',
      updateInProgress: 'Update läuft',
      commitsBehind: (count, branch) => `${count} commit${count === 1 ? '' : 's'} hinter ${branch}`,
      desktopVersion: version => `Hermes Desktop v${version}`,
      backendVersion: version => `Backend v${version}`,
      clientLabel: version => `Client v${version}`,
      connectionSsh: host => `SSH: ${host}`,
      connectionRemote: host => `Remote: ${host}`,
      connectionCloud: host => `Cloud: ${host}`,
      connectionCloudTooltip: host => `Hermes Cloud · ${host}`,
      connectionSshTooltip: host => `SSH · ${host}`,
      connectionRemoteTooltip: host => `Remote · ${host}`,
      backendLabel: version => `Backend v${version}`,
      commit: sha => `Commit ${sha}`,
      branch: branch => `Branch ${branch}`,
      closeCommandCenter: 'Command Center schließen',
      openCommandCenter: 'Command Center öffnen',
      showTerminal: 'Terminal anzeigen',
      hideTerminal: 'Terminal ausblenden',
      gateway: 'Gateway',
      gatewayReady: 'bereit',
      gatewayNeedsSetup: 'braucht Einrichtung',
      gatewayUnavailable: 'Inference nicht verfügbar',
      gatewayChecking: 'wird geprüft',
      gatewayConnecting: 'wird verbunden',
      gatewayOffline: 'offline',
      gatewayRestarting: 'wird neu gestartet…',
      gatewayTitle: 'Gateway',
      customizeTitle: 'In der Statusleiste anzeigen',
      hideStatusbar: 'Statusleiste ausblenden',
      resetStatusbar: 'Auf Standard zurücksetzen',
      toggleApprovalMode: 'Genehmigungen',
      toggleBackendVersion: 'Backend-Version',
      toggleCacheHitRate: 'Cache-Trefferquote',
      toggleCommandCenter: 'Command Center',
      toggleContextUsage: 'Kontext-Anzeige',
      toggleRunningTimer: 'Runden-Timer',
      toggleSessionTimer: 'Session-Timer',
      toggleTerminal: 'Terminal',
      toggleTokensPerSecond: 'Tokens pro Sekunde',
      toggleVersion: 'Version & Updates',
      toggleFreeTier: 'Gratis-Tarif',
      toggleWorkspace: 'Workspace',
      cacheHitRateTitle:
        'Prompt-Cache-Trefferquote dieser Session — gecachte Tokens kosten weniger, also ist höher günstiger',
      tokensPerSecondTitle: 'Ausgabe-Tokens pro Sekunde, gemittelt über die letzten 10 Modell-Aufrufe',
      agents: 'Agents',
      closeAgents: 'Agents schließen',
      openAgents: 'Agents öffnen',
      subagents: count => `${count} Subagent${count === 1 ? '' : 'en'}`,
      failed: count => `${count} fehlgeschlagen`,
      running: count => `${count} laufen`,
      cron: 'Cron',
      openCron: 'Cron-Jobs öffnen',
      webhooks: 'Webhooks',
      openWebhooks: 'Webhooks öffnen',
      starmap: 'Memory-Graph',
      openStarmap: 'Memory-Graph öffnen',
      turnRunning: 'Läuft',
      contextUsage: 'Kontext-Verbrauch',
      systemResources: {
        title: 'Systemressourcen',
        loading: 'Ressourcen…',
        gpuUtilization: 'GPU-Auslastung',
        gpuMemory: 'GPU-Speicher',
        ram: 'RAM',
        unifiedNote: 'Unified Memory — die GPU und das System teilen sich diesen Speicherpool.',
        toggle: 'Systemressourcen'
      },
      contextUsagePanel: {
        categories: {
          conversation: 'Konversation',
          mcp: 'MCP',
          memory: 'Memory',
          rules: 'Regeln',
          skills: 'Skills',
          subagent_definitions: 'Subagent-Definitionen',
          system_prompt: 'System-Prompt',
          tool_definitions: 'Tool-Definitionen'
        },
        empty: 'Noch keine Kontext-Daten',
        loading: 'Aufschlüsselung wird geladen…',
        percentFull: percent => `${percent}% voll`,
        title: 'Kontext-Verbrauch',
        tokenSummary: (used, max) => `${used} / ${max} Tokens`
      },
      focusedSince: 'Fokussiert seit',
      focusedSinceTitle: 'Zeit seit dem Fokussieren dieses Chats — nicht die Laufzeit eines Turns',
      yoloOn: 'YOLO an — gefährliche Befehle werden automatisch genehmigt. Shift+Klick schaltet global um.',
      yoloOff: 'YOLO aus. Shift+Klick schaltet global um.',
      modelNone: 'keines',
      noModel: 'kein Modell',
      switchModel: 'Modell wechseln',
      openModelPicker: 'Modell-Auswahl öffnen',
      modelPinned:
        'Von Ihnen angeheftet; neue Chats verwenden dieses Modell statt der Voreinstellung aus den Einstellungen',
      modelTitle: (provider, model) => `Modell · ${provider}: ${model}`,
      providerModelTitle: (provider, model) => `${provider} · ${model}`
    }
  },
  rightSidebar: {
    aria: 'Rechte Sidebar',
    panelsAria: 'Panels der rechten Sidebar',
    files: 'Dateisystem',
    terminal: 'Terminal',
    noFolderSelected: 'Kein Ordner ausgewählt',
    changeCwdTitle: 'Arbeitsverzeichnis ändern',
    remotePickerTitle: 'Remote-Ordner wählen',
    remotePickerDescription: 'Ordner auf dem verbundenen Backend durchsuchen.',
    remotePickerSelect: 'Ordner auswählen',
    remotePickerNewFolder: 'Neuer Ordner',
    remotePickerFolderName: 'Ordnername',
    remotePickerCreateFolder: 'Ordner erstellen',
    remotePickerInvalidFolderName: 'Gib einen einzelnen Ordnernamen ohne Schrägstriche ein.',
    remotePickerCreateFolderFailed: error => `Der Ordner konnte nicht erstellt werden (${error}).`,
    folderTip: cwd => cwd,
    openFolder: 'Ordner öffnen',
    refreshTree: 'Baum aktualisieren',
    collapseAll: 'Alle Ordner einklappen',
    showIgnored: 'Von Git ignorierte Dateien anzeigen',
    hideIgnored: 'Von Git ignorierte Dateien ausblenden',
    previewUnavailable: 'Vorschau nicht verfügbar',
    couldNotPreview: path => `Konnte ${path} nicht vorab anzeigen`,
    noProjectTitle: 'Kein Projekt',
    noProjectBody: 'Öffnen Sie ein Projekt, um seine Dateien zu durchsuchen und Änderungen zu prüfen.',
    noProjectOpen: 'Kein Projekt geöffnet',
    noDiffs: 'Keine Diffs',
    unreadableTitle: 'Nicht lesbar',
    unreadableBody: error => `Dieser Ordner konnte nicht gelesen werden (${error}).`,
    emptyTitle: 'Leer',
    emptyBody: 'Dieser Ordner ist leer.',
    treeErrorTitle: 'Baumfehler',
    treeErrorBody: 'Beim Anzeigen dieses Ordners ist ein Fehler im Dateibaum aufgetreten.',
    tryAgain: 'Erneut versuchen',
    loadingTree: 'Dateibaum wird geladen',
    loadingFiles: 'Dateien werden geladen',
    terminalHide: 'Terminal ausblenden',
    terminalsAria: 'Terminals',
    terminalNew: 'Neues Terminal',
    terminalCloseOthers: 'Andere schließen',
    terminalCloseAll: 'Alle schließen',
    addToChat: 'Zum Chat hinzufügen'
  },
  preview: {
    tab: 'Vorschau',
    closePane: 'Vorschau-Fenster schließen',
    loading: 'Vorschau wird geladen',
    unavailable: 'Vorschau nicht verfügbar',
    opening: 'Wird geöffnet…',
    hide: 'Ausblenden',
    openPreview: 'Vorschau öffnen',
    openInBrowser: 'Im Browser öffnen',
    openInExternal: 'In externem Programm öffnen',
    popIn: 'Eindocken',
    popOut: 'Abdocken',
    linkHint: '⌘/Strg-Klick für das Vorschau-Fenster',
    sourceLineTitle: 'Zum Auswählen klicken · zum Erweitern Umschalt-Klick · zum Composer ziehen',
    source: 'QUELLE',
    renderedPreview: 'VORSCHAU',
    diff: 'DIFF',
    unknownSize: 'unbekannte Größe',
    binaryTitle: 'Das sieht wie eine Binärdatei aus',
    binaryBody: label => `Die Vorschau von ${label} könnte unlesbaren Text zeigen.`,
    largeTitle: 'Diese Datei ist groß',
    largeBody: (label, size) => `${label} ist ${size}. Hermes zeigt nur die ersten 512 KB an.`,
    previewAnyway: 'Trotzdem anzeigen',
    truncated: 'Die ersten 512 KB werden angezeigt.',
    noInlineTitle: 'Keine Inline-Vorschau',
    noInlineBody: mimeType => `${mimeType || 'Dieser Dateityp'} kann trotzdem als Kontext angehängt werden.`,
    edit: 'Bearbeiten',
    editing: 'Wird bearbeitet',
    unsavedChanges: 'Nicht gespeicherte Änderungen',
    saveFailed: message => `Speichern fehlgeschlagen: ${message}`,
    diskChangedTitle: 'Datei auf der Festplatte geändert',
    diskChangedBody:
      'Diese Datei wurde geändert, seit Sie sie geöffnet haben. Mit Ihrer Version überschreiben oder Ihre Änderungen verwerfen und neu laden?',
    overwrite: 'Überschreiben',
    discardReload: 'Verwerfen & neu laden',
    console: {
      deselect: 'Eintrag abwählen',
      select: 'Eintrag auswählen',
      copyFailed: 'Konsolenausgabe konnte nicht kopiert werden',
      copyEntry: 'Diesen Eintrag kopieren',
      sendEntry: 'Diesen Eintrag an den Chat senden',
      messages: count => `${count} Konsolenmeldungen`,
      resize: 'Vorschau-Konsole in der Größe ändern',
      title: 'Vorschau-Konsole',
      selected: count => `${count} ausgewählt`,
      sendToChat: 'An Chat senden',
      copySelected: 'Ausgewähltes in die Zwischenablage kopieren',
      copyAll: 'Alles in die Zwischenablage kopieren',
      copy: 'Kopieren',
      clear: 'Leeren',
      empty: 'Noch keine Konsolenmeldungen vorhanden.',
      promptHeader: 'Vorschau-Konsole:',
      sentTitle: 'An Chat gesendet',
      sentMessage: count => `Einträge aus dem Log (${count}) wurden zum Composer hinzugefügt`
    },
    web: {
      appFailedToBoot: 'Die Vorschau-App konnte nicht gestartet werden',
      serverNotFound: 'Server nicht gefunden',
      remoteLoopback:
        'Diese Adresse verweist auf den Rechner, auf dem Ihr Agent läuft – nicht auf diesen. Das Browserfenster lädt Seiten lokal, daher braucht ein entfernter Entwicklungsserver eine Portweiterleitung oder einen erreichbaren Hostnamen.',
      failedToLoad: 'Vorschau konnte nicht geladen werden',
      tryAgain: 'Nochmal versuchen',
      restarting: 'Hermes wird neu gestartet …',
      askRestart: 'Hermes bitten, den Server neu zu starten',
      lookingRestart: taskId => `Hermes sucht nach einem Vorschau-Server zum Neustarten (${taskId})`,
      restartingTitle: 'Vorschau-Server wird neu gestartet',
      restartingMessage: 'Hermes arbeitet im Hintergrund. Beobachte im Fortschritt die Vorschau-Konsole.',
      startRestartFailed: message => `Server-Neustart konnte nicht gestartet werden: ${message}`,
      restartFailed: 'Server-Neustart fehlgeschlagen',
      hideConsole: 'Vorschau-Konsole ausblenden',
      showConsole: 'Vorschau-Konsole anzeigen',
      hideDevTools: 'Vorschau-DevTools ausblenden',
      openDevTools: 'Vorschau-DevTools öffnen',
      goBack: 'Zurück',
      goForward: 'Vor',
      reload: 'Seite neu laden',
      address: 'Adresse',
      addressPlaceholder: 'Adresse eingeben',
      blankPageBody: 'Geben Sie oben eine Adresse ein, um zu browsen, oder bitten Sie Hermes, eine Seite zu öffnen.',
      finishedRestarting: message => `Hermes hat den Vorschau-Server neu gestartet${message ? `: ${message}` : ''}`,
      failedRestarting: message => `Server-Neustart fehlgeschlagen: ${message}`,
      unknownError: 'unbekannter Fehler',
      restartedTitle: 'Vorschau-Server neu gestartet',
      reloadingNow: 'Die Vorschau wird jetzt neu geladen.',
      restartFailedTitle: 'Vorschau-Neustart fehlgeschlagen',
      restartFailedMessage: 'Hermes konnte den Server nicht neu starten.',
      stillWorking:
        'Hermes arbeitet noch, aber es ist noch kein Ergebnis des Neustarts eingetroffen. Der Server-Befehl läuft möglicherweise im Vordergrund.',
      workspaceReloading: 'Arbeitsbereich geändert, Vorschau wird neu geladen',
      fileChanged: url => `Datei geändert, Vorschau wird neu geladen: ${url}`,
      filesChanged: (count, url) => `${count} Dateiänderungen, Vorschau wird neu geladen: ${url}`,
      watchFailed: message => `Vorschau-Datei konnte nicht überwacht werden: ${message}`,
      moduleMimeDescription:
        'Modul-Skripte werden mit dem falschen MIME-Typ ausgeliefert. Das bedeutet meist, dass ein statischer Datei-Server eine Vite/React-App ausliefert statt des Projekt-Entwicklungs-Servers.',
      loadFailedConsole: (code, message) => `Laden fehlgeschlagen${code ? ` (${code})` : ''}: ${message}`,
      unreachableDescription: 'Die Vorschau-Seite konnte nicht erreicht werden.',
      openTarget: url => `${url} öffnen`,
      fallbackTitle: 'Vorschau',
      annotate: 'Annotieren',
      annotateOn: 'Annotation beenden',
      annotateNeedPage: 'Öffnen Sie zuerst eine Seite im In-App-Browser.',
      annotateFailed: 'Annotationsmodus konnte nicht gestartet werden',
      commenting: 'Kommentieren',
      addComments: count => (count === 1 ? '1 Kommentar hinzufügen' : `${count} Kommentare hinzufügen`),
      commentPlaceholder: 'Kommentar hinzufügen …',
      commentTitle: n => `Kommentar ${n}`,
      saveComment: 'Speichern',
      cancelComment: 'Kommentar abbrechen'
    }
  },
  interfaceMode: {
    title: 'Oberflächenmodus',
    hint: 'Ändert, was angezeigt wird, nicht was Hermes kann.',
    sessionNote:
      'Vom einfachen Modus festgelegt. Eine Änderung hier gilt für diese Session; wechseln Sie zu „Erweitert“, um sie dauerhaft zu übernehmen.',
    simple: {
      label: 'Einfach',
      description: 'Zum Chatten mit Hermes. Seitenleiste und Chat; keine Terminal-, Datei- oder Diff-Bereiche.'
    },
    advanced: {
      label: 'Erweitert',
      description: 'Für Entwickler. Terminal, Dateien, Diffs, Statusleiste und Layouts – so, wie Sie sie einrichten.'
    }
  },
  zones: {
    showTabStrip: 'Tabs anzeigen',
    hideTabStrip: 'Tabs ausblenden',
    showStripTab: title => `${title} anzeigen`,
    hideStripTab: title => `${title} ausblenden`,
    lastTabKeptTitle: 'Letzter Tab bleibt',
    lastTabKeptBody:
      'Diese Zone braucht mindestens einen sichtbaren Tab. Zeigen Sie zuerst einen anderen Tab an oder klappen Sie die ganze Seitenleiste ein.',
    toggleStripTab: title => `${title}-Tab umschalten`,
    minimize: 'Minimieren',
    restore: 'Wiederherstellen',
    closeRunningTitle: 'Laufenden Tab schließen?',
    closeRunningBody:
      'Dieser Chat arbeitet noch (oder wartet auf Ihre Eingabe). Das Schließen des Tabs blendet ihn nur aus – die Session behält ihren Fortschritt und kann über die Seitenleiste wieder geöffnet werden.',
    closeRunningConfirm: 'Tab schließen',
    reload: 'Neu laden',
    closeOthers: 'Andere schließen',
    closeToRight: 'Nach rechts schließen',
    closeAll: 'Alle schließen',
    newSessionTab: 'Neuer Session-Tab',
    newTab: 'Neuer Tab',
    pluginDisabled: pluginId => `Plugin "${pluginId}" deaktiviert`,
    pluginDisabledBody: 'Aktivieren Sie es wieder unter Einstellungen → Plugins, um das Panel zurückzuholen.',
    missingPane: paneId => `fehlendes Panel: ${paneId}`,
    editTitle: 'Layouts',
    editHint: 'Wählen Sie ein Layout oder ziehen Sie Panels zwischen den Zonen.',
    reset: 'Zurücksetzen',
    templates: 'Vorlagen',
    custom: 'Benutzerdefiniert',
    newGridLayout: 'Neues Grid-Layout',
    saveCurrentAs: 'Aktuelle Anordnung als Vorlage speichern',
    nameLayoutPlaceholder: 'Dieses Layout benennen…',
    deletePreset: name => `${name} löschen`,
    zoneEditorTitle: 'Zonen-Editor',
    editorHintPre: 'Klicken zum Teilen · ',
    editorHintPost: ' dreht die Linie · über Zonen ziehen zum Zusammenführen · gemeinsame Kanten ziehen zum Skalieren',
    templateColumns: 'Spalten',
    templateRows: 'Zeilen',
    templateGrid: 'Raster',
    templatePriority: 'Priorität',
    zoneTag: index => `Zone ${index}`,
    mergeZones: count => `${count} Zonen zusammenführen`,
    customZoneName: count => `Benutzerdefinierte ${count}-Zone`,
    layoutNamePlaceholder: fallback => `Layout-Name (${fallback})`,
    saveApply: 'Speichern & anwenden',
    notExpressible: 'diese Anordnung greift ineinander (Windrad) — noch nicht als verschachtelte Splits ausdrückbar',
    zoneCount: count => `${count} Zonen`,
    tabCount: count => `${count} Tabs`
  },
  contextMenu: {
    link: {
      openInApp: 'Im In-App-Browser öffnen',
      openExternal: 'Im externen Browser öffnen',
      copyUrl: 'URL kopieren',
      copyResolvedUrl: 'Aufgelöste URL kopieren'
    },
    image: {
      copyImage: 'Bild kopieren',
      copyImageAddress: 'Bildadresse kopieren',
      saveImageAs: 'Bild speichern unter…'
    },
    edit: {
      cut: 'Ausschneiden',
      paste: 'Einfügen',
      selectAll: 'Alles auswählen',
      addToDictionary: 'Zum Wörterbuch hinzufügen'
    },
    page: {
      copyPageUrl: 'Seiten-URL kopieren',
      inspectElement: 'Element untersuchen'
    }
  },
  assistant: {
    thread: {
      loadingSession: 'Session wird geladen',
      showEarlier: 'Frühere Nachrichten anzeigen',
      loadingResponse: 'Hermes lädt eine Antwort',
      loadingLocalModel: model => `${model} wird in den Speicher geladen`,
      processingPrompt: 'Verarbeite Prompt',
      resumeWhenBackgroundDone: count =>
        count === 1
          ? 'Wird fortgesetzt, wenn die Hintergrundaufgabe endet'
          : `Wird fortgesetzt, wenn ${count} Hintergrundaufgaben enden`,
      thinking: 'Denkt',
      thought: 'Gedanke',
      thoughtBriefly: 'Kurz nachgedacht',
      thoughtFor: duration => `${duration} nachgedacht`,
      turnDuration: duration => `Dieser Turn dauerte ${duration}`,
      today: time => `Heute, ${time}`,
      yesterday: time => `Gestern, ${time}`,
      copy: 'Kopieren',
      refresh: 'Aktualisieren',
      moreActions: 'Weitere Aktionen',
      branchNewChat: 'In neuem Chat abzweigen',
      react: 'Reagieren',
      dismissError: 'Fehler schließen',
      errorLayers: {
        auth: 'Authentifizierungsfehler',
        billing: 'Keine Credits mehr',
        disk: 'Festplatte voll',
        endpoint: 'Benutzerdefinierter Endpunktfehler',
        gateway: 'Gateway-Fehler',
        generic: 'Turn fehlgeschlagen',
        provider: 'Anbieterfehler',
        runtime: 'Lokaler Laufzeitfehler',
        streaming: 'Streaming-Verbindungsfehler'
      },
      errorLayerBodies: {
        auth: 'Der KI-Dienst hat Ihre Anmeldung abgelehnt. Prüfen Sie die Zugangsdaten für diesen Anbieter und senden Sie Ihre Nachricht erneut.',
        billing:
          'Ihr Konto hat bei diesem Anbieter kein Guthaben mehr. Laden Sie Guthaben auf oder wechseln Sie den Anbieter und senden Sie erneut.',
        disk: 'Ihre Festplatte ist voll, daher konnte Hermes dieses Gespräch nicht speichern. Geben Sie Speicherplatz frei und versuchen Sie es erneut.',
        endpoint:
          'Hermes erreicht Ihren eigenen Modellserver nicht. Prüfen Sie, ob er läuft, und senden Sie Ihre Nachricht erneut.',
        gateway:
          'Beim Starten dieser Antwort ist in Hermes ein internes Problem aufgetreten. Senden Sie Ihre Nachricht erneut; wenn es bestehen bleibt, senden Sie Diagnosedaten.',
        generic:
          'Beim Antworten ist etwas schiefgelaufen. Versuchen Sie es erneut oder kopieren Sie die Details, wenn es bestehen bleibt.',
        provider:
          'Der KI-Dienst konnte diese Anfrage nicht abschließen. Versuchen Sie es gleich erneut oder wechseln Sie den Anbieter.',
        runtime:
          'Beim Starten dieser Antwort ist in Hermes ein internes Problem aufgetreten. Senden Sie Ihre Nachricht erneut; wenn es bestehen bleibt, senden Sie Diagnosedaten.',
        streaming:
          'Die Verbindung ist abgebrochen, bevor die Antwort fertig war. Versuchen Sie es erneut, um sie noch einmal zu senden.'
      },
      errorCodes: {
        auth: {
          title: provider => `${provider} hat Ihre Anmeldung abgelehnt`,
          body: provider =>
            `Die für ${provider} gespeicherten Zugangsdaten wurden nicht akzeptiert. Korrigieren Sie sie in den Einstellungen oder wechseln Sie den Anbieter und senden Sie Ihre Nachricht erneut.`
        },
        auth_permanent: {
          title: provider => `${provider} hat Ihre Anmeldung abgelehnt`,
          body: provider =>
            `Die für ${provider} gespeicherten Zugangsdaten sind ungültig oder wurden widerrufen. Aktualisieren Sie sie oder wechseln Sie den Anbieter und senden Sie Ihre Nachricht erneut.`
        },
        billing: {
          title: 'Kein Guthaben mehr',
          body: provider =>
            `Ihr ${provider}-Konto hat kein Guthaben mehr. Laden Sie Guthaben auf oder wechseln Sie den Anbieter und senden Sie erneut.`
        },
        rate_limit: {
          title: 'Der KI-Dienst ist ausgelastet',
          body: provider =>
            `${provider} begrenzt gerade die Anfragen. Warten Sie eine Minute und versuchen Sie es erneut.`
        },
        upstream_rate_limit: {
          title: 'Der KI-Dienst ist ausgelastet',
          body: provider =>
            `${provider} begrenzt gerade die Anfragen. Warten Sie eine Minute und versuchen Sie es erneut.`
        },
        overloaded: {
          title: 'Der KI-Dienst ist überlastet',
          body: provider =>
            `${provider} hat gerade Probleme. Versuchen Sie es gleich erneut oder wechseln Sie den Anbieter.`
        },
        server_error: {
          title: 'Beim KI-Dienst ist ein Fehler aufgetreten',
          body: provider =>
            `${provider} hat einen Serverfehler zurückgegeben. Versuchen Sie es gleich erneut oder wechseln Sie den Anbieter.`
        },
        timeout: {
          title: 'Die Antwort hat zu lange gebraucht',
          body: provider =>
            `${provider} hat nicht rechtzeitig geantwortet. Versuchen Sie es erneut, um die Nachricht noch einmal zu senden.`
        },
        stream_drop: {
          title: 'Die Antwort wurde abgebrochen',
          body: 'Die Verbindung ist abgebrochen, bevor die Antwort fertig war. Versuchen Sie es erneut, um sie noch einmal zu senden.'
        },
        upstream_blocked: {
          title: 'Eine Firewall hat die Anfrage blockiert',
          body: (provider: string) =>
            `Eine Firewall oder ein CDN vor ${provider} hat die Anfrage blockiert, bevor sie das Modell erreicht hat – Ihr Schlüssel ist vermutlich in Ordnung. Setzen Sie in den Einstellungen über die extra_headers des Anbieters einen User-Agent-Header oder wechseln Sie den Anbieter und senden Sie die Nachricht dann erneut.`
        },
        ssl_cert_verification: {
          title: 'Sichere Verbindung fehlgeschlagen',
          body: provider =>
            `Hermes konnte die sichere Verbindung zu ${provider} nicht verifizieren. Prüfen Sie Ihre Netzwerk- oder Proxy-Einstellungen oder wechseln Sie den Anbieter und senden Sie Ihre Nachricht erneut.`
        },
        context_overflow: {
          title: 'Dieses Gespräch ist zu lang',
          body: 'Das Gespräch passt nicht mehr ins Modell. Komprimieren Sie es oder starten Sie einen neuen Chat und senden Sie erneut.'
        },
        payload_too_large: {
          title: 'Diese Nachricht ist zu groß',
          body: 'Die Anfrage war zu groß für das Modell. Komprimieren Sie das Gespräch oder starten Sie einen neuen Chat und senden Sie erneut.'
        },
        model_not_found: {
          title: 'Dieses Modell ist nicht verfügbar',
          body: provider =>
            `${provider} bietet dieses Modell für Ihr Konto nicht an. Wählen Sie ein anderes Modell und senden Sie Ihre Nachricht erneut.`
        },
        provider_policy_blocked: {
          title: 'Dieses Modell ist durch Ihre Kontoeinstellungen gesperrt',
          body: provider =>
            `${provider} leitet diese Anfrage mit den Daten- und Datenschutzeinstellungen Ihres Kontos nicht weiter. Wählen Sie ein anderes Modell oder wechseln Sie den Anbieter.`
        },
        content_policy_blocked: {
          title: 'Der KI-Dienst hat diese Anfrage abgelehnt',
          body: provider =>
            `${provider} wollte diese Nachricht nicht beantworten. Ändern Sie sie und senden Sie sie erneut.`
        },
        format_error: {
          title: 'Der KI-Dienst hat die Anfrage abgelehnt',
          body: provider =>
            `${provider} hat den Aufbau dieser Anfrage nicht akzeptiert. Wechseln Sie den Anbieter oder senden Sie Diagnosedaten, damit wir es prüfen können.`
        },
        truncated: {
          title: 'Die Antwort wurde abgeschnitten',
          body: 'Das Modell hat vor dem Ende abgebrochen. Versuchen Sie es erneut, um eine vollständige Antwort zu erhalten.'
        },
        invalid_response: {
          title: 'Der KI-Dienst hat eine unlesbare Antwort geschickt',
          body: provider =>
            `${provider} hat etwas zurückgegeben, das Hermes nicht lesen konnte. Versuchen Sie es gleich erneut.`
        },
        empty_response: {
          title: 'Der KI-Dienst hat eine leere Antwort geschickt',
          body: provider => `${provider} hat auf diese Nachricht nichts zurückgegeben. Versuchen Sie es gleich erneut.`
        },
        loop_error: {
          title: 'Hermes ist in einer Schleife hängen geblieben',
          body: 'Die Antwort hat dieselben Schritte wiederholt, daher hat Hermes sie gestoppt. Versuchen Sie es erneut oder starten Sie einen neuen Chat, wenn es wieder passiert.'
        },
        SESSION_NOT_OWNED: {
          title: 'Dieser Chat ist woanders offen',
          body: 'Dieser Chat ist gerade in einem anderen Hermes-Fenster oder Terminal geöffnet. Schließen Sie ihn dort und senden Sie Ihre Nachricht erneut, oder starten Sie hier einen neuen Chat.'
        },
        disk_full: {
          title: 'Festplatte voll',
          body: 'Ihre Festplatte ist voll, daher konnte Hermes dieses Gespräch nicht speichern. Geben Sie Speicherplatz frei und versuchen Sie es erneut.'
        },
        free_tier_disabled: {
          title: 'Chatten ohne Anmeldung ist gerade abgeschaltet',
          body: 'Melden Sie sich mit einem Nous-Konto an, um weiterzuschreiben – es ist kostenlos.'
        },
        free_tier_rate_limited: {
          title: 'Sie haben das Kontingent für Chats ohne Anmeldung aufgebraucht',
          body: 'Es wird bald wieder aufgefüllt. Melden Sie sich mit einem Nous-Konto an, um ein größeres Kontingent zu erhalten – es ist kostenlos.'
        },
        free_tier_at_capacity: {
          title: 'Chatten ohne Anmeldung ist gerade sehr stark ausgelastet',
          body: 'Melden Sie sich an, um die Warteschlange zu überspringen – es ist kostenlos –, oder versuchen Sie es später erneut.'
        },
        free_tier_model_not_free: {
          title: 'Dieses Modell gibt es ohne Anmeldung nicht',
          body: 'Hermes verwendet vorerst das kostenlose Modell. Melden Sie sich mit einem Nous-Konto an, um mehr Modelle zu nutzen – es ist kostenlos.'
        },
        free_tier_route: {
          title: 'Hermes hat das kostenlose Modell über diese Route nicht erreicht',
          body: 'Melden Sie sich mit einem Nous-Konto an – es ist kostenlos – oder prüfen Sie die Einstellung NOUS_INFERENCE_BASE_URL.'
        },
        free_tier_outage: {
          title: 'Das kostenlose Modell antwortet gerade schlecht',
          body: 'Versuchen Sie in einer Minute, Ihre Nachricht erneut zu senden.'
        },
        free_tier_refused: {
          title: 'Hermes konnte das ohne Anmeldung nicht senden',
          body: 'Eine Anmeldung mit einem Nous-Konto ist kostenlos.'
        }
      },
      errorAuthKinds: {
        api_key: {
          title: provider => `${provider} hat Ihren API-Key abgelehnt`,
          body: provider =>
            `Der für ${provider} gespeicherte Key ist ungültig oder wurde widerrufen. Aktualisieren Sie ihn und versuchen Sie es erneut.`
        },
        oauth: {
          title: provider => `Ihre ${provider}-Anmeldung ist abgelaufen`
        }
      },
      errorDetails: 'Details',
      errorGenericProvider: 'Der KI-Dienst',
      errorToastTitle: 'Hermes konnte die Antwort nicht fertigstellen',
      errorRetry: 'Erneut versuchen',
      errorLimitResets: (time: string) => `Limit wird um ${time} zurückgesetzt`,
      errorRetryAtReset: (time: string) => `Erneut versuchen, wenn das Limit zurückgesetzt wird (${time})`,
      errorRetryScheduled: (time: string, wait: string) => `Neuer Versuch um ${time} – in ${wait}`,
      errorRetryScheduledCancel: 'Abbrechen',
      errorStartNewSession: 'Neue Session starten',
      errorSwitchProvider: 'Anbieter wechseln',
      errorChooseModel: 'Modell wählen',
      errorCompressConversation: 'Gespräch komprimieren',
      errorCompressFailed: 'Das Gespräch konnte nicht komprimiert werden',
      errorOpenHermesFolder: 'Hermes-Ordner öffnen',
      errorOpenHermesFolderFailed: 'Der Hermes-Ordner konnte nicht geöffnet werden',
      errorUpdateApiKey: 'API-Key aktualisieren',
      errorSignInAgain: provider => `Erneut bei ${provider} anmelden`,
      errorSignInFreeTier: 'Mit einem Nous-Konto anmelden',
      errorOauthExpired: provider =>
        `Ihre Anmeldung bei ${provider} ist abgelaufen oder wurde widerrufen. Melden Sie sich erneut an, um weiterzuchatten.`,
      errorOpenLogs: 'Logs öffnen',
      errorOpenLogsFailed: 'Der Logs-Ordner konnte nicht geöffnet werden',
      errorOpenDesktopLogs: 'Desktop-Logs öffnen',
      errorCopyDiagnostics: 'Fehlerdetails kopieren',
      errorSendDiagnostics: 'Diagnose senden',
      filesChanged: count => (count === 1 ? '1 Datei geändert' : `${count} Dateien geändert`),
      reviewChanges: 'Prüfen',
      readAloudFailed: 'Vorlesen fehlgeschlagen',
      preparingAudio: 'Bereitet Audio vor...',
      stopReading: 'Vorlesen stoppen',
      readAloud: 'Vorlesen',
      editMessage: 'Nachricht bearbeiten',
      expandMessage: 'Nachricht aufklappen',
      scrollToBottom: 'Nach unten scrollen',
      stop: 'Stopp',
      restorePrevious: 'Vorherigen Checkpoint wiederherstellen',
      restoreCheckpoint: 'Checkpoint wiederherstellen',
      restoreFromHere: 'Checkpoint wiederherstellen — von diesem Prompt erneut ausführen',
      restoreTitle: 'Zu diesem Checkpoint wiederherstellen?',
      restoreBody: 'Alles nach diesem Prompt wird aus der Konversation entfernt, und der Prompt läuft von hier erneut.',
      restoreConfirm: 'Wiederherstellen & erneut ausführen',
      restoreNext: 'Nächsten Checkpoint wiederherstellen',
      goForward: 'Vorwärts',
      sendEdited: 'Bearbeitete Nachricht senden',
      attachingFile: 'Hängt an…'
    },
    approval: {
      gatewayDisconnected: 'Hermes-Gateway ist nicht verbunden',
      sendFailed: 'Genehmigungsantwort konnte nicht gesendet werden',
      reconnect: 'Neu verbinden',
      timedOutSystemLine:
        'Die Freigabe ist abgelaufen — der Befehl wurde nicht ausgeführt. Bitten Sie Hermes, es erneut zu versuchen, oder erhöhen Sie das Limit unter Einstellungen → Sicherheit → Freigabe-Timeout.',
      openSafetySettings: 'Sicherheitseinstellungen öffnen',
      run: 'Ausführen',
      command: 'Befehl',
      moreOptions: 'Weitere Genehmigungsoptionen',
      allowSession: 'Diese Session erlauben',
      alwaysAllowMenu: 'Immer erlauben…',
      jumpToApproval: 'Genehmigung erforderlich',
      reject: 'Ablehnen',
      alwaysTitle: 'Diesen Befehl immer erlauben?',
      alwaysDescription: pattern =>
        `Dies fügt das Muster „${pattern}“ Ihrer dauerhaften Zulassungsliste hinzu (~/.hermes/config.yaml). Hermes fragt bei solchen Befehlen nicht mehr nach – weder in dieser noch in zukünftigen Sessions.`,
      alwaysAllow: 'Immer erlauben'
    },
    clarify: {
      notReady: 'Klärungsanfrage ist noch nicht bereit',
      gatewayDisconnected: 'Hermes-Gateway ist nicht verbunden',
      sendFailed: 'Klärungsantwort konnte nicht gesendet werden',
      loadingQuestion: 'Frage wird geladen…',
      other: 'Anderes (Antwort eingeben)',
      placeholder: 'Geben Sie Ihre Antwort ein…',
      skip: 'Überspringen',
      skipped: 'Übersprungen',
      continueLabel: 'Weiter',
      confirmAndContinueLabel: 'Bestätigen und fortfahren',
      answeredBadge: 'Beantwortet',
      questionProgress: (answered, total) => `${answered} von ${total} beantwortet`,
      lateAnswer: (question, choice) => `Re: „${question}“ — meine Antwort: ${choice}`,
      lateAnswerTip: 'Diese Antwort als Folgenachricht entwerfen',
      lateAnswerHint: 'Dieser Prompt wartet nicht mehr. Wählen Sie eine Option, um sie als Folgenachricht zu entwerfen.'
    },
    catalogInstall: {
      preparing: 'Installation wird vorbereitet…',
      install: 'Installieren',
      advanced: 'Erweitert',
      skip: 'Überspringen',
      installing: 'Wird installiert…',
      installed: 'Installiert',
      notInstalled: 'Nicht installiert',
      failed: 'Fehlgeschlagen',
      showNames: 'Namen anzeigen',
      hideNames: 'Namen ausblenden',
      skill: (name: string) => `Skill ${name}`,
      kind: {
        plugin: 'Plugin',
        skill: 'Skill'
      },
      tier: {
        official: 'offiziell',
        community: 'Community'
      },
      targetProfile: (profile: string) => `Wird in Ihr Profil ${profile} installiert`,
      sendFailed: 'Ihre Antwort konnte nicht gesendet werden. Versuchen Sie es erneut.',
      commitLabel: 'Commit',
      subdirLabel: 'Ordner',
      securityHeading: 'Sicherheit',
      scan: {
        passed: 'Scan bestanden',
        warnings: 'Scan hat Warnungen gefunden',
        failed: 'Scan fehlgeschlagen'
      },
      requirementsLabel: 'Erfordert',
      credentialsHeading: 'Zugangsdaten'
    },
    mcpSetup: {
      installTitle: 'MCP-Server hinzufügen',
      enableTitle: 'MCP-Server aktivieren',
      authorizeTitle: 'MCP-Server autorisieren',
      installAction: 'Installieren',
      enableAction: 'Aktivieren',
      authorizeAction: 'Autorisieren',
      installed: server => `${server} installiert`,
      enabled: server => `${server} aktiviert`,
      authorized: server => `${server} autorisiert`,
      failed: server => `Einrichtung für ${server} fehlgeschlagen`,
      toolCount: count => (count === 1 ? '1 Tool' : `${count} Tools`),
      envRequired: 'Füllen Sie zuerst die erforderlichen Anmeldedaten aus',
      sendFailed: 'MCP-Einrichtungsantwort konnte nicht gesendet werden',
      reloadFailed:
        'Server gespeichert, aber das Neuladen der MCP-Tools schlug fehl — sie laden in der nächsten Session',
      gatewayDisconnected: 'Hermes-Gateway ist nicht verbunden'
    },
    tool: {
      copyCode: 'Code kopieren',
      renderingImage: 'Rendert Bild',
      copyOutput: 'Ausgabe kopieren',
      copyCommand: 'Befehl kopieren',
      copyContent: 'Inhalt kopieren',
      copyUrl: 'URL kopieren',
      copyResults: 'Ergebnisse kopieren',
      copyQuery: 'Abfrage kopieren',
      copyFile: 'Datei kopieren',
      copyPath: 'Pfad kopieren',
      failedCalls: count => `${count} Tool-Aufruf${count === 1 ? '' : 'e'} fehlgeschlagen`,
      skillActivity: {
        loading: 'Skill wird geladen',
        loaded: 'Skill geladen',
        loadFailed: 'Skill konnte nicht geladen werden',
        readingResource: 'Skill-Ressource wird gelesen',
        readResource: 'Skill-Ressource gelesen',
        resourceFailed: 'Skill-Ressource konnte nicht gelesen werden',
        listing: 'Skills werden aufgelistet',
        listed: 'Skills aufgelistet',
        listFailed: 'Skills konnten nicht aufgelistet werden',
        unavailable: 'Skill-Ergebnis nicht verfügbar'
      },
      outputAlt: 'Tool-Ausgabe',
      rawResponse: 'Rohantwort',
      copyActivity: 'Aktivität kopieren',
      recoveredOne: 'Nach 1 fehlgeschlagenem Schritt wiederhergestellt',
      recoveredMany: count => `Nach ${count} fehlgeschlagenen Schritten wiederhergestellt`,
      failedOne: '1 Schritt fehlgeschlagen',
      failedMany: count => `${count} Schritte fehlgeschlagen`,
      statusRunning: 'Läuft',
      statusError: 'Fehler',
      statusRecovered: 'Wiederhergestellt',
      statusDone: 'Fertig',
      resultUnavailable: 'Ergebnis nicht verfügbar',
      resultInterrupted: 'Unterbrochen',
      memoryWriteNoted: 'Speicher-Schreiben notiert',
      actions: {
        read: 'Lesen',
        reading: 'Liest',
        opened: 'Geöffnet',
        opening: 'Öffnet',
        failedToOpen: 'Öffnen fehlgeschlagen',
        searched: 'Durchsucht',
        searching: 'Durchsucht',
        ran: 'Ausgeführt',
        running: 'Läuft',
        ranCode: 'Code ausgeführt',
        runningCode: 'Skriptet'
      },
      prefixes: {
        browser: 'Browser',
        web: 'Web'
      },
      titleTemplates: {
        actionCommand: (action, command) => `${action} ${command}`,
        actionQuoted: (action, value) => `${action} „${value}“`,
        actionTarget: (action, target) => `${action} ${target}`,
        prefixedDone: (prefix, action) => `${prefix} ${action}`,
        runningPrefixedTool: (prefix, action) => `${prefix} ${action.toLowerCase()} wird ausgeführt`,
        runningTool: action => `${action} wird ausgeführt`
      },
      titles: {
        browser_click: {
          done: 'Seitenelement geklickt',
          pending: 'Klickt Seitenelement',
          pendingAction: 'Klickt'
        },
        browser_fill: {
          done: 'Formularfeld ausgefüllt',
          pending: 'Füllt Formularfeld',
          pendingAction: 'Füllt'
        },
        browser_navigate: {
          done: 'Seite geöffnet',
          pending: 'Öffnet Seite',
          pendingAction: 'Öffnet'
        },
        browser_snapshot: {
          done: 'Seiten-Snapshot erfasst',
          pending: 'Erfasst Seiten-Snapshot',
          pendingAction: 'Erfasst'
        },
        browser_take_screenshot: {
          done: 'Screenshot erfasst',
          pending: 'Erfasst Screenshot',
          pendingAction: 'Erfasst'
        },
        browser_type: {
          done: 'Auf Seite getippt',
          pending: 'Tippt auf Seite',
          pendingAction: 'Tippt'
        },
        clarify: {
          done: 'Frage gestellt',
          pending: 'Stellt Frage',
          pendingAction: 'Stellt'
        },
        cronjob: {
          done: 'Cron-Job',
          pending: 'Plant Cron-Job',
          pendingAction: 'Plant'
        },
        edit_file: {
          done: 'Datei bearbeitet',
          pending: 'Bearbeitet Datei',
          pendingAction: 'Bearbeitet'
        },
        execute_code: {
          done: 'Code ausgeführt',
          pending: 'Skriptet',
          pendingAction: 'Skriptet'
        },
        image_generate: {
          done: 'Bild generiert',
          pending: 'Generiert Bild',
          pendingAction: 'Generiert'
        },
        list_files: {
          done: 'Dateien aufgelistet',
          pending: 'Listet Dateien',
          pendingAction: 'Listet'
        },
        memory: {
          done: 'Im Speicher gespeichert',
          pending: 'Speichert im Speicher',
          pendingAction: 'Speichert'
        },
        patch: {
          done: 'Datei gepatcht',
          pending: 'Patcht Datei',
          pendingAction: 'PATCHT'
        },
        read_file: {
          done: 'Datei gelesen',
          pending: 'Liest Datei',
          pendingAction: 'Liest'
        },
        search_files: {
          done: 'Dateien durchsucht',
          pending: 'Durchsucht Dateien',
          pendingAction: 'Durchsucht'
        },
        session_search_recall: {
          done: 'Session-Verlauf durchsucht',
          pending: 'Durchsucht Session-Verlauf',
          pendingAction: 'Durchsucht'
        },
        terminal: {
          done: 'Befehl ausgeführt',
          pending: 'Führt Befehl aus',
          pendingAction: 'Führt aus'
        },
        todo: {
          done: 'Todos aktualisiert',
          pending: 'Aktualisiert Todos',
          pendingAction: 'Aktualisiert'
        },
        vision_analyze: {
          done: 'Bild analysiert',
          pending: 'Analysiert Bild',
          pendingAction: 'Analysiert'
        },
        web_extract: {
          done: 'Webseite gelesen',
          pending: 'Liest Webseite',
          pendingAction: 'Liest'
        },
        web_search: {
          done: 'Web durchsucht',
          pending: 'Durchsucht Web',
          pendingAction: 'Durchsucht'
        },
        write_file: {
          done: 'Datei bearbeitet',
          pending: 'Bearbeitet Datei',
          pendingAction: 'Bearbeitet'
        }
      }
    }
  },
  prompts: {
    gatewayDisconnected: 'Das Hermes-Gateway ist nicht verbunden',
    reconnect: 'Neu verbinden',
    sudoSendFailed: 'Sudo-Passwort konnte nicht gesendet werden',
    secretSendFailed: 'Geheimnis konnte nicht gesendet werden',
    sudoTitle: 'Administrator-Passwort',
    sudoDesc:
      'Hermes benötigt Ihr Sudo-Passwort, um einen privilegierten Befehl auszuführen. Es wird nur an Ihren lokalen Agenten gesendet.',
    sudoCommandUnavailable:
      'Dieser Agent hat den Befehl nicht mitgeliefert. Brechen Sie ab, wenn Sie ihn im Gespräch nicht überprüfen können.',
    sudoInstallDesc:
      'Hermes benötigt Ihr sudo-Passwort, um die Bot-Screen-Pakete (TigerVNC + Xfce) auf dem Gateway-Host zu installieren. Es wird nur an diesen Host gesendet.',
    sudoPlaceholder: 'Sudo-Passwort',
    secretTitle: 'Geheimnis erforderlich',
    secretDesc: 'Hermes benötigt eine Zugangsdaten, um fortzufahren.',
    secretPlaceholder: 'Geheimnis-Wert',
    vaultUnlockSendFailed: 'Master-Passwort konnte nicht gesendet werden',
    vaultUnlockTitle: name => `${name} entsperren`,
    vaultUnlockDesc: name =>
      `Der Agent möchte sich mit einem in ${name} gespeicherten Login bei einer Seite anmelden. Geben Sie Ihr Master-Passwort ein, um ihn für diese Session zu entsperren – es geht direkt an ${name} auf diesem Rechner und wird nie gespeichert oder dem Agenten gezeigt.`,
    vaultUnlockPlaceholder: 'Master-Passwort',
    vaultUnlockKeepLocked: 'Gesperrt lassen',
    vaultUnlockConfirm: 'Entsperren',
    vaultSaveSendFailed: 'Login konnte nicht gespeichert werden',
    vaultSaveTitle: site => `${site}-Login speichern?`,
    vaultSaveDesc: origin =>
      `Hermes ist auf eine Anmeldeseite unter ${origin} gestoßen und hat dafür keinen Login. Geben Sie ihn einmal hier ein; er wird auf diesem Rechner verschlüsselt und in die Seite eingetragen, ohne dass das Modell das Passwort je sieht.`,
    vaultSaveIdentifierLabel: 'E-Mail oder Benutzername',
    vaultSaveIdentifierPlaceholder: 'name@example.com',
    vaultSavePasswordPlaceholder: 'Passwort',
    vaultSaveFootnote: 'Verwalten Sie gespeicherte Logins unter Einstellungen → Passwörter & Logins.',
    vaultSaveDecline: 'Nicht speichern',
    vaultSaveConfirm: 'Speichern & anmelden',
    vaultCodeSendFailed: 'Code konnte nicht gesendet werden',
    vaultCodeTitle: site => `Bestätigungscode für ${site}`,
    vaultCodeDesc: site =>
      `${site} verlangt einen Einmalcode (SMS, E-Mail oder Authenticator-App). Geben Sie ihn hier ein, Hermes trägt ihn in die Seite ein; das Modell sieht ihn nie.`,
    vaultCodeLabel: 'Code',
    vaultCodeFootnote:
      'Tipp: Speichern Sie den Authentifizierungsschlüssel zusammen mit diesem Login unter Einstellungen → Passwörter & Logins, dann gibt Hermes die Codes für Sie ein.',
    vaultCodeSkip: 'Überspringen',
    vaultCodeConfirm: 'Code eingeben'
  },
  desktop: {
    audioReadFailed: 'Aufgenommenes Audio konnte nicht gelesen werden',
    sessionUnavailable: 'Session nicht verfügbar',
    createSessionFailed: 'Neue Session konnte nicht erstellt werden',
    promptFailed: 'Prompt fehlgeschlagen',
    staleSessionTitle: 'Chat veraltet',
    staleSessionBody:
      'Dieses Fenster war hinter einer anderen Ansicht desselben Chats. Die neuesten Nachrichten wurden geladen. Senden Sie erneut, wenn Sie noch möchten.',
    providerCredentialRequired:
      'Fügen Sie Anmeldedaten für einen Anbieter hinzu, bevor Sie Ihre erste Nachricht senden.',
    emptySlashCommand: 'leerer Slash-Befehl',
    desktopCommands: 'Desktop-Befehle',
    skillCommandsAvailable: count => `${count} Fähigkeitsbefehle verfügbar.`,
    warningLine: message => `Warnung: ${message}`,
    yoloArmed: 'YOLO für diesen Chat aktiviert',
    yoloOff: 'YOLO aus',
    yoloSystem: active => `YOLO ${active ? 'an' : 'aus'} für diese Session`,
    yoloTitle: 'YOLO',
    yoloToggleFailed: 'YOLO konnte nicht umgeschaltet werden',
    profileStatus: current =>
      `Profil: ${current}. Verwenden Sie /profile <name> oder die Auswahl „Neue Session“, um einen Chat in einem anderen Profil zu starten.`,
    unknownProfile: 'Unbekanntes Profil',
    noProfileNamed: (target, available) => `Kein Profil namens „${target}“. Verfügbar: ${available}`,
    newChatsProfile: name => `Neue Chats verwenden Profil ${name}.`,
    setProfileFailed: 'Profil konnte nicht gesetzt werden',
    sttDisabled: 'Sprach-zu-Text ist in den Einstellungen deaktiviert.',
    stopFailed: 'Stopp fehlgeschlagen',
    regenerateFailed: 'Regenerieren fehlgeschlagen',
    editFailed: 'Bearbeiten fehlgeschlagen',
    editTurnUnavailable: 'Dieser Turn ist nicht mehr in der Server-Historie (er wurde möglicherweise komprimiert).',
    resumeFailed: 'Fortsetzen fehlgeschlagen',
    readOnlyTranscriptTitle: 'Schreibgeschützt geöffnet',
    readOnlyTranscriptBody:
      'Noch kein verbundenes Backend beansprucht diesen älteren Chat, also wurde er als schreibgeschütztes Transkript geöffnet. Seine Historie ist intakt; Senden ist deaktiviert, bis ein Backend ihn beansprucht.',
    readOnlyTranscriptSendBlocked:
      'Dieser Chat ist als schreibgeschütztes Transkript geöffnet — Senden ist deaktiviert.',
    resumeStrandedTitle: 'Diese Session konnte nicht geladen werden',
    resumeStrandedBody:
      'Die Verbindung zu dieser Session ist fehlgeschlagen, und automatische Wiederholungen wurden eingestellt. Prüfen Sie, ob das Gateway läuft, und versuchen Sie es erneut.',
    poolSlotTimeoutBody:
      'Alle Backend-Plätze für lokale Profile sind belegt. Erhöhen Sie „Warme Bot-Backends“ unter Einstellungen → Erweitert, oder versuchen Sie es erneut, nachdem ein inaktives Backend entfernt wurde.',
    poolSlotTimeoutOpenSettings: 'Erweiterte Einstellungen öffnen',
    resumeRetry: 'Erneut versuchen',
    nothingToBranch: 'Nichts zum Abzweigen',
    branchNeedsChat: 'Starten Sie einen Chat oder setzen Sie ihn fort, bevor Sie abzweigen.',
    sessionBusy: 'Session beschäftigt',
    branchStopCurrent: 'Stoppen Sie den laufenden Turn, bevor Sie von diesem Chat abzweigen.',
    branchNoText: 'Diese Nachricht hat keinen Text zum Abzweigen.',
    branchTitle: n => `Entwurf: Branch #${n}`,
    branchFailed: 'Abzweigen fehlgeschlagen',
    deleteFailed: 'Löschen fehlgeschlagen',
    archived: 'Archiviert',
    archiveFailed: 'Archivieren fehlgeschlagen',
    restored: 'Wiederhergestellt',
    unarchiveFailed: 'Archivierung aufheben fehlgeschlagen',
    cwdChangeFailed: 'Arbeitsverzeichnis-Änderung fehlgeschlagen',
    cwdStagedTitle: 'Arbeitsverzeichnis bereitgestellt',
    cwdStagedMessage: 'Starten Sie das Desktop-Backend neu, um cwd-Änderungen auf diese aktive Session anzuwenden.',
    modelSwitchConfirmBody: 'Dieser Modellwechsel muss bestätigt werden.',
    modelSwitchConfirmLabel: 'Trotzdem wechseln',
    modelSwitchConfirmTitle: (model: string) => `Zu ${model} wechseln?`,
    modelSwitchConfirmTitleFallback: 'Modell wechseln?',
    modelSwitchFailed: 'Modellwechsel fehlgeschlagen',
    modelSwitchKeepLabel: 'Aktuelles Modell behalten',
    modelSwitchStaleNotice: 'Auswahl geändert – der Modellwechsel wurde nicht angewendet.',
    hydrationSyncing: (profile: string) => `Synchronisiert ${profile}…`,
    sessionExported: 'Session exportiert',
    sessionExportFailed: 'Session konnte nicht exportiert werden',
    imageSaved: 'Bild gespeichert',
    downloadStarted: 'Download gestartet',
    restartToUseSaveImage: 'Starten Sie Hermes Desktop neu, um „Bild speichern“ zu verwenden.',
    restartToSaveImages: 'Starten Sie Hermes Desktop neu, um Bilder zu speichern',
    imageDownloadFailed: 'Bild-Download fehlgeschlagen',
    openImage: 'Bild öffnen',
    downloadImage: 'Bild herunterladen',
    savingImage: 'Speichert Bild',
    imagePreviewFailed: 'Bildvorschau fehlgeschlagen',
    imageAttach: 'Bild anhängen',
    imageWriteFailed: 'Bild konnte nicht auf die Festplatte geschrieben werden.',
    imageAttachFailed: 'Bild-Anhängen fehlgeschlagen',
    pastedContent: 'Eingefügter Inhalt',
    pasteAttachFailed: 'Eingefügter Text konnte nicht angehängt werden',
    attachImages: 'Bilder anhängen',
    clipboard: 'Zwischenablage',
    noClipboardImage: 'Kein Bild in der Zwischenablage',
    clipboardPasteFailed: 'Zwischenablage-Einfügen fehlgeschlagen',
    dropFiles: 'Dateien ablegen',
    handoff: {
      pickPlatform: 'Ziel wählen',
      success: platform => `Übergeben an ${platform}. Jederzeit hier fortsetzen.`,
      systemNote: platform => `↻ Übergeben an ${platform} — jederzeit hier fortsetzen.`,
      failed: error => `Übergabe fehlgeschlagen: ${error}`,
      timedOut: 'Zeitüberschreitung beim Warten auf das Gateway. Läuft `hermes gateway`?',
      startMessaging: 'Messaging starten'
    }
  },
  tips: {
    close: 'Diesen Tip nicht mehr zeigen',
    items: {
      'new-session': {
        title: 'Frisch loslegen',
        text: 'Ein neuer Chat bekommt seinen eigenen Context, sein eigenes Terminal und Arbeitsverzeichnis.'
      },
      skills: {
        title: 'Einmal beibringen',
        text: 'Skills sind Ordner mit Anweisungen, die Hermes lädt, wenn die Arbeit danach verlangt.'
      },
      messaging: {
        title: 'Hermes abseits Ihres Schreibtischs',
        text: 'Verbinden Sie Telegram, Discord, Slack und mehr – derselbe Agent, dasselbe Gedächtnis.'
      },
      artifacts: {
        title: 'Alles, was Hermes gemacht hat',
        text: 'Bilder, Dateien und Links aus jeder Session, an einem Ort indexiert.'
      },
      cron: {
        title: 'Arbeit, die von selbst läuft',
        text: 'Planen Sie einen Prompt stündlich, nächtlich oder nach einem Cron-Ausdruck.'
      },
      'command-palette': {
        title: 'Eine Box für alles',
        text: 'Sessions, Einstellungen, Skills und Befehle gehorchen alle der Palette.'
      },
      profiles: {
        title: 'Profile sind getrennt',
        text: 'Jedes ist sein eigenes Hermes — eigene Schlüssel, eigenes Gedächtnis, eigene Sessions.'
      },
      'composer-mentions': {
        title: 'Anhängen und befehlen',
        text: 'Geben Sie @ ein, um eine Datei in die Konversation zu holen, oder /, um einen Befehl auszuführen.'
      },
      'local-runtime-update': {
        title: 'Ein Update für die lokale Engine ist verfügbar',
        text: 'Aktualisieren Sie die Engine, die Ihre lokalen Modelle ausführt. Laufende lokale Anfragen können unterbrochen werden.',
        action: 'Jetzt aktualisieren'
      },
      'local-setup': {
        title: 'Dieses Gerät kann Modelle lokal ausführen',
        text: 'Ihre Hardware kann ein lokales Modell ausführen. Chats bleiben auf Ihrem Computer und kosten nichts.',
        action: 'Einrichten'
      },
      'right-pane': {
        title: 'Der Arbeitsbereich',
        text: 'Dateien, Terminal, Review und der In-App-Browser teilen sich die rechte Seite.'
      }
    }
  },
  errors: {
    genericFailure: 'Etwas ist schiefgelaufen',
    boundaryTitle: 'Etwas ist in der Oberfläche kaputtgegangen',
    boundaryDesc:
      'In dieser Ansicht ist ein unerwarteter Fehler aufgetreten. Ihre Chats und Einstellungen sind sicher.',
    boundaryDetails: 'Details',
    sendDiagnostics: 'Diagnosedaten senden',
    reloadWindow: 'Fenster neu laden',
    openLogs: 'Logs öffnen'
  },
  ui: {
    search: {
      clear: 'Suche löschen'
    },
    pagination: {
      label: 'Seitennummerierung',
      previous: 'Zurück',
      previousAria: 'Zur vorherigen Seite',
      next: 'Weiter',
      nextAria: 'Zur nächsten Seite'
    },
    sidebar: {
      title: 'Seitenleiste',
      description: 'Zeigt die mobile Sidebar an.',
      toggle: open => `${open ? 'Anzeigen' : 'Ausblenden'}: Sidebar`
    }
  }
} satisfies TranslationOverrides

export const de = defineLocale(deOverrides)
