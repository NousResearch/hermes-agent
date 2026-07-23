import { defineFieldCopy } from '@/app/settings/field-copy'

import { defineLocale } from './define-locale'

// Deutsche Locale für die Hermes Desktop-App.
//
// Übersetzungslinie: UI-Chrome auf Deutsch, etablierte Produkt-/Tech-Begriffe
// bleiben englisch (Hermes, Gateway, Skill, Toolset, Agent, Cron, Token, MCP,
// Provider, YOLO, Pet/Petdex …). `defineLocale()` merged über `en` — alles, was
// hier fehlt (z. B. die schema-getriebenen Settings-Feldlabels), fällt
// automatisch auf Englisch zurück.

export const de = defineLocale({
  common: {
    apply: 'Übernehmen',
    back: 'Zurück',
    save: 'Speichern',
    saving: 'Speichern…',
    cancel: 'Abbrechen',
    change: 'Ändern',
    choose: 'Auswählen',
    clear: 'Leeren',
    close: 'Schließen',
    collapse: 'Einklappen',
    confirm: 'Bestätigen',
    connect: 'Verbinden',
    connecting: 'Verbinde',
    continue: 'Weiter',
    copied: 'Kopiert',
    copy: 'Kopieren',
    copyFailed: 'Kopieren fehlgeschlagen',
    delete: 'Löschen',
    docs: 'Doku',
    done: 'Fertig',
    error: 'Fehler',
    failed: 'Fehlgeschlagen',
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
    on: 'An',
    off: 'Aus'
  },

  boot: {
    ready: 'Hermes Desktop ist bereit',
    desktopBootFailedWithMessage: message => `Desktop-Start fehlgeschlagen: ${message}`,
    steps: {
      connectingGateway: 'Verbinde Live-Desktop-Gateway',
      loadingSettings: 'Lade Hermes-Einstellungen',
      loadingSessions: 'Lade letzte Sitzungen',
      startingDesktopConnection: 'Starte Desktop-Verbindung',
      startingHermesDesktop: 'Starte Hermes Desktop…'
    },
    errors: {
      backgroundExited: 'Der Hermes-Hintergrundprozess wurde beendet.',
      backgroundExitedDuringStartup: 'Der Hermes-Hintergrundprozess wurde beim Start beendet.',
      backendStopped: 'Backend gestoppt',
      desktopBootFailed: 'Desktop-Start fehlgeschlagen',
      gatewaySignInRequired: 'Gateway-Anmeldung erforderlich',
      ipcBridgeUnavailable: 'Die Desktop-IPC-Bridge ist nicht verfügbar.'
    },
    failure: {
      title: 'Hermes konnte nicht starten',
      description:
        'Das Hintergrund-Gateway ist nicht hochgekommen. Probiere einen der Wiederherstellungsschritte unten. Nichts davon löscht deine Chats oder Einstellungen.',
      remoteTitle: 'Anmeldung am Remote-Gateway erforderlich',
      remoteDescription:
        'Deine Remote-Gateway-Sitzung ist abgelaufen. Melde dich erneut an, um die Verbindung wiederherzustellen. Nichts davon löscht deine Chats oder Einstellungen.',
      retry: 'Erneut versuchen',
      repairInstall: 'Installation reparieren',
      useLocalGateway: 'Lokales Gateway verwenden',
      gatewaySettings: 'Gateway-Einstellungen',
      back: 'Zurück',
      openLogs: 'Logs öffnen',
      repairHint:
        'Die Reparatur führt den Installer erneut aus und kann auf einem frischen Rechner einige Minuten dauern.',
      remoteSignInHint: signInLabel =>
        `Meldet dich von der gespeicherten Remote-Browser-Sitzung ab und öffnet anschließend ${signInLabel}. „Lokales Gateway verwenden" wechselt stattdessen zum mitgelieferten Backend.`,
      signOutAndSignIn: 'Abmelden & neu anmelden',
      remoteFailureHint:
        'Prüfe die Gateway-URL und melde dich unter Gateway-Einstellungen an, oder wechsle zum lokalen Gateway.',
      hideRecentLogs: 'Letzte Logs ausblenden',
      showRecentLogs: 'Letzte Logs anzeigen',
      signedInTitle: 'Angemeldet',
      signedInMessage: 'Verbinde erneut mit dem Remote-Gateway…',
      signInIncompleteTitle: 'Anmeldung unvollständig',
      signInIncompleteMessage: 'Das Login-Fenster wurde geschlossen, bevor die Authentifizierung abgeschlossen war.',
      signInFailed: 'Anmeldung fehlgeschlagen',
      signInToRemoteGateway: 'Am Remote-Gateway anmelden',
      signInWithProvider: provider => `Mit ${provider} anmelden`,
      identityProvider: 'deinem Identitätsanbieter'
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
    copyDetailFailed: 'Benachrichtigungsdetail konnte nicht kopiert werden',
    backendOutOfDateTitle: 'Backend veraltet',
    backendOutOfDateMessage:
      'Dein Hermes-Backend ist älter als dieser Desktop-Build und funktioniert möglicherweise nicht korrekt. Aktualisiere, um beide abzugleichen.',
    updateHermes: 'Hermes aktualisieren',
    updateReadyTitle: 'Update bereit',
    updateReadyMessage: count => `${count} neue Änderung${count === 1 ? '' : 'en'} verfügbar.`,
    seeWhatsNew: 'Was ist neu',
    errors: {
      elevenLabsNeedsKey: 'ElevenLabs STT benötigt ELEVENLABS_API_KEY.',
      elevenLabsRejectedKey: 'ElevenLabs hat den API-Key abgelehnt (401).',
      methodNotAllowed:
        'Das Desktop-Backend hat die Anfrage abgelehnt (405 Method Not Allowed). Starte Hermes Desktop neu.',
      microphonePermission: 'Mikrofon-Berechtigung wurde verweigert.',
      openaiRejectedApiKey: 'OpenAI hat den API-Key abgelehnt.',
      openaiRejectedApiKeyWithStatus: status => `OpenAI hat den API-Key abgelehnt (${status} invalid_api_key).`,
      openaiTtsNeedsKey: 'OpenAI TTS benötigt VOICE_TOOLS_OPENAI_KEY oder OPENAI_API_KEY.'
    },
    voice: {
      configureSpeechToText: 'Richte die Spracherkennung ein, um den Sprachmodus zu nutzen.',
      couldNotStartSession: 'Sprachsitzung konnte nicht gestartet werden',
      microphoneAccessDenied: 'Mikrofon-Zugriff verweigert.',
      microphoneConstraintsUnsupported: 'Die Mikrofon-Vorgaben werden von diesem Gerät nicht unterstützt.',
      microphoneFailed: 'Mikrofon fehlgeschlagen',
      microphoneInUse: 'Das Mikrofon wird bereits von einer anderen App verwendet.',
      microphonePermissionDenied: 'Mikrofon-Berechtigung wurde verweigert.',
      microphoneStartFailed: 'Mikrofon-Aufnahme konnte nicht gestartet werden.',
      microphoneUnsupported: 'Diese Laufzeitumgebung unterstützt keine Mikrofon-Aufnahme.',
      noMicrophone: 'Kein Mikrofon gefunden.',
      noSpeechDetected: 'Keine Sprache erkannt',
      playbackFailed: 'Sprachwiedergabe fehlgeschlagen',
      recordingFailed: 'Sprachaufnahme fehlgeschlagen',
      transcriptionFailed: 'Transkription fehlgeschlagen',
      transcriptionUnavailable: 'Die Sprachtranskription ist noch nicht verfügbar.',
      tryRecordingAgain: 'Versuche die Aufnahme erneut.',
      unavailable: 'Sprache nicht verfügbar'
    },
    native: {
      approvalTitle: 'Freigabe erforderlich',
      approveAction: 'Freigeben',
      rejectAction: 'Ablehnen',
      inputTitle: 'Eingabe erforderlich',
      inputBody: 'Hermes wartet auf deine Antwort.',
      turnDoneTitle: 'Hermes ist fertig',
      turnDoneBody: 'Die Antwort ist bereit.',
      turnErrorTitle: 'Durchlauf fehlgeschlagen',
      backgroundDoneTitle: 'Hintergrund-Task abgeschlossen',
      backgroundFailedTitle: 'Hintergrund-Task fehlgeschlagen'
    }
  },

  remoteDisplayBanner: {
    message: reason =>
      `Software-Rendering aktiv — Remote-Display erkannt (${reason}). GPU-Beschleunigung ist deaktiviert, um Flackern zu vermeiden.`
  },

  titlebar: {
    hideSidebar: 'Seitenleiste ausblenden',
    showSidebar: 'Seitenleiste anzeigen',
    search: 'Suchen',
    searchTitle: 'Sitzungen, Ansichten und Aktionen durchsuchen',
    swapSidebarSides: 'Seitenleisten-Seiten tauschen',
    swapSidebarSidesTitle: 'Sitzungen- und Datei-Browser-Seite tauschen',
    hideRightSidebar: 'Rechte Seitenleiste ausblenden',
    showRightSidebar: 'Rechte Seitenleiste anzeigen',
    muteHaptics: 'Haptik stummschalten',
    unmuteHaptics: 'Haptik aktivieren',
    openSettings: 'Einstellungen öffnen',
    openKeybinds: 'Tastenkürzel'
  },

  keybinds: {
    title: 'Tastenkürzel',
    subtitle: open => `Klicke auf ein Kürzel, um es neu zu belegen · ${open} öffnet dieses Panel erneut.`,
    rebind: 'Neu belegen',
    reset: 'Auf Standard zurücksetzen',
    resetAll: 'Alle zurücksetzen',
    pressKey: 'Taste drücken…',
    set: 'gesetzt',
    conflictWith: label => `Ebenfalls belegt mit „${label}"`,
    categories: {
      composer: 'Eingabe',
      profiles: 'Profile',
      session: 'Sitzung',
      navigation: 'Navigation',
      view: 'Ansicht'
    },
    actions: {
      'keybinds.openPanel': 'Tastenkürzel öffnen',
      'nav.commandPalette': 'Befehlspalette öffnen',
      'nav.commandCenter': 'Command Center öffnen',
      'nav.settings': 'Einstellungen öffnen',
      'nav.profiles': 'Profile öffnen',
      'nav.skills': 'Skills öffnen',
      'nav.messaging': 'Messaging öffnen',
      'nav.artifacts': 'Artefakte öffnen',
      'nav.cron': 'Geplante Jobs öffnen',
      'nav.agents': 'Agents öffnen',
      'session.new': 'Neue Sitzung',
      'session.newWindow': 'Neue Sitzung im Fenster',
      'session.next': 'Nächste Sitzung',
      'session.prev': 'Vorherige Sitzung',
      'session.slot.1': 'Zu letzter Sitzung 1 wechseln',
      'session.slot.2': 'Zu letzter Sitzung 2 wechseln',
      'session.slot.3': 'Zu letzter Sitzung 3 wechseln',
      'session.slot.4': 'Zu letzter Sitzung 4 wechseln',
      'session.slot.5': 'Zu letzter Sitzung 5 wechseln',
      'session.slot.6': 'Zu letzter Sitzung 6 wechseln',
      'session.slot.7': 'Zu letzter Sitzung 7 wechseln',
      'session.slot.8': 'Zu letzter Sitzung 8 wechseln',
      'session.slot.9': 'Zu letzter Sitzung 9 wechseln',
      'session.focusSearch': 'Sitzungen durchsuchen',
      'session.togglePin': 'Aktuelle Sitzung anpinnen / lösen',
      'composer.focus': 'Eingabe fokussieren',
      'composer.modelPicker': 'Modell-Auswahl öffnen',
      'view.toggleSidebar': 'Sitzungs-Seitenleiste umschalten',
      'view.toggleRightSidebar': 'Datei-Browser umschalten',
      'view.showFiles': 'Datei-Browser anzeigen',
      'view.showTerminal': 'Terminal anzeigen',
      'view.terminalSelection': 'Terminal-Auswahl an Eingabe senden',
      'view.closePreviewTab': 'Vorschau-Tab schließen',
      'view.flipPanes': 'Seitenleisten-Seiten tauschen',
      'appearance.toggleMode': 'Hell / Dunkel umschalten',
      'profile.default': 'Zum Standard-Profil wechseln',
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
      'composer.newline': 'Zeilenumbruch einfügen',
      'composer.steer': 'Laufenden Durchlauf steuern',
      'composer.sendQueued': 'Nächsten Durchlauf aus der Warteschlange senden',
      'composer.mention': 'Dateien, Ordner, URLs referenzieren',
      'composer.slash': 'Slash-Befehlspalette',
      'composer.help': 'Schnellhilfe',
      'composer.history': 'Popover / Verlauf durchblättern',
      'composer.cancel': 'Popover schließen · Durchlauf abbrechen'
    }
  },

  language: {
    label: 'Sprache',
    description: 'Wähle die Sprache für die Desktop-Oberfläche.',
    saving: 'Sprache wird gespeichert…',
    saveError: 'Sprachaktualisierung fehlgeschlagen',
    switchTo: 'Sprache wechseln',
    searchPlaceholder: 'Sprachen durchsuchen…',
    noResults: 'Keine Sprachen gefunden'
  },

  settings: {
    closeSettings: 'Einstellungen schließen',
    exportConfig: 'Konfig exportieren',
    importConfig: 'Konfig importieren',
    resetToDefaults: 'Auf Standard zurücksetzen',
    resetConfirm: 'Alle Einstellungen auf die Hermes-Standardwerte zurücksetzen?',
    exportFailed: 'Export fehlgeschlagen',
    resetFailed: 'Zurücksetzen fehlgeschlagen',
    nav: {
      providers: 'Provider',
      providerAccounts: 'Konten',
      providerApiKeys: 'API-Keys',
      gateway: 'Gateway',
      apiKeys: 'Tools & Keys',
      keysTools: 'Tools',
      keysSettings: 'Einstellungen',
      mcp: 'MCP',
      archivedChats: 'Archivierte Chats',
      about: 'Über',
      notifications: 'Benachrichtigungen'
    },
    notifications: {
      title: 'Benachrichtigungen',
      intro:
        'Native Desktop-Benachrichtigungen, getrennt von den In-App-Hinweisen. Diese sind gerätelokal — jeder Rechner behält seine eigenen Einstellungen.',
      enableAll: 'Benachrichtigungen aktivieren',
      enableAllDesc: 'Hauptschalter. Schalte ihn aus, um alle Benachrichtigungen unten stummzuschalten.',
      focusedHint: 'Abschluss-Hinweise erscheinen nur, während Hermes im Hintergrund läuft.',
      kinds: {
        approval: {
          label: 'Freigabe erforderlich',
          description: 'Ein Befehl wartet darauf, dass du ihn freigibst oder ablehnst.'
        },
        input: {
          label: 'Eingabe erforderlich',
          description: 'Hermes hat eine Frage gestellt oder braucht ein Passwort bzw. ein Secret.'
        },
        turnDone: {
          label: 'Antwort bereit',
          description: 'Ein Durchlauf wurde abgeschlossen, während Hermes im Hintergrund war.'
        },
        turnError: {
          label: 'Durchlauf fehlgeschlagen',
          description: 'Ein Durchlauf endete mit einem Fehler.'
        },
        backgroundDone: {
          label: 'Hintergrund-Task abgeschlossen',
          description: 'Ein in den Hintergrund verschobener Terminal-Befehl wurde abgeschlossen.'
        }
      },
      test: 'Test-Benachrichtigung senden',
      testTitle: 'Hermes',
      testBody: 'Benachrichtigungen funktionieren.',
      testSent:
        'Test gesendet. Falls nichts erscheint, prüfe die Benachrichtigungs-Berechtigungen deines Systems sowie den Fokus-/Nicht-stören-Modus.',
      testUnsupported: 'Dieses System unterstützt keine nativen Benachrichtigungen.',
      completionSoundTitle: 'Abschluss-Sound',
      completionSoundDesc: 'Wird abgespielt, wenn ein Agent-Durchlauf endet. Wähle eine Vorlage und höre sie hier an.',
      completionSoundPreview: 'Vorhören'
    },
    sections: {
      model: 'Modell',
      chat: 'Chat',
      appearance: 'Darstellung',
      workspace: 'Arbeitsbereich',
      safety: 'Sicherheit',
      memory: 'Memory & Kontext',
      voice: 'Sprache',
      advanced: 'Erweitert'
    },
    searchPlaceholder: {
      about: 'Über Hermes Desktop',
      config: 'Einstellungen durchsuchen…',
      gateway: 'Gateway-Verbindung…',
      keys: 'API-Keys durchsuchen…',
      mcp: 'MCP-Server durchsuchen…',
      sessions: 'Archivierte Sitzungen durchsuchen…'
    },
    modeOptions: {
      light: { label: 'Hell', description: 'Helle Desktop-Oberflächen' },
      dark: { label: 'Dunkel', description: 'Blendarmer Arbeitsbereich' },
      system: { label: 'System', description: 'Der OS-Darstellung folgen' }
    },
    appearance: {
      title: 'Darstellung',
      intro:
        'Dies sind reine Desktop-Anzeigeeinstellungen. Der Modus steuert die Helligkeit; das Theme steuert die Akzentfarben und die Gestaltung der Chat-Oberfläche.',
      colorMode: 'Farbmodus',
      colorModeDesc: 'Wähle einen festen Modus oder lass Hermes deiner Systemeinstellung folgen.',
      toolViewTitle: 'Anzeige von Tool-Aufrufen',
      toolViewDesc: 'Produkt blendet rohe Tool-Daten aus; Technisch zeigt die vollständige Ein-/Ausgabe.',
      translucencyTitle: 'Fenster-Transluzenz',
      translucencyDesc: 'Sieh deinen Desktop durch das gesamte Fenster. Nur macOS und Windows.',
      product: 'Produkt',
      productDesc: 'Benutzerfreundliche Tool-Aktivität mit knappen Zusammenfassungen.',
      technical: 'Technisch',
      technicalDesc: 'Inklusive roher Tool-Argumente/-Ergebnisse und Low-Level-Details.',
      themeTitle: 'Theme',
      themeDesc: 'Nur Desktop-Paletten. Der gewählte Modus wird darüber angewendet.',
      themeProfileNote: profile => `Für das Profil „${profile}" gespeichert — jedes Profil behält sein eigenes Theme.`,
      installTitle: 'Aus VS Code installieren',
      installDesc:
        'Füge eine Marketplace-Extension-ID ein (z. B. dracula-theme.theme-dracula), um deren Farb-Theme in eine Desktop-Palette zu konvertieren.',
      installPlaceholder: 'publisher.extension',
      installButton: 'Installieren',
      installing: 'Installiere…',
      installError: 'Dieses Theme konnte nicht installiert werden.',
      installed: name => `„${name}" installiert.`,
      removeTheme: 'Theme entfernen',
      importedBadge: 'Importiert',
      pet: {
        title: 'Pet',
        intro:
          'Adoptiere ein animiertes Petdex-Maskottchen, das über der App schwebt und auf das reagiert, was Hermes gerade tut — es rennt, während Tools laufen, jubelt bei Erfolg und schmollt bei Fehlern.',
        restartHint:
          'Pets brauchen einen kurzen Neustart — die laufende App wurde gestartet, bevor dieses Feature hinzukam. Beende Hermes, öffne es erneut und komm hierher zurück.',
        on: 'An',
        off: 'Aus',
        scaleTitle: 'Größe',
        scaleDesc: 'Ändere die Größe des schwebenden Maskottchens. Wird überall sofort übernommen.',
        chooseTitle: 'Pet auswählen',
        chooseDesc: 'Bei der Auswahl wird es (falls nötig) installiert und aktiviert.',
        searchPlaceholder: 'Pets durchsuchen…',
        unreachable: 'Die Petdex-Galerie war nicht erreichbar. Prüfe deine Verbindung und öffne diese Seite erneut.',
        noMatch: query => `Keine Pets passen zu „${query}".`,
        installedTag: 'installiert',
        countCapped: (cap, total) => `Zeige ${cap} von ${total} — tippe, um einzugrenzen.`,
        count: n => `${n} ${n === 1 ? 'Pet' : 'Pets'}.`,
        uninstall: name => `${name} deinstallieren`,
        adoptFailed: slug => `${slug} konnte nicht adoptiert werden`,
        uninstallFailed: slug => `${slug} konnte nicht deinstalliert werden`,
        noneAvailable: 'Gerade keine Pets zum Aktivieren verfügbar.',
        turnOnFailed: 'Das Pet konnte nicht aktiviert werden.',
        turnOffFailed: 'Das Pet konnte nicht deaktiviert werden.'
      }
    },
    fieldLabels: defineFieldCopy({
      model: 'Standard-Modell',
      modelContextLength: 'Kontextfenster',
      fallbackProviders: 'Fallback-Modelle',
      toolsets: 'Aktivierte Toolsets',
      timezone: 'Zeitzone',
      display: {
        personality: 'Persönlichkeit',
        showReasoning: 'Reasoning-Blöcke'
      },
      agent: {
        maxTurns: 'Max. Agent-Schritte',
        imageInputMode: 'Bildanhänge',
        apiMaxRetries: 'API-Wiederholungen',
        serviceTier: 'Service-Tier',
        toolUseEnforcement: 'Tool-Use-Erzwingung'
      },
      terminal: {
        cwd: 'Arbeitsverzeichnis',
        backend: 'Ausführungs-Backend',
        timeout: 'Befehls-Timeout',
        persistentShell: 'Persistente Shell',
        envPassthrough: 'Umgebungs-Durchreichung',
        dockerImage: 'Docker-Image',
        singularityImage: 'Singularity-Image',
        modalImage: 'Modal-Image',
        daytonaImage: 'Daytona-Image'
      },
      fileReadMaxChars: 'Datei-Leselimit',
      toolOutput: {
        maxBytes: 'Terminal-Ausgabelimit',
        maxLines: 'Datei-Seitenlimit',
        maxLineLength: 'Zeilenlängen-Limit'
      },
      codeExecution: {
        mode: 'Code-Ausführungsmodus'
      },
      approvals: {
        mode: 'Freigabemodus',
        timeout: 'Freigabe-Timeout',
        mcpReloadConfirm: 'MCP-Neuladen bestätigen'
      },
      commandAllowlist: 'Befehls-Allowlist',
      security: {
        redactSecrets: 'Secrets schwärzen',
        allowPrivateUrls: 'Private URLs erlauben'
      },
      browser: {
        allowPrivateUrls: 'Private Browser-URLs',
        autoLocalForPrivateUrls: 'Lokaler Browser für private URLs'
      },
      checkpoints: {
        enabled: 'Datei-Checkpoints',
        maxSnapshots: 'Checkpoint-Limit'
      },
      voice: {
        recordKey: 'Sprach-Kürzel',
        maxRecordingSeconds: 'Max. Aufnahmelänge',
        autoTts: 'Antworten vorlesen'
      },
      stt: {
        enabled: 'Spracherkennung',
        provider: 'Spracherkennungs-Provider',
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
          tagAudioEvents: 'Audio-Events markieren',
          diarize: 'Sprecher-Diarisierung'
        }
      },
      tts: {
        provider: 'Text-To-Speech-Provider',
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
          voiceId: 'xAI-(Grok)-Stimme',
          language: 'xAI-Sprache'
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
        }
      },
      memory: {
        memoryEnabled: 'Persistentes Memory',
        userProfileEnabled: 'Benutzerprofil',
        memoryCharLimit: 'Memory-Budget',
        userCharLimit: 'Profil-Budget',
        provider: 'Memory-Provider'
      },
      context: {
        engine: 'Kontext-Engine'
      },
      compression: {
        enabled: 'Auto-Komprimierung',
        threshold: 'Komprimierungs-Schwelle',
        targetRatio: 'Komprimierungs-Ziel',
        protectLastN: 'Geschützte letzte Nachrichten'
      },
      delegation: {
        model: 'Subagent-Modell',
        provider: 'Subagent-Provider',
        maxIterations: 'Subagent-Durchlauf-Limit',
        maxConcurrentChildren: 'Parallele Subagenten',
        childTimeoutSeconds: 'Subagent-Timeout',
        reasoningEffort: 'Subagent-Reasoning-Aufwand'
      },
      updates: {
        nonInteractiveLocalChanges: 'In-App-Update: lokale Änderungen'
      }
    }),
    fieldDescriptions: defineFieldCopy({
      model: 'Wird für neue Chats verwendet, sofern du in der Eingabe kein anderes Modell wählst.',
      modelContextLength: 'Auf 0 lassen, um das erkannte Kontextfenster des gewählten Modells zu nutzen.',
      fallbackProviders: 'Backup-Einträge provider:model, die versucht werden, falls das Standard-Modell fehlschlägt.',
      display: {
        personality: 'Standard-Assistenten-Stil für neue Sitzungen.',
        showReasoning: 'Reasoning-Abschnitte anzeigen, wenn das Backend sie liefert.'
      },
      timezone: 'Wird genutzt, wenn Hermes lokalen Zeitkontext braucht. Leer nutzt die System-Zeitzone.',
      agent: {
        imageInputMode: 'Steuert, wie Bildanhänge an das Modell gesendet werden.',
        maxTurns: 'Obergrenze für Tool-Aufruf-Durchläufe, bevor Hermes einen Lauf stoppt.'
      },
      terminal: {
        cwd: 'Standard-Projektordner für Tool- und Terminal-Arbeit.',
        persistentShell: 'Shell-Zustand zwischen Befehlen behalten, wenn das Backend es unterstützt.',
        envPassthrough: 'Umgebungsvariablen, die in die Tool-Ausführung durchgereicht werden.',
        dockerImage: 'Container-Image, das verwendet wird, wenn das Ausführungs-Backend Docker ist.',
        singularityImage: 'Image, das verwendet wird, wenn das Ausführungs-Backend Singularity ist.',
        modalImage: 'Image, das verwendet wird, wenn das Ausführungs-Backend Modal ist.',
        daytonaImage: 'Image, das verwendet wird, wenn das Ausführungs-Backend Daytona ist.'
      },
      codeExecution: {
        mode: 'Wie streng die Code-Ausführung auf das aktuelle Projekt eingegrenzt wird.'
      },
      fileReadMaxChars: 'Maximale Zeichenzahl, die Hermes aus einer einzelnen Datei-Anfrage lesen kann.',
      approvals: {
        mode: 'Wie Hermes mit Befehlen umgeht, die eine explizite Freigabe brauchen.',
        timeout: 'Wie lange Freigabe-Abfragen warten, bevor sie ablaufen.'
      },
      security: {
        redactSecrets: 'Erkannte Secrets nach Möglichkeit aus modell-sichtbarem Inhalt verbergen.'
      },
      checkpoints: {
        enabled: 'Rollback-Snapshots vor Datei-Änderungen erstellen.'
      },
      memory: {
        memoryEnabled: 'Dauerhafte Erinnerungen speichern, die künftigen Sitzungen helfen können.',
        userProfileEnabled: 'Ein kompaktes Profil der Nutzerpräferenzen pflegen.'
      },
      context: {
        engine: 'Strategie für die Verwaltung langer Unterhaltungen nahe am Kontextlimit.'
      },
      compression: {
        enabled: 'Älteren Kontext zusammenfassen, wenn Unterhaltungen groß werden.'
      },
      voice: {
        autoTts: 'Assistenten-Antworten automatisch vorlesen.'
      },
      tts: {
        xai: {
          voiceId: 'xAI-Voice-ID (z. B. eve) oder eine eigene Voice-ID.',
          language: 'Gesprochener Sprachcode, z. B. en.'
        },
        neutts: {
          device: 'Lokales Inferenz-Gerät für NeuTTS.'
        }
      },
      stt: {
        enabled: 'Lokale oder provider-gestützte Sprachtranskription aktivieren.',
        elevenlabs: {
          languageCode: 'Optionaler ISO-639-3-Sprachcode. Leer lässt ElevenLabs automatisch erkennen.'
        }
      },
      updates: {
        nonInteractiveLocalChanges:
          'Wenn Hermes sich aus der App aktualisiert (ohne Terminal-Abfrage), lokale Quellcode-Änderungen behalten (stash) oder verwerfen (discard). Terminal-Updates fragen immer nach.'
      }
    }),
    about: {
      heading: 'Hermes Desktop',
      version: value => `Version ${value}`,
      versionUnavailable: 'Version nicht verfügbar',
      updates: 'Updates',
      checkNow: 'Jetzt prüfen',
      checking: 'Prüfe…',
      seeWhatsNew: 'Was ist neu',
      updateNow: 'Jetzt aktualisieren',
      releaseNotes: 'Release Notes',
      onLatest: 'Du nutzt die neueste Version.',
      installing: 'Ein Update wird gerade installiert.',
      cantUpdate: 'Dieser Build kann sich nicht aus der App heraus selbst aktualisieren.',
      cantReach: 'Der Update-Server war nicht erreichbar.',
      tapCheck: 'Tippe auf „Jetzt prüfen", um nach Updates zu suchen.',
      updateReady: count => `Ein neues Update ist bereit (${count} Änderung${count === 1 ? '' : 'en'} enthalten).`,
      lastChecked: age => `Zuletzt geprüft ${age}`,
      justNowSuffix: ' · gerade eben',
      automaticUpdates: 'Automatische Updates',
      automaticUpdatesDesc:
        'Hermes sucht im Hintergrund automatisch nach Updates und meldet sich, sobald eines bereit ist.',
      branchCommit: (branch, commit) => `Branch ${branch} · Commit ${commit}`,
      never: 'nie',
      justNow: 'gerade eben',
      minAgo: count => `vor ${count} Min`,
      hoursAgo: count => `vor ${count} Std`,
      daysAgo: count => `vor ${count} Tagen`
    },
    config: {
      none: 'Keine',
      noneParen: '(keine)',
      notSet: 'Nicht gesetzt',
      commaSeparated: 'kommagetrennte Werte',
      loading: 'Lade Hermes-Konfiguration…',
      emptyTitle: 'Nichts zu konfigurieren',
      emptyDesc: 'Dieser Bereich hat keine einstellbaren Optionen.',
      failedLoad: 'Einstellungen konnten nicht geladen werden',
      autosaveFailed: 'Automatisches Speichern fehlgeschlagen',
      imported: 'Konfig importiert',
      invalidJson: 'Ungültiges Konfig-JSON'
    },
    credentials: {
      pasteKey: 'Key einfügen',
      pasteLabelKey: label => `${label}-Key einfügen`,
      optional: 'Optional',
      enterValueFirst: 'Gib zuerst einen Wert ein.',
      couldNotSave: 'Zugangsdaten konnten nicht gespeichert werden.',
      remove: 'Entfernen',
      getKey: 'Key holen',
      saving: 'Speichern'
    },
    envActions: {
      actionsFor: label => `Aktionen für ${label}`,
      credentialActions: 'Zugangsdaten-Aktionen',
      docs: 'Doku',
      hideValue: 'Wert verbergen',
      revealValue: 'Wert anzeigen',
      replace: 'Ersetzen',
      set: 'Setzen',
      clear: 'Leeren'
    },
    gateway: {
      loading: 'Lade Gateway-Einstellungen…',
      unavailableTitle: 'Gateway-Einstellungen nicht verfügbar',
      unavailableDesc: 'Die Desktop-IPC-Bridge stellt keine Gateway-Einstellungen bereit.',
      title: 'Gateway-Verbindung',
      envOverride: 'Env-Override',
      intro:
        'Hermes Desktop startet standardmäßig sein eigenes lokales Gateway. Nutze ein Remote-Gateway, wenn diese App ein bereits laufendes Hermes-Backend auf einem anderen Rechner oder hinter einem vertrauenswürdigen Proxy steuern soll. Wähle unten ein Profil, um ihm einen eigenen Remote-Host zu geben.',
      appliesTo: 'Gilt für',
      allProfiles: 'Alle Profile',
      defaultConnection: 'Standardverbindung für jedes Profil ohne eigenen Override.',
      profileConnection: profile =>
        `Verbindung, die nur genutzt wird, wenn „${profile}" das aktive Profil ist. Setze sie auf Lokal, um den Standard zu übernehmen.`,
      envOverrideTitle: 'Umgebungsvariablen steuern diese Desktop-Sitzung.',
      envOverrideDesc:
        'Hebe HERMES_DESKTOP_REMOTE_URL und HERMES_DESKTOP_REMOTE_TOKEN auf, um die gespeicherte Einstellung unten zu verwenden.',
      localTitle: 'Lokales Gateway',
      localDesc: 'Startet ein privates Hermes-Backend auf localhost. Das ist der Standard und funktioniert offline.',
      remoteTitle: 'Remote-Gateway',
      remoteDesc:
        'Verbindet diese Desktop-Hülle mit einem entfernten Hermes-Backend. Gehostete Gateways nutzen OAuth oder Benutzername und Passwort; selbst gehostete ggf. ein Session-Token.',
      remoteUrlTitle: 'Remote-URL',
      remoteUrlDesc: 'Basis-URL für das entfernte Dashboard-Backend. Pfad-Präfixe werden unterstützt, z. B. /hermes.',
      probing: 'Prüfe, wie sich dieses Gateway authentifiziert…',
      probeError:
        'Dieses Gateway war noch nicht erreichbar. Prüfe die URL — die Auth-Methode erscheint, sobald es antwortet.',
      signedIn: 'Angemeldet',
      signIn: 'Anmelden',
      signOut: 'Abmelden',
      signInWith: provider => `Mit ${provider} anmelden`,
      authTitle: 'Authentifizierung',
      authSignedInPassword:
        'Dieses Gateway nutzt Benutzername und Passwort. Du bist angemeldet; die Sitzung wird automatisch erneuert.',
      authSignedInOauth: 'Dieses Gateway nutzt OAuth. Du bist angemeldet; die Sitzung wird automatisch erneuert.',
      authNeedsPassword:
        'Dieses Gateway nutzt Benutzername und Passwort. Melde dich an, um diese Desktop-App zu autorisieren.',
      authNeedsOauth: provider =>
        `Dieses Gateway nutzt OAuth. Melde dich mit ${provider} an, um diese Desktop-App zu autorisieren.`,
      tokenTitle: 'Session-Token',
      tokenDesc:
        'Das Dashboard-Session-Token für REST- und WebSocket-Zugriff. Leer lassen, um das gespeicherte Token zu behalten.',
      existingToken: value => `Vorhandenes Token ${value}`,
      savedToken: 'gespeichert',
      pasteSessionToken: 'Session-Token einfügen',
      testRemote: 'Remote testen',
      saveForRestart: 'Für nächsten Neustart speichern',
      saveAndReconnect: 'Speichern und neu verbinden',
      diagnostics: 'Diagnose',
      diagnosticsDesc: 'Zeigt desktop.log im Dateimanager — hilfreich, wenn das Gateway nicht startet.',
      openLogs: 'Logs öffnen',
      incompleteTitle: 'Remote-Gateway unvollständig',
      incompleteSignIn: 'Gib eine Remote-URL ein und melde dich an, bevor du auf Remote wechselst.',
      incompleteToken: 'Gib eine Remote-URL und ein Session-Token ein, bevor du auf Remote wechselst.',
      incompleteSignInTest: 'Gib eine Remote-URL ein und melde dich an, bevor du testest.',
      incompleteTokenTest: 'Gib eine Remote-URL und ein Session-Token ein, bevor du testest.',
      enterUrlFirst: 'Gib zuerst eine Remote-URL ein.',
      restartingTitle: 'Gateway-Verbindung wird neu gestartet',
      savedTitle: 'Gateway-Einstellungen gespeichert',
      restartingMessage: 'Hermes Desktop verbindet sich mit den gespeicherten Einstellungen neu.',
      savedMessage: 'Für den nächsten Neustart gespeichert.',
      connectedTo: (baseUrl, version) => `Verbunden mit ${baseUrl}${version ? ` · Hermes ${version}` : ''}`,
      reachableTitle: 'Remote-Gateway erreichbar',
      signedOutTitle: 'Abgemeldet',
      signedOutMessage: 'Die Remote-Gateway-Sitzung wurde geleert.',
      failedLoad: 'Gateway-Einstellungen konnten nicht geladen werden',
      signInFailed: 'Anmeldung fehlgeschlagen',
      signOutFailed: 'Abmeldung fehlgeschlagen',
      testFailed: 'Remote-Gateway-Test fehlgeschlagen',
      applyFailed: 'Gateway-Einstellungen konnten nicht angewendet werden',
      saveFailed: 'Gateway-Einstellungen konnten nicht gespeichert werden'
    },
    keys: {
      loading: 'Lade API-Keys und Zugangsdaten…',
      failedLoad: 'API-Keys konnten nicht geladen werden',
      empty: 'In dieser Kategorie ist noch nichts konfiguriert.'
    },
    mcp: {
      loading: 'Lade MCP-Server…',
      failedLoad: 'MCP-Konfig konnte nicht geladen werden',
      nameRequiredTitle: 'Name erforderlich',
      nameRequiredMessage: 'Gib diesem MCP-Server einen Konfig-Schlüssel.',
      objectRequired: 'Die Server-Konfig muss ein JSON-Objekt sein',
      invalidJson: 'Ungültiges MCP-JSON',
      saveFailed: 'Speichern fehlgeschlagen',
      removeFailed: 'Entfernen fehlgeschlagen',
      gatewayUnavailableTitle: 'Gateway nicht verfügbar',
      gatewayUnavailableMessage: 'Verbinde das Gateway neu, bevor du MCP neu lädst.',
      reloadedTitle: 'MCP-Tools neu geladen',
      reloadedMessage: 'Neue Tool-Schemas gelten ab dem nächsten Durchlauf.',
      reloadFailed: 'MCP-Neuladen fehlgeschlagen',
      savedTitle: 'MCP-Server gespeichert',
      savedMessage: name => `${name} wird nach dem MCP-Neuladen wirksam.`,
      newServer: 'Neuer Server',
      reload: 'MCP neu laden',
      reloading: 'Lade neu…',
      emptyTitle: 'Keine MCP-Server',
      emptyDesc: 'Füge einen stdio- oder HTTP-Server hinzu, um MCP-Tools bereitzustellen.',
      disabled: 'deaktiviert',
      editServer: 'Server bearbeiten',
      name: 'Name',
      serverJson: 'Server-JSON',
      remove: 'Entfernen',
      saveServer: 'Server speichern'
    },
    model: {
      loading: 'Lade Modell-Konfiguration…',
      appliesDesc:
        'Gilt für neue Sitzungen. Nutze die Modell-Auswahl in der Eingabe, um das aktive Chat-Modell direkt zu wechseln.',
      provider: 'Provider',
      model: 'Modell',
      applying: 'Wende an…',
      defaultsLabel: 'Standards',
      reasoning: 'Reasoning',
      reasoningOff: 'Aus',
      defaultsFailed: 'Modell-Standards konnten nicht gespeichert werden',
      auxiliaryTitle: 'Hilfsmodelle',
      resetAllToMain: 'Alle auf Hauptmodell zurücksetzen',
      auxiliaryDesc:
        'Hilfsaufgaben laufen standardmäßig auf dem Hauptmodell. Weise einer Aufgabe ein eigenes Modell zu, um das zu überschreiben.',
      setToMain: 'Auf Hauptmodell setzen',
      change: 'Ändern',
      autoUseMain: 'auto · Hauptmodell verwenden',
      providerDefault: '(Provider-Standard)',
      tasks: {
        vision: { label: 'Vision', hint: 'Bildanalyse' },
        web_extract: { label: 'Web-Extraktion', hint: 'Seiten-Zusammenfassung' },
        compression: { label: 'Komprimierung', hint: 'Kontext-Verdichtung' },
        skills_hub: { label: 'Skills-Hub', hint: 'Skill-Suche' },
        approval: { label: 'Freigabe', hint: 'Smarte Auto-Freigabe' },
        mcp: { label: 'MCP', hint: 'MCP-Tool-Routing' },
        title_generation: { label: 'Titel-Generierung', hint: 'Sitzungstitel' },
        curator: { label: 'Curator', hint: 'Skill-Nutzungs-Review' }
      }
    },
    providers: {
      connectAccount: 'Konto verbinden',
      haveApiKey: 'Hast du stattdessen einen API-Key?',
      intro:
        'Melde dich mit einem Abo an — kein API-Key zum Kopieren. Hermes führt die Browser-Anmeldung direkt hier in der App für dich aus.',
      connected: 'Verbunden',
      collapse: 'Einklappen',
      connectAnother: 'Weiteren Provider verbinden',
      otherProviders: 'Andere Provider',
      disconnect: 'Trennen',
      disconnectInTerminal: 'Trennen (führt den Entfernungsbefehl im Terminal aus)',
      removeConfirm: provider => `${provider} entfernen?`,
      removeExternalGeneric: provider => `${provider} wird über seine eigene CLI verwaltet — entferne es dort.`,
      removeKeyManaged: provider => `${provider} ist über einen API-Key konfiguriert. Entferne es unter API-Keys.`,
      removeTerminalConfirm: (provider, command) =>
        `${provider} trennen? Dazu wird „${command}" im Terminal ausgeführt, um die Zugangsdaten zu löschen.`,
      removeTerminalRunning: provider => `Führe ${provider}-Trennung im Terminal aus…`,
      removedTitle: 'Konto entfernt',
      removedMessage: provider => `${provider} wurde entfernt.`,
      failedRemove: provider => `${provider} konnte nicht entfernt werden`,
      noProviderKeys: 'Keine Provider-API-Keys verfügbar.',
      searchKeys: 'Provider durchsuchen…',
      noKeysMatch: 'Keine Provider passen zu deiner Suche.',
      loading: 'Lade Provider…'
    },
    sessions: {
      loading: 'Lade archivierte Sitzungen…',
      archivedTitle: 'Archivierte Sitzungen',
      archivedIntro:
        'Archivierte Chats sind aus der Seitenleiste ausgeblendet, behalten aber alle Nachrichten. Strg/⌘-Klick auf einen Chat in der Seitenleiste archiviert ihn.',
      emptyArchivedTitle: 'Nichts archiviert',
      emptyArchivedDesc: 'Archiviere einen Chat, um ihn hier auszublenden.',
      unarchive: 'Aus Archiv holen',
      deletePermanently: 'Endgültig löschen',
      messages: count => `${count} ${count === 1 ? 'Nachricht' : 'Nachrichten'}`,
      restored: 'Wiederhergestellt',
      deleteConfirm: title => `„${title}" endgültig löschen? Das kann nicht rückgängig gemacht werden.`,
      defaultDirTitle: 'Standard-Projektverzeichnis',
      defaultDirDesc:
        'Neue Sitzungen starten in diesem Ordner, sofern du keinen anderen wählst. Leer lassen, um dein Home-Verzeichnis zu nutzen.',
      defaultDirUpdated:
        'Standard-Projektverzeichnis aktualisiert — starte einen neuen Chat (Strg/⌘+N), damit es wirksam wird',
      defaultsTo: label => `Standard ist ${label}.`,
      change: 'Ändern',
      choose: 'Auswählen',
      clear: 'Leeren',
      notSet: 'Nicht gesetzt',
      failedLoad: 'Archivierte Sitzungen konnten nicht geladen werden',
      unarchiveFailed: 'Aus-Archiv-holen fehlgeschlagen',
      deleteFailed: 'Löschen fehlgeschlagen',
      updateDirFailed: 'Standardverzeichnis konnte nicht aktualisiert werden',
      clearDirFailed: 'Standardverzeichnis konnte nicht geleert werden'
    },
    toolsets: {
      loadingConfig: 'Lade Konfiguration',
      savedTitle: 'Zugangsdaten gespeichert',
      savedMessage: key => `${key} aktualisiert.`,
      removedTitle: 'Zugangsdaten entfernt',
      removedMessage: key => `${key} entfernt.`,
      failedSave: key => `${key} konnte nicht gespeichert werden`,
      failedRemove: key => `${key} konnte nicht entfernt werden`,
      failedReveal: key => `${key} konnte nicht angezeigt werden`,
      removeConfirm: key => `${key} aus der .env entfernen?`,
      set: 'Setzen',
      notSet: 'Nicht gesetzt',
      selectedTitle: 'Provider ausgewählt',
      selectedMessage: provider => `${provider} ist jetzt aktiv.`,
      failedSelect: provider => `${provider} konnte nicht ausgewählt werden`,
      failedLoad: 'Tool-Konfiguration konnte nicht geladen werden',
      noProviderOptions:
        'Dieses Toolset hat keine Provider-Optionen — aktiviere es und es funktioniert mit deinem aktuellen Setup.',
      noProviders: 'Für dieses Toolset sind gerade keine Provider verfügbar.',
      ready: 'Bereit',
      nousIncluded: 'In einem Nous-Abo enthalten — melde dich im Nous Portal an, um es zu aktivieren.',
      noApiKeyRequired: 'Kein API-Key erforderlich.',
      postSetupHint: step =>
        `Dieses Backend benötigt eine einmalige Installation (${step}). Läuft auf diesem Rechner — kann ein paar Minuten dauern.`,
      postSetupRun: 'Setup ausführen',
      postSetupRunning: 'Installiere…',
      postSetupStarting: 'Starte…',
      postSetupCompleteTitle: 'Setup abgeschlossen',
      postSetupCompleteMessage: step => `${step} installiert.`,
      postSetupErrorTitle: 'Setup mit Fehlern beendet',
      postSetupErrorMessage: step => `Prüfe das ${step}-Log.`,
      postSetupFailed: step => `${step}-Setup konnte nicht ausgeführt werden`
    }
  },

  skills: {
    tabSkills: 'Skills',
    tabToolsets: 'Toolsets',
    all: 'Alle',
    searchSkills: 'Skills durchsuchen…',
    searchToolsets: 'Toolsets durchsuchen…',
    refresh: 'Skills aktualisieren',
    refreshing: 'Aktualisiere Skills',
    loading: 'Lade Fähigkeiten…',
    noSkillsTitle: 'Keine Skills gefunden',
    noSkillsDesc: 'Versuche eine breitere Suche oder eine andere Kategorie.',
    noToolsetsTitle: 'Keine Toolsets gefunden',
    noToolsetsDesc: 'Versuche eine breitere Suchanfrage.',
    noDescription: 'Keine Beschreibung.',
    configured: 'Konfiguriert',
    needsKeys: 'Benötigt Keys',
    toolsetsEnabled: (enabled, total) => `${enabled}/${total} Toolsets aktiviert`,
    configureToolset: label => `${label} konfigurieren`,
    toggleToolset: label => `Toolset ${label} umschalten`,
    skillsLoadFailed: 'Skills konnten nicht geladen werden',
    toolsetsRefreshFailed: 'Toolsets konnten nicht aktualisiert werden',
    skillEnabled: 'Skill aktiviert',
    skillDisabled: 'Skill deaktiviert',
    toolsetEnabled: 'Toolset aktiviert',
    toolsetDisabled: 'Toolset deaktiviert',
    appliesToNewSessions: name => `${name} gilt für neue Sitzungen.`,
    failedToUpdate: name => `${name} konnte nicht aktualisiert werden`
  },

  agents: {
    close: 'Agents schließen',
    title: 'Spawn-Baum',
    subtitle: 'Live-Aktivität der Subagenten für den aktuellen Durchlauf.',
    emptyTitle: 'Keine Live-Subagenten',
    emptyDesc: 'Wenn ein Durchlauf Arbeit delegiert, streamen die Kind-Agenten ihren Fortschritt hier.',
    running: 'Läuft',
    failed: 'Fehlgeschlagen',
    done: 'Fertig',
    streaming: 'Streamt',
    files: 'Dateien',
    moreFiles: count => `+${count} weitere Dateien`,
    delegation: index => `Delegation ${index}`,
    workers: count => `${count} Worker`,
    workersActive: count => `${count} aktiv`,
    agentsCount: count => `${count} ${count === 1 ? 'Agent' : 'Agents'}`,
    activeCount: count => `${count} aktiv`,
    failedCount: count => `${count} fehlgeschlagen`,
    toolsCount: count => `${count} Tools`,
    filesCount: count => `${count} Dateien`,
    updatedAgo: age => `aktualisiert ${age}`,
    ageNow: 'jetzt',
    ageSeconds: seconds => `vor ${seconds}s`,
    ageMinutes: minutes => `vor ${minutes}m`,
    ageHours: hours => `vor ${hours}h`,
    durationSeconds: seconds => `${seconds}s`,
    durationMinutes: (minutes, seconds) => `${minutes}m ${seconds}s`,
    tokens: value => `${value} Tok`
  },

  commandCenter: {
    close: 'Command Center schließen',
    paletteTitle: 'Befehlspalette',
    back: 'Zurück',
    searchPlaceholder: 'Sitzungen, Ansichten und Aktionen durchsuchen',
    goTo: 'Gehe zu',
    goToSession: 'Zur Sitzung gehen',
    commandCenter: 'Command Center',
    appearance: 'Darstellung',
    settings: 'Einstellungen',
    changeTheme: 'Theme ändern',
    changeColorMode: 'Farbmodus ändern…',
    pets: {
      title: 'Pets',
      placeholder: 'Pets durchsuchen…',
      loading: 'Lade Petdex-Galerie…',
      error: 'Die Petdex-Galerie war nicht erreichbar.',
      staleBackend: 'Starte Hermes neu, um Pets zu nutzen — das Backend ist älter als dieses Feature.',
      empty: 'Keine passenden Pets.',
      turnOff: 'Ausschalten',
      turnOn: 'Einschalten',
      installed: 'Installiert',
      adoptFailed: 'Dieses Pet konnte nicht adoptiert werden.',
      toggleFailed: 'Das Pet konnte nicht umgeschaltet werden.',
      noneAvailable: 'Keine Pets verfügbar — wähle unten eines zum Installieren.'
    },
    installTheme: {
      title: 'Theme installieren…',
      placeholder: 'Den VS-Code-Marketplace durchsuchen…',
      loading: 'Durchsuche den Marketplace…',
      error: 'Der Marketplace war nicht erreichbar.',
      empty: 'Keine passenden Themes.',
      install: 'Installieren',
      installing: 'Installiere…',
      installed: 'Installiert',
      installs: count => `${count} Installationen`
    },
    settingsFields: 'Einstellungsfelder',
    mcpServers: 'MCP-Server',
    archivedChats: 'Archivierte Chats',
    sections: { sessions: 'Sitzungen', system: 'System', usage: 'Nutzung' },
    sectionDescriptions: {
      sessions: 'Sitzungen suchen und verwalten',
      system: 'Status, Logs und Systemaktionen',
      usage: 'Token-, Kosten- und Skill-Aktivität über die Zeit'
    },
    nav: {
      newChat: { title: 'Neue Sitzung', detail: 'Eine frische Sitzung starten' },
      settings: { title: 'Einstellungen', detail: 'Hermes Desktop konfigurieren' },
      skills: { title: 'Skills & Tools', detail: 'Skills, Toolsets und Provider aktivieren' },
      messaging: { title: 'Messaging', detail: 'Telegram, Slack, Discord und mehr einrichten' },
      artifacts: { title: 'Artefakte', detail: 'Generierte Ausgaben durchsuchen' }
    },
    sectionEntries: {
      sessions: { title: 'Sitzungs-Panel', detail: 'Sitzungen suchen, anpinnen und verwalten' },
      system: { title: 'System-Panel', detail: 'Gateway-Status, Logs, Neustart/Update' },
      usage: { title: 'Nutzungs-Panel', detail: 'Token-, Kosten- und Skill-Aktivität' }
    },
    providerNavigate: 'Navigieren',
    providerSessions: 'Sitzungen',
    refresh: 'Aktualisieren',
    refreshing: 'Aktualisiere…',
    noResults: 'Keine passenden Ergebnisse gefunden.',
    pinSession: 'Sitzung anpinnen',
    unpinSession: 'Sitzung lösen',
    exportSession: 'Sitzung exportieren',
    deleteSession: 'Sitzung löschen',
    noSessions: 'Noch keine Sitzungen.',
    gatewayRunning: 'Messaging-Gateway läuft',
    gatewayStopped: 'Messaging-Gateway gestoppt',
    hermesActiveSessions: (version, count) => `Hermes ${version} · Aktive Sitzungen ${count}`,
    restartGateway: 'Gateway neu starten',
    gatewayRestartFailed: 'Gateway-Neustart fehlgeschlagen.',
    updateHermes: 'Hermes aktualisieren',
    actionRunning: 'läuft',
    actionDone: 'fertig',
    actionFailed: 'fehlgeschlagen',
    actionStartedWaiting: 'Aktion gestartet, warte auf Status…',
    loadingStatus: 'Lade Status…',
    recentLogs: 'Letzte Logs',
    noLogs: 'Noch keine Logs geladen.',
    days: count => `${count} T`,
    statSessions: 'Sitzungen',
    statApiCalls: 'API-Aufrufe',
    statTokens: 'Token ein/aus',
    statCost: 'Gesch. Kosten',
    actualCost: cost => `tatsächlich ${cost}`,
    loadingUsage: 'Lade Nutzung…',
    noUsage: period => `Keine Nutzung in den letzten ${period} Tagen.`,
    retry: 'Erneut versuchen',
    dailyTokens: 'Tägliche Token',
    input: 'Eingabe',
    output: 'Ausgabe',
    noDailyActivity: 'Keine tägliche Aktivität.',
    topModels: 'Top-Modelle',
    noModelUsage: 'Noch keine Modell-Nutzung.',
    topSkills: 'Top-Skills',
    noSkillActivity: 'Noch keine Skill-Aktivität.',
    actions: count => `${count} Aktionen`
  },

  messaging: {
    search: 'Messaging durchsuchen…',
    loading: 'Lade Messaging-Plattformen…',
    loadFailed: 'Messaging-Plattformen konnten nicht geladen werden',
    states: {
      connected: 'Verbunden',
      connecting: 'Verbinde',
      disabled: 'Deaktiviert',
      fatal: 'Fehler',
      gateway_stopped: 'Messaging-Gateway gestoppt',
      not_configured: 'Einrichtung nötig',
      pending_restart: 'Neustart nötig',
      retrying: 'Versuche erneut',
      startup_failed: 'Start fehlgeschlagen'
    },
    unknown: 'Unbekannt',
    hintPendingRestart: 'Starte das Gateway über die Statusleiste neu, um diese Änderung anzuwenden.',
    hintGatewayStopped: 'Starte das Gateway über die Statusleiste, um zu verbinden.',
    credentialsSet: 'Zugangsdaten gesetzt',
    needsSetup: 'Einrichtung nötig',
    gatewayStopped: 'Messaging-Gateway gestoppt',
    getCredentials: 'Hol dir deine Zugangsdaten',
    openSetupGuide: 'Einrichtungs-Anleitung öffnen',
    required: 'Erforderlich',
    recommended: 'Empfohlen',
    advanced: count => `Erweitert (${count})`,
    noTokenNeeded: 'Diese Plattform braucht hier kein Token. Nutze die Anleitung oben und aktiviere sie dann unten.',
    enabled: 'Aktiviert',
    disabled: 'Deaktiviert',
    unsavedChanges: 'Ungespeicherte Änderungen',
    saving: 'Speichere…',
    saveChanges: 'Änderungen speichern',
    saved: 'Gespeichert',
    replaceValue: 'Aktuellen Wert ersetzen',
    openDocs: 'Doku öffnen',
    clearField: key => `${key} leeren`,
    enableAria: name => `${name} aktivieren`,
    disableAria: name => `${name} deaktivieren`,
    platformEnabled: name => `${name} aktiviert`,
    platformDisabled: name => `${name} deaktiviert`,
    restartToApply: 'Diese Änderung wird nach einem Gateway-Neustart wirksam.',
    setupSaved: name => `${name}-Einrichtung gespeichert`,
    restartToReconnect: 'Neue Zugangsdaten werden nach einem Gateway-Neustart wirksam.',
    keyCleared: key => `${key} geleert`,
    setupUpdated: name => `${name}-Einrichtung wurde aktualisiert.`,
    failedUpdate: name => `${name} konnte nicht aktualisiert werden`,
    failedSave: name => `${name} konnte nicht gespeichert werden`,
    failedClear: key => `${key} konnte nicht geleert werden`,
    fieldCopy: {
      TELEGRAM_BOT_TOKEN: {
        label: 'Bot-Token',
        help: 'Erstelle einen Bot mit @BotFather und füge dann das Token ein, das er dir gibt.',
        placeholder: 'Telegram-Bot-Token einfügen'
      },
      TELEGRAM_ALLOWED_USERS: {
        label: 'Erlaubte Telegram-User-IDs',
        help: 'Empfohlen. Kommagetrennte numerische IDs von @userinfobot. Ohne dies kann dir jeder schreiben.'
      },
      TELEGRAM_PROXY: { label: 'Proxy-URL', help: 'Nur in Netzwerken nötig, in denen Telegram blockiert ist.' },
      DISCORD_BOT_TOKEN: {
        label: 'Bot-Token',
        help: 'Erstelle eine Anwendung im Discord Developer Portal, füge einen Bot hinzu und füge dann dessen Token ein.'
      },
      DISCORD_ALLOWED_USERS: {
        label: 'Erlaubte Discord-User-IDs',
        help: 'Empfohlen. Kommagetrennte Discord-User-IDs.'
      },
      DISCORD_REPLY_TO_MODE: { label: 'Antwort-Stil', help: 'first, all oder off.' },
      DISCORD_ALLOW_ALL_USERS: {
        label: 'Alle Discord-User erlauben',
        help: 'Nur für Entwicklung. Wenn true, kann jeder dem Bot ohne Allowlist schreiben.'
      },
      DISCORD_HOME_CHANNEL: {
        label: 'Home-Channel-ID',
        help: 'Channel, in den der Bot proaktive Nachrichten sendet (Cron-Ausgabe, Erinnerungen).'
      },
      DISCORD_HOME_CHANNEL_NAME: {
        label: 'Home-Channel-Name',
        help: 'Anzeigename für den Home-Channel in Logs und Status-Ausgabe.'
      },
      BLUEBUBBLES_ALLOW_ALL_USERS: {
        label: 'Alle iMessage-User erlauben',
        help: 'Wenn true, wird die BlueBubbles-Allowlist übersprungen.'
      },
      MATTERMOST_ALLOW_ALL_USERS: { label: 'Alle Mattermost-User erlauben' },
      MATTERMOST_HOME_CHANNEL: { label: 'Home-Channel' },
      QQ_ALLOW_ALL_USERS: { label: 'Alle QQ-User erlauben' },
      QQBOT_HOME_CHANNEL: { label: 'QQ-Home-Channel', help: 'Standard-Channel oder -Gruppe für die Cron-Zustellung.' },
      QQBOT_HOME_CHANNEL_NAME: { label: 'QQ-Home-Channel-Name' },
      SLACK_BOT_TOKEN: {
        label: 'Slack-Bot-Token',
        help: 'Nutze das Bot-Token aus „OAuth & Permissions", nachdem du deine Slack-App installiert hast.',
        placeholder: 'Slack-Bot-Token einfügen'
      },
      SLACK_APP_TOKEN: {
        label: 'Slack-App-Token',
        help: 'Nutze das App-Level-Token, das für den Socket Mode erforderlich ist.',
        placeholder: 'Slack-App-Token einfügen'
      },
      SLACK_ALLOWED_USERS: { label: 'Erlaubte Slack-User-IDs', help: 'Empfohlen. Kommagetrennte Slack-User-IDs.' },
      MATTERMOST_URL: { label: 'Server-URL', placeholder: 'https://mattermost.example.com' },
      MATTERMOST_TOKEN: { label: 'Bot-Token' },
      MATTERMOST_ALLOWED_USERS: {
        label: 'Erlaubte User-IDs',
        help: 'Empfohlen. Kommagetrennte Mattermost-User-IDs.'
      },
      MATRIX_HOMESERVER: { label: 'Homeserver-URL', placeholder: 'https://matrix.org' },
      MATRIX_ACCESS_TOKEN: { label: 'Access-Token' },
      MATRIX_USER_ID: { label: 'Bot-User-ID', placeholder: '@hermes:example.org' },
      MATRIX_ALLOWED_USERS: {
        label: 'Erlaubte Matrix-User-IDs',
        help: 'Empfohlen. Kommagetrennte User-IDs im Format @user:server.'
      },
      SIGNAL_HTTP_URL: {
        label: 'Signal-Bridge-URL',
        placeholder: 'http://127.0.0.1:8080',
        help: 'URL einer laufenden signal-cli-REST-Bridge.'
      },
      SIGNAL_ACCOUNT: { label: 'Telefonnummer', help: 'Die bei deiner signal-cli-Bridge registrierte Nummer.' },
      SIGNAL_ALLOWED_USERS: { label: 'Erlaubte Signal-User', help: 'Empfohlen. Kommagetrennte Signal-Kennungen.' },
      WHATSAPP_ENABLED: {
        label: 'WhatsApp-Bridge aktivieren',
        help: 'Wird automatisch durch den Schalter unten gesetzt. Nicht anfassen, außer du weißt, dass du es brauchst.'
      },
      WHATSAPP_MODE: { label: 'Bridge-Modus' },
      WHATSAPP_ALLOWED_USERS: {
        label: 'Erlaubte WhatsApp-User',
        help: 'Empfohlen. Kommagetrennte Telefonnummern oder WhatsApp-IDs.'
      }
    },
    platformIntro: {}
  },

  profiles: {
    close: 'Profile schließen',
    nameHint:
      'Kleinbuchstaben, Ziffern, Bindestriche und Unterstriche. Muss mit einem Buchstaben oder einer Ziffer beginnen.',
    title: 'Profile',
    count: count => `${count} ${count === 1 ? 'Profil' : 'Profile'}`,
    loading: 'Lade Profile…',
    newProfile: 'Neues Profil',
    allProfiles: 'Alle Profile',
    showAllProfiles: 'Alle Profile anzeigen',
    switchToProfile: name => `Zu ${name} wechseln`,
    manageProfiles: 'Profile verwalten…',
    actionsFor: name => `Aktionen für ${name}`,
    color: 'Farbe…',
    colorFor: name => `Farbe für ${name}`,
    setColor: color => `Farbe ${color} setzen`,
    autoColor: 'Auto',
    noProfiles: 'Noch keine Profile.',
    selectPrompt: 'Wähle ein Profil, um seine Details zu sehen.',
    refresh: 'Profile aktualisieren',
    refreshing: 'Aktualisiere Profile',
    default: 'Standard',
    skills: count => `${count} ${count === 1 ? 'Skill' : 'Skills'}`,
    env: 'env',
    defaultBadge: 'Standard',
    rename: 'Umbenennen',
    copySetup: 'Setup kopieren',
    copying: 'Kopiere…',
    modelLabel: 'Modell',
    skillsLabel: 'Skills',
    notSet: 'Nicht gesetzt',
    soulDesc: 'Der System-Prompt und die Persona-Anweisungen, die in dieses Profil eingebacken sind.',
    soulOptional: 'optional',
    soulPlaceholder: mode =>
      `Der System-Prompt / die Persona für dieses Profil.\nLeer lassen, um den ${mode}-Standard zu behalten.`,
    soulPlaceholderCloned: 'geklont',
    soulPlaceholderEmpty: 'leer',
    unsavedChanges: 'Ungespeicherte Änderungen',
    loadingSoul: 'Lade SOUL.md…',
    emptySoul: 'Leere SOUL.md — beginne, die Persona zu schreiben…',
    saving: 'Speichere…',
    saveSoul: 'SOUL.md speichern',
    deleteTitle: 'Profil löschen?',
    deleteDescPrefix: 'Dies löscht ',
    deleteDescMid: ' und entfernt sein ',
    deleteDescSuffix: '-Verzeichnis. Das kann nicht rückgängig gemacht werden.',
    deleting: 'Lösche…',
    createDesc: 'Profile sind unabhängige Hermes-Umgebungen: eigene Konfig, Skills und SOUL.md.',
    nameLabel: 'Name',
    cloneFrom: 'Klonen von',
    cloneFromNone: 'Keine (leer)',
    cloneFromDesc: 'Kopiert Konfig, Skills und SOUL.md aus dem gewählten Quell-Profil.',
    cloneFromDefault: 'Vom Standard klonen',
    cloneFromDefaultDesc: 'Kopiert Konfig, Skills und SOUL.md aus deinem Standard-Profil.',
    invalidName: hint => `Ungültiger Name. ${hint}`,
    nameRequired: 'Name ist erforderlich.',
    creating: 'Erstelle…',
    createAction: 'Profil erstellen',
    renameTitle: 'Profil umbenennen',
    renameDescPrefix: 'Das Umbenennen aktualisiert das Profil-Verzeichnis und alle Wrapper-Skripte in ',
    renameDescSuffix: '.',
    newNameLabel: 'Neuer Name',
    renaming: 'Benenne um…',
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

  cron: {
    close: 'Cron schließen',
    search: 'Cron-Jobs durchsuchen…',
    loading: 'Lade Cron-Jobs…',
    states: {
      enabled: 'aktiviert',
      scheduled: 'geplant',
      running: 'läuft',
      paused: 'pausiert',
      disabled: 'deaktiviert',
      error: 'Fehler',
      completed: 'abgeschlossen'
    },
    deliveryLabels: {
      local: 'Dieser Desktop',
      telegram: 'Telegram',
      discord: 'Discord',
      slack: 'Slack',
      email: 'E-Mail'
    },
    scheduleLabels: {
      daily: 'Täglich',
      weekdays: 'Wochentags',
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
      hourly: 'Zur vollen Stunde',
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
    weekdaysAt: time => `Wochentags um ${time}`,
    everyDayOfWeekAt: (day, time) => `Jeden ${day} um ${time}`,
    monthlyOnDayAt: (dayOfMonth, time) => `Monatlich am ${dayOfMonth}. um ${time}`,
    topOfHour: 'Zur vollen Stunde',
    everyHourAt: minute => `Jede Stunde um :${minute}`,
    newCron: 'Neuer Cron',
    emptyDescNew:
      'Plane einen Prompt, der nach einem Cron-Ausdruck läuft. Hermes führt ihn aus und liefert die Ergebnisse an das von dir gewählte Ziel.',
    emptyDescSearch: 'Versuche eine breitere Suchanfrage.',
    emptyTitleNew: 'Noch keine geplanten Jobs',
    emptyTitleSearch: 'Keine Treffer',
    last: 'Zuletzt:',
    next: 'Nächste:',
    noRuns: 'Noch keine Durchläufe',
    manage: 'Verwalten',
    showRuns: 'Durchläufe anzeigen',
    hideRuns: 'Durchläufe ausblenden',
    runHistory: 'Durchlauf-Verlauf',
    actionsFor: title => `Aktionen für ${title}`,
    actionsTitle: 'Cron-Job-Aktionen',
    resume: 'Cron fortsetzen',
    pause: 'Cron pausieren',
    resumeTitle: 'Fortsetzen',
    pauseTitle: 'Pausieren',
    triggerNow: 'Jetzt auslösen',
    edit: 'Cron bearbeiten',
    deleteTitle: 'Cron-Job löschen?',
    deleteDescPrefix: 'Dies entfernt ',
    deleteDescSuffix: ' endgültig. Er hört sofort auf zu feuern.',
    deleting: 'Lösche…',
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
    editDesc: 'Aktualisiere Zeitplan, Prompt oder Zustellungsziel. Änderungen gelten ab dem nächsten Durchlauf.',
    createDesc:
      'Plane einen Prompt, der automatisch läuft. Nutze Cron-Syntax oder einen natürlichen Ausdruck wie „alle 15 Minuten".',
    nameLabel: 'Name',
    namePlaceholder: 'Morgen-Briefing',
    promptLabel: 'Prompt',
    promptPlaceholder: 'Fasse meine ungelesenen Slack-Threads zusammen und maile mir die Top 5…',
    frequencyLabel: 'Häufigkeit',
    deliverLabel: 'Zustellen an',
    customScheduleLabel: 'Benutzerdefinierter Zeitplan',
    customPlaceholder: '0 9 * * * oder wochentags um 9 Uhr',
    customHint: 'Cron-Ausdruck oder Ausdrücke wie „jede Stunde" oder „wochentags um 9 Uhr".',
    optional: 'Optional',
    promptScheduleRequired: 'Prompt und Zeitplan sind erforderlich.',
    saveChanges: 'Änderungen speichern',
    createAction: 'Cron erstellen'
  },

  artifacts: {
    search: 'Artefakte durchsuchen…',
    refresh: 'Artefakte aktualisieren',
    refreshing: 'Aktualisiere Artefakte',
    indexing: 'Indexiere Artefakte der letzten Sitzungen',
    tabAll: 'Alle',
    tabImages: 'Bilder',
    tabFiles: 'Dateien',
    tabLinks: 'Links',
    noArtifactsTitle: 'Keine Artefakte gefunden',
    noArtifactsDesc: 'Generierte Bilder und Datei-Ausgaben erscheinen hier, sobald Sitzungen sie erzeugen.',
    failedLoad: 'Artefakte konnten nicht geladen werden',
    openFailed: 'Öffnen fehlgeschlagen',
    itemsImage: 'Bilder',
    itemsLink: 'Links',
    itemsFile: 'Dateien',
    itemsGeneric: 'Elemente',
    zero: '0',
    rangeOf: (start, end, total) => `${start}-${end} von ${total}`,
    goToPage: (itemLabel, page) => `Zu ${itemLabel}-Seite ${page} gehen`,
    colTitleLink: 'Link-Titel',
    colTitleFile: 'Name',
    colTitleDefault: 'Titel / Name',
    colLocationLink: 'URL',
    colLocationFile: 'Pfad',
    colLocationDefault: 'Ort',
    colSession: 'Sitzung',
    kindImage: 'Bild',
    kindFile: 'Datei',
    kindLink: 'Link',
    chat: 'Chat',
    copyUrl: 'URL kopieren',
    copyPath: 'Pfad kopieren'
  },

  sidebar: {
    nav: {
      'new-session': 'Neue Sitzung',
      skills: 'Skills & Tools',
      messaging: 'Messaging',
      artifacts: 'Artefakte'
    },
    searchAria: 'Sitzungen durchsuchen',
    searchPlaceholder: 'Sitzungen durchsuchen…',
    clearSearch: 'Suche leeren',
    noMatch: query => `Keine Sitzungen passen zu „${query}".`,
    results: 'Ergebnisse',
    pinned: 'Angepinnt',
    sessions: 'Sitzungen',
    cronJobs: 'Cron-Jobs',
    groupAriaGrouped: 'Sitzungen als einzelne Liste anzeigen',
    groupAriaUngrouped: 'Sitzungen nach Arbeitsbereich gruppieren',
    groupTitleGrouped: 'Gruppierung aufheben',
    groupTitleUngrouped: 'Nach Arbeitsbereich gruppieren',
    allPinned: 'Hier ist alles angepinnt. Löse einen Chat, um ihn in den Letzten anzuzeigen.',
    shiftClickHint: 'Shift-Klick auf einen Chat zum Anpinnen',
    noWorkspace: 'Kein Arbeitsbereich',
    newSessionIn: label => `Neue Sitzung in ${label}`,
    showMoreIn: (count, label) => `${count} weitere in ${label} anzeigen`,
    loading: 'Lädt…',
    loadMore: 'Mehr laden',
    loadCount: step => `${step} weitere laden`,
    row: {
      pin: 'Anpinnen',
      unpin: 'Lösen',
      copyId: 'ID kopieren',
      export: 'Exportieren',
      rename: 'Umbenennen',
      archive: 'Archivieren',
      newWindow: 'Neues Fenster',
      copyIdFailed: 'Sitzungs-ID konnte nicht kopiert werden',
      actionsFor: title => `Aktionen für ${title}`,
      sessionActions: 'Sitzungs-Aktionen',
      sessionRunning: 'Sitzung läuft',
      needsInput: 'Braucht deine Eingabe',
      waitingForAnswer: 'Wartet auf deine Antwort',
      handoffOrigin: platform => `Übergeben von ${platform}`,
      renamed: 'Umbenannt',
      renameFailed: 'Umbenennen fehlgeschlagen',
      renameTitle: 'Sitzung umbenennen',
      renameDesc: 'Gib diesem Chat einen einprägsamen Titel. Leer lassen zum Löschen.',
      untitledPlaceholder: 'Unbenannte Sitzung',
      ageNow: 'jetzt',
      ageDay: 'T',
      ageHour: 'Std',
      ageMin: 'm'
    }
  },

  composer: {
    message: 'Nachricht',
    wakingProfile: profile => `Wecke ${profile} auf…`,
    placeholderStarting: 'Starte Hermes…',
    placeholderReconnecting: 'Verbinde erneut mit Hermes…',
    placeholderFollowUp: 'Nachfrage senden',
    newSessionPlaceholders: [
      'Was bauen wir?',
      'Gib Hermes eine Aufgabe',
      'Was beschäftigt dich?',
      'Beschreibe, was du brauchst',
      'Was packen wir an?',
      'Frag irgendwas',
      'Starte mit einem Ziel'
    ],
    followUpPlaceholders: [
      'Sende eine Nachfrage',
      'Füge mehr Kontext hinzu',
      'Verfeinere die Anfrage',
      'Was kommt als Nächstes?',
      'Mach weiter',
      'Treib es weiter voran',
      'Anpassen oder fortfahren'
    ],
    startVoice: 'Sprachunterhaltung starten',
    queueMessage: 'Nachricht in Warteschlange',
    steer: 'Aktuellen Durchlauf steuern',
    stop: 'Stopp',
    send: 'Senden',
    speaking: 'Spreche',
    transcribing: 'Transkribiere',
    thinking: 'Denke nach',
    muted: 'Stumm',
    listening: 'Höre zu',
    muteMic: 'Mikrofon stummschalten',
    unmuteMic: 'Mikrofon aktivieren',
    stopListening: 'Zuhören beenden und senden',
    stopShort: 'Stopp',
    endConversation: 'Sprachunterhaltung beenden',
    endShort: 'Beenden',
    stopDictation: 'Diktat beenden',
    transcribingDictation: 'Transkribiere Diktat',
    voiceDictation: 'Sprachdiktat',
    lookupLoading: 'Suche…',
    lookupNoMatches: 'Keine Treffer.',
    lookupTry: 'Versuche',
    lookupOr: 'oder',
    commonCommands: 'Häufige Befehle',
    hotkeys: 'Hotkeys',
    helpFooter: 'öffnet das volle Panel · Backspace schließt',
    commandDescs: {
      '/help': 'Vollständige Liste der Befehle + Hotkeys',
      '/clear': 'Eine neue Sitzung starten',
      '/resume': 'Eine frühere Sitzung fortsetzen',
      '/details': 'Detailgrad des Transkripts steuern',
      '/copy': 'Auswahl oder letzte Assistenten-Nachricht kopieren',
      '/quit': 'Hermes beenden'
    },
    hotkeyDescs: {
      'composer.mention': 'Dateien, Ordner, URLs, Git referenzieren',
      'composer.slash': 'Slash-Befehlspalette',
      'composer.help': 'Diese Schnellhilfe (Entf zum Schließen)',
      'composer.sendNewline': 'senden · Shift+Enter für Zeilenumbruch',
      'composer.sendQueued': 'nächsten Durchlauf aus der Warteschlange senden',
      'keybinds.openPanel': 'alle Tastenkürzel',
      'composer.cancel': 'Popover schließen · Durchlauf abbrechen',
      'composer.history': 'Popover / Verlauf durchblättern'
    },
    attachUrlTitle: 'URL anhängen',
    attachUrlDesc: 'Hermes ruft die Seite ab und nimmt sie als Kontext für diesen Durchlauf auf.',
    urlPlaceholder: 'https://example.com/post',
    urlHintPre: 'Gib die vollständige URL an, z. B. ',
    attach: 'Anhängen',
    queued: count => `${count} in Warteschlange`,
    attachmentOnly: 'Durchlauf nur mit Anhang',
    emptyTurn: 'Leerer Durchlauf',
    attachments: count => `${count} ${count === 1 ? 'Anhang' : 'Anhänge'}`,
    editingInComposer: 'Bearbeite in der Eingabe',
    editingQueuedInComposer: 'Bearbeite Durchlauf aus Warteschlange in der Eingabe',
    queueEdit: 'Bearbeiten',
    queueSendNext: 'Nächste',
    queueSend: 'Senden',
    queueDelete: 'Löschen',
    queueStuckTitle: 'Nachricht in Warteschlange nicht gesendet',
    queueStuckBody:
      'Ein Durchlauf in der Warteschlange ließ sich wiederholt nicht senden. Er ist noch in der Warteschlange — versuche, ihn erneut zu senden.',
    previewUnavailable: 'Vorschau nicht verfügbar',
    previewLabel: label => `Vorschau ${label}`,
    couldNotPreview: label => `Vorschau für ${label} nicht möglich`,
    removeAttachment: label => `${label} entfernen`,
    dictating: 'Diktiere',
    preparingAudio: 'Bereite Audio vor',
    speakingResponse: 'Spreche Antwort',
    readingAloud: 'Lese vor',
    themeSuggestions: 'Desktop-Theme-Vorschläge',
    noMatchingThemes: 'Keine passenden Themes.',
    themeTryPre: 'Versuche ',
    themeTryPost: '.',
    attachLabel: 'Anhängen',
    files: 'Dateien…',
    folder: 'Ordner…',
    images: 'Bilder…',
    pasteImage: 'Bild einfügen',
    url: 'URL…',
    promptSnippets: 'Prompt-Snippets…',
    tipPre: 'Tipp: tippe ',
    tipPost: ', um Dateien inline zu referenzieren.',
    snippetsTitle: 'Prompt-Snippets',
    snippetsDesc: 'Wähle einen Start-Prompt, um ihn in die Eingabe zu übernehmen.',
    dropFiles: 'Dateien hierher ziehen zum Anhängen',
    dropSession: 'Hier ablegen, um diesen Chat zu verknüpfen',
    snippets: {
      codeReview: {
        label: 'Code-Review',
        description: 'Prüfe die aktuelle Änderung auf Regressionen, übersehene Edge-Cases und fehlende Tests.',
        text: 'Bitte prüfe das auf Bugs, Regressionen und fehlende Tests.'
      },
      implementationPlan: {
        label: 'Umsetzungsplan',
        description: 'Skizziere einen Ansatz, bevor du Code anfasst, damit der Diff fokussiert bleibt.',
        text: 'Bitte erstelle einen knappen Umsetzungsplan, bevor du Code änderst.'
      },
      explainThis: {
        label: 'Erklär das',
        description: 'Geh durch, wie der ausgewählte Code funktioniert, und verlinke die wichtigsten Dateien.',
        text: 'Bitte erkläre, wie das funktioniert, und zeig mir die wichtigsten Dateien.'
      }
    }
  },

  statusStack: {
    agents: 'Agents',
    background: count => `${count} im Hintergrund`,
    subagents: count => `${count} ${count === 1 ? 'Subagent' : 'Subagents'}`,
    todos: (done, total) => `Aufgaben ${done}/${total}`,
    running: 'Läuft',
    stop: 'Stopp',
    dismiss: 'Schließen',
    exit: code => `Exit ${code}`
  },

  updates: {
    stages: {
      idle: 'Mache mich bereit…',
      prepare: 'Mache mich bereit…',
      fetch: 'Lade herunter…',
      pull: 'Fast geschafft…',
      pydeps: 'Schließe ab…',
      update: 'Aktualisiere Hermes…',
      rebuild: 'Baue die Desktop-App neu…',
      restart: 'Starte Hermes neu…',
      done: 'Update abgeschlossen',
      manual: 'Update über dein Terminal',
      guiSkew: 'Desktop-App aktualisieren',
      error: 'Update pausiert'
    },
    checking: 'Suche nach Updates…',
    checkFailedTitle: 'Suche nach Updates fehlgeschlagen',
    tryAgain: 'Erneut versuchen',
    notAvailableTitle: 'Update nicht verfügbar',
    unsupportedMessage: 'Diese Hermes-Version kann sich nicht aus der App heraus selbst aktualisieren.',
    connectionRetry: 'Prüfe deine Verbindung und versuche es erneut.',
    latestBody: 'Du nutzt die neueste Version.',
    latestBodyBackend: 'Das Backend läuft auf der neuesten Version.',
    allSetTitle: 'Alles bereit',
    availableTitle: 'Neues Update verfügbar',
    availableBody: 'Eine neue Hermes-Version ist installationsbereit.',
    availableTitleBackend: 'Backend-Update verfügbar',
    availableBodyBackend: 'Eine neuere Version des verbundenen Hermes-Backends ist installationsbereit.',
    availableBodyNoChangelog:
      'Eine neuere Version ist bereit. Für diesen Installationstyp sind keine Release Notes verfügbar.',
    updateNow: 'Jetzt aktualisieren',
    maybeLater: 'Vielleicht später',
    moreChanges: count => `+ ${count} weitere Änderung${count === 1 ? '' : 'en'} enthalten.`,
    manualTitle: 'Update über dein Terminal',
    manualBody:
      'Du hast Hermes über die Kommandozeile installiert, daher laufen Updates auch dort. Füge dies in dein Terminal ein:',
    manualPickedUp: 'Hermes übernimmt die neue Version, wenn du es das nächste Mal startest.',
    guiSkewTitle: 'Desktop-App aktualisieren',
    guiSkewBody:
      'Das Backend wurde aktualisiert, aber dieses Desktop-App-Paket wurde nicht geändert. Aktualisiere oder installiere die Hermes-Desktop-App (deine AppImage / .deb / .rpm) neu, damit beides zusammenpasst.',
    copy: 'Kopieren',
    copied: 'Kopiert',
    done: 'Fertig',
    applyingBody:
      'Der Hermes-Updater übernimmt in seinem eigenen Fenster und öffnet Hermes automatisch wieder, wenn er fertig ist. Bitte öffne Hermes nicht selbst, während es aktualisiert wird.',
    applyingBodyBackend:
      'Das Remote-Backend wendet das Update an und startet neu. Hermes verbindet sich automatisch wieder, sobald es zurück ist.',
    applyingClose: 'Dieses Fenster schließt sich, während das Update läuft, dann öffnet sich Hermes von selbst wieder.',
    errorTitle: 'Update nicht abgeschlossen',
    errorBody: 'Keine Sorge — nichts ging verloren. Du kannst es jetzt erneut versuchen.',
    notNow: 'Jetzt nicht',
    applyStatus: {
      preparing: 'Aktualisiere Backend…',
      pulling: 'Backend aktualisiert…',
      restarting: 'Backend startet neu, um das Update zu laden…',
      notAvailable: 'Update für dieses Backend nicht verfügbar.',
      failed: 'Backend-Update fehlgeschlagen.',
      noReturn:
        'Das Backend ist nicht wieder online gekommen. Das Update wurde womöglich nicht abgeschlossen — prüfe den Backend-Host.'
    }
  },

  install: {
    stageStates: {
      pending: 'Ausstehend',
      running: 'Installiere',
      succeeded: 'Fertig',
      skipped: 'Übersprungen',
      failed: 'Fehlgeschlagen'
    },
    oneTimeTitle: 'Hermes braucht eine einmalige Installation',
    unsupportedDesc: platform =>
      `Die automatische Erstinstallation ist auf ${platform} noch nicht verfügbar. Öffne das Terminal, führe den Befehl unten aus und starte diese App neu. Spätere Starts überspringen diesen Schritt.`,
    installCommand: 'Installationsbefehl',
    copyCommand: 'Befehl kopieren',
    viewDocs: 'Installations-Doku ansehen',
    installTo: 'Wird installiert nach',
    retryAfterRun: 'Habe ihn ausgeführt — erneut versuchen',
    failedTitle: 'Installation fehlgeschlagen',
    settingUpTitle: 'Richte Hermes Agent ein',
    finishingTitle: 'Schließe ab',
    failedDesc:
      'Einer der Installationsschritte ist fehlgeschlagen. Unter Windows kann das passieren, wenn eine andere Hermes-CLI oder Desktop-Instanz läuft. Stoppe alle laufenden Hermes-Instanzen und versuche es erneut. Prüfe die Details unten oder das Desktop-Log für das vollständige Protokoll.',
    activeDesc:
      'Dies ist eine einmalige Einrichtung. Der Hermes-Installer lädt Abhängigkeiten herunter und konfiguriert deinen Rechner. Spätere Starts überspringen diesen Schritt.',
    progress: (completed, total) => `${completed} von ${total} Schritten abgeschlossen`,
    currentStage: stage => ` — jetzt: ${stage}`,
    fetchingManifest: 'Lade Installer-Manifest…',
    error: 'Fehler',
    hideOutput: 'Installer-Ausgabe ausblenden',
    showOutput: 'Installer-Ausgabe anzeigen',
    lines: count => `${count} Zeile${count === 1 ? '' : 'n'}`,
    noOutput: 'Noch keine Ausgabe.',
    cancelling: 'Breche ab…',
    cancelInstall: 'Installation abbrechen',
    transcriptSaved: 'Vollständiges Protokoll gespeichert unter',
    copiedOutput: 'Kopiert!',
    copyOutput: 'Ausgabe kopieren',
    reloadRetry: 'Neu laden und erneut versuchen'
  },

  onboarding: {
    headerTitle: 'Lass uns dich mit Hermes Agent einrichten',
    headerDesc: 'Verbinde einen Modell-Provider, um zu chatten. Die meisten Optionen brauchen nur einen Klick.',
    preparingInstall: 'Hermes schließt die Installation ab. Beim ersten Start dauert das meist unter einer Minute.',
    starting: 'Starte Hermes…',
    lookingUpProviders: 'Suche Provider…',
    collapse: 'Einklappen',
    otherProviders: 'Andere Provider',
    haveApiKey: 'Ich habe einen API-Key',
    chooseLater: 'Ich wähle später einen Provider',
    recommended: 'Empfohlen',
    connected: 'Verbunden',
    featuredPitch: 'Ein Abo, 300+ Frontier-Modelle — der empfohlene Weg, Hermes zu betreiben',
    openRouterPitch: 'Ein Key, hunderte Modelle — ein solider Standard',
    apiKeyOptions: {
      openrouter: {
        short: 'ein Key, viele Modelle',
        description: 'Hostet hunderte Modelle hinter einem einzigen Key. Guter Standard für neue Installationen.'
      },
      openai: { short: 'GPT-Klasse-Modelle', description: 'Direkter Zugriff auf OpenAI-Modelle.' },
      gemini: { short: 'Gemini-Modelle', description: 'Direkter Zugriff auf Google-Gemini-Modelle.' },
      xai: { short: 'Grok-Modelle', description: 'Direkter Zugriff auf xAI-Grok-Modelle.' },
      local: {
        short: 'selbst gehostet',
        description:
          'Richte Hermes auf einen lokalen oder selbst gehosteten OpenAI-kompatiblen Endpoint aus (vLLM, llama.cpp, Ollama usw.).'
      }
    },
    backToSignIn: 'Zurück zur Anmeldung',
    getKey: 'Key holen',
    replaceCurrent: 'Aktuellen Wert ersetzen',
    pasteApiKey: 'API-Key einfügen',
    localApiKeyPlaceholder: 'API-Key (optional — nur falls dein Endpoint einen verlangt)',
    couldNotSave: 'Zugangsdaten konnten nicht gespeichert werden.',
    connecting: 'Verbinde',
    update: 'Aktualisieren',
    flowSubtitles: {
      pkce: 'Öffnet deinen Browser zur Anmeldung, dann geht es hier weiter',
      device_code: 'Öffnet eine Verifizierungsseite in deinem Browser — Hermes verbindet sich automatisch',
      loopback: 'Öffnet deinen Browser zur Anmeldung — Hermes verbindet sich automatisch',
      external: 'Melde dich einmal in deinem Terminal an, dann komm zurück zum Chatten'
    },
    startingSignIn: provider => `Starte Anmeldung für ${provider}…`,
    verifyingCode: provider => `Verifiziere deinen Code mit ${provider}…`,
    connectedProvider: provider => `${provider} verbunden`,
    connectedPicking: provider => `${provider} verbunden. Wähle ein Standard-Modell…`,
    signInFailed: 'Anmeldung fehlgeschlagen. Versuche es erneut.',
    pickDifferentProvider: 'Anderen Provider wählen',
    signInWith: provider => `Mit ${provider} anmelden`,
    openedBrowser: provider => `Wir haben ${provider} in deinem Browser geöffnet.`,
    authorizeThere: 'Autorisiere Hermes dort.',
    copyAuthCode: 'Kopiere den Autorisierungscode und füge ihn unten ein.',
    pasteAuthCode: 'Autorisierungscode einfügen',
    reopenAuthPage: 'Autorisierungsseite erneut öffnen',
    autoBrowser: provider =>
      `Wir haben ${provider} in deinem Browser geöffnet. Autorisiere Hermes dort und du wirst automatisch verbunden — nichts zu kopieren oder einzufügen.`,
    reopenSignInPage: 'Anmeldeseite erneut öffnen',
    waitingAuthorize: 'Warte auf deine Autorisierung…',
    externalPending: provider =>
      `${provider} meldet sich über seine eigene CLI an. Führe diesen Befehl in einem Terminal aus, komm dann zurück und wähle „Ich habe mich angemeldet":`,
    signedIn: 'Ich habe mich angemeldet',
    deviceCodeOpened: provider => `Wir haben ${provider} in deinem Browser geöffnet. Gib dort diesen Code ein:`,
    reopenVerification: 'Verifizierungsseite erneut öffnen',
    copy: 'Kopieren',
    defaultModel: 'Standard-Modell',
    freeTier: 'Kostenlose Stufe',
    pro: 'Pro',
    free: 'Kostenlos',
    price: (input, output) => `${input} ein / ${output} aus pro Mtok`,
    change: 'Ändern',
    startChatting: 'Loslegen',
    docs: provider => `${provider}-Doku`
  },

  modelPicker: {
    title: 'Modell wechseln',
    current: 'aktuell:',
    unknown: '(unbekannt)',
    search: 'Provider und Modelle filtern…',
    noModels: 'Keine Modelle gefunden.',
    addProvider: 'Provider hinzufügen',
    loadFailed: 'Modelle konnten nicht geladen werden',
    noAuthenticatedProviders: 'Keine authentifizierten Provider.',
    pro: 'Pro',
    proNeedsSubscription: 'Pro-Modelle benötigen ein kostenpflichtiges Nous-Abo.',
    free: 'Kostenlos',
    freeTier: 'Kostenlose Stufe',
    priceTitle: 'Ein-/Ausgabe-Preis pro Million Token'
  },

  modelVisibility: {
    title: 'Modelle',
    search: 'Modelle durchsuchen',
    noAuthenticatedProviders: 'Keine authentifizierten Provider.',
    addProvider: 'Provider hinzufügen…'
  },

  shell: {
    windowControls: 'Fenster-Steuerung',
    paneControls: 'Bereichs-Steuerung',
    appControls: 'App-Steuerung',
    modelMenu: {
      search: 'Modelle durchsuchen',
      noModels: 'Keine Modelle gefunden',
      editModels: 'Modelle bearbeiten…',
      refreshModels: 'Modelle aktualisieren',
      fast: 'Schnell',
      medium: 'Mittel'
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
      max: 'Max',
      updateFailed: 'Aktualisierung der Modell-Option fehlgeschlagen',
      fastFailed: 'Aktualisierung des Schnell-Modus fehlgeschlagen'
    },
    gatewayMenu: {
      gateway: 'Gateway',
      connected: 'Verbunden',
      connecting: 'Verbinde',
      offline: 'Offline',
      inferenceReady: 'Inferenz bereit',
      inferenceNotReady: 'Inferenz nicht bereit',
      checkingInference: 'Prüfe Inferenz',
      disconnected: 'Getrennt',
      openSystem: 'System-Panel öffnen',
      connection: label => `Verbindung: ${label}`,
      recentActivity: 'Letzte Aktivität',
      viewAllLogs: 'Alle Logs ansehen →',
      messagingPlatforms: 'Messaging-Plattformen'
    },
    statusbar: {
      unknown: 'unbekannt',
      restart: 'Neustart',
      update: 'Update',
      updateInProgress: 'Update läuft',
      commitsBehind: (count, branch) => `${count} Commit${count === 1 ? '' : 's'} hinter ${branch}`,
      desktopVersion: version => `Hermes Desktop v${version}`,
      backendVersion: version => `Backend v${version}`,
      clientLabel: version => `Client v${version}`,
      backendLabel: version => `Backend v${version}`,
      commit: sha => `Commit ${sha}`,
      branch: branch => `Branch ${branch}`,
      closeCommandCenter: 'Command Center schließen',
      openCommandCenter: 'Command Center öffnen',
      showTerminal: 'Terminal anzeigen',
      hideTerminal: 'Terminal ausblenden',
      gateway: 'Gateway',
      gatewayReady: 'bereit',
      gatewayNeedsSetup: 'Einrichtung nötig',
      gatewayChecking: 'prüfe',
      gatewayConnecting: 'verbinde',
      gatewayOffline: 'offline',
      gatewayRestarting: 'starte neu…',
      gatewayTitle: 'Status des Hermes-Inferenz-Gateways',
      agents: 'Agents',
      closeAgents: 'Agents schließen',
      openAgents: 'Agents öffnen',
      subagents: count => `${count} ${count === 1 ? 'Subagent' : 'Subagents'}`,
      failed: count => `${count} fehlgeschlagen`,
      running: count => `${count} laufen`,
      cron: 'Cron',
      openCron: 'Cron-Jobs öffnen',
      turnRunning: 'Läuft',
      currentTurnElapsed: 'Vergangene Zeit des aktuellen Durchlaufs',
      contextUsage: 'Kontext-Nutzung',
      session: 'Sitzung',
      runtimeSessionElapsed: 'Vergangene Laufzeit der Sitzung',
      yoloOn:
        'YOLO an — gefährliche Befehle werden automatisch freigegeben. Klick zum Ausschalten. Shift+Klick schaltet es global um.',
      yoloOff: 'YOLO aus — klicke, um gefährliche Befehle automatisch freizugeben. Shift+Klick schaltet es global um.',
      modelNone: 'keins',
      noModel: 'kein Modell',
      switchModel: 'Modell wechseln',
      openModelPicker: 'Modell-Auswahl öffnen',
      modelTitle: (provider, model) => `Modell · ${provider}: ${model}`,
      providerModelTitle: (provider, model) => `${provider} · ${model}`
    }
  },

  rightSidebar: {
    aria: 'Rechte Seitenleiste',
    panelsAria: 'Panels der rechten Seitenleiste',
    files: 'Dateisystem',
    terminal: 'Terminal',
    noFolderSelected: 'Kein Ordner ausgewählt',
    changeCwdTitle: 'Arbeitsverzeichnis ändern',
    remotePickerTitle: 'Remote-Ordner wählen',
    remotePickerDescription: 'Durchsuche Ordner auf dem verbundenen Backend.',
    remotePickerSelect: 'Ordner auswählen',
    folderTip: cwd => `${cwd} — klicken, um den Ordner zu wechseln`,
    openFolder: 'Ordner öffnen',
    refreshTree: 'Baum aktualisieren',
    collapseAll: 'Alle Ordner einklappen',
    previewUnavailable: 'Vorschau nicht verfügbar',
    couldNotPreview: path => `Vorschau für ${path} nicht möglich`,
    noProjectTitle: 'Kein Projekt',
    noProjectBody: 'Setze über die Statusleiste ein Arbeitsverzeichnis, um Dateien zu durchsuchen.',
    unreadableTitle: 'Nicht lesbar',
    unreadableBody: error => `Dieser Ordner konnte nicht gelesen werden (${error}).`,
    emptyTitle: 'Leer',
    emptyBody: 'Dieser Ordner ist leer.',
    treeErrorTitle: 'Baum-Fehler',
    treeErrorBody: 'Beim Rendern dieses Ordners ist im Datei-Baum ein Fehler aufgetreten.',
    tryAgain: 'Erneut versuchen',
    loadingTree: 'Lade Datei-Baum',
    loadingFiles: 'Lade Dateien',
    terminalHide: 'Terminal ausblenden',
    addToChat: 'Zum Chat hinzufügen'
  },

  preview: {
    tab: 'Vorschau',
    closeTab: label => `${label} schließen`,
    closePane: 'Vorschau-Bereich schließen',
    loading: 'Lade Vorschau',
    unavailable: 'Vorschau nicht verfügbar',
    opening: 'Öffne…',
    hide: 'Ausblenden',
    openPreview: 'Vorschau öffnen',
    openInBrowser: 'Im Browser öffnen',
    sourceLineTitle: 'Klicken zum Auswählen · Shift-Klick zum Erweitern · in die Eingabe ziehen',
    source: 'QUELLE',
    renderedPreview: 'VORSCHAU',
    unknownSize: 'unbekannte Größe',
    binaryTitle: 'Das sieht nach einer Binärdatei aus',
    binaryBody: label => `Die Vorschau von ${label} zeigt möglicherweise unlesbaren Text.`,
    largeTitle: 'Diese Datei ist groß',
    largeBody: (label, size) => `${label} ist ${size}. Hermes zeigt nur die ersten 512 KB.`,
    previewAnyway: 'Trotzdem Vorschau',
    truncated: 'Zeige die ersten 512 KB.',
    noInlineTitle: 'Keine Inline-Vorschau',
    noInlineBody: mimeType => `${mimeType || 'Dieser Dateityp'} kann trotzdem als Kontext angehängt werden.`,
    console: {
      deselect: 'Eintrag abwählen',
      select: 'Eintrag auswählen',
      copyFailed: 'Konsolen-Ausgabe konnte nicht kopiert werden',
      copyEntry: 'Diesen Eintrag kopieren',
      sendEntry: 'Diesen Eintrag an den Chat senden',
      messages: count => `${count} Konsolen-Nachrichten`,
      resize: 'Vorschau-Konsole skalieren',
      title: 'Vorschau-Konsole',
      selected: count => `${count} ausgewählt`,
      sendToChat: 'An den Chat senden',
      copySelected: 'Auswahl in die Zwischenablage kopieren',
      copyAll: 'Alles in die Zwischenablage kopieren',
      copy: 'Kopieren',
      clear: 'Leeren',
      empty: 'Noch keine Konsolen-Nachrichten.',
      promptHeader: 'Vorschau-Konsole:',
      sentTitle: 'An den Chat gesendet',
      sentMessage: count => `${count} Log-Eintr${count === 1 ? 'ag' : 'äge'} zur Eingabe hinzugefügt`
    },
    web: {
      appFailedToBoot: 'Vorschau-App konnte nicht starten',
      serverNotFound: 'Server nicht gefunden',
      failedToLoad: 'Vorschau konnte nicht geladen werden',
      tryAgain: 'Erneut versuchen',
      restarting: 'Hermes startet neu…',
      askRestart: 'Hermes bitten, den Server neu zu starten',
      lookingRestart: taskId => `Hermes sucht einen Vorschau-Server zum Neustarten (${taskId})`,
      restartingTitle: 'Starte Vorschau-Server neu',
      restartingMessage: 'Hermes arbeitet im Hintergrund. Beobachte die Vorschau-Konsole für den Fortschritt.',
      startRestartFailed: message => `Server-Neustart konnte nicht gestartet werden: ${message}`,
      restartFailed: 'Server-Neustart fehlgeschlagen',
      hideConsole: 'Vorschau-Konsole ausblenden',
      showConsole: 'Vorschau-Konsole anzeigen',
      hideDevTools: 'Vorschau-DevTools ausblenden',
      openDevTools: 'Vorschau-DevTools öffnen',
      finishedRestarting: message => `Hermes hat den Vorschau-Server neu gestartet${message ? `: ${message}` : ''}`,
      failedRestarting: message => `Server-Neustart fehlgeschlagen: ${message}`,
      unknownError: 'unbekannter Fehler',
      restartedTitle: 'Vorschau-Server neu gestartet',
      reloadingNow: 'Lade die Vorschau jetzt neu.',
      restartFailedTitle: 'Vorschau-Neustart fehlgeschlagen',
      restartFailedMessage: 'Hermes konnte den Server nicht neu starten.',
      stillWorking:
        'Hermes arbeitet noch, aber es ist noch kein Neustart-Ergebnis eingetroffen. Der Server-Befehl läuft möglicherweise im Vordergrund.',
      workspaceReloading: 'Arbeitsbereich geändert, lade Vorschau neu',
      fileChanged: url => `Datei geändert, lade Vorschau neu: ${url}`,
      filesChanged: (count, url) => `${count} Datei-Änderungen, lade Vorschau neu: ${url}`,
      watchFailed: message => `Vorschau-Datei konnte nicht überwacht werden: ${message}`,
      moduleMimeDescription:
        'Modul-Skripte werden mit dem falschen MIME-Typ ausgeliefert. Das bedeutet meist, dass ein statischer Datei-Server eine Vite/React-App ausliefert statt des Projekt-Dev-Servers.',
      loadFailedConsole: (code, message) => `Laden fehlgeschlagen${code ? ` (${code})` : ''}: ${message}`,
      unreachableDescription: 'Die Vorschau-Seite war nicht erreichbar.',
      openTarget: url => `${url} öffnen`,
      fallbackTitle: 'Vorschau'
    }
  },

  assistant: {
    thread: {
      loadingSession: 'Lade Sitzung',
      showEarlier: 'Frühere Nachrichten anzeigen',
      loadingResponse: 'Hermes lädt eine Antwort',
      thinking: 'Denke nach',
      today: time => `Heute, ${time}`,
      yesterday: time => `Gestern, ${time}`,
      copy: 'Kopieren',
      refresh: 'Aktualisieren',
      moreActions: 'Weitere Aktionen',
      branchNewChat: 'In neuem Chat verzweigen',
      dismissError: 'Fehler schließen',
      readAloudFailed: 'Vorlesen fehlgeschlagen',
      preparingAudio: 'Bereite Audio vor…',
      stopReading: 'Vorlesen stoppen',
      readAloud: 'Vorlesen',
      editMessage: 'Nachricht bearbeiten',
      scrollToBottom: 'Nach unten scrollen',
      stop: 'Stopp',
      restorePrevious: 'Vorherigen Checkpoint wiederherstellen',
      restoreCheckpoint: 'Checkpoint wiederherstellen',
      restoreFromHere: 'Checkpoint wiederherstellen — ab diesem Prompt erneut ausführen',
      restoreTitle: 'Auf diesen Checkpoint wiederherstellen?',
      restoreBody:
        'Alles nach diesem Prompt wird aus der Unterhaltung entfernt, und der Prompt läuft von hier aus erneut.',
      restoreConfirm: 'Wiederherstellen & erneut ausführen',
      restoreNext: 'Nächsten Checkpoint wiederherstellen',
      goForward: 'Vorwärts',
      sendEdited: 'Bearbeitete Nachricht senden',
      attachingFile: 'Hänge an…'
    },
    approval: {
      gatewayDisconnected: 'Das Hermes-Gateway ist nicht verbunden',
      sendFailed: 'Freigabe-Antwort konnte nicht gesendet werden',
      run: 'Ausführen',
      command: 'Befehl',
      moreOptions: 'Weitere Freigabe-Optionen',
      allowSession: 'Diese Sitzung erlauben',
      alwaysAllowMenu: 'Immer erlauben…',
      jumpToApproval: 'Freigabe erforderlich',
      reject: 'Ablehnen',
      alwaysTitle: 'Diesen Befehl immer erlauben?',
      alwaysDescription: pattern =>
        `Das fügt das Muster „${pattern}" zu deiner permanenten Allowlist hinzu (~/.hermes/config.yaml). Hermes fragt bei solchen Befehlen nicht mehr nach — weder in dieser Sitzung noch in zukünftigen.`,
      alwaysAllow: 'Immer erlauben'
    },
    clarify: {
      notReady: 'Die Rückfrage ist noch nicht bereit',
      gatewayDisconnected: 'Das Hermes-Gateway ist nicht verbunden',
      sendFailed: 'Rückfrage-Antwort konnte nicht gesendet werden',
      loadingQuestion: 'Lade Frage…',
      other: 'Andere (tippe deine Antwort)',
      placeholder: 'Tippe deine Antwort…',
      skip: 'Überspringen',
      continueLabel: 'Weiter'
    },
    tool: {
      code: 'Code',
      copyCode: 'Code kopieren',
      renderingImage: 'Rendere Bild',
      copyOutput: 'Ausgabe kopieren',
      copyCommand: 'Befehl kopieren',
      copyContent: 'Inhalt kopieren',
      copyUrl: 'URL kopieren',
      copyResults: 'Ergebnisse kopieren',
      copyQuery: 'Query kopieren',
      copyFile: 'Datei kopieren',
      copyPath: 'Pfad kopieren',
      outputAlt: 'Tool-Ausgabe',
      rawResponse: 'Rohe Antwort',
      copyActivity: 'Aktivität kopieren',
      recoveredOne: 'Nach 1 fehlgeschlagenen Schritt erholt',
      recoveredMany: count => `Nach ${count} fehlgeschlagenen Schritten erholt`,
      failedOne: '1 Schritt fehlgeschlagen',
      failedMany: count => `${count} Schritte fehlgeschlagen`,
      statusRunning: 'Läuft',
      statusError: 'Fehler',
      statusRecovered: 'Erholt',
      statusDone: 'Fertig'
    }
  },

  prompts: {
    gatewayDisconnected: 'Das Hermes-Gateway ist nicht verbunden',
    sudoSendFailed: 'sudo-Passwort konnte nicht gesendet werden',
    secretSendFailed: 'Secret konnte nicht gesendet werden',
    sudoTitle: 'Administrator-Passwort',
    sudoDesc:
      'Hermes braucht dein sudo-Passwort, um einen privilegierten Befehl auszuführen. Es wird nur an deinen lokalen Agenten gesendet.',
    sudoPlaceholder: 'sudo-Passwort',
    secretTitle: 'Secret erforderlich',
    secretDesc: 'Hermes braucht eine Zugangsinformation, um fortzufahren.',
    secretPlaceholder: 'Secret-Wert'
  },

  desktop: {
    audioReadFailed: 'Aufgenommenes Audio konnte nicht gelesen werden',
    sessionUnavailable: 'Sitzung nicht verfügbar',
    createSessionFailed: 'Neue Sitzung konnte nicht erstellt werden',
    promptFailed: 'Prompt fehlgeschlagen',
    providerCredentialRequired: 'Füge Provider-Zugangsdaten hinzu, bevor du deine erste Nachricht sendest.',
    emptySlashCommand: 'leerer Slash-Befehl',
    desktopCommands: 'Desktop-Befehle',
    skillCommandsAvailable: count => `${count} Skill-Befehle verfügbar.`,
    warningLine: message => `Warnung: ${message}`,
    yoloArmed: 'YOLO für diesen Chat scharfgeschaltet',
    yoloOff: 'YOLO aus',
    yoloSystem: active => `YOLO ${active ? 'an' : 'aus'} für diese Sitzung`,
    yoloTitle: 'YOLO',
    yoloToggleFailed: 'YOLO konnte nicht umgeschaltet werden',
    profileStatus: current =>
      `Profil: ${current}. Nutze /profile <name> oder die Auswahl „Neue Sitzung", um einen Chat in einem anderen Profil zu starten.`,
    unknownProfile: 'Unbekanntes Profil',
    noProfileNamed: (target, available) => `Kein Profil namens „${target}". Verfügbar: ${available}`,
    newChatsProfile: name => `Neue Chats nutzen Profil ${name}.`,
    setProfileFailed: 'Profil konnte nicht gesetzt werden',
    sttDisabled: 'Spracherkennung ist in den Einstellungen deaktiviert.',
    stopFailed: 'Stoppen fehlgeschlagen',
    regenerateFailed: 'Neu generieren fehlgeschlagen',
    editFailed: 'Bearbeiten fehlgeschlagen',
    resumeFailed: 'Fortsetzen fehlgeschlagen',
    resumeStrandedTitle: 'Diese Sitzung konnte nicht geladen werden',
    resumeStrandedBody:
      'Die Verbindung zu dieser Sitzung ist fehlgeschlagen und automatische Wiederholungen haben aufgegeben. Prüfe, ob das Gateway läuft, und versuche es erneut.',
    resumeRetry: 'Erneut versuchen',
    nothingToBranch: 'Nichts zum Verzweigen',
    branchNeedsChat: 'Starte oder setze einen Chat fort, bevor du verzweigst.',
    sessionBusy: 'Sitzung beschäftigt',
    branchStopCurrent: 'Stoppe den aktuellen Durchlauf, bevor du diesen Chat verzweigst.',
    branchNoText: 'Diese Nachricht hat keinen Text, von dem aus verzweigt werden kann.',
    branchTitle: n => `Entwurf: Verzweigung #${n}`,
    branchFailed: 'Verzweigen fehlgeschlagen',
    deleteFailed: 'Löschen fehlgeschlagen',
    archived: 'Archiviert',
    archiveFailed: 'Archivieren fehlgeschlagen',
    cwdChangeFailed: 'Wechsel des Arbeitsverzeichnisses fehlgeschlagen',
    cwdStagedTitle: 'Arbeitsverzeichnis vorgemerkt',
    cwdStagedMessage: 'Starte das Desktop-Backend neu, um die cwd-Änderungen auf diese aktive Sitzung anzuwenden.',
    modelSwitchFailed: 'Modell-Wechsel fehlgeschlagen',
    sessionExported: 'Sitzung exportiert',
    sessionExportFailed: 'Sitzung konnte nicht exportiert werden',
    imageSaved: 'Bild gespeichert',
    downloadStarted: 'Download gestartet',
    restartToUseSaveImage: 'Starte Hermes Desktop neu, um „Bild speichern" zu nutzen.',
    restartToSaveImages: 'Starte Hermes Desktop neu, um Bilder zu speichern',
    imageDownloadFailed: 'Bild-Download fehlgeschlagen',
    openImage: 'Bild öffnen',
    downloadImage: 'Bild herunterladen',
    savingImage: 'Speichere Bild',
    imagePreviewFailed: 'Bild-Vorschau fehlgeschlagen',
    imageAttach: 'Bild anhängen',
    imageWriteFailed: 'Bild konnte nicht auf die Festplatte geschrieben werden.',
    imageAttachFailed: 'Bild anhängen fehlgeschlagen',
    attachImages: 'Bilder anhängen',
    clipboard: 'Zwischenablage',
    noClipboardImage: 'Kein Bild in der Zwischenablage gefunden',
    clipboardPasteFailed: 'Einfügen aus der Zwischenablage fehlgeschlagen',
    dropFiles: 'Dateien ablegen',
    handoff: {
      pickPlatform: 'Ein Ziel wählen',
      success: platform => `An ${platform} übergeben. Hier jederzeit fortsetzbar.`,
      systemNote: platform => `↻ An ${platform} übergeben — hier jederzeit fortsetzbar.`,
      failed: error => `Übergabe fehlgeschlagen: ${error}`,
      timedOut: 'Zeitüberschreitung beim Warten auf das Gateway. Läuft `hermes gateway`?'
    }
  },

  errors: {
    genericFailure: 'Etwas ist schiefgelaufen',
    boundaryTitle: 'In der Oberfläche ist etwas kaputtgegangen',
    boundaryDesc: 'Die Ansicht ist auf einen unerwarteten Fehler gestoßen. Deine Chats und Einstellungen sind sicher.',
    reloadWindow: 'Fenster neu laden',
    openLogs: 'Logs öffnen'
  },

  ui: {
    search: {
      clear: 'Suche leeren'
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
      description: 'Zeigt die mobile Seitenleiste.',
      toggle: 'Seitenleiste umschalten'
    }
  }
})
