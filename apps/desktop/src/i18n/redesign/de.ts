import type { RedesignOverlay } from './types'

// German wording for the redesigned shell: only the keys whose text differs from the base catalog.
export const deRedesign: RedesignOverlay = {
  sharedMetrics: {
    collectedMilestones:
      'Einrichtung: welche Schritte abgeschlossen sind, Anbieterverbindungen, wie viele Skills, Plugins und Routinen',
    sending:
      'Wenn Sie teilen, sendet Hermes Statistiken einmal täglich mit einer zufälligen ID für diesen Agenten an Nous. Abgesehen von einem einmaligen Hinweis, dass Hermes installiert wurde, werden Statistiken von vor Ihrer Zustimmung nie gesendet. Sie können das jederzeit in den Einstellungen ändern.',
    sendDesc:
      'Nur Zählwerte, einmal täglich mit einer zufälligen ID für diesen Agenten gesendet. Niemals Ihre Nachrichten, Dateien, Pfade oder Fehlertexte.',
    consentBody:
      'Hermes kann zählen, wie Sie es nutzen: Chatdauer, welche Modelle und Tools laufen und wann etwas fehlschlägt. Ihre Nachrichten, Dateien, Pfade oder Fehlertexte werden nie aufgezeichnet.',
    collectedActivity: 'Chats: Dauer, Ergebnis, Fehlertyp, aktive Zeit pro Tag'
  },
  connectorsPage: {
    dialog: {
      disconnectBody:
        'Das Konto wird für alle Agenten getrennt, die dasselbe Nous-Konto verwenden, nicht nur für diesen Agenten. Sie können es jederzeit wieder verbinden.',
      nousLine: 'Nous-Apps folgen Ihrem Konto, nicht dem Agenten.',
      appSwitch: (name: string) => `Der Agent darf ${name} verwenden`,
      bothOn: (name: string) => `Beide sind an, daher sieht der Agent jedes ${name}-Tool doppelt.`,
      rulesSignIn: 'Melden Sie sich an, um festzulegen, was der Agent hier darf.'
    },
    accounts: {
      removeBody:
        'Das verbundene Konto wird für alle Agenten entfernt, die dasselbe Nous-Konto verwenden, nicht nur für diesen Agenten. Ihre anderen Konten bleiben verbunden.'
    },
    page: {
      noMatchBody: 'Keine Treffer. Fügen Sie Ihren eigenen MCP-Server hinzu, damit der Agent mehr Tools bekommt.'
    },
    tools: {
      summaryTitle: (name: string) => `Was der Agent mit ${name} tun darf`,
      summaryPreviewTitle: (name: string) => `Was der Agent mit ${name} tun könnte, sobald Sie verbunden sind`,
      goneBody:
        'Der Agent kann ihn nicht mehr aufrufen. Die Zeile bleibt, bis Sie sie entfernen, damit nichts verschwindet.'
    }
  },
  common: {
    bots: 'Agenten'
  },
  boot: {
    steps: {
      connectingGateway: 'Verbindung zum Rechner wird hergestellt',
      loadingSessions: 'Letzte Chats werden geladen'
    },
    errors: {
      gatewayConnectionLost: 'Verbindung zum Rechner verloren',
      gatewayConnectionLostDetail:
        'Im Hintergrund wird weiterhin versucht, die Verbindung herzustellen. Sie können weiterlesen und weiterschreiben – öffnen Sie die Rechnereinstellungen, falls das anhält.',
      gatewaySignInRequired: 'Anmeldung am Rechner erforderlich'
    },
    failure: {
      description:
        'Der Hintergrunddienst von Hermes ist nicht gestartet. Probieren Sie einen der Wiederherstellungsschritte unten. Keiner davon löscht Ihre Chats oder Einstellungen.',
      remoteTitle: 'Anmeldung am Remote-Rechner erforderlich',
      remoteDescription:
        'Ihre Session auf dem Remote-Rechner ist abgelaufen. Melden Sie sich erneut an, um die Verbindung wiederherzustellen. Keiner dieser Schritte löscht Ihre Chats oder Einstellungen.',
      useLocalGateway: 'Lokalen Rechner verwenden',
      gatewaySettings: 'Rechnereinstellungen',
      remoteSignInHint: signInLabel =>
        `Meldet Sie von der gespeicherten Remote-Browser-Session ab und öffnet dann ${signInLabel}. Verwenden Sie den lokalen Rechner, um stattdessen zum integrierten Backend zu wechseln.`,
      remoteFailureHint:
        'Überprüfen Sie die Rechner-URL und die Anmeldung in den Rechnereinstellungen, oder wechseln Sie zum lokalen Rechner.',
      cloudDownDescription:
        'Der von Nous verwaltete Cloud-Agent, mit dem sich diese App verbindet, meldet einen Serverfehler. Er kann von hier aus nicht neu gestartet werden – prüfen Sie seinen Status, wechseln Sie zum lokalen Rechner oder wenden Sie sich an den Support.',
      signedInMessage: 'Wird mit dem Remote-Rechner neu verbunden…',
      signInToRemoteGateway: 'Am Remote-Rechner anmelden'
    }
  },
  notifications: {
    errors: {
      gatewayAuthFailed:
        'Dieses Hermes akzeptiert Ihre gespeicherte Anmeldung nicht mehr. Öffnen Sie Rechner und melden Sie sich erneut an (oder fügen Sie ein neues Zugriffstoken ein), und versuchen Sie es dann erneut.'
    },
    actions: {
      openGateways: 'Rechner öffnen'
    },
    voice: {
      couldNotStartSession: 'Sprach-Chat konnte nicht gestartet werden',
      liveEnded: 'Live-Sprach-Chat beendet',
      liveEndedConnectionLost: 'Der Live-Sprach-Chat hat die Verbindung verloren.',
      liveEndedClosed: 'Der Live-Sprach-Chat wurde vom Dienst geschlossen.',
      liveDelegationFailed: 'Anfrage konnte nicht an den Agenten übergeben werden'
    },
    native: {
      inputBody: 'Der Agent wartet auf Ihre Antwort.',
      turnDoneTitle: 'Agent fertig'
    }
  },
  promptNotices: {
    desktopCommands: {
      configFooter:
        'Ändere Einstellungen unter Einstellungen oder bearbeite config.yaml mit `hermes config edit` auf dem verbundenen Rechner.',
      saveLegacyRefused:
        'Nicht gespeichert: Hermes auf diesem Rechner kann keine geschwärzte, benannte oder Nicht-JSON-Kopie exportieren. Aktualisiere Hermes oder führe /save ohne Optionen aus, um rohes JSON auf dem verbundenen Rechner zu schreiben.',
      savedOnHost: (path: string) => `Auf dem verbundenen Rechner gespeichert: ${path}`,
      exportRemote:
        '/export packt einen Agenten auf dem verbundenen Rechner, und das ist nicht dieser Rechner. Führe `hermes profile export` auf dem verbundenen Rechner aus.',
      importRemote:
        '/import installiert einen Agenten auf dem verbundenen Rechner, und das ist nicht dieser Rechner. Führe `hermes profile import <archive>` auf dem verbundenen Rechner aus.',
      historyHermes: 'Agent',
      checkpointsDisabled:
        'Checkpoints sind für diesen Chat nicht aktiviert (checkpoints.enabled in config.yaml setzen).'
    },
    sessionCommands: {
      busy: (command: string) =>
        `Chat beschäftigt — der Agent antwortet noch. Beende zuerst die aktuelle Antwort (Stopp-Taste oder Strg+C im Terminal), dann führe /${command} aus.`,
      busyCompressing: (command: string) =>
        `Chat beschäftigt — ein /compress läuft; führe /${command} aus, wenn es fertig ist.`,
      titleSet: (title, queued) => `Chat-Titel gesetzt: ${title}${queued ? ' (wartet auf den Start des Chats)' : ''}`,
      titleCleared: 'Chat-Titel entfernt.'
    }
  },
  sendDiagnostics: {
    privacyNotice:
      'Damit laden Sie ein Debug-Paket in den internen Nous-Speicher hoch (kein öffentliches Paste). Es enthält Systeminfos (Betriebssystem, Versionen, Provider, welche API-Keys konfiguriert sind – niemals die Keys selbst) sowie vollständige Logs des Agenten, des Hermes-Dienstes und der Desktop-App (bis zu 512 KB je Datei), die sehr wahrscheinlich Gesprächsinhalte, Tool-Ausgaben und Dateipfade enthalten. Geheimnisse werden vor dem Upload geschwärzt. Das Paket ist nur für Nous-Mitarbeitende und freigeschaltete Discord-Moderatoren einsehbar und wird nach 14 Tagen automatisch gelöscht.'
  },
  keybinds: {
    categories: {
      profiles: 'Agenten'
    },
    actions: {
      'nav.profiles': 'Agenten öffnen',
      'nav.cron': 'Routinen öffnen',
      'view.toggleProfileRail': 'Agenten-Leiste ein-/ausblenden',
      'profile.default': 'Zum Standardagenten wechseln',
      'profile.switch.1': 'Zu Agent 1 wechseln',
      'profile.switch.2': 'Zu Agent 2 wechseln',
      'profile.switch.3': 'Zu Agent 3 wechseln',
      'profile.switch.4': 'Zu Agent 4 wechseln',
      'profile.switch.5': 'Zu Agent 5 wechseln',
      'profile.switch.6': 'Zu Agent 6 wechseln',
      'profile.switch.7': 'Zu Agent 7 wechseln',
      'profile.switch.8': 'Zu Agent 8 wechseln',
      'profile.switch.9': 'Zu Agent 9 wechseln',
      'profile.switch.10': 'Zu Agent 10 wechseln',
      'profile.switch.11': 'Zu Agent 11 wechseln',
      'profile.switch.12': 'Zu Agent 12 wechseln',
      'profile.switch.13': 'Zu Agent 13 wechseln',
      'profile.switch.14': 'Zu Agent 14 wechseln',
      'profile.switch.15': 'Zu Agent 15 wechseln',
      'profile.switch.16': 'Zu Agent 16 wechseln',
      'profile.switch.17': 'Zu Agent 17 wechseln',
      'profile.switch.18': 'Zu Agent 18 wechseln',
      'profile.next': 'Nächster Agent',
      'profile.prev': 'Vorheriger Agent',
      'profile.toggleAll': 'Ansicht aller Agenten umschalten',
      'profile.create': 'Agent erstellen',
      'agents.new': 'Neuer Agent (Seite „Agenten“)',
      'nav.artifacts': 'Bibliothek öffnen',
      'session.new': 'Neuer Chat',
      'session.newTab': 'Neuer Chat-Tab',
      'session.next': 'Nächster Chat',
      'session.prev': 'Vorheriger Chat',
      'session.slot.1': 'Zu letztem Chat 1 wechseln',
      'session.slot.2': 'Zu letztem Chat 2 wechseln',
      'session.slot.3': 'Zu letztem Chat 3 wechseln',
      'session.slot.4': 'Zu letztem Chat 4 wechseln',
      'session.slot.5': 'Zu letztem Chat 5 wechseln',
      'session.slot.6': 'Zu letztem Chat 6 wechseln',
      'session.slot.7': 'Zu letztem Chat 7 wechseln',
      'session.slot.8': 'Zu letztem Chat 8 wechseln',
      'session.slot.9': 'Zu letztem Chat 9 wechseln',
      'session.focusSearch': 'Chats durchsuchen',
      'session.togglePin': 'Aktuellen Chat anheften / lösen',
      'session.archive': 'Aktuellen Chat archivieren',
      'view.toggleSidebar': 'Chat-Sidebar umschalten',
      'view.cycleSidebarGrouping': 'Chat-Gruppierung wechseln'
    }
  },
  settings: {
    subpages: {
      gatewayConnection: 'Primärer Rechner'
    },
    nav: {
      gateway: 'Rechner'
    },
    plugins: {
      installModal: {
        profileLabel: 'Für Agenten installieren',
        desktopTargetFromPackage: 'Aus dem Paket oben in diese App geladen — gleich für jeden Agenten'
      }
    },
    searchPlaceholder: {
      gateway: 'Rechnerverbindung…',
      sessions: 'Archivierte Chats durchsuchen…'
    },
    appearance: {
      vibeHeartsDesc:
        'Schwebende Herzen, wenn Sie sich bedanken oder ein Herz senden. Unabhängig von den Nachrichten-Reaktionen oben.',
      themeProfileNote: profile => `Für den Agenten ${profile} gespeichert — jeder Agent behält sein eigenes Theme.`,
      sessionDensityTitle: 'Dichte der Chat-Liste',
      sessionDensityDesc: 'Wählen Sie, wie viel Kontext unter den Chat-Titeln in der Seitenleiste erscheint.',
      reactionsDesc:
        'Emoji-Tapbacks im iMessage-Stil – reagieren Sie auf Nachrichten, und der Agent kann auf Ihre reagieren.',
      tipsDesc:
        'Gelegentliche Hinweise der App und Ihres Agenten. Jeder Tipp erscheint nur einmal. Schaltet sich nach Ihren ersten 30 Tagen automatisch aus; Sie können ihn wieder einschalten.',
      toursDesc:
        'Lassen Sie den Agenten jeden Schritt hervorheben, während er Sie durch die App führt. Schaltet sich nach Ihren ersten 30 Tagen automatisch aus; Sie können es wieder einschalten.',
      pet: {
        intro:
          'Adoptieren Sie ein animiertes Petdex-Maskottchen, das über der App schwebt und darauf reagiert, was der Agent gerade tut – es rennt, während Tools laufen, feiert bei Erfolg und schmollt bei Fehlern.'
      }
    },
    uninstallSection: {
      options: {
        full: {
          description:
            'Entfernt die App, den Agenten und alle Benutzerdaten – Konfiguration, Chats, Routinen, Geheimnisse, Logs.'
        }
      }
    },
    about: {
      channel: {
        heldBody: (v: string) =>
          `Dieser Build ist neuer als das stabile Release ${v}, und deine Chats nutzen bereits seine Hermes-Laufzeit, die ${v} nicht öffnen kann. Hermes bleibt bei diesem Build, bis ein neueres stabiles Release erscheint.`
      }
    },
    connections: {
      title: 'Registrierte Rechner',
      intro:
        'Verwalten Sie dieses Gerät und jeden Rechner mit Hermes, den es über Remote-, SSH- oder Cloud-Verbindungen erreichen kann.',
      stagedNote:
        'Jeder Rechner bleibt verbunden. Agenten, Chats, Nachrichten und Routinen bleiben auf ihrem Rechner, und Chats kann alle gleichzeitig anzeigen.',
      launchModeTitle: 'Beim Start zu Chats auf dem zuletzt verwendeten Rechner zurückkehren',
      launchModeDesc: 'Wenn deaktiviert, öffnet Chats auf dem primären Rechner.',
      searchPlaceholder: 'Rechner durchsuchen…',
      noSearchResults: 'Keine Rechner entsprechen Ihrer Suche.',
      kindRemote: 'Remote-Rechner',
      kindRemoteDesc: 'Ein Rechner mit Hermes, der über HTTP(S) erreichbar ist – LAN, Tailscale oder das Internet.',
      urlTitle: 'Rechner-URL',
      headersTitle: 'Zusätzliche Verbindungsheader',
      headersDesc:
        'Wird mit jeder HTTP- und WebSocket-Anfrage an diesen Rechner gesendet – für Zugriffs-Proxys wie Cloudflare Access (CF-Access-Client-Id / CF-Access-Client-Secret). Werte werden verschlüsselt gespeichert. Header, die Hermes verwaltet (Authorization, Cookie, Host…), werden ignoriert.',
      duplicateUrl: (label: string) => `Eine Verbindung zu dieser Rechner-URL existiert bereits („${label}“).`
    },
    managedUpdates: {
      intro:
        'Aktualisiert von Desktop verwaltete SSH-Installationen transaktional: laufende Chats werden beendet, das Remote-Repository wird aktualisiert und jeder Agent wird mit einer zugehörigen Quittung wiederhergestellt.',
      progress:
        'Laufende Chats werden beendet, die Remote-Installation wird aktualisiert und Agenten werden wiederhergestellt…',
      scopesRestored: (profiles: string) => `Wiederhergestellte Agenten: ${profiles}`,
      scopeNotRestored: (profile: string, error: string) => `Agent „${profile}“ nicht wiederhergestellt: ${error}`
    },
    gateway: {
      loading: 'Rechnereinstellungen werden geladen…',
      unavailableTitle: 'Rechnereinstellungen nicht verfügbar',
      unavailableDesc: 'Die Desktop-IPC-Brücke stellt keine Rechnereinstellungen bereit.',
      webappHostDesc:
        'Die Webapp läuft immer auf dem Rechner, der sie bereitstellt. Die gespeicherten Verbindungen unten sind die Liste dieses Rechners, geteilt mit der Hermes-Desktop-App; andere Rechner werden über diesen Rechner erreicht.',
      webappOauthPaste: {
        description:
          'Wenn der Anmelde-Tab auf einer Adresse endete, die nicht lädt (dieser Browser läuft nicht auf dem Rechner mit Hermes), füge diese Adresse hier ein.'
      },
      webAccess: {
        pendingOn: 'Gespeichert. Hermes stellt die Web-App nach seinem Neustart bereit.',
        pendingOff: 'Ausgeschaltet, aber Hermes stellt die Web-App bis zu seinem Neustart weiter bereit.',
        unavailableDetail: reason => `Hermes konnte die Web-App nicht bereitstellen: ${reason}`,
        restartBusy:
          'Gespeichert. Ein Chat läuft, deshalb wurde Hermes nicht neu gestartet. Wähle „Hermes neu starten“, sobald er beendet ist.',
        restartFailed:
          'Gespeichert, aber Hermes konnte nicht neu starten. Wähle „Hermes neu starten“, um es erneut zu versuchen.',
        restartFailedBecause: reason => `Hermes konnte nicht neu starten: ${reason}`,
        startTimedOut: 'Hermes hat die Web-App nicht innerhalb von 90 Sekunden bereitgestellt.',
        stopTimedOut: 'Hermes hat die Bereitstellung der Web-App nicht innerhalb von 90 Sekunden beendet.',
        startStages: {
          restarting: 'Hermes wird neu gestartet, um die Web-App bereitzustellen.'
        },
        restartGateway: 'Hermes neu starten'
      },
      title: 'Rechnerverbindung',
      intro:
        'Standardmäßig lokal. Verwenden Sie Remote, wenn diese App Hermes an einem anderen Ort ausführen soll. Verbinden Sie einen Rechner, um seine Agenten zu entdecken.',
      localTitle: 'Lokaler Rechner',
      remoteTitle: 'Remote-Rechner',
      remoteAuthHint:
        'Gehostete Rechner verwenden OAuth oder Benutzername und Passwort; selbst gehostete können ein Session-Token verwenden.',
      cloudSavedTitle: 'Gespeicherte Cloud-Rechner',
      cloudSavedDesc:
        'Verwenden Sie einen gespeicherten Rechner, ohne Ihren Standard zu ändern. Melden Sie sich unten an, um Instanzen hinzuzufügen. Namen und Anmeldung verwalten Sie in der Liste der gespeicherten Verbindungen.',
      cloudUseSaved: 'Rechner verwenden',
      probing: 'Authentifizierungsmethode dieses Rechners wird geprüft…',
      probeError:
        'Dieser Rechner ist noch nicht erreichbar. Prüfen Sie die URL – die Authentifizierungsmethode erscheint, sobald er antwortet.',
      authSignedInPassword:
        'Dieser Rechner verwendet Benutzername und Passwort. Sie sind angemeldet; die Session wird automatisch aktualisiert.',
      authSignedInOauth:
        'Dieser Rechner verwendet OAuth. Sie sind angemeldet; die Session wird automatisch aktualisiert.',
      authNeedsPassword:
        'Dieser Rechner verwendet Benutzername und Passwort. Melden Sie sich an, um diese Desktop-App zu autorisieren.',
      authNeedsOauth: provider =>
        `Dieser Rechner verwendet OAuth. Melden Sie sich mit ${provider} an, um diese Desktop-App zu autorisieren.`,
      plainTextConfirmTitle: 'Verbindungstoken im Klartext speichern?',
      keychainEncryptionDesc:
        'Standardmäßig aus. Wenn aktiv, werden Verbindungstokens und Anmeldedaten mit dem Schlüsselbund Ihres Systems verschlüsselt (Schlüsselbundverwaltung, GNOME Keyring oder Windows DPAPI) – Ihr System fragt möglicherweise nach einer Erlaubnis oder einem Passwort. Wenn aus, werden sie als Klartextdateien gespeichert, die nur Ihr Benutzerkonto lesen kann.',
      diagnosticsDesc: 'Zeigt desktop.log in Ihrem Dateimanager an – nützlich, wenn Hermes nicht startet.',
      incompleteTitle: 'Remote-Rechner unvollständig eingerichtet',
      restartingTitle: 'Rechnerverbindung wird wiederhergestellt',
      savedTitle: 'Rechnereinstellungen gespeichert',
      reachableTitle: 'Remote-Rechner erreichbar',
      signedOutMessage: 'Die Session auf dem Remote-Rechner wurde geleert.',
      failedLoad: 'Rechnereinstellungen konnten nicht geladen werden',
      testFailed: 'Verbindungstest zum Remote-Rechner fehlgeschlagen',
      applyFailed: 'Rechnereinstellungen konnten nicht angewendet werden',
      saveFailed: 'Rechnereinstellungen konnten nicht gespeichert werden',
      webappHostTitle: 'Hermes-Rechner',
      envOverrideTitle: 'Diese Verbindung wurde durch den Start von Hermes festgelegt.',
      cloudSignedInDesc:
        'Sie sind angemeldet. Wählen Sie unten einen Agenten; die Session wird automatisch aktualisiert.',
      sshDesc:
        'Hermes wird per SSH auf dem Remote-Rechner gestartet und in diese App getunnelt – Sie müssen nichts selbst starten oder freigeben. Erfordert funktionierenden, schlüsselbasierten SSH-Zugriff auf den Host.'
    },
    profileScope: {
      editsProfile: profile => `Änderungen auf dieser Seite gelten für den Agenten „${profile}“.`
    },
    model: {
      tasks: {
        profile_describer: {
          label: 'Agenten-Beschreiber',
          hint: 'Automatische Agentenbeschreibungen'
        },
        title_generation: {
          hint: 'Chat-Titel'
        }
      },
      subagents: {
        routing: {
          modes: {
            profile_routes: {
              label: 'Agentenrouten',
              description:
                'Subagenten laufen als einer Ihrer Agenten, mit dessen eigenem Modell, Skills und Schlüsseln.'
            }
          },
          pluginMissing: 'Modellrouten und Agentenrouten stammen aus dem Plugin Advanced Delegation.',
          pluginDisabled: 'Advanced Delegation ist installiert, aber für diesen Agenten ausgeschaltet.',
          blockedDisabled: 'Subagenten starten erst, wenn Advanced Delegation für diesen Agenten eingeschaltet ist.',
          profileRoutesTitle: 'Agenten, als die Subagenten laufen können',
          profileRoutesHint:
            'Jeder ausgewählte Agent bringt sein eigenes Modell, Skills, Anweisungen und Schlüssel mit.',
          profileDescriptionPlaceholder: 'Wann er genutzt wird (optional)',
          profileEmpty: 'Noch keine weiteren Agenten. Erstellen Sie einen über die Agenten-Leiste.',
          profileMissing: name => `${name} (Agent existiert nicht mehr)`,
          inactiveProfiles: profile =>
            `Agentenrouten wählen den Agenten, als den jeder Subagent läuft. Subagenten ohne gewählten Agenten laufen als „${profile}“.`,
          inactiveProfilesNoDefault:
            'Agentenrouten wählen den Agenten, als den jeder Subagent läuft. Subagenten ohne gewählten Agenten verwenden das Modell des Hauptagenten.'
        }
      },
      appliesDesc:
        'Gilt für neue Chats. Verwenden Sie die Modellauswahl im Composer, um im aktiven Chat schnell zu wechseln.',
      mainAppliedMessage: model => `Neue Chats verwenden ${model}.`
    },
    billing: {
      errors: {
        remoteSpendingReconnect: (who: string) =>
          `${who} Verbinden Sie sich unter Einstellungen -> Rechner erneut, um dieses Gerät neu zu autorisieren.`,
        sessionRevoked: {
          message: 'Ihre Session wurde abgemeldet. Melden Sie sich unter Einstellungen → Rechner erneut an.'
        },
        transport: {
          message: 'Die Abrechnungsanfrage ist fehlgeschlagen, bevor sie Hermes erreicht hat.'
        }
      }
    },
    vault: {
      otpHint:
        'Der „Einrichtungsschlüssel", den die Seite beim Aktivieren von 2FA anzeigt. Ist er gespeichert, erzeugt der Agent die Codes selbst.'
    },
    notifications: {
      kinds: {
        input: {
          description: 'Der Agent hat eine Frage gestellt oder braucht ein Passwort oder Geheimnis.'
        }
      }
    },
    fieldDescriptions: {
      'display.personality': 'Standard-Assistentenstil für neue Chats.',
      'memory.memoryEnabled': 'Dauerhafte Erinnerungen speichern, die zukünftigen Chats helfen können.'
    },
    memoryLayers: {
      selectedDesc: current => `Noch nicht aktiv: Chats nutzen weiter ${current}, bis diese Auswahl gespeichert ist.`,
      noProviderDesc:
        'Der Agent nutzt nur die integrierte Ebene unten. Wähle unter Einstellungen › Speicher & Kontext einen externen Anbieter, um einen hinzuzufügen.',
      providerDesc: name => `${name} ergänzt jeden Chat um eigene Erinnerungen, zusätzlich zur integrierten Ebene.`,
      builtinDesc:
        'MEMORY.md und USER.md sind der eigene Speicher des Agenten, unabhängig von externen Anbietern. Beide werden standardmäßig in Chats geladen; die Schalter für Speicher und Benutzerprofil unter Einstellungen › Speicher & Kontext schalten sie einzeln ab. Ein Anbieter ergänzt diese Ebene, er ersetzt sie nie.'
    },
    sessions: {
      loading: 'Archivierte Chats werden geladen…',
      archivedTitle: 'Archivierte Chats',
      defaultDirDesc:
        'Neue Chats starten in diesem Ordner, sofern Sie keinen anderen wählen. Nicht festlegen, um Ihr Home-Verzeichnis zu verwenden.',
      failedLoad: 'Archivierte Chats konnten nicht geladen werden'
    },
    toolsets: {
      modelSelectedMessage: model => `${model} gilt für neue Chats.`,
      terminalBackend: {
        selectedMessage: backend => `Terminal-Befehle laufen jetzt über ${backend}. Gilt für neue Chats.`,
        needsSetupConfirmDescription: (detail: string) =>
          `${detail} Chats, die nach dieser Änderung starten, haben keine Terminal- oder Datei-Tools, bis die Einrichtung abgeschlossen ist.`,
        needsSetupConfirmDescriptionGeneric:
          'Dieses Backend ist noch nicht eingerichtet. Chats, die nach dieser Änderung starten, haben keine Terminal- oder Datei-Tools, bis die Einrichtung abgeschlossen ist.',
        switchedToLocal: 'Terminalbefehle laufen jetzt lokal. Gilt für neue Chats.'
      },
      browserRealProfile: {
        description:
          'Kopiert die Anmeldungen und Cookies Ihres Standardbrowsers in einen verwalteten Schnappschuss, mit dem der Agent browst, und meldet auch den In-App-Browser damit an. Ihr Live-Profil wird nie direkt geöffnet. Gilt für neue Chats.',
        enabledMessage: 'Neue Chats browsen mit einem Schnappschuss Ihres Standard-Browserprofils.',
        disabledMessage:
          'Der Profil-Schnappschuss wird gelöscht und der In-App-Browser davon abgemeldet; neue Chats verwenden einen sauberen Browser.',
        prompt: {
          body: 'Lassen Sie den Agenten mit einem Schnappschuss Ihres Standard-Browserprofils browsen, damit Websites bereits angemeldet öffnen, für den Agenten und im In-App-Browser.'
        }
      }
    }
  },
  skills: {
    plugins: {
      agentBlurb: 'Erweitert den gewählten Agenten — Tools, Hooks, Anbieter. Wirkt nach einem Neustart von Hermes.',
      halfDesktopHint: 'diese App, gleich für jeden Agenten',
      halfAgentIn: profile => `Agent ${profile}`,
      installAgentHereTip: profile =>
        `Die Desktop-Hälfte ist in dieser App geladen, aber die Agent-Hälfte ist für ${profile} nicht installiert. Installieren Sie sie dort.`,
      installAgentHereNoOrigin:
        'Die Agent-Hälfte ist für diesen Agenten nicht installiert, und dieses Paket wurde von Hand hineinkopiert (kein Katalogeintrag und kein Git-Remote), daher kann es nicht von hier aus installiert werden. Kopieren Sie seinen Ordner in das Verzeichnis des Agenten oder installieren Sie es neu aus Git.',
      empty: 'Für diesen Agenten sind keine Plugins installiert.',
      alreadyInstalled: name => `${name} ist für diesen Agenten bereits installiert.`,
      updated: name =>
        `${name} wurde auf den aktuellen Katalog-Pin aktualisiert. Starten Sie Hermes neu, damit es wirksam wird.`,
      autoUpdateTip: (hours: number) =>
        `Plugin-Updates automatisch anwenden, wenn Hermes prüft (alle ${hours} h). Neuer Code wird in neuen Chats geladen.`,
      uninstallConfirmBody: (name: string, profile: string) =>
        `Dadurch werden die Dateien des Plugins aus dem Verzeichnis des Agenten ${profile} gelöscht. Eine mitgelieferte Desktop-Hälfte wird ebenfalls entfernt. Sie können es jederzeit aus dem Katalog oder aus Git neu installieren.`,
      uninstalled: (name: string) => `${name} deinstalliert. Starten Sie Hermes neu, um es zu entladen.`,
      settingsForm: {
        secretStoredAs: (env: string) =>
          `Wird in der .env des Agenten als ${env} gespeichert, nie in config.yaml; leer lassen, um den aktuellen Wert zu behalten.`
      },
      defaultProfile: 'Standardagent'
    },
    remotePluginHint:
      'Öffnen Sie diesen Agenten auf seinem Rechner, um seine Plugins zu verwalten. Die Plugin-Installation verwendet den verbundenen Rechner.',
    openAgentSettings: 'Agenten-Einstellungen',
    appliesToNewSessions: name => `${name} gilt für neue Chats.`,
    bulkUpdated: count => `${count} ${count === 1 ? 'Element' : 'Elemente'} für neue Chats aktualisiert.`,
    changesApplyNewSessions: 'Änderungen gelten für neue Chats.',
    openToolLibrary: 'Werkzeugkatalog durchsuchen'
  },
  gatewayGroupOpen: {
    lookupFailed: 'Diese Gruppe konnte auf dem Rechner nicht gefunden werden. Versuche es erneut.'
  },
  commandCenter: {
    sectionEntries: {
      system: {
        detail: 'Rechnerstatus, Logs, Neustart/Update'
      },
      sessions: {
        title: 'Chat-Bereich',
        detail: 'Chats durchsuchen, anpinnen und verwalten'
      }
    },
    gatewayRunning: 'Messaging läuft',
    gatewayStopped: 'Messaging gestoppt',
    restartGateway: 'Hermes neu starten',
    gatewayRestartFailed: 'Hermes-Neustart fehlgeschlagen.',
    sharedGatewayRestartTitle: 'Hermes für alle Agenten neu starten?',
    sharedGatewayRestartDescription: bots => `Alle Agenten auf diesem Gerät verbinden sich neu: ${bots}`,
    sharedGatewayRestarted: count => `Hermes neu gestartet (${count} ${count === 1 ? 'Agent' : 'Agenten'})`,
    searchPlaceholder: 'Chats, Ansichten und Aktionen durchsuchen',
    goToSession: 'Zum Chat gehen',
    newSessionInProject: project => `Neuer Chat in ${project}`,
    nav: {
      newChat: {
        title: 'Neuer Chat',
        detail: 'Einen neuen Chat starten'
      },
      artifacts: {
        title: 'Bibliothek'
      }
    },
    pinSession: 'Chat anpinnen',
    unpinSession: 'Chat lösen',
    exportSession: 'Chat exportieren',
    deleteSession: 'Chat löschen',
    noSessions: 'Noch keine Chats.',
    hermesActiveSessions: (version, count) => `Hermes ${version} · Aktive Chats ${count}`,
    maintenance: {
      backupDesc: 'Konfiguration, Memories, Skills und Chats als ZIP packen',
      resetUser: 'Benutzerprofil zurücksetzen'
    }
  },
  messaging: {
    states: {
      gateway_stopped: 'Messaging gestoppt'
    },
    hintPendingRestart: 'Starten Sie Hermes über die Statusleiste neu, um diese Änderung zu übernehmen.',
    sharedListenerUrl: 'Erreichbar unter',
    hintGatewayStopped: 'Starten Sie Hermes über die Statusleiste, um die Verbindung herzustellen.',
    gatewayStopped: 'Messaging gestoppt',
    restartToApply: 'Diese Änderung wird nach einem Hermes-Neustart wirksam.',
    restartToReconnect: 'Neue Zugangsdaten werden nach einem Hermes-Neustart wirksam.',
    appliedLive: 'Ohne Neustart von Hermes angewendet.',
    connectingLive: 'Hermes verbindet sich mit den neuen Zugangsdaten.',
    restartNeeded: 'Gespeichert. Starten Sie Hermes neu, damit die neuen Einstellungen wirksam werden.',
    restartFailedManual:
      'Hermes-Neustart fehlgeschlagen – starten Sie Hermes manuell neu und prüfen Sie die Hermes-Logs.',
    relayLink: {
      managedNote: (platform: string) =>
        `Team Gateway übernimmt auf dieser Instanz das Messaging, daher schaltet Hermes die direkte ${platform}-Verbindung bei jedem Start ab. Um sie parallel zum Relay zu betreiben, setze GATEWAY_RELAY_ALLOW_DIRECT_PLATFORMS=true in der Umgebung von Hermes (eine direkte Verbindung verhindert das Herunterskalieren auf null und kann Nachrichten doppelt zustellen).`
    },
    telegramQr: {
      savedRestarting: 'Telegram gespeichert; Hermes wird neu gestartet…',
      savedRestartFailed: detail => `Telegram gespeichert; Hermes-Neustart fehlgeschlagen${detail}`
    },
    fieldCopy: {
      DISCORD_HOME_CHANNEL: {
        help: 'Kanal, in den der Bot proaktive Nachrichten sendet (Ergebnisse von Routinen, Erinnerungen).'
      },
      QQBOT_HOME_CHANNEL: {
        help: 'Standard-Kanal oder -Gruppe für Ergebnisse von Routinen.'
      }
    }
  },
  webhooks: {
    disabledBody:
      'Webhooks funktionieren unabhängig von Chat-Kanälen. Aktivieren Sie sie hier, um eingehende HTTP-Events anzunehmen; Chat-Kanäle werden nur benötigt, wenn ein Abonnement an Telegram, Discord, Slack oder einen anderen Kanal zustellt.',
    restarting: 'Hermes wird neu gestartet …',
    restartNeeded:
      'Webhooks sind aktiviert, aber Hermes braucht vor dem Onlinegehen des Empfängers noch einen Neustart.',
    restartGateway: 'Hermes neu starten',
    restartFailed: (detail: string) => `Hermes-Neustart fehlgeschlagen${detail}`,
    enabledRestarting: 'Webhooks aktiviert; Hermes wird neu gestartet …'
  },
  profiles: {
    close: 'Agenten schließen',
    title: 'Agenten',
    count: count => `${count} ${count === 1 ? 'Agent' : 'Agenten'}`,
    search: 'Agenten durchsuchen...',
    loading: 'Agenten werden geladen...',
    newProfile: 'Neuer Agent',
    importProfile: 'Agent importieren…',
    exportProfile: 'Agent exportieren…',
    imported: 'Agent importiert',
    exported: 'Agent exportiert',
    failedImport: 'Agent konnte nicht importiert werden',
    failedExport: 'Agent konnte nicht exportiert werden',
    exportNeedsHost: name =>
      `Der Browser kann kein Agentenarchiv speichern. Führe auf dem Rechner mit Hermes aus: hermes profile export ${name}`,
    allProfiles: 'Alle Agenten',
    showAllProfiles: 'Alle Agenten anzeigen',
    manageProfiles: 'Agenten verwalten…',
    connectGateway: 'Rechner verwalten…',
    fleet: {
      allOnGateway: 'Alle Agenten auf diesem Rechner',
      gateway: gateway => `Agenten auf ${gateway}`,
      localDevice:
        'Dieses Gerät (lokales Backend — installiert Hermes, falls es fehlt, sonst öffnet es einen neuen Chat)',
      installDeviceDesc:
        'Hermes wird lokal installiert und danach ein neuer Chat auf diesem Computer geöffnet. Ohne Bestätigung startet keine Installation.'
    },
    remoteOverride: {
      description:
        'Chats mit diesem Agenten laufen auf dem von Ihnen festgelegten Remote-Hermes statt auf diesem Computer.',
      collisionWarning: (label: string) =>
        `Ein Rechner mit dem Namen “${label}” existiert bereits in den Einstellungen. Diese Agenten-Verbindung ist davon getrennt und ändert ihn nicht.`,
      confirmTitle: 'Diesen Agenten mit einem Remote-Rechner verbinden?',
      confirmNote: (profile: string, host: string) =>
        `Neue Chats mit ${profile} laufen auf ${host}. Befehle werden dort ausgeführt und Dateien dort gelesen, nicht auf diesem Rechner. Verbinden Sie sich nur mit einem Rechner, dem Sie vertrauen.`,
      savedTitle: 'Agent verbunden',
      menuItem: 'Mit Remote-Rechner verbinden…',
      title: (profile: string) => `${profile} mit einem Remote-Rechner verbinden`,
      authFailedTitle: 'Remote-Rechner hat das gespeicherte Token abgelehnt'
    },
    defaultProfile: 'Standardagent',
    defaultDescription:
      'Wird beim Öffnen von Hermes und für neue Chats verwendet. Bestehende Chats bleiben bei ihren Agenten.',
    failedSetDefault: 'Der Standardagent konnte nicht festgelegt werden',
    noProfiles: 'Noch keine Agenten.',
    selectPrompt: 'Wählen Sie einen Agenten aus, um seine Details zu sehen.',
    refresh: 'Agenten aktualisieren',
    refreshing: 'Agenten werden aktualisiert',
    soulDesc: 'Die Anweisungen und die Persönlichkeit dieses Agenten.',
    soulPlaceholder: mode =>
      `Die Anweisungen und die Persönlichkeit für diesen Agenten.\nLeer lassen, um den ${mode}-Standard zu behalten.`,
    deleteTitle: 'Agent löschen?',
    createDesc: 'Jeder Agent hat eigene Anweisungen, Einstellungen, Skills und Chats.',
    cloneFromDesc: 'Kopiert Einstellungen, Skills und Anweisungen vom gewählten Agenten.',
    cloneFromDefault: 'Vom Standardagenten klonen',
    cloneFromDefaultDesc: 'Einstellungen, Skills und Anweisungen von Ihrem Standardagenten kopieren.',
    createAction: 'Agent erstellen',
    renameTitle: 'Agent umbenennen',
    renameDescPrefix: 'Durch das Umbenennen werden das Verzeichnis des Agenten und alle Wrapper-Scripts in ',
    displayNameTitle: 'Diesen Agenten benennen',
    displayNameDesc:
      'Legt einen Anzeigenamen fest, der in der ganzen App angezeigt wird. Die interne ID bleibt "default".',
    created: 'Agent erstellt',
    renamed: 'Agent umbenannt',
    deleted: 'Agent gelöscht',
    failedLoad: 'Agenten konnten nicht geladen werden',
    failedDelete: 'Agent konnte nicht gelöscht werden',
    failedCreate: 'Agent konnte nicht erstellt werden',
    failedRename: 'Agent konnte nicht umbenannt werden',
    status: {
      unread: (count: number) => (count === 1 ? '1 ungelesener Chat' : `${count} ungelesene Chats`),
      needsInput: (count: number) =>
        count === 1 ? '1 Chat wartet auf Ihre Antwort' : `${count} Chats warten auf Ihre Antwort`,
      working: (count: number) => (count === 1 ? '1 laufender Chat' : `${count} laufende Chats`)
    }
  },
  cron: {
    close: 'Routinen schließen',
    title: 'Routinen',
    count: count => `${count} ${count === 1 ? 'Routine' : 'Routinen'}`,
    search: 'Routinen durchsuchen...',
    loading: 'Routinen werden geladen...',
    editJob: 'Routine bearbeiten',
    newCron: 'Neue Routine',
    emptyDescNew:
      'Führen Sie einen Prompt automatisch zu den von Ihnen gewählten Zeiten aus und senden Sie die Ergebnisse an das gewünschte Ziel.',
    emptyTitleNew: 'Noch keine Routinen',
    actionsTitle: 'Routine-Aktionen',
    resume: 'Routine fortsetzen',
    pause: 'Routine pausieren',
    edit: 'Routine bearbeiten',
    deleteTitle: 'Routine löschen?',
    deleteDescSuffix: ' dauerhaft. Es werden keine weiteren Ausführungen gestartet.',
    resumed: 'Routine fortgesetzt',
    paused: 'Routine pausiert',
    triggered: 'Routine gestartet',
    deleted: 'Routine gelöscht',
    created: 'Routine erstellt',
    updated: 'Routine aktualisiert',
    failedLoad: 'Routinen konnten nicht geladen werden',
    failedUpdate: 'Routine konnte nicht aktualisiert werden',
    failedTrigger: 'Routine konnte nicht gestartet werden',
    failedDelete: 'Routine konnte nicht gelöscht werden',
    failedSave: 'Routine konnte nicht gespeichert werden',
    editTitle: 'Routine bearbeiten',
    createTitle: 'Neue Routine',
    createDesc:
      'Legen Sie fest, wann die Aufgabe automatisch ausgeführt wird. Nutzen Sie die Cron-Syntax oder einen Ausdruck wie „every 15 minutes“.',
    scriptOnlyEditHint: 'Routine für ein Skript (ohne KI-Prompt). Routine-ID:',
    createAction: 'Routine erstellen',
    tabs: {
      jobs: 'Routinen'
    },
    blueprints: {
      subtitle: 'Vorlagen für Routinen',
      dialogDesc: 'Füllen Sie die Details aus und legen Sie fest, wann sie läuft.',
      scheduled: 'Routine aus Vorlage erstellt',
      emptyDesc: 'Auf diesem Backend sind keine Vorlagen für Routinen verfügbar.'
    }
  },
  sidebar: {
    filter: {
      profile: 'Agenten'
    },
    gatewayGroups: {
      grouping: 'Rechner & Agent',
      aliasHint: 'Nur der Anzeigename; Rechner- und Agentenname bleiben unverändert.'
    },
    profileRail: 'Agenten-Leiste',
    nav: {
      cron: 'Routinen',
      'new-session': 'Neuer Chat',
      artifacts: 'Bibliothek'
    },
    cronJobs: 'Routinen',
    storageCorrupt: {
      action:
        'Beenden Sie Hermes für diesen Agenten und prüfen Sie die Datei dann, ohne sie zu ändern, oder stellen Sie einen Snapshot wieder her:',
      title: 'Die Chat-Datenbank ist beschädigt',
      body: (profiles: string) =>
        `Hermes kann nicht den gesamten Chat-Verlauf für ${profiles} lesen. Chats, die in dieser Liste fehlen, wurden nicht gelöscht; die Datei, in der sie gespeichert sind, ist beschädigt.`
    },
    projects: {
      createdInPreviousContext:
        'Das Projekt wurde auf der vorherigen Verbindung oder beim vorherigen Agenten erstellt. Wechsle zurück; IDEA.md wurde nicht geschrieben.',
      showAllSessions: 'Alle Chats anzeigen',
      moveFailed: 'Chat konnte nicht verschoben werden',
      toggle: (label, open) => `${open ? 'Anzeigen' : 'Ausblenden'} ${label} Chats`,
      showAllCount: (count: number) => `Alle ${count} Chats anzeigen`
    },
    row: {
      ownedByProfile: profile => `Agent: ${profile}`,
      copyIdFailed: 'Chat-ID konnte nicht kopiert werden',
      sessionActions: 'Chat-Aktionen',
      sessionRunning: 'Chat läuft',
      renameTitle: 'Chat umbenennen',
      untitledPlaceholder: 'Unbenannter Chat',
      deleteTitle: 'Chat löschen?',
      deleted: 'Chat gelöscht'
    },
    searchAria: 'Chats durchsuchen',
    searchPlaceholder: 'Chats durchsuchen…',
    noMatch: query => `Keine Chats passen zu “${query}”.`,
    groupAriaGrouped: 'Chats als einzelne Liste anzeigen',
    groupAriaUngrouped: 'Chats nach Workspace gruppieren',
    showSessions: 'Chats anzeigen',
    groupTitleGrouped: 'Chats aufteilen',
    projectEmpty: 'Noch keine Chats',
    projectLoadFailed: 'Chats konnten nicht geladen werden',
    noSessions: 'Noch keine Chats',
    noFilterMatches: 'Keine Chats passen zu diesen Filtern',
    newSessionIn: label => `Neuer Chat in ${label}`
  },
  composer: {
    commandDescs: {
      '/export': 'Einen Agenten (Einstellungen, Skills, Theme) als teilbares Archiv exportieren',
      '/import': 'Ein geteiltes Agentenarchiv als neuen Agenten importieren',
      '/profile': 'Aktiven Agenten wechseln',
      '/suggestions': 'Vorgeschlagene Routinen prüfen (annehmen/verwerfen)',
      '/blueprint': 'Eine Routine aus einer Blueprint-Vorlage einrichten',
      '/platform': 'Eine fehlerhafte Messaging-Plattform pausieren, fortsetzen oder auflisten',
      '/clear': 'neuen Chat starten',
      '/resume': 'Gespeicherten Chat fortsetzen',
      '/topic': 'Telegram-DM-Themen-Chats aktivieren oder prüfen',
      '/title': 'Aktuellen Chat umbenennen',
      '/handoff': 'Diesen Chat an eine Messaging-Plattform übergeben',
      '/bg': 'Einen Prompt in einem separaten Hintergrund-Chat ausführen',
      '/goal': 'Ein dauerhaftes Ziel festlegen, an dem der Agent über mehrere Turns arbeitet, bis es erreicht ist',
      '/heartbeat': 'Einen wiederkehrenden Prompt festlegen, der bei Leerlauf in diesen Chat zurückkehrt',
      '/loop': 'Einen Prompt in diesem Chat in regelmäßigen Abständen erneut ausführen',
      '/status': 'Status des aktuellen Chats anzeigen',
      '/model': 'Modell für diesen Chat wechseln'
    },
    queueLostDiscardTip:
      'Hermes wurde mitten im Turn neu gestartet und kann ihn nicht abschließen. Verwirf ihn, damit die eingereihten Turns dahinter laufen.',
    newSessionPlaceholders: [
      'Was bauen wir?',
      'Geben Sie dem Agenten eine Aufgabe',
      'Was ist Ihnen wichtig?',
      'Beschreiben Sie, was Sie brauchen',
      'Was sollen wir angehen?',
      'Fragen Sie irgendetwas',
      'Beginnen Sie mit einem Ziel'
    ],
    voiceEngineChained: 'Sprache-zu-Text + Agenten-Stimme',
    voiceEngineLive: 'GPT-Live (Vollduplex, delegiert an den Agenten)',
    attachUrlDesc: 'Der Agent ruft die Seite ab und fügt sie als Kontext für diesen Turn hinzu.',
    queueDroppedBody:
      'Dieser Hintergrund-Eintrag wurde verworfen, da sein Chat nach wiederholten Versuchen nicht fortgesetzt werden konnte. Die übrige Warteschlange ist unverändert.',
    browseAllSessions: 'Alle Chats durchsuchen…'
  },
  updates: {
    bundleOutOfSyncDesc:
      'Die Hermes-Laufzeit wurde aktualisiert, die Desktop-App selbst ist aber noch ein älterer Build – neue Oberflächenfunktionen (wie der Agenten-Modus) fehlen, bis sie aktualisiert wird. Führen Sie das Update unten aus, um die App neu zu bauen. Falls das die Warnung nicht behebt, installieren Sie den neuesten Desktop-Installer neu.',
    applyStatus: {
      noReturn:
        'Das Backend kam nicht wieder online. Das Update wurde möglicherweise nicht abgeschlossen — prüfen Sie den Rechner, auf dem Hermes läuft.'
    }
  },
  install: {
    setupChoiceDesc:
      'Verbinden Sie diese App mit einem Rechner, auf dem Hermes bereits läuft, oder installieren Sie Hermes lokal auf diesem Computer.',
    remoteSetupDesc:
      'Geben Sie die URL Ihres Rechners ein. Hermes Desktop erkennt, ob ein Token oder eine Browser-Anmeldung nötig ist.',
    remoteUrlTitle: 'Rechner-URL',
    remoteUrlDesc:
      'Verwenden Sie die Basis-URL von Hermes auf dem Rechner, bei Remote-Rechnern einschließlich https://.',
    probing: 'Authentifizierung des Rechners wird erkannt...',
    probeError: 'Dieser Rechner mit Hermes konnte nicht erreicht werden.',
    authNeedsOauth: provider =>
      `Melden Sie sich mit ${provider} an, bevor Sie die Verbindung zu diesem Rechner testen.`,
    enterUrlFirst: 'Geben Sie zuerst eine Rechner-URL ein.',
    tokenDesc: 'Fügen Sie das Session-Token aus der .env-Datei von Hermes auf dem Remote-Rechner ein.',
    incompleteSignInTest: 'Melden Sie sich an, bevor Sie die OAuth-geschützte Verbindung zu diesem Rechner testen.',
    incompleteTokenTest: 'Geben Sie ein Session-Token ein, bevor Sie die Verbindung zu diesem Rechner testen.',
    connectExistingDesc:
      'Einen Remote-Rechner mit Session-Token oder Browser-Anmeldung verwenden. Es wird keine lokale Installation gestartet.'
  },
  shell: {
    gatewayMenu: {
      gateway: 'Rechner',
      reconnectGateway: 'Rechner neu verbinden',
      machineSettings: 'Rechner',
      restartTitle: 'Hermes auf diesem Rechner neu starten?'
    },
    statusbar: {
      gateway: 'Rechner',
      gatewayRestarting: 'Hermes wird neu gestartet…',
      gatewayTitle: 'Rechner',
      cron: 'Routinen',
      openCron: 'Routinen öffnen',
      toggleSessionTimer: 'Chat-Timer',
      cacheHitRateTitle:
        'Prompt-Cache-Trefferquote dieses Chats — gecachte Tokens kosten weniger, also ist höher günstiger'
    }
  },
  rightSidebar: {
    terminalGatewayRemoved: 'Rechner entfernt',
    terminalGatewayRemovedBody:
      'Der Rechner, auf dem dieses Terminal lief, wurde entfernt; es kann nicht wieder geöffnet werden. Schließe diesen Tab.',
    terminalGatewayNoShellBody:
      'Dieser Rechner bietet keinen Shell-Zugang; dieses Terminal kann nicht wieder geöffnet werden. Schließe diesen Tab.'
  },
  preview: {
    saveScopeChanged:
      'Wechsle zur ursprünglichen Verbindung und zum ursprünglichen Agenten zurück, um diesen Entwurf zu speichern.',
    web: {
      restarting: 'Der Agent startet den Server neu …',
      askRestart: 'Den Agenten bitten, den Server neu zu starten',
      lookingRestart: taskId => `Der Agent sucht nach einem Vorschau-Server zum Neustarten (${taskId})`,
      restartingMessage: 'Der Agent arbeitet im Hintergrund. Verfolge den Fortschritt in der Vorschau-Konsole.',
      blankPageBody:
        'Geben Sie oben eine Adresse ein, um zu browsen, oder bitten Sie den Agenten, eine Seite zu öffnen.',
      finishedRestarting: message => `Der Agent hat den Vorschau-Server neu gestartet${message ? `: ${message}` : ''}`,
      restartFailedMessage: 'Der Agent konnte den Server nicht neu starten.',
      stillWorking:
        'Der Agent arbeitet noch, aber es ist noch kein Ergebnis des Neustarts eingetroffen. Der Server-Befehl läuft möglicherweise im Vordergrund.'
    }
  },
  assistant: {
    thread: {
      errorCodes: {
        cron_session_read_only: {
          title: 'Dieser Chat gehört zu einer Routine',
          body: 'Er führt nur die Aufgabe dieser Routine aus, daher wurde Ihre Nachricht nicht ausgeführt. Starten Sie einen neuen Chat, um sie zu senden.'
        },
        no_reply: {
          body: 'Der Agent hat diesen Durchlauf beendet, aber keine Antwort hat dieses Fenster erreicht. Versuchen Sie es erneut, um sie noch einmal zu senden.'
        },
        loop_error: {
          title: 'Der Agent konnte diesen Schritt nicht abschließen'
        },
        launch_credentials_unavailable: {
          body: 'Dieser Chat wurde mit einem API-Schlüssel gestartet, der inzwischen geändert oder entfernt wurde, daher wurde Ihre Nachricht nicht gesendet. Ein offener Chat wechselt nie von selbst zu einem anderen Schlüssel. Starten Sie einen neuen Chat, um Ihren aktuellen Schlüssel zu verwenden.'
        }
      },
      loadingSession: 'Chat wird geladen',
      loadingResponse: 'Der Agent lädt eine Antwort',
      errorLayers: {
        generic: 'Der Agent konnte diese Antwort nicht fertigstellen',
        runtime: 'Beim Agenten ist ein Problem aufgetreten'
      },
      errorLayerBodies: {
        generic:
          'Beim Antworten des Agenten ist etwas schiefgelaufen. Versuchen Sie es erneut oder kopieren Sie die Details, wenn es bestehen bleibt.'
      },
      errorToastTitle: 'Der Agent konnte die Antwort nicht fertigstellen',
      errorStartNewSession: 'Neuen Chat starten'
    },
    approval: {
      gatewayDisconnected: 'Hermes ist nicht verbunden',
      timedOutSystemLine:
        'Die Freigabe ist abgelaufen — der Befehl wurde nicht ausgeführt. Bitten Sie den Agenten, es erneut zu versuchen, oder erhöhen Sie das Limit unter Einstellungen → Sicherheit → Freigabe-Timeout.',
      allowSession: 'Für diesen Chat erlauben',
      alwaysDescription: pattern =>
        `Dies fügt das Muster „${pattern}“ Ihrer dauerhaften Zulassungsliste hinzu (~/.hermes/config.yaml). Der Agent fragt bei solchen Befehlen nicht mehr nach – weder in diesem noch in zukünftigen Chats.`
    },
    clarify: {
      gatewayDisconnected: 'Hermes ist nicht verbunden'
    },
    catalogInstall: {
      targetProfile: (profile: string) => `Wird für Ihren Agenten ${profile} installiert`
    },
    mcpSetup: {
      gatewayDisconnected: 'Hermes ist nicht verbunden',
      reloadFailed: 'Server gespeichert, aber das Neuladen der MCP-Tools schlug fehl — sie laden im nächsten Chat'
    },
    tool: {
      titles: {
        cronjob: {
          done: 'Routine',
          pending: 'Erstellt Routine',
          pendingAction: 'Richtet ein'
        },
        session_search_recall: {
          done: 'Chat-Verlauf durchsucht',
          pending: 'Durchsucht Chat-Verlauf'
        }
      }
    },
    themePicker: {
      basicDetail: 'Zum Chatten mit Ihrem Agenten.'
    }
  },
  prompts: {
    gatewayDisconnected: 'Hermes ist nicht verbunden',
    sudoInstallDesc:
      'Der Agent benötigt Ihr sudo-Passwort, um die Pakete für den Agenten-Bildschirm (TigerVNC + Xfce) auf dem verbundenen Rechner zu installieren. Es wird nur an diesen Rechner gesendet.',
    sudoDesc:
      'Prüfen Sie den Befehl, bevor Sie Ihr Sudo-Passwort eingeben. Ihr Passwort wird an den Agenten gesendet, der ihn ausführt, und für diesen Chat zwischengespeichert.',
    secretDesc: 'Der Agent benötigt Zugangsdaten, um fortzufahren.',
    vaultSaveDesc: origin =>
      `Der Agent ist auf eine Anmeldeseite unter ${origin} gestoßen und hat dafür keinen Login. Geben Sie ihn einmal hier ein; er wird auf diesem Rechner verschlüsselt und in die Seite eingetragen, ohne dass das Modell das Passwort je sieht.`,
    vaultCodeDesc: site =>
      `${site} verlangt einen Einmalcode (SMS, E-Mail oder Authenticator-App). Geben Sie ihn hier ein, der Agent trägt ihn in die Seite ein; das Modell sieht ihn nie.`,
    vaultCodeFootnote:
      'Tipp: Speichern Sie den Authentifizierungsschlüssel zusammen mit diesem Login unter Einstellungen → Passwörter & Logins, dann gibt der Agent die Codes für Sie ein.'
  },
  desktop: {
    profileStatus: current =>
      `Agent: ${current}. Verwenden Sie /profile <name> oder die Auswahl „Neuer Chat“, um einen Chat mit einem anderen Agenten zu starten.`,
    unknownProfile: 'Unbekannter Agent',
    noProfileNamed: (target, available) => `Kein Agent namens „${target}“. Verfügbar: ${available}`,
    newChatsProfile: name => `Neue Chats verwenden Agent ${name}.`,
    setProfileFailed: 'Agent konnte nicht gesetzt werden',
    resumeStrandedBody:
      'Die Verbindung zu diesem Chat ist fehlgeschlagen, und automatische Wiederholungen wurden eingestellt. Prüfen Sie, ob Hermes auf dem Rechner läuft, und versuchen Sie es erneut.',
    poolSlotTimeoutBody:
      'Alle Backend-Plätze für lokale Agenten sind belegt. Erhöhen Sie „Warme Agenten-Backends“ unter Einstellungen → Erweitert, oder versuchen Sie es erneut, nachdem ein inaktives Backend entfernt wurde.',
    handoff: {
      timedOut:
        'Hermes konnte Ihre Messaging-Verbindung nicht erreichen. Starten Sie sie unter Einstellungen → Messaging und versuchen Sie die Übergabe dann erneut.'
    },
    sessionUnavailable: 'Chat nicht verfügbar',
    createSessionFailed: 'Neuer Chat konnte nicht erstellt werden',
    yoloSystem: active => `YOLO ${active ? 'an' : 'aus'} für diesen Chat`,
    resumeStrandedTitle: 'Dieser Chat konnte nicht geladen werden',
    sessionBusy: 'Chat beschäftigt',
    cwdStagedMessage: 'Starten Sie das Desktop-Backend neu, um cwd-Änderungen auf diesen aktiven Chat anzuwenden.',
    sessionExported: 'Chat exportiert',
    sessionExportFailed: 'Chat konnte nicht exportiert werden',
    sessionOpenFailed: 'Dieser Chat konnte nicht geöffnet werden'
  },
  tips: {
    items: {
      cron: {
        title: 'Routinen laufen automatisch',
        text: 'Routinen führen einen Prompt stündlich, nächtlich oder nach einem Cron-Ausdruck aus.'
      },
      profiles: {
        title: 'Agenten sind unabhängig',
        text: 'Jeder Agent hat eigene Anweisungen, Einstellungen, Speicher und Chats.'
      },
      skills: {
        text: 'Skills sind Ordner mit Anweisungen, die der Agent lädt, wenn die Arbeit danach verlangt.'
      },
      messaging: {
        title: 'Ihr Agent abseits Ihres Schreibtischs'
      },
      artifacts: {
        title: 'Alles, was Ihr Agent gemacht hat',
        text: 'Bilder, Dateien und Links aus jedem Chat, gesammelt in Ihrer Bibliothek.'
      },
      'command-palette': {
        text: 'Chats, Einstellungen, Skills und Befehle gehorchen alle der Palette.'
      }
    }
  },
  intro: {
    stock: {
      none: [
        'Stellen Sie eine Frage, fügen Sie einen Fehler ein oder zeigen Sie mir ein Repo. Ich kann Code lesen, Tools ausführen und Ihnen beim Ausliefern helfen.',
        'Beschreiben Sie die Aufgabe in eigenen Worten. Ich wähle die passenden Tools, erkläre meinen Plan und frage vor riskanten Schritten nach.',
        'Geben Sie einen Dateipfad, einen Traceback oder eine grobe Idee an. Ich untersuche, schlage nächste Schritte vor und halte alles umkehrbar.',
        'Repo durchsuchen, Dateien bearbeiten, Tests ausführen, PRs öffnen. Nennen Sie das Ziel, und ich übernehme den mechanischen Teil.',
        'Geben Sie eine Aufgabe, Frage oder ein Snippet ein. Ich merke mir den Chat, nenne meine Quellen und frage nach, wenn ich unsicher bin.'
      ]
    }
  },
  firstRun: {
    guide: {
      upgradeRequired:
        'Hermes auf diesem Rechner benötigt ein Update für die Einrichtung ohne Chat. Führen Sie auf dem Rechner hermes update aus, starten Sie Hermes neu und versuchen Sie es erneut. Ihre Chats bleiben unverändert.'
    },
    intro: {
      greeting: 'Hallo, ich bin Ihr Agent.'
    }
  },
  connectors: {
    unavailable: 'Verbindungen sind für diesen Chat nicht verfügbar.',
    disclaimer: 'Das Verbinden ist freiwillig. Geben Sie nur den Apps Zugriff, die der Agent verwenden soll.',
    namedAccount: (app: string, alias: string) => `Der Agent hat dieses ${app}-Konto ${alias} genannt.`
  },
  sessionImport: {
    action: 'Chat importieren',
    search: 'Geladene Chats durchsuchen',
    scanError: 'Chats konnten nicht gefunden werden',
    emptyHelp: 'Claude Code- und Codex-Chats auf diesem Rechner werden hier angezeigt.',
    searchHelp: 'Versuchen Sie es mit einem anderen Titel oder Ordner, oder laden Sie weitere Chats.',
    more: 'Weitere Chats laden',
    chooseHelp: 'Wählen Sie einen Chat, um seinen Verlauf zu lesen, bevor Sie ihn in Hermes übernehmen.'
  },
  titlebar: {
    searchTitle: 'Chats, Ansichten und Aktionen durchsuchen',
    unreadSessions: count => (count === 1 ? '1 ungelesener Chat' : `${count} ungelesene Chats`)
  },
  starmap: {
    emptyDesc: 'Sobald Ihr Agent Skills und Erinnerungen zu Ihrer Arbeit aufbaut, erscheinen sie hier.'
  },
  artifacts: {
    search: 'Bibliothek durchsuchen...',
    refresh: 'Bibliothek aktualisieren',
    refreshing: 'Bibliothek wird aktualisiert',
    indexing: 'Ausgaben aus den letzten Chats werden indiziert',
    noArtifactsTitle: 'Noch nichts in Ihrer Bibliothek',
    noArtifactsDesc: 'Generierte Bilder und Datei-Ausgaben erscheinen hier, sobald Chats sie erstellen.',
    failedLoad: 'Bibliothek konnte nicht geladen werden',
    partialIndex: (failed, total) => `${failed} von ${total} letzten Chats beim Indizieren der Bibliothek übersprungen.`
  },
  artifactPreview: {
    missingTitle: 'Eintrag nicht verfügbar',
    missingBody: 'Dieser Eintrag ist nicht mehr in der Bibliothek.'
  },
  statusStack: {
    control: {
      waitSession: target => `Warten auf Chat ${target}`,
      loopDeferredNotice: 'Ein aktives Ziel steuert derzeit den Chat.',
      continuationBusy:
        'Ziel fortgesetzt — Chat ist beschäftigt; stoppe zuerst die aktuelle Antwort (Stopp-Button oder Esc), um fortzufahren',
      controlUnavailable: msg => `Chat-Steuerung nicht verfügbar: ${msg}`
    },
    coding: {
      agentShip: 'Den Agenten bitten, einen PR zu öffnen'
    }
  },
  onboarding: {
    headerTitle: 'Richten wir Ihren Agenten ein'
  },
  freeTier: {
    busyBody: wait =>
      `Hermes konnte Ihre Anmeldung nicht abschließen, weil der Nous-Dienst ausgelastet ist. Versuchen Sie es in ${wait} erneut. Ihr Chat bleibt so lange erhalten.`,
    unreachableBody:
      'Hermes konnte den Nous-Dienst nicht erreichen, um Ihre Anmeldung abzuschließen. Prüfen Sie Ihre Internetverbindung und versuchen Sie es erneut. Ihr Chat bleibt erhalten.',
    setupFailed: {
      locked:
        'Dieser Chat kann ohne Anmeldung nicht fortgesetzt werden. Melden Sie sich an oder legen Sie ein kostenloses Nous-Konto an, um weiterzumachen.'
    }
  },
  interfaceMode: {
    hint: 'Ändert, was angezeigt wird, nicht was der Agent kann.',
    simple: {
      description: 'Zum Chatten mit Ihrem Agenten. Seitenleiste und Chat; keine Terminal-, Datei- oder Diff-Bereiche.'
    }
  },
  zones: {
    closeRunningBody:
      'Dieser Chat arbeitet noch (oder wartet auf Ihre Eingabe). Das Schließen des Tabs blendet ihn nur aus – der Chat behält seinen Fortschritt und kann über die Seitenleiste wieder geöffnet werden.',
    newSessionTab: 'Neuer Chat-Tab'
  }
}
