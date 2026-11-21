// Notification settings copy used when the desktop locale loads.
export const deNotifications = {
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
  browserPermission: {
    title: 'Browser-Benachrichtigungen',
    allow: 'Erlauben',
    states: {
      default:
        'Hinweise dieses Browsers anzeigen, während der Tab im Hintergrund ist. Bis du es erlaubst, blinkt nur der Tab-Titel.',
      granted: 'Erlaubt. Hinweise erscheinen, während dieser Tab im Hintergrund ist; der Tab-Titel blinkt ebenfalls.',
      denied: 'In den Website-Einstellungen dieses Browsers blockiert. Der Tab-Titel blinkt trotzdem.',
      insecure: 'Erfordert HTTPS, etwa über Tailscale serve. Über reines HTTP blinkt nur der Tab-Titel.',
      unsupported: 'Dieser Browser kann keine Benachrichtigungen anzeigen. Der Tab-Titel blinkt trotzdem.'
    }
  },
  phonePresence: {
    title: 'Dieser Computer für Telefon-Benachrichtigungen',
    description: 'Was dieses Fenster Hermes darüber mitteilt, ob du hier bist. Ein Plugin für Telefon-Benachrichtigungen entscheidet, was daraus folgt.',
    unsupported: 'Telefon-Benachrichtigungen hängen auf diesem Backend nicht von deiner Aktivität ab.',
    disconnected: 'Nicht verbunden.',
    watchingComputer: (time: string) => `Solange in diesem Fenster eine Unterhaltung geöffnet ist, erfährt Hermes, wann du diesen Computer zuletzt benutzt hast. Letzte Eingabe vor ${time}.`,
    watchingWindow: (time: string) => `Solange in diesem Fenster eine Unterhaltung geöffnet ist, erfährt Hermes, wann du dieses Fenster zuletzt benutzt hast. Letzte Eingabe vor ${time}.`,
    watchingWindowNoInput: 'Solange in diesem Fenster eine Unterhaltung geöffnet ist, erfährt Hermes, wann du dieses Fenster zuletzt benutzt hast. Noch keine Eingabe bemerkt.'
  },
  test: 'Testbenachrichtigung senden',
  testTitle: 'Hermes',
  testBody: 'Benachrichtigungen funktionieren.',
  testSent:
    'Test gesendet. Wenn nichts erscheint, überprüfen Sie die Benachrichtigungsberechtigungen Ihres Betriebssystems und Fokus/Nicht stören.',
  testUnsupported: 'Dieses System unterstützt keine nativen Benachrichtigungen.',
  muteSoundsTitle: 'Oberflächentöne stummschalten',
  muteSoundsDesc:
    'Schaltet automatische Abschluss-, Aktivierungs- und Denktöne, haptisches Feedback und die Töne von Systembenachrichtigungen aus. Klangvorschauen und gesprochene Antworten werden weiterhin abgespielt.',
  completionSoundTitle: 'Abschluss-Sound',
  completionSoundDesc:
    'Wird abgespielt, wenn ein Agent-Turn endet. Wählen Sie eine Vorlage aus und hören Sie sie hier an.',
  completionSoundPreview: 'Vorschau'
}
