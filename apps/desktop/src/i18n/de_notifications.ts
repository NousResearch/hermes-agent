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
  completionSoundPreview: 'Vorschau',
  completionSoundVolume: 'Lautstärke'
}
