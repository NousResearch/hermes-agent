export const deSharedMetrics = {
  consentTitle: 'Hermes verbessern helfen?',
  consentBody:
    'Geteilte Metriken enthalten nur begrenzte Zähler. Niemals Prompts, Dateien, Pfade oder Fehlertexte. Die Erfassung bleibt lokal. Das Senden an Nous ist eine separate Zustimmung.',
  whatIsCollected: 'Was erfasst wird',
  collectedIntro: 'Nur begrenzte Zähler:',
  collectedActivity:
    'Aktivität, Session-Länge, Ergebnisse und Fehlerklassen, einschließlich eines Grunds aus einer festen Liste, wenn ein Speicherschreibvorgang oder eine Kontextkomprimierung abgelehnt wird, fehlschlägt oder übersprungen wird',
  collectedModels: 'Modellrouten und Token-Summen',
  collectedNames: 'Namen integrierter Tools, Befehle und Katalogeinträge',
  collectedMilestones: 'Gruppierte Einrichtungszahlen',
  collectedReliability:
    'Ergebnisse und Dauer von Updates und Installationen (mit einem Grund aus einer festen Liste und der Phase, wenn etwas fehlschlägt, einschließlich einer Neuinstallation, die auf diesem Gerät festgehalten und erst nach deiner Zustimmung gezählt wird), Abstürze, Start- und Antwortzeiten, Zustand der Messaging-Plattformen',
  collectedUsage:
    'Wie Hermes genutzt wird: Genauigkeit und Effizienz des Agenten (Treffer bei Bearbeitungen, Schleifen, Erholung nach Fehlern, Tokens und Tool-Aufrufe pro Aufgabe, Cache-Brüche), aktive Zeit pro Oberfläche und Desktop-Modus, welche App-Bereiche, Aktionen und Einstellungen genutzt, schnell geschlossen oder abgeschaltet werden, sowie Ergebnisse der Anbietereinrichtung',
  collectedMachine:
    'Grobe Gerätedaten: RAM-Bereich, GPU-Typ, Alter und Kanal der Hermes-Version, Anzahl ausstehender Updates, ob ein lokaler Modellserver genutzt wird',
  installId:
    'Beim Senden wird jedes Tagespaket an den Nous-Telemetriedienst hochgeladen. Pakete tragen die Installations-ID dieses Profils: eine feste zufällige UUID ohne persönliche Daten, zurückgesetzt durch Löschen des Shared-Metrics-Ordners.',
  consentWindow:
    'Gesendet werden nur Pakete, deren gesamter Erfassungszeitraum in ein erfasstes Zustimmungsfenster fällt. Abgesehen vom Hinweis auf eine Neuinstallation (auf diesem Gerät festgehalten und erst nach Ihrer Zustimmung gezählt) bleiben Daten von vor Ihrer Zustimmung oder aus Lücken, in denen das Senden aus war, auf diesem Rechner. Das Senden lässt sich jederzeit wieder abschalten.',
  readDocs: 'Alle Details lesen',
  share: 'Erfassen und an Nous senden',
  local: 'Nur lokal erfassen',
  off: 'Nein, danke',
  changeLater: 'Sie können das jederzeit unter Einstellungen → Sicherheit ändern.',
  saveFailed: 'Ihre Auswahl konnte nicht gespeichert werden',
  collectLabel: 'Nutzungsstatistiken erfassen',
  collectDesc: 'Begrenzte Zähler auf diesem Gerät. Niemals Prompts, Dateien, Pfade oder Fehlertexte.',
  sendLabel: 'Nutzungsstatistiken an Nous senden',
  sendDesc:
    'Jedes Tagespaket an den Nous-Telemetriedienst hochladen. Nur Daten aus einem Zustimmungsfenster werden gesendet. Erfordert aktive Erfassung.',
  unavailable: 'Aktualisieren Sie das Hermes-Backend, um diese Einstellung zu ändern.',
  stripBody: 'Nur begrenzte Zähler, niemals Prompts oder Dateien.',
  stripReaskBody:
    'Wir fragen noch einmal: Eine frühere Version konnte „Nein danke“ speichern, bevor Sie diese Frage gesehen haben.',
  stripChoices: { share: 'An Nous senden', local: 'Nur lokal', off: 'Nein danke' },
  stripDetails: 'Details'
}
