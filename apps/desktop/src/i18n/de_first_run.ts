import type { FirstRunCopy } from './types_first_run'

export const deFirstRun: FirstRunCopy = {
  guide: {
    title: 'Hermes einrichten',
    resume: 'Einrichtung fortsetzen',
    upgradeRequired:
      'Hermes auf diesem Rechner benötigt ein Update für die Einrichtung ohne Sitzung. Führen Sie auf dem Rechner hermes update aus, starten Sie Hermes neu und versuchen Sie es erneut. Ihre Chats bleiben unverändert.',
    intro:
      'Ein paar optionale Einstellungen, dann Ihr normaler Chat. Die Einrichtung erstellt niemals eine Unterhaltung.',
    appearance: 'Darstellung',
    model: 'Modell auswählen',
    ready: 'Ihr Modell ist bereit.',
    providerNeeded:
      'Verbinden Sie einen Anbieter oder nutzen Sie das kostenlose Modell, bevor Sie eine Nachricht senden.',
    freeModel: 'Kostenloses Modell nutzen',
    freeUnavailable:
      'Das kostenlose Modell ist hier nicht verfügbar. Verbinden Sie einen Anbieter oder wählen Sie später einen aus.',
    providers: 'Anbieter verbinden',
    finish: 'Einrichtung abschließen',
    skip: 'Einrichtung überspringen',
    back: 'Zurück',
    check: 'Bereitschaft prüfen',
    checking: 'Wird geprüft…',
    privacy: 'Das Teilen von Nutzungsmetriken ist optional.'
  },
  beats: {
    name: 'Name',
    theme: 'Akzentfarbe',
    layout: 'Layout',
    connectors: 'Apps',
    plugins: 'Plugins',
    tour: 'Tour',
    build: 'Erste Aufgabe'
  },
  namePlaceholder: 'Dein Name',
  useName: name => `„${name}“ verwenden`,
  install: 'Installieren',
  installed: 'Installiert',
  loading: 'Wird geladen…',
  noPlugins: 'Gerade keine Plugins verfügbar.',
  noneChosen: 'Keine',
  noAnswer: 'Keine Antwort',
  answerRefused: 'Antwort nicht angenommen',
  tours: {
    quick: 'Kurze Tour',
    full: 'Zeig mir alles',
    skip: 'Überspringen, lass uns etwas bauen'
  },
  skippedAll: 'Einrichtung übersprungen',
  skip: 'Einrichtung überspringen',
  skipTip: 'Beendet die geführte Einrichtung und öffnet einen neuen Chat.',
  skipFailed: 'Einrichtung konnte nicht übersprungen werden',
  slashOffTitle: 'Slash-Befehle sind während der Einrichtung aus',
  slashOffBody: 'Nach der Einrichtung funktionieren sie wie gewohnt.',
  localModel: {
    title: 'Dieser Computer kann ein Modell lokal ausführen',
    text: (model, size) =>
      `${model} (${size}) passt zu deiner Hardware. Es läuft kostenlos, und deine Chats bleiben auf diesem Computer.`,
    setUp: 'Einrichten',
    notNow: 'Nicht jetzt'
  },
  intro: {
    greeting: 'Hallo, ich bin Hermes.',
    plan: 'Richten wir alles für Sie ein. Danach machen wir etwas Cooles.'
  },
  rerun: {
    title: 'Onboarding erneut starten',
    description: 'Öffnet einen neuen Einrichtungs-Chat, der die ersten Schritte mit deinem aktuellen Modell durchgeht.',
    action: 'Erneut starten',
    failed: 'Onboarding konnte nicht gestartet werden'
  }
}
