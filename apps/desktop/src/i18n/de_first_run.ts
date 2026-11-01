import type { FirstRunCopy } from './types_first_run'

export const deFirstRun: FirstRunCopy = {
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
