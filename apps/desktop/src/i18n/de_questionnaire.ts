import type { Translations } from './types'

export const deQuestionnaire: Translations['questionnaire'] = {
  skipSetup: 'Einrichtung überspringen',
  back: 'Zurück',
  trailLabel: 'Ihre Antworten',
  changeAnswer: 'Diese Antwort ändern',
  retry: 'Erneut versuchen',
  saveFailed: 'Einrichtung konnte nicht als erledigt gespeichert werden',
  skipped: 'Übersprungen',
  otherLabel: 'Andere Antwort',
  kinds: {
    accent: 'Akzentfarbe',
    layout: 'Layout',
    local: 'Lokales Modell',
    apps: 'Apps auf diesem Gerät',
    connectors: 'Connectors'
  },
  name: {
    greeting: 'Hallo, ich bin Hermes.',
    question: 'Wie soll ich Sie nennen?',
    placeholder: 'Dein Name'
  },
  accent: {
    title: 'Wählen Sie eine Farbe',
    custom: 'Eigene Farbe'
  },
  layout: {
    title: 'Wie möchten Sie arbeiten?',
    basic: 'Basic',
    basicDetail: 'Zum Chatten mit Hermes.',
    elite: 'Elite',
    eliteDetail: 'Für Entwickler: Terminal, Dateien, Diffs.'
  },
  local: {
    titleSpark: 'Ihr Spark kann Hermes offline ausführen',
    title: kind => `Dieser ${kind} kann ein Modell lokal ausführen`,
    download: model => `${model} herunterladen`,
    downloadDetail: 'Lädt im Hintergrund, sobald Ihr erster Chat beginnt.',
    notNow: 'Nicht jetzt',
    notNowDetail: 'Bleibt im Modellmenü als „Lokal ausführen“.',
    chip: 'Chip',
    gpu: 'GPU',
    memory: 'Arbeitsspeicher',
    os: 'Betriebssystem',
    memoryValue: gb => `${gb} GB`
  },
  apps: {
    title: 'Hermes in diesen Apps nutzen?',
    plugin: 'Plugin',
    pluginNotRunning: 'Plugin · App läuft nicht',
    approve: 'Sie bestätigen jede Installation in Ihrem ersten Chat.',
    found: 'Auf diesem Gerät gefunden. Nicht installierte Apps werden nicht aufgeführt.'
  },
  connectors: {
    title: 'Ihre Apps verbinden?',
    checking: 'Prüfe, welche Apps Ihr Konto verbinden kann…',
    offered: 'Diese Liste bietet Ihr Konto derzeit an. Sie melden sich bei jeder App in Ihrem ersten Chat an.'
  },
  task: {
    title: 'Womit fangen wir an?',
    question: 'Wählen Sie eins oder schreiben Sie Ihr eigenes.',
    blender: 'Eine Szene in Blender bauen',
    nvidia: 'Meine GPU optimieren',
    broadcast: 'Mikrofon und Kamera einrichten',
    brief: 'Überblick über den heutigen Tag',
    tidy: 'Meinen Downloads-Ordner aufräumen'
  },
  tour: {
    title: 'Eine kurze Tour?',
    question: 'Danach kurz umsehen?',
    quick: 'Zeig mir alles',
    quickDetail: 'Ein paar Stationen, unter einer Minute.',
    none: 'Ich finde mich zurecht',
    noneDetail: 'Sie können Hermes später nach einer Tour fragen.'
  },
  start: {
    title: name => (name ? `Bereit, ${name}.` : 'Bereit.'),
    sub: 'Das ist die erste Nachricht, die Hermes erhält. Die Bitte steht oben, der Rest folgt darunter.',
    aboutMe: 'Über mich:',
    beforeYouStart: 'Bevor Sie anfangen:',
    noTask: 'Keine Aufgabe gewählt: Hermes fragt, was Sie tun möchten.',
    action: 'Starten',
    waiting: 'Ihr kostenloses Konto wird vorbereitet…',
    failed: 'Ihr erster Chat konnte nicht gestartet werden'
  },
  summary: {
    custom: 'Eigene',
    local: model => `Lokal: ${model}`,
    localNo: 'Lokal: nicht jetzt',
    noApps: 'Keine Apps',
    noConnectors: 'Keine Connectors',
    more: (label, count) => `${label} +${count}`,
    tour: 'Tour',
    noTour: 'Keine Tour'
  },
  status: {
    settingUp: 'Ihr kostenloses Konto wird eingerichtet',
    stillSettingUp: 'Ihr kostenloses Konto wird noch eingerichtet',
    unavailable: 'Kostenloses Konto nicht verfügbar',
    ready: 'Nous · kostenlos',
    downloadingEngine: 'Lokale Engine wird heruntergeladen',
    downloading: model => `${model} wird heruntergeladen`
  },
  settings: {
    title: 'Einrichtung erneut ausführen',
    description: 'Öffnet die Fragen beim ersten Start erneut. Ihre Antworten gelten für das Standardprofil.',
    action: 'Einrichtung starten',
    failed: 'Einrichtung konnte nicht gestartet werden'
  }
}
