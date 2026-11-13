import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

/** The redesigned Routines page's own words (Figma "Routines — proposal + mock"). The redesign renders
 *  every one of them, so they hold the redesign terms directly; shared nouns (the page title, New
 *  routine, Pause, the missed-run choices) come from the catalog's redesign overlay. */
export interface RoutinesCopy {
  allRoutines: string
  groups: { attention: string; scheduled: string; paused: string; finished: string }
  filterLabel: string
  filters: { all: string; attention: string; scheduled: string; paused: string }
  search: string
  columns: { routine: string; agent: string; when: string; lastRun: string; status: string }
  status: {
    failed: string
    delivery: string
    overdue: string
    running: string
    on: string
    paused: string
    finished: string
  }
  notRunYet: string
  runningNow: string
  footnote: (routines: number, agents: number) => string
  noMatchesTitle: string
  noMatchesDescription: string
  emptyDescription: string
  banner: {
    failedOnce: (name: string) => string
    failedStreak: (name: string, runs: number) => string
    delivery: (name: string) => string
    overdue: (name: string) => string
    overdueHint: string
    retryAt: (when: string) => string
    /** A run refused because the routine's agent has no gateway running here. */
    agentNotRunning: (agent: string) => string
    /** Discloses the backend's full error text under the plain sentence. */
    details: string
  }
  actions: {
    runNow: string
    retry: string
    edit: string
    change: string
    view: string
    delete: string
    more: string
  }
  whatItDoes: string
  whenAndWhere: string
  whenToRun: string
  sendResultsTo: string
  nextRun: (when: string) => string
  keptInRuns: string
  scriptRoutine: string
  advanced: string
  advancedSummary: string
  model: string
  modelDefault: string
  runLimit: string
  noLimit: string
  limitProgress: (done: number, total: number) => string
  stopAfter: string
  runsUnit: string
  rememberLastRun: string
  rememberHint: string
  on: string
  off: string
  expression: string
  routineId: string
  script: string
  runs: string
  runsHint: string
  noRuns: string
  startingRun: string
  finishedIn: (duration: string) => string
  runChips: { running: string; failed: string; delivery: string; stopped: string }
  deleteRoutine: string
  deleteHint: (agent: string) => string
  pickDescription: string
  forAgent: string
  readyMade: string
  searchReadyMade: string
  use: string
  showAll: (count: number) => string
  showFewer: string
  noReadyMade: string
  describeInChat: string
  askInChat: string
  startBlank: string
  agent: string
  name: string
  namePlaceholder: string
  instruction: string
  instructionPlaceholder: string
  frequencies: {
    once: string
    hourly: string
    daily: string
    weekdays: string
    weekly: string
    monthly: string
    custom: string
  }
  at: string
  onDay: string
  customPlaceholder: string
  customHint: string
  customHintDeveloper: string
  /** A ready-made routine's time field: the example is the field's own default, in the locale's clock. */
  localTimeHint: (example: string) => string
  everyRunKept: string
  nothingSelected: string
  agentChat: (agent: string) => string
  needsHomeChannel: string
  noChannels: string
  instructionRequired: string
  scheduleRequired: string
  saveChanges: string
  schedule: {
    once: (date: string, time: string) => string
    hourly: string
    hourlyAt: (minute: string) => string
    daily: (time: string) => string
    weekdays: (time: string) => string
    weekly: (day: string, time: string) => string
    monthly: (day: string, time: string) => string
    everyMinutes: (count: number) => string
    everyHours: (count: number) => string
    everyDays: (count: number) => string
    custom: string
  }
  railAttention: (count: number) => string
  rename: string
  renameTitle: string
  editInSettings: (agent: string) => string
  editInAgent: (agent: string) => string
  editCardTitle: string
  editCardHint: string
  askInChatPrompt: string
  chipTooltip: (count: number, agent: string) => string
  chipFailing: (name: string) => string
  chipOpens: string
}

const plural = (count: number, one: string, other: string) => (count === 1 ? one : other)

const en: RoutinesCopy = {
  allRoutines: 'All routines',
  groups: { attention: 'Needs attention', scheduled: 'Scheduled', paused: 'Paused', finished: 'Finished' },
  filterLabel: 'Show',
  filters: { all: 'All', attention: 'Needs attention', scheduled: 'Scheduled', paused: 'Paused' },
  search: 'Search routines',
  columns: { routine: 'Routine', agent: 'Agent', when: 'When', lastRun: 'Last run', status: 'Status' },
  status: {
    failed: 'Failed',
    delivery: 'Not delivered',
    overdue: 'Overdue',
    running: 'Running',
    on: 'On',
    paused: 'Paused',
    finished: 'Finished'
  },
  notRunYet: 'Not run yet',
  runningNow: 'Running now',
  footnote: (routines, agents) =>
    `${routines} ${plural(routines, 'routine', 'routines')} across ${agents} ${plural(agents, 'agent', 'agents')}`,
  noMatchesTitle: 'No matches',
  noMatchesDescription: 'No routine matches this search or filter.',
  emptyDescription: 'Let an agent do something on its own at the times you choose, and send you the results.',
  banner: {
    failedOnce: name => `${name} failed its last run`,
    failedStreak: (name, runs) => `${name} failed the last ${runs} runs`,
    delivery: name => `${name} ran, but its results were not delivered`,
    overdue: name => `${name} has not run since it was due`,
    overdueHint: 'Hermes may not be running on this machine.',
    retryAt: when => `It will try again ${when}.`,
    agentNotRunning: agent => `${agent} isn’t running on this machine right now.`,
    details: 'Details'
  },
  actions: {
    runNow: 'Run now',
    retry: 'Retry',
    edit: 'Edit',
    change: 'Change',
    view: 'View',
    delete: 'Delete',
    more: 'More actions'
  },
  whatItDoes: 'What it does',
  whenAndWhere: 'When and where',
  whenToRun: 'When to run',
  sendResultsTo: 'Send results to',
  nextRun: when => `next ${when}`,
  keptInRuns: 'Kept in Runs only',
  scriptRoutine: 'Runs a script on its own, without the agent.',
  advanced: 'Advanced',
  advancedSummary: 'Model · Missed runs · Run limit · Remember last run',
  model: 'Model',
  modelDefault: 'Default model',
  runLimit: 'Run limit',
  noLimit: 'No limit',
  limitProgress: (done, total) => `${done} of ${total} runs done`,
  stopAfter: 'Stop after',
  runsUnit: 'runs',
  rememberLastRun: 'Remember last run',
  rememberHint: 'Each run sees what the previous run reported, so it can pick up where it left off.',
  on: 'On',
  off: 'Off',
  expression: 'Schedule expression',
  routineId: 'Routine ID',
  script: 'Script',
  runs: 'Runs',
  runsHint: 'Runs stay here. They don’t appear in Chats.',
  noRuns: 'No runs yet',
  startingRun: 'Starting a run…',
  finishedIn: duration => `Finished in ${duration}`,
  runChips: { running: 'Running', failed: 'Failed', delivery: 'Not delivered', stopped: 'Stopped' },
  deleteRoutine: 'Delete routine',
  deleteHint: agent => `Stops it for good and removes it from ${agent}.`,
  pickDescription: 'What kind of routine? Pick a starting point; you can change everything next.',
  forAgent: 'For',
  readyMade: 'Ready-made',
  searchReadyMade: 'Search ready-made routines',
  use: 'Use',
  showAll: count => `Show all ${count}`,
  showFewer: 'Show fewer',
  noReadyMade: 'No ready-made routine matches.',
  describeInChat: 'Or describe it in a chat with this agent',
  askInChat: 'Ask in chat',
  startBlank: 'Start blank',
  agent: 'Agent',
  name: 'Name',
  namePlaceholder: 'Morning brief',
  instruction: 'Instruction',
  instructionPlaceholder:
    'What should the agent do each time? For example: summarize my unread email and list anything urgent.',
  frequencies: {
    once: 'Once',
    hourly: 'Every hour',
    daily: 'Every day',
    weekdays: 'Weekdays',
    weekly: 'Every week',
    monthly: 'Every month',
    custom: 'Custom'
  },
  at: 'at',
  onDay: 'on day',
  customPlaceholder: 'every 2 hours',
  customHint: 'Say it in plain words, like “every 2 hours” or “every monday 9am”.',
  customHintDeveloper: 'Plain words like “every 2 hours”, or a cron expression like “0 9 * * 1-5”.',
  localTimeHint: example => `Your local time, e.g. ${example}`,
  everyRunKept: 'Every run is also kept in Runs',
  nothingSelected: 'Nothing selected? Results are still saved in Runs.',
  agentChat: agent => `Chat with ${agent}`,
  needsHomeChannel: 'Set a home channel first',
  noChannels: 'No messaging channel is connected to this agent yet.',
  instructionRequired: 'Tell the agent what to do.',
  scheduleRequired: 'Choose when it runs.',
  saveChanges: 'Save changes',
  schedule: {
    once: (date, time) => `Once, on ${date} at ${time}`,
    hourly: 'Every hour',
    hourlyAt: minute => `Every hour at :${minute}`,
    daily: time => `Every day at ${time}`,
    weekdays: time => `Monday to Friday at ${time}`,
    weekly: (day, time) => `Every ${day} at ${time}`,
    monthly: (day, time) => `Day ${day} of every month at ${time}`,
    everyMinutes: count => `Every ${count} ${plural(count, 'minute', 'minutes')}`,
    everyHours: count => `Every ${count} ${plural(count, 'hour', 'hours')}`,
    everyDays: count => `Every ${count} ${plural(count, 'day', 'days')}`,
    custom: 'On a custom schedule'
  },
  rename: 'Rename',
  renameTitle: 'Rename routine',
  editInSettings: agent => `Edit in ${agent} settings`,
  editInAgent: agent => `Edit in ${agent}`,
  editCardTitle: 'Change what it does, when, or where results go',
  editCardHint: 'Routines are edited in their agent’s settings.',
  askInChatPrompt: 'Set up a routine that ',
  chipTooltip: (count, agent) => `${count} ${plural(count, 'routine', 'routines')} for ${agent}`,
  chipFailing: name => `${name} failed · opens Routines`,
  chipOpens: 'Opens Routines',
  railAttention: count => `${count} ${plural(count, 'routine needs', 'routines need')} attention`
}

const fr: RoutinesCopy = {
  allRoutines: 'Toutes les routines',
  groups: { attention: 'À vérifier', scheduled: 'Planifiées', paused: 'En pause', finished: 'Terminées' },
  filterLabel: 'Afficher',
  filters: { all: 'Toutes', attention: 'À vérifier', scheduled: 'Planifiées', paused: 'En pause' },
  search: 'Rechercher des routines',
  columns: { routine: 'Routine', agent: 'Agent', when: 'Quand', lastRun: 'Dernière exécution', status: 'État' },
  status: {
    failed: 'Échec',
    delivery: 'Non envoyée',
    overdue: 'En retard',
    running: 'En cours',
    on: 'Active',
    paused: 'En pause',
    finished: 'Terminée'
  },
  notRunYet: 'Pas encore exécutée',
  runningNow: 'En cours',
  footnote: (routines, agents) =>
    `${routines} ${plural(routines, 'routine', 'routines')} pour ${agents} ${plural(agents, 'agent', 'agents')}`,
  noMatchesTitle: 'Aucun résultat',
  noMatchesDescription: 'Aucune routine ne correspond à cette recherche ou à ce filtre.',
  emptyDescription: 'Laissez un agent agir seul aux moments choisis et vous envoyer les résultats.',
  banner: {
    failedOnce: name => `${name} a échoué lors de sa dernière exécution`,
    failedStreak: (name, runs) => `${name} a échoué lors des ${runs} dernières exécutions`,
    delivery: name => `${name} s’est exécutée, mais ses résultats n’ont pas été envoyés`,
    overdue: name => `${name} ne s’est pas exécutée à l’heure prévue`,
    overdueHint: 'Hermes n’est peut-être pas lancé sur cette machine.',
    retryAt: when => `Nouvel essai ${when}.`,
    agentNotRunning: agent => `${agent} n’est pas en cours d’exécution sur cet ordinateur pour le moment.`,
    details: 'Détails'
  },
  actions: {
    runNow: 'Exécuter maintenant',
    retry: 'Réessayer',
    edit: 'Modifier',
    change: 'Modifier',
    view: 'Voir',
    delete: 'Supprimer',
    more: 'Plus d’actions'
  },
  whatItDoes: 'Ce qu’elle fait',
  whenAndWhere: 'Quand et où',
  whenToRun: 'Quand l’exécuter',
  sendResultsTo: 'Envoyer les résultats à',
  nextRun: when => `prochaine ${when}`,
  keptInRuns: 'Conservés dans les exécutions uniquement',
  scriptRoutine: 'Exécute un script seul, sans l’agent.',
  advanced: 'Avancé',
  advancedSummary: 'Modèle · Exécutions manquées · Limite · Mémoire de la dernière exécution',
  model: 'Modèle',
  modelDefault: 'Modèle par défaut',
  runLimit: 'Limite d’exécutions',
  noLimit: 'Aucune limite',
  limitProgress: (done, total) => `${done} exécutions sur ${total}`,
  stopAfter: 'Arrêter après',
  runsUnit: 'exécutions',
  rememberLastRun: 'Se souvenir de la dernière exécution',
  rememberHint: 'Chaque exécution voit ce que la précédente a rapporté et reprend là où elle s’était arrêtée.',
  on: 'Activé',
  off: 'Désactivé',
  expression: 'Expression d’horaire',
  routineId: 'ID de la routine',
  script: 'Script',
  runs: 'Exécutions',
  runsHint: 'Les exécutions restent ici. Elles n’apparaissent pas dans les discussions.',
  noRuns: 'Aucune exécution pour le moment',
  startingRun: 'Lancement d’une exécution…',
  finishedIn: duration => `Terminée en ${duration}`,
  runChips: { running: 'En cours', failed: 'Échec', delivery: 'Non envoyée', stopped: 'Arrêtée' },
  deleteRoutine: 'Supprimer la routine',
  deleteHint: agent => `L’arrête définitivement et la retire de ${agent}.`,
  pickDescription: 'Quel type de routine ? Choisissez un point de départ ; tout reste modifiable ensuite.',
  forAgent: 'Pour',
  readyMade: 'Prêtes à l’emploi',
  searchReadyMade: 'Rechercher une routine prête à l’emploi',
  use: 'Utiliser',
  showAll: count => `Tout afficher (${count})`,
  showFewer: 'Afficher moins',
  noReadyMade: 'Aucune routine prête à l’emploi ne correspond.',
  describeInChat: 'Ou décrivez-la dans une discussion avec cet agent',
  askInChat: 'Demander dans une discussion',
  startBlank: 'Partir de zéro',
  agent: 'Agent',
  name: 'Nom',
  namePlaceholder: 'Point du matin',
  instruction: 'Instruction',
  instructionPlaceholder:
    'Que doit faire l’agent à chaque fois ? Par exemple : résumer mes e-mails non lus et signaler ce qui est urgent.',
  frequencies: {
    once: 'Une fois',
    hourly: 'Toutes les heures',
    daily: 'Tous les jours',
    weekdays: 'En semaine',
    weekly: 'Chaque semaine',
    monthly: 'Chaque mois',
    custom: 'Personnalisé'
  },
  at: 'à',
  onDay: 'le',
  customPlaceholder: 'toutes les 2 heures',
  customHint: 'Dites-le simplement, par exemple « every 2 hours » ou « every monday 9am ».',
  customHintDeveloper: 'Des mots simples comme « every 2 hours », ou une expression cron comme « 0 9 * * 1-5 ».',
  localTimeHint: example => `Votre heure locale, par ex. ${example}`,
  everyRunKept: 'Chaque exécution est aussi conservée',
  nothingSelected: 'Rien de sélectionné ? Les résultats restent enregistrés dans les exécutions.',
  agentChat: agent => `Discussion avec ${agent}`,
  needsHomeChannel: 'Définissez d’abord un canal principal',
  noChannels: 'Aucune messagerie n’est encore connectée à cet agent.',
  instructionRequired: 'Dites à l’agent quoi faire.',
  scheduleRequired: 'Choisissez quand elle s’exécute.',
  saveChanges: 'Enregistrer',
  schedule: {
    once: (date, time) => `Une fois, le ${date} à ${time}`,
    hourly: 'Toutes les heures',
    hourlyAt: minute => `Toutes les heures à :${minute}`,
    daily: time => `Tous les jours à ${time}`,
    weekdays: time => `Du lundi au vendredi à ${time}`,
    weekly: (day, time) => `Chaque ${day} à ${time}`,
    monthly: (day, time) => `Le ${day} de chaque mois à ${time}`,
    everyMinutes: count => `Toutes les ${count} minutes`,
    everyHours: count => (count === 1 ? 'Toutes les heures' : `Toutes les ${count} heures`),
    everyDays: count => (count === 1 ? 'Tous les jours' : `Tous les ${count} jours`),
    custom: 'Selon un horaire personnalisé'
  },
  rename: 'Renommer',
  renameTitle: 'Renommer la routine',
  editInSettings: agent => `Modifier dans les réglages de ${agent}`,
  editInAgent: agent => `Modifier dans ${agent}`,
  editCardTitle: 'Changer ce qu’elle fait, quand, ou où vont les résultats',
  editCardHint: 'Les routines se modifient dans les réglages de leur agent.',
  askInChatPrompt: 'Crée une routine qui ',
  chipTooltip: (count, agent) => `${count} ${plural(count, 'routine', 'routines')} pour ${agent}`,
  chipFailing: name => `${name} a échoué · ouvre les routines`,
  chipOpens: 'Ouvre les routines',
  railAttention: count => `${count} ${plural(count, 'routine à vérifier', 'routines à vérifier')}`
}

const de: RoutinesCopy = {
  allRoutines: 'Alle Routinen',
  groups: { attention: 'Braucht Aufmerksamkeit', scheduled: 'Geplant', paused: 'Pausiert', finished: 'Beendet' },
  filterLabel: 'Anzeigen',
  filters: { all: 'Alle', attention: 'Braucht Aufmerksamkeit', scheduled: 'Geplant', paused: 'Pausiert' },
  search: 'Routinen suchen',
  columns: { routine: 'Routine', agent: 'Agent', when: 'Wann', lastRun: 'Letzter Lauf', status: 'Status' },
  status: {
    failed: 'Fehlgeschlagen',
    delivery: 'Nicht zugestellt',
    overdue: 'Überfällig',
    running: 'Läuft',
    on: 'An',
    paused: 'Pausiert',
    finished: 'Beendet'
  },
  notRunYet: 'Noch nicht gelaufen',
  runningNow: 'Läuft gerade',
  footnote: (routines, agents) =>
    `${routines} ${plural(routines, 'Routine', 'Routinen')} bei ${agents} ${plural(agents, 'Agent', 'Agenten')}`,
  noMatchesTitle: 'Keine Treffer',
  noMatchesDescription: 'Keine Routine passt zu dieser Suche oder diesem Filter.',
  emptyDescription: 'Lass einen Agenten zu festen Zeiten selbst etwas erledigen und dir die Ergebnisse schicken.',
  banner: {
    failedOnce: name => `${name}: Der letzte Lauf ist fehlgeschlagen`,
    failedStreak: (name, runs) => `${name}: Die letzten ${runs} Läufe sind fehlgeschlagen`,
    delivery: name => `${name} ist gelaufen, aber die Ergebnisse wurden nicht zugestellt`,
    overdue: name => `${name} ist seit dem fälligen Termin nicht gelaufen`,
    overdueHint: 'Hermes läuft auf diesem Rechner möglicherweise nicht.',
    retryAt: when => `Nächster Versuch ${when}.`,
    agentNotRunning: agent => `${agent} läuft gerade nicht auf diesem Computer.`,
    details: 'Details'
  },
  actions: {
    runNow: 'Jetzt ausführen',
    retry: 'Erneut versuchen',
    edit: 'Bearbeiten',
    change: 'Ändern',
    view: 'Ansehen',
    delete: 'Löschen',
    more: 'Weitere Aktionen'
  },
  whatItDoes: 'Was sie tut',
  whenAndWhere: 'Wann und wohin',
  whenToRun: 'Wann ausführen',
  sendResultsTo: 'Ergebnisse senden an',
  nextRun: when => `nächster Lauf ${when}`,
  keptInRuns: 'Nur in den Läufen gespeichert',
  scriptRoutine: 'Führt ein Skript allein aus, ohne den Agenten.',
  advanced: 'Erweitert',
  advancedSummary: 'Modell · Verpasste Läufe · Laufgrenze · Letzten Lauf merken',
  model: 'Modell',
  modelDefault: 'Standardmodell',
  runLimit: 'Laufgrenze',
  noLimit: 'Keine Grenze',
  limitProgress: (done, total) => `${done} von ${total} Läufen erledigt`,
  stopAfter: 'Beenden nach',
  runsUnit: 'Läufen',
  rememberLastRun: 'Letzten Lauf merken',
  rememberHint: 'Jeder Lauf sieht, was der vorige berichtet hat, und macht dort weiter.',
  on: 'An',
  off: 'Aus',
  expression: 'Zeitplan-Ausdruck',
  routineId: 'Routinen-ID',
  script: 'Skript',
  runs: 'Läufe',
  runsHint: 'Läufe bleiben hier. Sie erscheinen nicht in den Chats.',
  noRuns: 'Noch keine Läufe',
  startingRun: 'Lauf wird gestartet…',
  finishedIn: duration => `Fertig in ${duration}`,
  runChips: { running: 'Läuft', failed: 'Fehlgeschlagen', delivery: 'Nicht zugestellt', stopped: 'Gestoppt' },
  deleteRoutine: 'Routine löschen',
  deleteHint: agent => `Stoppt sie endgültig und entfernt sie von ${agent}.`,
  pickDescription: 'Welche Art von Routine? Wähle einen Ausgangspunkt – danach kannst du alles ändern.',
  forAgent: 'Für',
  readyMade: 'Vorlagen',
  searchReadyMade: 'Vorlagen durchsuchen',
  use: 'Verwenden',
  showAll: count => `Alle ${count} anzeigen`,
  showFewer: 'Weniger anzeigen',
  noReadyMade: 'Keine Vorlage passt.',
  describeInChat: 'Oder beschreibe sie in einem Chat mit diesem Agenten',
  askInChat: 'Im Chat fragen',
  startBlank: 'Leer beginnen',
  agent: 'Agent',
  name: 'Name',
  namePlaceholder: 'Morgenbriefing',
  instruction: 'Anweisung',
  instructionPlaceholder:
    'Was soll der Agent jedes Mal tun? Zum Beispiel: meine ungelesenen E-Mails zusammenfassen und Dringendes auflisten.',
  frequencies: {
    once: 'Einmal',
    hourly: 'Stündlich',
    daily: 'Täglich',
    weekdays: 'Werktags',
    weekly: 'Wöchentlich',
    monthly: 'Monatlich',
    custom: 'Benutzerdefiniert'
  },
  at: 'um',
  onDay: 'am Tag',
  customPlaceholder: 'every 2 hours',
  customHint: 'Beschreibe es in einfachen Worten, etwa „every 2 hours“ oder „every monday 9am“.',
  customHintDeveloper: 'Einfache Worte wie „every 2 hours“ oder ein Cron-Ausdruck wie „0 9 * * 1-5“.',
  localTimeHint: example => `Deine Ortszeit, z. B. ${example}`,
  everyRunKept: 'Jeder Lauf wird auch gespeichert',
  nothingSelected: 'Nichts ausgewählt? Die Ergebnisse bleiben in den Läufen gespeichert.',
  agentChat: agent => `Chat mit ${agent}`,
  needsHomeChannel: 'Lege zuerst einen Heimkanal fest',
  noChannels: 'Mit diesem Agenten ist noch kein Messaging-Kanal verbunden.',
  instructionRequired: 'Sag dem Agenten, was er tun soll.',
  scheduleRequired: 'Wähle, wann sie läuft.',
  saveChanges: 'Änderungen speichern',
  schedule: {
    once: (date, time) => `Einmal, am ${date} um ${time}`,
    hourly: 'Jede Stunde',
    hourlyAt: minute => `Jede Stunde um :${minute}`,
    daily: time => `Jeden Tag um ${time}`,
    weekdays: time => `Montag bis Freitag um ${time}`,
    weekly: (day, time) => `Jeden ${day} um ${time}`,
    monthly: (day, time) => `Am ${day}. jedes Monats um ${time}`,
    everyMinutes: count => `Alle ${count} ${plural(count, 'Minute', 'Minuten')}`,
    everyHours: count => `Alle ${count} ${plural(count, 'Stunde', 'Stunden')}`,
    everyDays: count => `Alle ${count} ${plural(count, 'Tag', 'Tage')}`,
    custom: 'Nach eigenem Zeitplan'
  },
  rename: 'Umbenennen',
  renameTitle: 'Routine umbenennen',
  editInSettings: agent => `In den Einstellungen von ${agent} bearbeiten`,
  editInAgent: agent => `In ${agent} bearbeiten`,
  editCardTitle: 'Ändern, was sie tut, wann sie läuft oder wohin die Ergebnisse gehen',
  editCardHint: 'Routinen werden in den Einstellungen ihres Agenten bearbeitet.',
  askInChatPrompt: 'Richte eine Routine ein, die ',
  chipTooltip: (count, agent) => `${count} ${plural(count, 'Routine', 'Routinen')} für ${agent}`,
  chipFailing: name => `${name} fehlgeschlagen · öffnet Routinen`,
  chipOpens: 'Öffnet Routinen',
  railAttention: count => `${count} ${plural(count, 'Routine braucht', 'Routinen brauchen')} Aufmerksamkeit`
}

const es: RoutinesCopy = {
  allRoutines: 'Todas las rutinas',
  groups: { attention: 'Requieren atención', scheduled: 'Programadas', paused: 'En pausa', finished: 'Terminadas' },
  filterLabel: 'Mostrar',
  filters: { all: 'Todas', attention: 'Requieren atención', scheduled: 'Programadas', paused: 'En pausa' },
  search: 'Buscar rutinas',
  columns: { routine: 'Rutina', agent: 'Agente', when: 'Cuándo', lastRun: 'Última ejecución', status: 'Estado' },
  status: {
    failed: 'Falló',
    delivery: 'No entregada',
    overdue: 'Atrasada',
    running: 'En curso',
    on: 'Activa',
    paused: 'En pausa',
    finished: 'Terminada'
  },
  notRunYet: 'Aún no se ha ejecutado',
  runningNow: 'Ejecutándose ahora',
  footnote: (routines, agents) =>
    `${routines} ${plural(routines, 'rutina', 'rutinas')} en ${agents} ${plural(agents, 'agente', 'agentes')}`,
  noMatchesTitle: 'Sin resultados',
  noMatchesDescription: 'Ninguna rutina coincide con esta búsqueda o filtro.',
  emptyDescription: 'Deja que un agente haga algo por su cuenta a las horas que elijas y te envíe los resultados.',
  banner: {
    failedOnce: name => `${name} falló en su última ejecución`,
    failedStreak: (name, runs) => `${name} falló en las últimas ${runs} ejecuciones`,
    delivery: name => `${name} se ejecutó, pero sus resultados no se entregaron`,
    overdue: name => `${name} no se ha ejecutado desde la hora prevista`,
    overdueHint: 'Puede que Hermes no esté en marcha en esta máquina.',
    retryAt: when => `Volverá a intentarlo ${when}.`,
    agentNotRunning: agent => `${agent} no se está ejecutando en este equipo ahora mismo.`,
    details: 'Detalles'
  },
  actions: {
    runNow: 'Ejecutar ahora',
    retry: 'Reintentar',
    edit: 'Editar',
    change: 'Cambiar',
    view: 'Ver',
    delete: 'Eliminar',
    more: 'Más acciones'
  },
  whatItDoes: 'Qué hace',
  whenAndWhere: 'Cuándo y dónde',
  whenToRun: 'Cuándo ejecutarla',
  sendResultsTo: 'Enviar resultados a',
  nextRun: when => `próxima ${when}`,
  keptInRuns: 'Solo se guardan en Ejecuciones',
  scriptRoutine: 'Ejecuta un script por su cuenta, sin el agente.',
  advanced: 'Avanzado',
  advancedSummary: 'Modelo · Ejecuciones perdidas · Límite · Recordar la última ejecución',
  model: 'Modelo',
  modelDefault: 'Modelo predeterminado',
  runLimit: 'Límite de ejecuciones',
  noLimit: 'Sin límite',
  limitProgress: (done, total) => `${done} de ${total} ejecuciones hechas`,
  stopAfter: 'Detener tras',
  runsUnit: 'ejecuciones',
  rememberLastRun: 'Recordar la última ejecución',
  rememberHint: 'Cada ejecución ve lo que informó la anterior y continúa donde lo dejó.',
  on: 'Activado',
  off: 'Desactivado',
  expression: 'Expresión del horario',
  routineId: 'ID de la rutina',
  script: 'Script',
  runs: 'Ejecuciones',
  runsHint: 'Las ejecuciones se quedan aquí. No aparecen en Chats.',
  noRuns: 'Aún no hay ejecuciones',
  startingRun: 'Iniciando una ejecución…',
  finishedIn: duration => `Terminó en ${duration}`,
  runChips: { running: 'En curso', failed: 'Falló', delivery: 'No entregada', stopped: 'Detenida' },
  deleteRoutine: 'Eliminar rutina',
  deleteHint: agent => `La detiene para siempre y la quita de ${agent}.`,
  pickDescription: '¿Qué tipo de rutina? Elige un punto de partida; después puedes cambiarlo todo.',
  forAgent: 'Para',
  readyMade: 'Listas para usar',
  searchReadyMade: 'Buscar rutinas listas para usar',
  use: 'Usar',
  showAll: count => `Mostrar las ${count}`,
  showFewer: 'Mostrar menos',
  noReadyMade: 'Ninguna rutina lista para usar coincide.',
  describeInChat: 'O descríbela en un chat con este agente',
  askInChat: 'Preguntar en un chat',
  startBlank: 'Empezar desde cero',
  agent: 'Agente',
  name: 'Nombre',
  namePlaceholder: 'Resumen matutino',
  instruction: 'Instrucción',
  instructionPlaceholder:
    '¿Qué debe hacer el agente cada vez? Por ejemplo: resumir mi correo sin leer y destacar lo urgente.',
  frequencies: {
    once: 'Una vez',
    hourly: 'Cada hora',
    daily: 'Cada día',
    weekdays: 'Entre semana',
    weekly: 'Cada semana',
    monthly: 'Cada mes',
    custom: 'Personalizado'
  },
  at: 'a las',
  onDay: 'el día',
  customPlaceholder: 'every 2 hours',
  customHint: 'Dilo con palabras sencillas, como «every 2 hours» o «every monday 9am».',
  customHintDeveloper: 'Palabras sencillas como «every 2 hours» o una expresión cron como «0 9 * * 1-5».',
  localTimeHint: example => `Tu hora local, p. ej. ${example}`,
  everyRunKept: 'Cada ejecución también se guarda',
  nothingSelected: '¿Nada seleccionado? Los resultados se siguen guardando en Ejecuciones.',
  agentChat: agent => `Chat con ${agent}`,
  needsHomeChannel: 'Configura primero un canal principal',
  noChannels: 'Este agente aún no tiene ningún canal de mensajería conectado.',
  instructionRequired: 'Dile al agente qué hacer.',
  scheduleRequired: 'Elige cuándo se ejecuta.',
  saveChanges: 'Guardar cambios',
  schedule: {
    once: (date, time) => `Una vez, el ${date} a las ${time}`,
    hourly: 'Cada hora',
    hourlyAt: minute => `Cada hora en el minuto ${minute}`,
    daily: time => `Cada día a las ${time}`,
    weekdays: time => `De lunes a viernes a las ${time}`,
    weekly: (day, time) => `Cada ${day} a las ${time}`,
    monthly: (day, time) => `El día ${day} de cada mes a las ${time}`,
    everyMinutes: count => `Cada ${count} ${plural(count, 'minuto', 'minutos')}`,
    everyHours: count => `Cada ${count} ${plural(count, 'hora', 'horas')}`,
    everyDays: count => `Cada ${count} ${plural(count, 'día', 'días')}`,
    custom: 'Con un horario personalizado'
  },
  rename: 'Cambiar nombre',
  renameTitle: 'Cambiar el nombre de la rutina',
  editInSettings: agent => `Editar en los ajustes de ${agent}`,
  editInAgent: agent => `Editar en ${agent}`,
  editCardTitle: 'Cambiar qué hace, cuándo o adónde van los resultados',
  editCardHint: 'Las rutinas se editan en los ajustes de su agente.',
  askInChatPrompt: 'Configura una rutina que ',
  chipTooltip: (count, agent) => `${count} ${plural(count, 'rutina', 'rutinas')} de ${agent}`,
  chipFailing: name => `${name} falló · abre Rutinas`,
  chipOpens: 'Abre Rutinas',
  railAttention: count => `${count} ${plural(count, 'rutina requiere', 'rutinas requieren')} atención`
}

const ja: RoutinesCopy = {
  allRoutines: 'すべてのルーティン',
  groups: { attention: '要確認', scheduled: 'スケジュール済み', paused: '一時停止中', finished: '終了' },
  filterLabel: '表示',
  filters: { all: 'すべて', attention: '要確認', scheduled: 'スケジュール済み', paused: '一時停止中' },
  search: 'ルーティンを検索',
  columns: {
    routine: 'ルーティン',
    agent: 'エージェント',
    when: '実行タイミング',
    lastRun: '前回の実行',
    status: '状態'
  },
  status: {
    failed: '失敗',
    delivery: '未配信',
    overdue: '遅延',
    running: '実行中',
    on: 'オン',
    paused: '一時停止中',
    finished: '終了'
  },
  notRunYet: 'まだ実行されていません',
  runningNow: '実行中',
  footnote: (routines, agents) => `${agents} 体のエージェントに ${routines} 件のルーティン`,
  noMatchesTitle: '一致なし',
  noMatchesDescription: 'この検索・フィルターに一致するルーティンはありません。',
  emptyDescription: '決めた時刻にエージェントが自動で作業し、結果を届けます。',
  banner: {
    failedOnce: name => `${name} の前回の実行が失敗しました`,
    failedStreak: (name, runs) => `${name} が直近 ${runs} 回連続で失敗しました`,
    delivery: name => `${name} は実行されましたが、結果を配信できませんでした`,
    overdue: name => `${name} が予定時刻を過ぎても実行されていません`,
    overdueHint: 'このマシンで Hermes が動いていない可能性があります。',
    retryAt: when => `次は ${when} に再試行します。`,
    agentNotRunning: agent => `${agent} は現在このコンピューターで実行されていません。`,
    details: '詳細'
  },
  actions: {
    runNow: '今すぐ実行',
    retry: '再試行',
    edit: '編集',
    change: '変更',
    view: '表示',
    delete: '削除',
    more: 'その他の操作'
  },
  whatItDoes: '内容',
  whenAndWhere: 'タイミングと送信先',
  whenToRun: '実行タイミング',
  sendResultsTo: '結果の送信先',
  nextRun: when => `次回 ${when}`,
  keptInRuns: '実行履歴にのみ保存',
  scriptRoutine: 'エージェントを使わずにスクリプトだけを実行します。',
  advanced: '詳細',
  advancedSummary: 'モデル · 実行漏れ · 実行回数の上限 · 前回の実行を覚える',
  model: 'モデル',
  modelDefault: 'デフォルトのモデル',
  runLimit: '実行回数の上限',
  noLimit: '上限なし',
  limitProgress: (done, total) => `${total} 回中 ${done} 回実行済み`,
  stopAfter: '上限',
  runsUnit: '回で停止',
  rememberLastRun: '前回の実行を覚える',
  rememberHint: '毎回、前回の報告内容を踏まえて続きから進めます。',
  on: 'オン',
  off: 'オフ',
  expression: 'スケジュール式',
  routineId: 'ルーティン ID',
  script: 'スクリプト',
  runs: '実行履歴',
  runsHint: '実行履歴はここに残り、チャットには表示されません。',
  noRuns: 'まだ実行されていません',
  startingRun: '実行を開始しています…',
  finishedIn: duration => `${duration} で完了`,
  runChips: { running: '実行中', failed: '失敗', delivery: '未配信', stopped: '停止' },
  deleteRoutine: 'ルーティンを削除',
  deleteHint: agent => `完全に停止し、${agent} から削除します。`,
  pickDescription: 'どんなルーティンにしますか？出発点を選んでください。あとで何でも変更できます。',
  forAgent: '対象',
  readyMade: 'テンプレート',
  searchReadyMade: 'テンプレートを検索',
  use: '使う',
  showAll: count => `${count} 件すべて表示`,
  showFewer: '表示を減らす',
  noReadyMade: '一致するテンプレートはありません。',
  describeInChat: 'または、このエージェントとのチャットで説明する',
  askInChat: 'チャットで依頼',
  startBlank: '白紙から作成',
  agent: 'エージェント',
  name: '名前',
  namePlaceholder: '朝のブリーフィング',
  instruction: '指示',
  instructionPlaceholder: '毎回エージェントに何をさせますか？例：未読メールを要約して、急ぎの件を挙げる。',
  frequencies: {
    once: '1 回だけ',
    hourly: '毎時',
    daily: '毎日',
    weekdays: '平日',
    weekly: '毎週',
    monthly: '毎月',
    custom: 'カスタム'
  },
  at: '時刻',
  onDay: '日付',
  customPlaceholder: 'every 2 hours',
  customHint: '「every 2 hours」「every monday 9am」のように、ふつうの言葉で書いてください。',
  customHintDeveloper: '「every 2 hours」のような言葉か、「0 9 * * 1-5」のような cron 式で指定します。',
  localTimeHint: example => `現地時刻（例: ${example}）`,
  everyRunKept: '各実行は履歴にも保存されます',
  nothingSelected: '何も選ばなくても、結果は実行履歴に保存されます。',
  agentChat: agent => `${agent} とのチャット`,
  needsHomeChannel: '先にホームチャンネルを設定してください',
  noChannels: 'このエージェントにはまだメッセージチャンネルが接続されていません。',
  instructionRequired: 'エージェントにしてほしいことを書いてください。',
  scheduleRequired: '実行タイミングを選んでください。',
  saveChanges: '変更を保存',
  schedule: {
    once: (date, time) => `1 回だけ、${date} ${time}`,
    hourly: '毎時',
    hourlyAt: minute => `毎時 ${minute} 分`,
    daily: time => `毎日 ${time}`,
    weekdays: time => `月曜〜金曜 ${time}`,
    weekly: (day, time) => `毎週${day} ${time}`,
    monthly: (day, time) => `毎月 ${day} 日 ${time}`,
    everyMinutes: count => `${count} 分ごと`,
    everyHours: count => `${count} 時間ごと`,
    everyDays: count => `${count} 日ごと`,
    custom: 'カスタムスケジュール'
  },
  rename: '名前を変更',
  renameTitle: 'ルーティン名を変更',
  editInSettings: agent => `${agent} の設定で編集`,
  editInAgent: agent => `${agent} で編集`,
  editCardTitle: '内容・実行タイミング・結果の送信先を変更',
  editCardHint: 'ルーティンはエージェントの設定で編集します。',
  askInChatPrompt: '次のことをするルーティンを作って：',
  chipTooltip: (count, agent) => `${agent} のルーティン ${count} 件`,
  chipFailing: name => `${name} が失敗 · ルーティンを開きます`,
  chipOpens: 'ルーティンを開きます',
  railAttention: count => `要確認のルーティンが ${count} 件あります`
}

const zh: RoutinesCopy = {
  allRoutines: '全部例行任务',
  groups: { attention: '需要处理', scheduled: '已计划', paused: '已暂停', finished: '已结束' },
  filterLabel: '显示',
  filters: { all: '全部', attention: '需要处理', scheduled: '已计划', paused: '已暂停' },
  search: '搜索例行任务',
  columns: { routine: '例行任务', agent: '智能体', when: '时间', lastRun: '上次运行', status: '状态' },
  status: {
    failed: '失败',
    delivery: '未送达',
    overdue: '已逾期',
    running: '运行中',
    on: '已开启',
    paused: '已暂停',
    finished: '已结束'
  },
  notRunYet: '尚未运行',
  runningNow: '正在运行',
  footnote: (routines, agents) => `${agents} 个智能体，共 ${routines} 个例行任务`,
  noMatchesTitle: '没有匹配项',
  noMatchesDescription: '没有例行任务符合当前搜索或筛选条件。',
  emptyDescription: '让智能体在你选定的时间自动完成工作，并把结果发给你。',
  banner: {
    failedOnce: name => `${name} 上次运行失败`,
    failedStreak: (name, runs) => `${name} 最近 ${runs} 次运行都失败了`,
    delivery: name => `${name} 已运行，但结果未能送达`,
    overdue: name => `${name} 到点后一直没有运行`,
    overdueHint: '这台机器上的 Hermes 可能没有在运行。',
    retryAt: when => `将于${when}重试。`,
    agentNotRunning: agent => `${agent} 目前未在这台电脑上运行。`,
    details: '详细信息'
  },
  actions: {
    runNow: '立即运行',
    retry: '重试',
    edit: '编辑',
    change: '更改',
    view: '查看',
    delete: '删除',
    more: '更多操作'
  },
  whatItDoes: '任务内容',
  whenAndWhere: '时间与去向',
  whenToRun: '运行时间',
  sendResultsTo: '结果发送至',
  nextRun: when => `下次 ${when}`,
  keptInRuns: '仅保存在运行记录中',
  scriptRoutine: '单独运行一个脚本，不调用智能体。',
  advanced: '高级',
  advancedSummary: '模型 · 错过的运行 · 运行次数上限 · 记住上次运行',
  model: '模型',
  modelDefault: '默认模型',
  runLimit: '运行次数上限',
  noLimit: '不限',
  limitProgress: (done, total) => `已运行 ${done}/${total} 次`,
  stopAfter: '运行',
  runsUnit: '次后停止',
  rememberLastRun: '记住上次运行',
  rememberHint: '每次运行都能看到上一次的报告，从上次停下的地方继续。',
  on: '开启',
  off: '关闭',
  expression: '计划表达式',
  routineId: '例行任务 ID',
  script: '脚本',
  runs: '运行记录',
  runsHint: '运行记录保留在这里，不会出现在对话中。',
  noRuns: '还没有运行记录',
  startingRun: '正在开始运行…',
  finishedIn: duration => `用时 ${duration}`,
  runChips: { running: '运行中', failed: '失败', delivery: '未送达', stopped: '已停止' },
  deleteRoutine: '删除例行任务',
  deleteHint: agent => `将永久停止它，并从 ${agent} 中移除。`,
  pickDescription: '想要哪种例行任务？先选一个起点，之后一切都可以修改。',
  forAgent: '用于',
  readyMade: '现成模板',
  searchReadyMade: '搜索现成模板',
  use: '使用',
  showAll: count => `显示全部 ${count} 个`,
  showFewer: '收起',
  noReadyMade: '没有匹配的现成模板。',
  describeInChat: '或在与此智能体的对话中描述它',
  askInChat: '在对话中提出',
  startBlank: '从空白开始',
  agent: '智能体',
  name: '名称',
  namePlaceholder: '晨间简报',
  instruction: '指令',
  instructionPlaceholder: '智能体每次要做什么？例如：总结我的未读邮件，并列出紧急事项。',
  frequencies: {
    once: '仅一次',
    hourly: '每小时',
    daily: '每天',
    weekdays: '工作日',
    weekly: '每周',
    monthly: '每月',
    custom: '自定义'
  },
  at: '时间',
  onDay: '日期',
  customPlaceholder: 'every 2 hours',
  customHint: '用简单的话描述，例如 “every 2 hours” 或 “every monday 9am”。',
  customHintDeveloper: '简单的话，如 “every 2 hours”，或 cron 表达式，如 “0 9 * * 1-5”。',
  localTimeHint: example => `本地时间，例如 ${example}`,
  everyRunKept: '每次运行也都会保存在运行记录中',
  nothingSelected: '什么都没选？结果仍会保存在运行记录中。',
  agentChat: agent => `与 ${agent} 的对话`,
  needsHomeChannel: '请先设置主频道',
  noChannels: '这个智能体还没有连接任何消息渠道。',
  instructionRequired: '请告诉智能体要做什么。',
  scheduleRequired: '请选择运行时间。',
  saveChanges: '保存更改',
  schedule: {
    once: (date, time) => `仅一次，${date} ${time}`,
    hourly: '每小时',
    hourlyAt: minute => `每小时第 ${minute} 分`,
    daily: time => `每天 ${time}`,
    weekdays: time => `周一至周五 ${time}`,
    weekly: (day, time) => `每${day} ${time}`,
    monthly: (day, time) => `每月 ${day} 日 ${time}`,
    everyMinutes: count => `每 ${count} 分钟`,
    everyHours: count => `每 ${count} 小时`,
    everyDays: count => `每 ${count} 天`,
    custom: '自定义计划'
  },
  rename: '重命名',
  renameTitle: '重命名例行任务',
  editInSettings: agent => `在 ${agent} 的设置中编辑`,
  editInAgent: agent => `在 ${agent} 中编辑`,
  editCardTitle: '更改任务内容、运行时间或结果去向',
  editCardHint: '例行任务在其智能体的设置中编辑。',
  askInChatPrompt: '帮我设置一个例行任务：',
  chipTooltip: (count, agent) => `${agent} 的 ${count} 个例行任务`,
  chipFailing: name => `${name} 失败 · 打开例行任务`,
  chipOpens: '打开例行任务',
  railAttention: count => `${count} 个例行任务需要处理`
}

const zhHant: RoutinesCopy = {
  allRoutines: '所有例行任務',
  groups: { attention: '需要處理', scheduled: '已排程', paused: '已暫停', finished: '已結束' },
  filterLabel: '顯示',
  filters: { all: '全部', attention: '需要處理', scheduled: '已排程', paused: '已暫停' },
  search: '搜尋例行任務',
  columns: { routine: '例行任務', agent: '智慧體', when: '時間', lastRun: '上次執行', status: '狀態' },
  status: {
    failed: '失敗',
    delivery: '未送達',
    overdue: '已逾期',
    running: '執行中',
    on: '已開啟',
    paused: '已暫停',
    finished: '已結束'
  },
  notRunYet: '尚未執行',
  runningNow: '正在執行',
  footnote: (routines, agents) => `${agents} 個智慧體，共 ${routines} 個例行任務`,
  noMatchesTitle: '沒有符合的項目',
  noMatchesDescription: '沒有例行任務符合目前的搜尋或篩選條件。',
  emptyDescription: '讓智慧體在你選定的時間自動完成工作，並把結果傳給你。',
  banner: {
    failedOnce: name => `${name} 上次執行失敗`,
    failedStreak: (name, runs) => `${name} 最近 ${runs} 次執行都失敗了`,
    delivery: name => `${name} 已執行，但結果未能送達`,
    overdue: name => `${name} 過了預定時間仍未執行`,
    overdueHint: '這台機器上的 Hermes 可能沒有在執行。',
    retryAt: when => `將於${when}重試。`,
    agentNotRunning: agent => `${agent} 目前未在這台電腦上執行。`,
    details: '詳細資訊'
  },
  actions: {
    runNow: '立即執行',
    retry: '重試',
    edit: '編輯',
    change: '變更',
    view: '查看',
    delete: '刪除',
    more: '更多動作'
  },
  whatItDoes: '任務內容',
  whenAndWhere: '時間與去向',
  whenToRun: '執行時間',
  sendResultsTo: '結果傳送至',
  nextRun: when => `下次 ${when}`,
  keptInRuns: '僅保存在執行紀錄中',
  scriptRoutine: '單獨執行一個腳本，不呼叫智慧體。',
  advanced: '進階',
  advancedSummary: '模型 · 錯過的執行 · 執行次數上限 · 記住上次執行',
  model: '模型',
  modelDefault: '預設模型',
  runLimit: '執行次數上限',
  noLimit: '不限',
  limitProgress: (done, total) => `已執行 ${done}/${total} 次`,
  stopAfter: '執行',
  runsUnit: '次後停止',
  rememberLastRun: '記住上次執行',
  rememberHint: '每次執行都能看到上一次的回報，從上次停下的地方繼續。',
  on: '開啟',
  off: '關閉',
  expression: '排程表示式',
  routineId: '例行任務 ID',
  script: '腳本',
  runs: '執行紀錄',
  runsHint: '執行紀錄保留在這裡，不會出現在對話中。',
  noRuns: '還沒有執行紀錄',
  startingRun: '正在開始執行…',
  finishedIn: duration => `耗時 ${duration}`,
  runChips: { running: '執行中', failed: '失敗', delivery: '未送達', stopped: '已停止' },
  deleteRoutine: '刪除例行任務',
  deleteHint: agent => `將永久停止它，並從 ${agent} 移除。`,
  pickDescription: '想要哪種例行任務？先選一個起點，之後一切都能修改。',
  forAgent: '用於',
  readyMade: '現成範本',
  searchReadyMade: '搜尋現成範本',
  use: '使用',
  showAll: count => `顯示全部 ${count} 個`,
  showFewer: '收合',
  noReadyMade: '沒有符合的現成範本。',
  describeInChat: '或在與此智慧體的對話中描述它',
  askInChat: '在對話中提出',
  startBlank: '從空白開始',
  agent: '智慧體',
  name: '名稱',
  namePlaceholder: '晨間簡報',
  instruction: '指令',
  instructionPlaceholder: '智慧體每次要做什麼？例如：摘要我的未讀郵件，並列出緊急事項。',
  frequencies: {
    once: '僅一次',
    hourly: '每小時',
    daily: '每天',
    weekdays: '平日',
    weekly: '每週',
    monthly: '每月',
    custom: '自訂'
  },
  at: '時間',
  onDay: '日期',
  customPlaceholder: 'every 2 hours',
  customHint: '用簡單的話描述，例如「every 2 hours」或「every monday 9am」。',
  customHintDeveloper: '簡單的話，如「every 2 hours」，或 cron 表示式，如「0 9 * * 1-5」。',
  localTimeHint: example => `當地時間，例如 ${example}`,
  everyRunKept: '每次執行也都會保存在執行紀錄中',
  nothingSelected: '什麼都沒選？結果仍會保存在執行紀錄中。',
  agentChat: agent => `與 ${agent} 的對話`,
  needsHomeChannel: '請先設定主頻道',
  noChannels: '這個智慧體還沒有連接任何訊息管道。',
  instructionRequired: '請告訴智慧體要做什麼。',
  scheduleRequired: '請選擇執行時間。',
  saveChanges: '儲存變更',
  schedule: {
    once: (date, time) => `僅一次，${date} ${time}`,
    hourly: '每小時',
    hourlyAt: minute => `每小時第 ${minute} 分`,
    daily: time => `每天 ${time}`,
    weekdays: time => `週一至週五 ${time}`,
    weekly: (day, time) => `每${day} ${time}`,
    monthly: (day, time) => `每月 ${day} 日 ${time}`,
    everyMinutes: count => `每 ${count} 分鐘`,
    everyHours: count => `每 ${count} 小時`,
    everyDays: count => `每 ${count} 天`,
    custom: '自訂排程'
  },
  rename: '重新命名',
  renameTitle: '重新命名例行任務',
  editInSettings: agent => `在 ${agent} 的設定中編輯`,
  editInAgent: agent => `在 ${agent} 中編輯`,
  editCardTitle: '變更任務內容、執行時間或結果去向',
  editCardHint: '例行任務在其智慧體的設定中編輯。',
  askInChatPrompt: '幫我設定一個例行任務：',
  chipTooltip: (count, agent) => `${agent} 的 ${count} 個例行任務`,
  chipFailing: name => `${name} 失敗 · 開啟例行任務`,
  chipOpens: '開啟例行任務',
  railAttention: count => `${count} 個例行任務需要處理`
}

const RU_PLURAL = (count: number, one: string, few: string, many: string) => {
  const mod10 = count % 10
  const mod100 = count % 100

  if (mod10 === 1 && mod100 !== 11) {
    return one
  }

  return mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14) ? few : many
}

const ru: RoutinesCopy = {
  allRoutines: 'Все регулярные задачи',
  groups: {
    attention: 'Требуют внимания',
    scheduled: 'По расписанию',
    paused: 'Приостановлены',
    finished: 'Завершены'
  },
  filterLabel: 'Показать',
  filters: { all: 'Все', attention: 'Требуют внимания', scheduled: 'По расписанию', paused: 'Приостановлены' },
  search: 'Поиск регулярных задач',
  columns: { routine: 'Задача', agent: 'Агент', when: 'Когда', lastRun: 'Последний запуск', status: 'Состояние' },
  status: {
    failed: 'Ошибка',
    delivery: 'Не доставлено',
    overdue: 'Просрочено',
    running: 'Выполняется',
    on: 'Включена',
    paused: 'Приостановлена',
    finished: 'Завершена'
  },
  notRunYet: 'Ещё не запускалась',
  runningNow: 'Выполняется сейчас',
  footnote: (routines, agents) =>
    `${routines} ${RU_PLURAL(routines, 'регулярная задача', 'регулярные задачи', 'регулярных задач')} у ${agents} ${RU_PLURAL(agents, 'агента', 'агентов', 'агентов')}`,
  noMatchesTitle: 'Ничего не найдено',
  noMatchesDescription: 'Ни одна регулярная задача не подходит под этот поиск или фильтр.',
  emptyDescription: 'Пусть агент сам выполняет работу в выбранное время и присылает вам результаты.',
  banner: {
    failedOnce: name => `${name}: последний запуск завершился ошибкой`,
    failedStreak: (name, runs) => `${name}: последние ${runs} запуска завершились ошибкой`,
    delivery: name => `${name}: запуск прошёл, но результаты не доставлены`,
    overdue: name => `${name}: задача не запустилась в назначенное время`,
    overdueHint: 'Возможно, Hermes не запущен на этом компьютере.',
    retryAt: when => `Следующая попытка — ${when}.`,
    agentNotRunning: agent => `${agent} сейчас не запущен на этом компьютере.`,
    details: 'Подробнее'
  },
  actions: {
    runNow: 'Запустить сейчас',
    retry: 'Повторить',
    edit: 'Изменить',
    change: 'Изменить',
    view: 'Открыть',
    delete: 'Удалить',
    more: 'Другие действия'
  },
  whatItDoes: 'Что делает',
  whenAndWhere: 'Когда и куда',
  whenToRun: 'Когда запускать',
  sendResultsTo: 'Куда отправлять результаты',
  nextRun: when => `следующий запуск ${when}`,
  keptInRuns: 'Только в истории запусков',
  scriptRoutine: 'Выполняет скрипт сам по себе, без агента.',
  advanced: 'Дополнительно',
  advancedSummary: 'Модель · Пропущенные запуски · Лимит · Помнить прошлый запуск',
  model: 'Модель',
  modelDefault: 'Модель по умолчанию',
  runLimit: 'Лимит запусков',
  noLimit: 'Без ограничений',
  limitProgress: (done, total) => `Выполнено ${done} из ${total}`,
  stopAfter: 'Остановить после',
  runsUnit: 'запусков',
  rememberLastRun: 'Помнить прошлый запуск',
  rememberHint: 'Каждый запуск видит отчёт предыдущего и продолжает с того же места.',
  on: 'Вкл.',
  off: 'Выкл.',
  expression: 'Выражение расписания',
  routineId: 'ID регулярной задачи',
  script: 'Скрипт',
  runs: 'Запуски',
  runsHint: 'Запуски хранятся здесь и не появляются в чатах.',
  noRuns: 'Запусков пока не было',
  startingRun: 'Запуск начинается…',
  finishedIn: duration => `Готово за ${duration}`,
  runChips: { running: 'Выполняется', failed: 'Ошибка', delivery: 'Не доставлено', stopped: 'Остановлен' },
  deleteRoutine: 'Удалить регулярную задачу',
  deleteHint: agent => `Полностью останавливает её и убирает у агента ${agent}.`,
  pickDescription: 'Какую регулярную задачу создать? Выберите отправную точку — потом всё можно изменить.',
  forAgent: 'Для',
  readyMade: 'Готовые',
  searchReadyMade: 'Поиск готовых задач',
  use: 'Выбрать',
  showAll: count => `Показать все (${count})`,
  showFewer: 'Свернуть',
  noReadyMade: 'Подходящих готовых задач нет.',
  describeInChat: 'Или опишите её в чате с этим агентом',
  askInChat: 'Попросить в чате',
  startBlank: 'Начать с нуля',
  agent: 'Агент',
  name: 'Название',
  namePlaceholder: 'Утренняя сводка',
  instruction: 'Инструкция',
  instructionPlaceholder:
    'Что агенту делать каждый раз? Например: кратко пересказать непрочитанную почту и перечислить срочное.',
  frequencies: {
    once: 'Один раз',
    hourly: 'Каждый час',
    daily: 'Каждый день',
    weekdays: 'По будням',
    weekly: 'Каждую неделю',
    monthly: 'Каждый месяц',
    custom: 'Своё'
  },
  at: 'в',
  onDay: 'число',
  customPlaceholder: 'every 2 hours',
  customHint: 'Опишите простыми словами, например «every 2 hours» или «every monday 9am».',
  customHintDeveloper: 'Простые слова, например «every 2 hours», или cron-выражение, например «0 9 * * 1-5».',
  localTimeHint: example => `Ваше местное время, например ${example}`,
  everyRunKept: 'Каждый запуск сохраняется в истории',
  nothingSelected: 'Ничего не выбрано? Результаты всё равно сохранятся в истории запусков.',
  agentChat: agent => `Чат с ${agent}`,
  needsHomeChannel: 'Сначала задайте основной канал',
  noChannels: 'К этому агенту пока не подключён ни один мессенджер.',
  instructionRequired: 'Напишите, что должен сделать агент.',
  scheduleRequired: 'Выберите, когда запускать.',
  saveChanges: 'Сохранить изменения',
  schedule: {
    once: (date, time) => `Один раз, ${date} в ${time}`,
    hourly: 'Каждый час',
    hourlyAt: minute => `Каждый час в :${minute}`,
    daily: time => `Каждый день в ${time}`,
    weekdays: time => `С понедельника по пятницу в ${time}`,
    weekly: (day, time) => `Еженедельно: ${day}, ${time}`,
    monthly: (day, time) => `${day}-го числа каждого месяца в ${time}`,
    everyMinutes: count => `Каждые ${count} ${RU_PLURAL(count, 'минуту', 'минуты', 'минут')}`,
    everyHours: count => `Каждые ${count} ${RU_PLURAL(count, 'час', 'часа', 'часов')}`,
    everyDays: count => `Каждые ${count} ${RU_PLURAL(count, 'день', 'дня', 'дней')}`,
    custom: 'По своему расписанию'
  },
  rename: 'Переименовать',
  renameTitle: 'Переименовать регулярную задачу',
  editInSettings: agent => `Изменить в настройках ${agent}`,
  editInAgent: agent => `Изменить в ${agent}`,
  editCardTitle: 'Изменить, что она делает, когда запускается и куда отправляет результаты',
  editCardHint: 'Регулярные задачи меняются в настройках их агента.',
  askInChatPrompt: 'Настрой регулярную задачу, которая ',
  chipTooltip: (count, agent) =>
    `${agent}: ${count} ${RU_PLURAL(count, 'регулярная задача', 'регулярные задачи', 'регулярных задач')}`,
  chipFailing: name => `${name}: ошибка · открывает регулярные задачи`,
  chipOpens: 'Открывает регулярные задачи',
  railAttention: count =>
    `${count} ${RU_PLURAL(count, 'регулярная задача требует', 'регулярные задачи требуют', 'регулярных задач требуют')} внимания`
}

const ar: RoutinesCopy = {
  allRoutines: 'كل المهام الروتينية',
  groups: { attention: 'تحتاج إلى انتباه', scheduled: 'مجدولة', paused: 'متوقفة مؤقتًا', finished: 'منتهية' },
  filterLabel: 'عرض',
  filters: { all: 'الكل', attention: 'تحتاج إلى انتباه', scheduled: 'مجدولة', paused: 'متوقفة مؤقتًا' },
  search: 'البحث في المهام الروتينية',
  columns: { routine: 'المهمة', agent: 'الوكيل', when: 'التوقيت', lastRun: 'آخر تشغيل', status: 'الحالة' },
  status: {
    failed: 'فشلت',
    delivery: 'لم تُرسل',
    overdue: 'متأخرة',
    running: 'قيد التشغيل',
    on: 'مفعّلة',
    paused: 'متوقفة مؤقتًا',
    finished: 'منتهية'
  },
  notRunYet: 'لم تُشغَّل بعد',
  runningNow: 'قيد التشغيل الآن',
  footnote: (routines, agents) => `المهام الروتينية: ${routines} · الوكلاء: ${agents}`,
  noMatchesTitle: 'لا توجد نتائج',
  noMatchesDescription: 'لا توجد مهمة روتينية تطابق هذا البحث أو عامل التصفية.',
  emptyDescription: 'دع وكيلًا ينجز عملًا بنفسه في الأوقات التي تختارها ويرسل إليك النتائج.',
  banner: {
    failedOnce: name => `فشل آخر تشغيل لـ ${name}`,
    failedStreak: (name, runs) => `فشلت آخر ${runs} مرات تشغيل لـ ${name}`,
    delivery: name => `تم تشغيل ${name}، لكن نتائجها لم تُرسل`,
    overdue: name => `لم تُشغَّل ${name} منذ موعدها`,
    overdueHint: 'ربما لا يعمل Hermes على هذا الجهاز.',
    retryAt: when => `ستُعاد المحاولة ${when}.`,
    agentNotRunning: agent => `${agent} لا يعمل على هذا الجهاز الآن.`,
    details: 'التفاصيل'
  },
  actions: {
    runNow: 'تشغيل الآن',
    retry: 'إعادة المحاولة',
    edit: 'تحرير',
    change: 'تغيير',
    view: 'عرض',
    delete: 'حذف',
    more: 'المزيد من الإجراءات'
  },
  whatItDoes: 'ما الذي تفعله',
  whenAndWhere: 'متى وإلى أين',
  whenToRun: 'وقت التشغيل',
  sendResultsTo: 'إرسال النتائج إلى',
  nextRun: when => `التالي ${when}`,
  keptInRuns: 'تُحفظ في سجل التشغيل فقط',
  scriptRoutine: 'تشغّل نصًا برمجيًا وحده، دون الوكيل.',
  advanced: 'متقدم',
  advancedSummary: 'النموذج · مرات التشغيل الفائتة · حد التشغيل · تذكّر آخر تشغيل',
  model: 'النموذج',
  modelDefault: 'النموذج الافتراضي',
  runLimit: 'حد التشغيل',
  noLimit: 'بلا حد',
  limitProgress: (done, total) => `اكتمل ${done} من ${total}`,
  stopAfter: 'الإيقاف بعد',
  runsUnit: 'مرات تشغيل',
  rememberLastRun: 'تذكّر آخر تشغيل',
  rememberHint: 'يرى كل تشغيل ما أبلغ عنه التشغيل السابق ويكمل من حيث توقف.',
  on: 'مفعّل',
  off: 'متوقف',
  expression: 'تعبير الجدول',
  routineId: 'معرّف المهمة الروتينية',
  script: 'النص البرمجي',
  runs: 'سجل التشغيل',
  runsHint: 'تبقى مرات التشغيل هنا ولا تظهر في المحادثات.',
  noRuns: 'لا توجد مرات تشغيل بعد',
  startingRun: 'جارٍ بدء التشغيل…',
  finishedIn: duration => `انتهى خلال ${duration}`,
  runChips: { running: 'قيد التشغيل', failed: 'فشل', delivery: 'لم يُرسل', stopped: 'أُوقف' },
  deleteRoutine: 'حذف المهمة الروتينية',
  deleteHint: agent => `يوقفها نهائيًا ويزيلها من ${agent}.`,
  pickDescription: 'ما نوع المهمة الروتينية؟ اختر نقطة بداية، ويمكنك تغيير كل شيء بعد ذلك.',
  forAgent: 'لـ',
  readyMade: 'جاهزة',
  searchReadyMade: 'البحث في المهام الجاهزة',
  use: 'استخدام',
  showAll: count => `عرض الكل (${count})`,
  showFewer: 'عرض أقل',
  noReadyMade: 'لا توجد مهمة جاهزة مطابقة.',
  describeInChat: 'أو صِفها في محادثة مع هذا الوكيل',
  askInChat: 'الطلب في محادثة',
  startBlank: 'البدء من الصفر',
  agent: 'الوكيل',
  name: 'الاسم',
  namePlaceholder: 'موجز الصباح',
  instruction: 'التعليمات',
  instructionPlaceholder: 'ماذا يفعل الوكيل في كل مرة؟ مثلًا: لخّص بريدي غير المقروء واذكر ما هو عاجل.',
  frequencies: {
    once: 'مرة واحدة',
    hourly: 'كل ساعة',
    daily: 'كل يوم',
    weekdays: 'أيام العمل',
    weekly: 'كل أسبوع',
    monthly: 'كل شهر',
    custom: 'مخصص'
  },
  at: 'في',
  onDay: 'في اليوم',
  customPlaceholder: 'every 2 hours',
  customHint: 'اكتبه بكلمات بسيطة، مثل «every 2 hours» أو «every monday 9am».',
  customHintDeveloper: 'كلمات بسيطة مثل «every 2 hours»، أو تعبير cron مثل «0 9 * * 1-5».',
  localTimeHint: example => `بتوقيتك المحلي، مثل ${example}`,
  everyRunKept: 'يُحفظ كل تشغيل في السجل أيضًا',
  nothingSelected: 'لم تختر شيئًا؟ تبقى النتائج محفوظة في سجل التشغيل.',
  agentChat: agent => `المحادثة مع ${agent}`,
  needsHomeChannel: 'عيّن قناة رئيسية أولًا',
  noChannels: 'لا توجد قناة مراسلة متصلة بهذا الوكيل بعد.',
  instructionRequired: 'أخبر الوكيل بما يفعله.',
  scheduleRequired: 'اختر وقت التشغيل.',
  saveChanges: 'حفظ التغييرات',
  schedule: {
    once: (date, time) => `مرة واحدة، ${date} في ${time}`,
    hourly: 'كل ساعة',
    hourlyAt: minute => `كل ساعة عند الدقيقة ${minute}`,
    daily: time => `كل يوم في ${time}`,
    weekdays: time => `من الاثنين إلى الجمعة في ${time}`,
    weekly: (day, time) => `كل ${day} في ${time}`,
    monthly: (day, time) => `اليوم ${day} من كل شهر في ${time}`,
    everyMinutes: count => `كل ${count} دقيقة`,
    everyHours: count => `كل ${count} ساعة`,
    everyDays: count => `كل ${count} يوم`,
    custom: 'وفق جدول مخصص'
  },
  rename: 'إعادة التسمية',
  renameTitle: 'إعادة تسمية المهمة الروتينية',
  editInSettings: agent => `التحرير في إعدادات ${agent}`,
  editInAgent: agent => `التحرير في ${agent}`,
  editCardTitle: 'غيّر ما تفعله أو وقت تشغيلها أو وجهة نتائجها',
  editCardHint: 'تُحرَّر المهام الروتينية في إعدادات وكيلها.',
  askInChatPrompt: 'أعدّ لي مهمة روتينية ',
  chipTooltip: (count, agent) => `مهام ${agent} الروتينية: ${count}`,
  chipFailing: name => `فشلت ${name} · يفتح المهام الروتينية`,
  chipOpens: 'يفتح المهام الروتينية',
  railAttention: count => `مهام روتينية تحتاج إلى انتباه: ${count}`
}

const COPY: Record<BundledLocale, RoutinesCopy> = { en, fr, de, es, ja, zh, 'zh-hant': zhHant, ru, ar }

export function routinesCopy(locale: string): RoutinesCopy {
  return COPY[isBundledLocale(locale) ? locale : 'en']
}

export function useRoutinesCopy(): RoutinesCopy {
  return routinesCopy(useI18n().locale)
}
