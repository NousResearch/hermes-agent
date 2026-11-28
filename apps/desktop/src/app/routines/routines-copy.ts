import { isBundledLocale, useI18n } from '@/i18n'
import type { BundledLocale } from '@/i18n/types'

/** The redesigned Routines page's own words (Figma "Routines final rev" 1081:29252). The redesign
 *  renders every one of them, so they hold the redesign terms directly; shared nouns (the page title,
 *  New routine, Pause routine, the missed-run choices) come from the catalog's redesign overlay. */
/** One run, said by its day: today's, yesterday's, a dated one, or (no date known) the last. */
export interface RunDayWords {
  today: (date: string) => string
  yesterday: (date: string) => string
  date: (date: string) => string
  last: string
}

export interface RoutinesCopy {
  groups: { attention: string; scheduled: string; paused: string; finished: string }
  search: string
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
  noMatchesDescription: string
  emptyDescription: string
  banner: {
    failed: RunDayWords
    notDelivered: RunDayWords
    failedStreak: (runs: number) => string
    overdue: string
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
    change: string
    view: string
    delete: string
    /** Run now's label while the routine is paused. */
    runRoutine: string
    runNowHint: string
    viewError: string
    undo: string
  }
  sendResultsTo: string
  keptInRuns: string
  scriptRoutine: string
  advanced: string
  model: string
  modelDefault: string
  runLimit: string
  runLimitHint: string
  stopAfter: string
  runsUnit: string
  rememberLastRun: string
  rememberHint: string
  routineId: string
  runs: string
  noRuns: string
  startingRun: string
  runChips: { running: string; failed: string; delivery: string; stopped: string; succeeded: string; missed: string }
  deleteRoutine: string
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
  everyRunKept: string
  nothingSelected: string
  agentChat: (agent: string) => string
  needsHomeChannel: string
  noChannels: string
  instructionRequired: string
  scheduleRequired: string
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
  askInChatPrompt: string
  railAttention: (count: number) => string
  frequency: string
  selectFrequency: string
  lastRan: string
  nextRunLabel: string
  notScheduled: string
  missedRunsHint: string
  /** The accessible name of a field's info button, whose tooltip holds its description. */
  aboutField: (label: string) => string
  runColumns: { when: string; duration: string; outcome: string }
  loadingMoreRuns: string
  newestRuns: (count: number) => string
  deleteConfirm: string
  deleteConfirmHint: string
  keepIt: string
  deleted: (name: string) => string
  deleteFailed: (name: string) => string
  undoFailed: (name: string) => string
  createTitle: string
  editTitle: (name: string) => string
  startFrom: string
  blank: string
  blankHint: string
  template: string
  templateHint: string
  searchTemplates: string
  noTemplates: string
  templatesFailed: string
  /** The catalog's own categories; a plugin's category shows as it names it. */
  templateCategories: { daily: string; weekly: string; email: string; general: string }
  fromTemplate: (title: string) => string
  agentLocked: string
  optional: string
  firstRun: (when: string) => string
  runsOnce: (when: string) => string
  createInChat: string
  saveRoutine: string
  agentRoutinesHint: string
  openRoutines: string
}

const plural = (count: number, one: string, other: string) => (count === 1 ? one : other)

const en: RoutinesCopy = {
  groups: { attention: 'Needs attention', scheduled: 'Scheduled', paused: 'Paused', finished: 'Finished' },
  search: 'Search routines',
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
  noMatchesDescription: 'No routine matches this search.',
  emptyDescription: 'Let an agent do something on its own at the times you choose, and send you the results.',
  banner: {
    failed: {
      today: () => 'Today’s run failed',
      yesterday: () => 'Yesterday’s run failed',
      date: date => `The run on ${date} failed`,
      last: 'The last run failed'
    },
    notDelivered: {
      today: () => 'Today’s run wasn’t delivered',
      yesterday: () => 'Yesterday’s run wasn’t delivered',
      date: date => `The run on ${date} wasn’t delivered`,
      last: 'The last run wasn’t delivered'
    },
    failedStreak: runs => `The last ${runs} runs failed`,
    overdue: 'It hasn’t run since it was due',
    overdueHint: 'Hermes may not be running on this machine.',
    retryAt: when => `It will try again ${when}.`,
    agentNotRunning: agent => `${agent} isn’t running on this machine right now.`,
    details: 'Details'
  },
  actions: {
    runNow: 'Run now',
    retry: 'Retry',
    change: 'Change',
    view: 'View',
    delete: 'Delete',
    runRoutine: 'Run routine',
    runNowHint: 'Run once now. Your schedule stays the same.',
    viewError: 'View error',
    undo: 'Undo'
  },
  sendResultsTo: 'Send results to',
  keptInRuns: 'Kept in Runs only',
  scriptRoutine: 'Runs a script on its own, without the agent.',
  advanced: 'Advanced',
  model: 'Model',
  modelDefault: 'Default model',
  runLimit: 'Run limit',
  runLimitHint: 'Stops the routine for good after this many runs in total, over its whole life — not retries within a single run.',
  stopAfter: 'Stop after',
  runsUnit: 'runs',
  rememberLastRun: 'Remember last run',
  rememberHint: 'Each run sees what the previous run reported. It can continue where it left off.',
  routineId: 'Routine ID',
  runs: 'Run history',
  noRuns: 'No runs yet',
  startingRun: 'Starting a run…',
  runChips: { running: 'Running', failed: 'Failed', delivery: 'Not delivered', stopped: 'Stopped', succeeded: 'Succeeded', missed: 'Missed' },
  deleteRoutine: 'Delete routine',
  agent: 'Agent',
  name: 'Name',
  namePlaceholder: 'Morning brief',
  instruction: 'Instruction',
  instructionPlaceholder: 'What should the agent do each time? For example: summarize my unread email and list anything urgent.',
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
  everyRunKept: 'Every run is also kept in Runs',
  nothingSelected: 'Nothing selected? Results are still saved in Runs.',
  agentChat: agent => `Chat with ${agent}`,
  needsHomeChannel: 'Set a home channel first',
  noChannels: 'No messaging channel is connected to this agent yet.',
  instructionRequired: 'Tell the agent what to do.',
  scheduleRequired: 'Choose when it runs.',
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
  askInChatPrompt: 'Set up a routine that ',
  railAttention: count => `${count} ${plural(count, 'routine needs', 'routines need')} attention`,
  frequency: 'Frequency',
  selectFrequency: 'Select frequency',
  lastRan: 'Last ran',
  nextRunLabel: 'Next run',
  notScheduled: 'Not scheduled',
  missedRunsHint: 'What to do with runs missed while Hermes was not running.',
  aboutField: label => `About ${label}`,
  runColumns: { when: 'When', duration: 'Duration', outcome: 'Outcome' },
  loadingMoreRuns: 'Loading more runs…',
  newestRuns: count => `Showing the newest ${count} runs.`,
  deleteConfirm: 'Delete this routine?',
  deleteConfirmHint: 'Its run history will also be deleted.',
  keepIt: 'Keep it',
  deleted: name => `${name} deleted.`,
  deleteFailed: name => `Couldn’t delete ${name}. It’s paused and still listed.`,
  undoFailed: name => `${name} is back, but couldn’t be resumed. It’s paused.`,
  createTitle: 'Create a new routine',
  editTitle: name => `Edit “${name}”`,
  startFrom: 'Start from',
  blank: 'Blank',
  blankHint: 'Create your own routine',
  template: 'Template',
  templateHint: 'Pick from templates',
  searchTemplates: 'Search templates',
  noTemplates: 'No template matches.',
  templatesFailed: 'Templates couldn’t be loaded.',
  templateCategories: { daily: 'Daily', weekly: 'Weekly', email: 'Email', general: 'General' },
  fromTemplate: title => `From template: ${title}`,
  agentLocked: 'An agent can’t be changed after a routine is created.',
  optional: '(optional)',
  firstRun: when => `First run: ${when}`,
  runsOnce: when => `Runs once: ${when}`,
  createInChat: 'Create in chat',
  saveRoutine: 'Save & update routine',
  agentRoutinesHint: 'Routines are created and changed on the Routines page.',
  openRoutines: 'Open Routines'
}

const fr: RoutinesCopy = {
  groups: { attention: 'À vérifier', scheduled: 'Planifiées', paused: 'En pause', finished: 'Terminées' },
  search: 'Rechercher des routines',
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
  noMatchesDescription: 'Aucune routine ne correspond à cette recherche ou à ce filtre.',
  emptyDescription: 'Laissez un agent agir seul aux moments choisis et vous envoyer les résultats.',
  banner: {
    failed: {
      today: () => 'L’exécution d’aujourd’hui a échoué',
      yesterday: () => 'L’exécution d’hier a échoué',
      date: date => `L’exécution du ${date} a échoué`,
      last: 'La dernière exécution a échoué'
    },
    notDelivered: {
      today: () => 'L’exécution d’aujourd’hui n’a pas été envoyée',
      yesterday: () => 'L’exécution d’hier n’a pas été envoyée',
      date: date => `L’exécution du ${date} n’a pas été envoyée`,
      last: 'La dernière exécution n’a pas été envoyée'
    },
    failedStreak: runs => `Les ${runs} dernières exécutions ont échoué`,
    overdue: 'Elle ne s’est pas exécutée à l’heure prévue',
    overdueHint: 'Hermes n’est peut-être pas lancé sur cette machine.',
    retryAt: when => `Nouvel essai ${when}.`,
    agentNotRunning: agent => `${agent} n’est pas en cours d’exécution sur cet ordinateur pour le moment.`,
    details: 'Détails'
  },
  actions: {
    runNow: 'Exécuter maintenant',
    retry: 'Réessayer',
    change: 'Modifier',
    view: 'Voir',
    delete: 'Supprimer',
    runRoutine: 'Exécuter la routine',
    runNowHint: 'Exécuter une fois maintenant. L’horaire ne change pas.',
    viewError: 'Voir l’erreur',
    undo: 'Annuler'
  },
  sendResultsTo: 'Envoyer les résultats à',
  keptInRuns: 'Conservés dans les exécutions uniquement',
  scriptRoutine: 'Exécute un script seul, sans l’agent.',
  advanced: 'Avancé',
  model: 'Modèle',
  modelDefault: 'Modèle par défaut',
  runLimit: 'Limite d’exécutions',
  runLimitHint: 'Arrête définitivement la routine après ce nombre total d’exécutions sur toute sa durée de vie — pas les nouvelles tentatives d’une même exécution.',
  stopAfter: 'Arrêter après',
  runsUnit: 'exécutions',
  rememberLastRun: 'Se souvenir de la dernière exécution',
  rememberHint: 'Chaque exécution voit ce que la précédente a rapporté et reprend là où elle s’était arrêtée.',
  routineId: 'ID de la routine',
  runs: 'Historique des exécutions',
  noRuns: 'Aucune exécution pour le moment',
  startingRun: 'Lancement d’une exécution…',
  runChips: { running: 'En cours', failed: 'Échec', delivery: 'Non envoyée', stopped: 'Arrêtée', succeeded: 'Réussie', missed: 'Manquée' },
  deleteRoutine: 'Supprimer la routine',
  agent: 'Agent',
  name: 'Nom',
  namePlaceholder: 'Point du matin',
  instruction: 'Instruction',
  instructionPlaceholder: 'Que doit faire l’agent à chaque fois ? Par exemple : résumer mes e-mails non lus et signaler ce qui est urgent.',
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
  everyRunKept: 'Chaque exécution est aussi conservée',
  nothingSelected: 'Rien de sélectionné ? Les résultats restent enregistrés dans les exécutions.',
  agentChat: agent => `Discussion avec ${agent}`,
  needsHomeChannel: 'Définissez d’abord un canal principal',
  noChannels: 'Aucune messagerie n’est encore connectée à cet agent.',
  instructionRequired: 'Dites à l’agent quoi faire.',
  scheduleRequired: 'Choisissez quand elle s’exécute.',
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
  askInChatPrompt: 'Crée une routine qui ',
  railAttention: count => `${count} ${plural(count, 'routine à vérifier', 'routines à vérifier')}`,
  frequency: 'Fréquence',
  selectFrequency: 'Choisir une fréquence',
  lastRan: 'Dernière exécution',
  nextRunLabel: 'Prochaine exécution',
  notScheduled: 'Non planifiée',
  missedRunsHint: 'Que faire des exécutions manquées pendant que Hermes ne tournait pas.',
  aboutField: label => `À propos : ${label}`,
  runColumns: { when: 'Quand', duration: 'Durée', outcome: 'Résultat' },
  loadingMoreRuns: 'Chargement d’autres exécutions…',
  newestRuns: count => `Les ${count} exécutions les plus récentes.`,
  deleteConfirm: 'Supprimer cette routine ?',
  deleteConfirmHint: 'Son historique d’exécutions sera aussi supprimé.',
  keepIt: 'La garder',
  deleted: name => `${name} supprimée.`,
  deleteFailed: name => `Impossible de supprimer ${name}. Elle est en pause et reste dans la liste.`,
  undoFailed: name => `${name} est revenue, mais n’a pas pu reprendre. Elle est en pause.`,
  createTitle: 'Créer une routine',
  editTitle: name => `Modifier « ${name} »`,
  startFrom: 'Partir de',
  blank: 'Vide',
  blankHint: 'Créez votre propre routine',
  template: 'Modèle',
  templateHint: 'Choisir parmi les modèles',
  searchTemplates: 'Rechercher des modèles',
  noTemplates: 'Aucun modèle ne correspond.',
  templatesFailed: 'Impossible de charger les modèles.',
  templateCategories: { daily: 'Quotidien', weekly: 'Hebdomadaire', email: 'E-mail', general: 'Général' },
  fromTemplate: title => `Modèle : ${title}`,
  agentLocked: 'L’agent ne peut plus être changé une fois la routine créée.',
  optional: '(facultatif)',
  firstRun: when => `Première exécution : ${when}`,
  runsOnce: when => `Une seule exécution : ${when}`,
  createInChat: 'Créer dans une discussion',
  saveRoutine: 'Enregistrer la routine',
  agentRoutinesHint: 'Les routines se créent et se modifient sur la page Routines.',
  openRoutines: 'Ouvrir Routines'
}

const de: RoutinesCopy = {
  groups: { attention: 'Braucht Aufmerksamkeit', scheduled: 'Geplant', paused: 'Pausiert', finished: 'Beendet' },
  search: 'Routinen suchen',
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
  noMatchesDescription: 'Keine Routine passt zu dieser Suche oder diesem Filter.',
  emptyDescription: 'Lass einen Agenten zu festen Zeiten selbst etwas erledigen und dir die Ergebnisse schicken.',
  banner: {
    failed: {
      today: () => 'Der heutige Lauf ist fehlgeschlagen',
      yesterday: () => 'Der gestrige Lauf ist fehlgeschlagen',
      date: date => `Der Lauf am ${date} ist fehlgeschlagen`,
      last: 'Der letzte Lauf ist fehlgeschlagen'
    },
    notDelivered: {
      today: () => 'Der heutige Lauf wurde nicht zugestellt',
      yesterday: () => 'Der gestrige Lauf wurde nicht zugestellt',
      date: date => `Der Lauf am ${date} wurde nicht zugestellt`,
      last: 'Der letzte Lauf wurde nicht zugestellt'
    },
    failedStreak: runs => `Die letzten ${runs} Läufe sind fehlgeschlagen`,
    overdue: 'Sie ist seit dem fälligen Termin nicht gelaufen',
    overdueHint: 'Hermes läuft auf diesem Rechner möglicherweise nicht.',
    retryAt: when => `Nächster Versuch ${when}.`,
    agentNotRunning: agent => `${agent} läuft gerade nicht auf diesem Computer.`,
    details: 'Details'
  },
  actions: {
    runNow: 'Jetzt ausführen',
    retry: 'Erneut versuchen',
    change: 'Ändern',
    view: 'Ansehen',
    delete: 'Löschen',
    runRoutine: 'Routine ausführen',
    runNowHint: 'Jetzt einmal ausführen. Der Zeitplan bleibt gleich.',
    viewError: 'Fehler ansehen',
    undo: 'Rückgängig'
  },
  sendResultsTo: 'Ergebnisse senden an',
  keptInRuns: 'Nur in den Läufen gespeichert',
  scriptRoutine: 'Führt ein Skript allein aus, ohne den Agenten.',
  advanced: 'Erweitert',
  model: 'Modell',
  modelDefault: 'Standardmodell',
  runLimit: 'Laufgrenze',
  runLimitHint: 'Beendet die Routine endgültig nach dieser Gesamtzahl von Läufen über ihre gesamte Lebensdauer – nicht die Wiederholungen innerhalb eines Laufs.',
  stopAfter: 'Beenden nach',
  runsUnit: 'Läufen',
  rememberLastRun: 'Letzten Lauf merken',
  rememberHint: 'Jeder Lauf sieht, was der vorige berichtet hat, und macht dort weiter.',
  routineId: 'Routinen-ID',
  runs: 'Laufverlauf',
  noRuns: 'Noch keine Läufe',
  startingRun: 'Lauf wird gestartet…',
  runChips: { running: 'Läuft', failed: 'Fehlgeschlagen', delivery: 'Nicht zugestellt', stopped: 'Gestoppt', succeeded: 'Erfolgreich', missed: 'Verpasst' },
  deleteRoutine: 'Routine löschen',
  agent: 'Agent',
  name: 'Name',
  namePlaceholder: 'Morgenbriefing',
  instruction: 'Anweisung',
  instructionPlaceholder: 'Was soll der Agent jedes Mal tun? Zum Beispiel: meine ungelesenen E-Mails zusammenfassen und Dringendes auflisten.',
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
  everyRunKept: 'Jeder Lauf wird auch gespeichert',
  nothingSelected: 'Nichts ausgewählt? Die Ergebnisse bleiben in den Läufen gespeichert.',
  agentChat: agent => `Chat mit ${agent}`,
  needsHomeChannel: 'Lege zuerst einen Heimkanal fest',
  noChannels: 'Mit diesem Agenten ist noch kein Messaging-Kanal verbunden.',
  instructionRequired: 'Sag dem Agenten, was er tun soll.',
  scheduleRequired: 'Wähle, wann sie läuft.',
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
  askInChatPrompt: 'Richte eine Routine ein, die ',
  railAttention: count => `${count} ${plural(count, 'Routine braucht', 'Routinen brauchen')} Aufmerksamkeit`,
  frequency: 'Häufigkeit',
  selectFrequency: 'Häufigkeit wählen',
  lastRan: 'Zuletzt gelaufen',
  nextRunLabel: 'Nächster Lauf',
  notScheduled: 'Nicht geplant',
  missedRunsHint: 'Was mit Läufen geschieht, die verpasst wurden, während Hermes nicht lief.',
  aboutField: label => `Über „${label}“`,
  runColumns: { when: 'Wann', duration: 'Dauer', outcome: 'Ergebnis' },
  loadingMoreRuns: 'Weitere Läufe werden geladen…',
  newestRuns: count => `Die neuesten ${count} Läufe.`,
  deleteConfirm: 'Diese Routine löschen?',
  deleteConfirmHint: 'Ihr Laufverlauf wird ebenfalls gelöscht.',
  keepIt: 'Behalten',
  deleted: name => `${name} gelöscht.`,
  deleteFailed: name => `${name} konnte nicht gelöscht werden. Sie ist pausiert und bleibt in der Liste.`,
  undoFailed: name => `${name} ist zurück, konnte aber nicht fortgesetzt werden. Sie ist pausiert.`,
  createTitle: 'Neue Routine erstellen',
  editTitle: name => `„${name}“ bearbeiten`,
  startFrom: 'Beginnen mit',
  blank: 'Leer',
  blankHint: 'Eigene Routine erstellen',
  template: 'Vorlage',
  templateHint: 'Aus Vorlagen wählen',
  searchTemplates: 'Vorlagen durchsuchen',
  noTemplates: 'Keine Vorlage passt.',
  templatesFailed: 'Vorlagen konnten nicht geladen werden.',
  templateCategories: { daily: 'Täglich', weekly: 'Wöchentlich', email: 'E-Mail', general: 'Allgemein' },
  fromTemplate: title => `Aus Vorlage: ${title}`,
  agentLocked: 'Der Agent kann nach dem Erstellen einer Routine nicht mehr geändert werden.',
  optional: '(optional)',
  firstRun: when => `Erster Lauf: ${when}`,
  runsOnce: when => `Läuft einmal: ${when}`,
  createInChat: 'Im Chat erstellen',
  saveRoutine: 'Routine speichern',
  agentRoutinesHint: 'Routinen werden auf der Seite Routinen erstellt und geändert.',
  openRoutines: 'Routinen öffnen'
}

const es: RoutinesCopy = {
  groups: { attention: 'Requieren atención', scheduled: 'Programadas', paused: 'En pausa', finished: 'Terminadas' },
  search: 'Buscar rutinas',
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
  noMatchesDescription: 'Ninguna rutina coincide con esta búsqueda o filtro.',
  emptyDescription: 'Deja que un agente haga algo por su cuenta a las horas que elijas y te envíe los resultados.',
  banner: {
    failed: {
      today: () => 'La ejecución de hoy falló',
      yesterday: () => 'La ejecución de ayer falló',
      date: date => `La ejecución del ${date} falló`,
      last: 'La última ejecución falló'
    },
    notDelivered: {
      today: () => 'La ejecución de hoy no se entregó',
      yesterday: () => 'La ejecución de ayer no se entregó',
      date: date => `La ejecución del ${date} no se entregó`,
      last: 'La última ejecución no se entregó'
    },
    failedStreak: runs => `Las últimas ${runs} ejecuciones fallaron`,
    overdue: 'No se ha ejecutado desde la hora prevista',
    overdueHint: 'Puede que Hermes no esté en marcha en esta máquina.',
    retryAt: when => `Volverá a intentarlo ${when}.`,
    agentNotRunning: agent => `${agent} no se está ejecutando en este equipo ahora mismo.`,
    details: 'Detalles'
  },
  actions: {
    runNow: 'Ejecutar ahora',
    retry: 'Reintentar',
    change: 'Cambiar',
    view: 'Ver',
    delete: 'Eliminar',
    runRoutine: 'Ejecutar rutina',
    runNowHint: 'Ejecutar una vez ahora. Tu horario no cambia.',
    viewError: 'Ver error',
    undo: 'Deshacer'
  },
  sendResultsTo: 'Enviar resultados a',
  keptInRuns: 'Solo se guardan en Ejecuciones',
  scriptRoutine: 'Ejecuta un script por su cuenta, sin el agente.',
  advanced: 'Avanzado',
  model: 'Modelo',
  modelDefault: 'Modelo predeterminado',
  runLimit: 'Límite de ejecuciones',
  runLimitHint: 'Detiene la rutina definitivamente tras este número total de ejecuciones a lo largo de su vida, no los reintentos dentro de una misma ejecución.',
  stopAfter: 'Detener tras',
  runsUnit: 'ejecuciones',
  rememberLastRun: 'Recordar la última ejecución',
  rememberHint: 'Cada ejecución ve lo que informó la anterior y continúa donde lo dejó.',
  routineId: 'ID de la rutina',
  runs: 'Historial de ejecuciones',
  noRuns: 'Aún no hay ejecuciones',
  startingRun: 'Iniciando una ejecución…',
  runChips: { running: 'En curso', failed: 'Falló', delivery: 'No entregada', stopped: 'Detenida', succeeded: 'Correcta', missed: 'Perdida' },
  deleteRoutine: 'Eliminar rutina',
  agent: 'Agente',
  name: 'Nombre',
  namePlaceholder: 'Resumen matutino',
  instruction: 'Instrucción',
  instructionPlaceholder: '¿Qué debe hacer el agente cada vez? Por ejemplo: resumir mi correo sin leer y destacar lo urgente.',
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
  everyRunKept: 'Cada ejecución también se guarda',
  nothingSelected: '¿Nada seleccionado? Los resultados se siguen guardando en Ejecuciones.',
  agentChat: agent => `Chat con ${agent}`,
  needsHomeChannel: 'Configura primero un canal principal',
  noChannels: 'Este agente aún no tiene ningún canal de mensajería conectado.',
  instructionRequired: 'Dile al agente qué hacer.',
  scheduleRequired: 'Elige cuándo se ejecuta.',
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
  askInChatPrompt: 'Configura una rutina que ',
  railAttention: count => `${count} ${plural(count, 'rutina requiere', 'rutinas requieren')} atención`,
  frequency: 'Frecuencia',
  selectFrequency: 'Elegir frecuencia',
  lastRan: 'Última ejecución',
  nextRunLabel: 'Próxima ejecución',
  notScheduled: 'Sin programar',
  missedRunsHint: 'Qué hacer con las ejecuciones perdidas mientras Hermes no estaba en marcha.',
  aboutField: label => `Acerca de «${label}»`,
  runColumns: { when: 'Cuándo', duration: 'Duración', outcome: 'Resultado' },
  loadingMoreRuns: 'Cargando más ejecuciones…',
  newestRuns: count => `Las ${count} ejecuciones más recientes.`,
  deleteConfirm: '¿Eliminar esta rutina?',
  deleteConfirmHint: 'Su historial de ejecuciones también se eliminará.',
  keepIt: 'Conservarla',
  deleted: name => `${name} eliminada.`,
  deleteFailed: name => `No se pudo eliminar ${name}. Está en pausa y sigue en la lista.`,
  undoFailed: name => `${name} ha vuelto, pero no se pudo reanudar. Está en pausa.`,
  createTitle: 'Crear una rutina',
  editTitle: name => `Editar «${name}»`,
  startFrom: 'Empezar desde',
  blank: 'En blanco',
  blankHint: 'Crea tu propia rutina',
  template: 'Plantilla',
  templateHint: 'Elige entre plantillas',
  searchTemplates: 'Buscar plantillas',
  noTemplates: 'Ninguna plantilla coincide.',
  templatesFailed: 'No se pudieron cargar las plantillas.',
  templateCategories: { daily: 'Diarias', weekly: 'Semanales', email: 'Correo', general: 'Generales' },
  fromTemplate: title => `Desde la plantilla: ${title}`,
  agentLocked: 'El agente no se puede cambiar una vez creada la rutina.',
  optional: '(opcional)',
  firstRun: when => `Primera ejecución: ${when}`,
  runsOnce: when => `Se ejecuta una vez: ${when}`,
  createInChat: 'Crear en un chat',
  saveRoutine: 'Guardar rutina',
  agentRoutinesHint: 'Las rutinas se crean y se cambian en la página Rutinas.',
  openRoutines: 'Abrir Rutinas'
}

const ja: RoutinesCopy = {
  groups: { attention: '要確認', scheduled: 'スケジュール済み', paused: '一時停止中', finished: '終了' },
  search: 'ルーティンを検索',
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
  noMatchesDescription: 'この検索・フィルターに一致するルーティンはありません。',
  emptyDescription: '決めた時刻にエージェントが自動で作業し、結果を届けます。',
  banner: {
    failed: {
      today: () => '今日の実行が失敗しました',
      yesterday: () => '昨日の実行が失敗しました',
      date: date => `${date} の実行が失敗しました`,
      last: '前回の実行が失敗しました'
    },
    notDelivered: {
      today: () => '今日の実行結果が配信されませんでした',
      yesterday: () => '昨日の実行結果が配信されませんでした',
      date: date => `${date} の実行結果が配信されませんでした`,
      last: '前回の実行結果が配信されませんでした'
    },
    failedStreak: runs => `直近 ${runs} 回連続で失敗しました`,
    overdue: '予定時刻を過ぎても実行されていません',
    overdueHint: 'このマシンで Hermes が動いていない可能性があります。',
    retryAt: when => `次は ${when} に再試行します。`,
    agentNotRunning: agent => `${agent} は現在このコンピューターで実行されていません。`,
    details: '詳細'
  },
  actions: {
    runNow: '今すぐ実行',
    retry: '再試行',
    change: '変更',
    view: '表示',
    delete: '削除',
    runRoutine: 'ルーティンを実行',
    runNowHint: '今すぐ1回実行します。スケジュールは変わりません。',
    viewError: 'エラーを表示',
    undo: '元に戻す'
  },
  sendResultsTo: '結果の送信先',
  keptInRuns: '実行履歴にのみ保存',
  scriptRoutine: 'エージェントを使わずにスクリプトだけを実行します。',
  advanced: '詳細',
  model: 'モデル',
  modelDefault: 'デフォルトのモデル',
  runLimit: '実行回数の上限',
  runLimitHint: 'ルーティンの寿命を通じた総実行回数に達すると完全に停止します。1回の実行内の再試行の上限ではありません。',
  stopAfter: '上限',
  runsUnit: '回で停止',
  rememberLastRun: '前回の実行を覚える',
  rememberHint: '毎回、前回の報告内容を踏まえて続きから進めます。',
  routineId: 'ルーティン ID',
  runs: '実行履歴',
  noRuns: 'まだ実行されていません',
  startingRun: '実行を開始しています…',
  runChips: { running: '実行中', failed: '失敗', delivery: '未配信', stopped: '停止', succeeded: '成功', missed: '未実行' },
  deleteRoutine: 'ルーティンを削除',
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
  everyRunKept: '各実行は履歴にも保存されます',
  nothingSelected: '何も選ばなくても、結果は実行履歴に保存されます。',
  agentChat: agent => `${agent} とのチャット`,
  needsHomeChannel: '先にホームチャンネルを設定してください',
  noChannels: 'このエージェントにはまだメッセージチャンネルが接続されていません。',
  instructionRequired: 'エージェントにしてほしいことを書いてください。',
  scheduleRequired: '実行タイミングを選んでください。',
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
  askInChatPrompt: '次のことをするルーティンを作って：',
  railAttention: count => `要確認のルーティンが ${count} 件あります`,
  frequency: '頻度',
  selectFrequency: '頻度を選択',
  lastRan: '前回の実行',
  nextRunLabel: '次回の実行',
  notScheduled: '予定なし',
  missedRunsHint: 'Hermes が起動していなかった間に実行できなかった分の扱いです。',
  aboutField: label => `${label} について`,
  runColumns: { when: '日時', duration: '所要時間', outcome: '結果' },
  loadingMoreRuns: 'さらに読み込んでいます…',
  newestRuns: count => `最新の ${count} 件を表示しています。`,
  deleteConfirm: 'このルーティンを削除しますか？',
  deleteConfirmHint: '実行履歴も削除されます。',
  keepIt: '残す',
  deleted: name => `${name} を削除しました。`,
  deleteFailed: name => `${name} を削除できませんでした。一時停止したままリストに残っています。`,
  undoFailed: name => `${name} を元に戻しましたが、再開できませんでした。一時停止中です。`,
  createTitle: '新しいルーティンを作成',
  editTitle: name => `「${name}」を編集`,
  startFrom: '作成方法',
  blank: '白紙',
  blankHint: '自分でルーティンを作成',
  template: 'テンプレート',
  templateHint: 'テンプレートから選ぶ',
  searchTemplates: 'テンプレートを検索',
  noTemplates: '一致するテンプレートはありません。',
  templatesFailed: 'テンプレートを読み込めませんでした。',
  templateCategories: { daily: '毎日', weekly: '毎週', email: 'メール', general: '一般' },
  fromTemplate: title => `テンプレート: ${title}`,
  agentLocked: 'ルーティンの作成後はエージェントを変更できません。',
  optional: '（任意）',
  firstRun: when => `初回の実行: ${when}`,
  runsOnce: when => `1回だけ実行: ${when}`,
  createInChat: 'チャットで作成',
  saveRoutine: 'ルーティンを保存',
  agentRoutinesHint: 'ルーティンの作成と変更はルーティンページで行います。',
  openRoutines: 'ルーティンを開く'
}

const zh: RoutinesCopy = {
  groups: { attention: '需要处理', scheduled: '已计划', paused: '已暂停', finished: '已结束' },
  search: '搜索例行任务',
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
  noMatchesDescription: '没有例行任务符合当前搜索或筛选条件。',
  emptyDescription: '让智能体在你选定的时间自动完成工作，并把结果发给你。',
  banner: {
    failed: {
      today: () => '今天的运行失败了',
      yesterday: () => '昨天的运行失败了',
      date: date => `${date} 的运行失败了`,
      last: '上次运行失败了'
    },
    notDelivered: {
      today: () => '今天的运行结果未送达',
      yesterday: () => '昨天的运行结果未送达',
      date: date => `${date} 的运行结果未送达`,
      last: '上次运行的结果未送达'
    },
    failedStreak: runs => `最近 ${runs} 次运行都失败了`,
    overdue: '到点后一直没有运行',
    overdueHint: '这台机器上的 Hermes 可能没有在运行。',
    retryAt: when => `将于${when}重试。`,
    agentNotRunning: agent => `${agent} 目前未在这台电脑上运行。`,
    details: '详细信息'
  },
  actions: {
    runNow: '立即运行',
    retry: '重试',
    change: '更改',
    view: '查看',
    delete: '删除',
    runRoutine: '运行例行任务',
    runNowHint: '立即运行一次。时间安排保持不变。',
    viewError: '查看错误',
    undo: '撤销'
  },
  sendResultsTo: '结果发送至',
  keptInRuns: '仅保存在运行记录中',
  scriptRoutine: '单独运行一个脚本，不调用智能体。',
  advanced: '高级',
  model: '模型',
  modelDefault: '默认模型',
  runLimit: '运行次数上限',
  runLimitHint: '达到这个总运行次数后，例行任务将永久停止；它指的是整个生命周期内的运行总数，而不是单次运行内的重试。',
  stopAfter: '运行',
  runsUnit: '次后停止',
  rememberLastRun: '记住上次运行',
  rememberHint: '每次运行都能看到上一次的报告，从上次停下的地方继续。',
  routineId: '例行任务 ID',
  runs: '运行记录',
  noRuns: '还没有运行记录',
  startingRun: '正在开始运行…',
  runChips: { running: '运行中', failed: '失败', delivery: '未送达', stopped: '已停止', succeeded: '成功', missed: '已错过' },
  deleteRoutine: '删除例行任务',
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
  everyRunKept: '每次运行也都会保存在运行记录中',
  nothingSelected: '什么都没选？结果仍会保存在运行记录中。',
  agentChat: agent => `与 ${agent} 的对话`,
  needsHomeChannel: '请先设置主频道',
  noChannels: '这个智能体还没有连接任何消息渠道。',
  instructionRequired: '请告诉智能体要做什么。',
  scheduleRequired: '请选择运行时间。',
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
  askInChatPrompt: '帮我设置一个例行任务：',
  railAttention: count => `${count} 个例行任务需要处理`,
  frequency: '频率',
  selectFrequency: '选择频率',
  lastRan: '上次运行',
  nextRunLabel: '下次运行',
  notScheduled: '未安排',
  missedRunsHint: 'Hermes 未运行期间错过的运行如何处理。',
  aboutField: label => `关于“${label}”`,
  runColumns: { when: '时间', duration: '耗时', outcome: '结果' },
  loadingMoreRuns: '正在加载更多运行记录…',
  newestRuns: count => `显示最近 ${count} 次运行。`,
  deleteConfirm: '删除此例行任务？',
  deleteConfirmHint: '它的运行记录也会被删除。',
  keepIt: '保留',
  deleted: name => `已删除 ${name}。`,
  deleteFailed: name => `无法删除 ${name}。它已暂停，仍在列表中。`,
  undoFailed: name => `${name} 已恢复，但无法继续运行。它处于暂停状态。`,
  createTitle: '新建例行任务',
  editTitle: name => `编辑“${name}”`,
  startFrom: '起点',
  blank: '空白',
  blankHint: '自己创建例行任务',
  template: '模板',
  templateHint: '从模板中选择',
  searchTemplates: '搜索模板',
  noTemplates: '没有匹配的模板。',
  templatesFailed: '无法加载模板。',
  templateCategories: { daily: '每天', weekly: '每周', email: '邮件', general: '通用' },
  fromTemplate: title => `来自模板：${title}`,
  agentLocked: '例行任务创建后无法更改智能体。',
  optional: '（可选）',
  firstRun: when => `首次运行：${when}`,
  runsOnce: when => `仅运行一次：${when}`,
  createInChat: '在对话中创建',
  saveRoutine: '保存例行任务',
  agentRoutinesHint: '例行任务在“例行任务”页面创建和修改。',
  openRoutines: '打开例行任务'
}

const zhHant: RoutinesCopy = {
  groups: { attention: '需要處理', scheduled: '已排程', paused: '已暫停', finished: '已結束' },
  search: '搜尋例行任務',
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
  noMatchesDescription: '沒有例行任務符合目前的搜尋或篩選條件。',
  emptyDescription: '讓智慧體在你選定的時間自動完成工作，並把結果傳給你。',
  banner: {
    failed: {
      today: () => '今天的執行失敗了',
      yesterday: () => '昨天的執行失敗了',
      date: date => `${date} 的執行失敗了`,
      last: '上次執行失敗了'
    },
    notDelivered: {
      today: () => '今天的執行結果未送達',
      yesterday: () => '昨天的執行結果未送達',
      date: date => `${date} 的執行結果未送達`,
      last: '上次執行的結果未送達'
    },
    failedStreak: runs => `最近 ${runs} 次執行都失敗了`,
    overdue: '過了預定時間仍未執行',
    overdueHint: '這台機器上的 Hermes 可能沒有在執行。',
    retryAt: when => `將於${when}重試。`,
    agentNotRunning: agent => `${agent} 目前未在這台電腦上執行。`,
    details: '詳細資訊'
  },
  actions: {
    runNow: '立即執行',
    retry: '重試',
    change: '變更',
    view: '查看',
    delete: '刪除',
    runRoutine: '執行例行任務',
    runNowHint: '立即執行一次。時間安排維持不變。',
    viewError: '查看錯誤',
    undo: '復原'
  },
  sendResultsTo: '結果傳送至',
  keptInRuns: '僅保存在執行紀錄中',
  scriptRoutine: '單獨執行一個腳本，不呼叫智慧體。',
  advanced: '進階',
  model: '模型',
  modelDefault: '預設模型',
  runLimit: '執行次數上限',
  runLimitHint: '達到這個總執行次數後，例行任務將永久停止；它指的是整個生命週期內的執行總數，而不是單次執行內的重試。',
  stopAfter: '執行',
  runsUnit: '次後停止',
  rememberLastRun: '記住上次執行',
  rememberHint: '每次執行都能看到上一次的回報，從上次停下的地方繼續。',
  routineId: '例行任務 ID',
  runs: '執行紀錄',
  noRuns: '還沒有執行紀錄',
  startingRun: '正在開始執行…',
  runChips: { running: '執行中', failed: '失敗', delivery: '未送達', stopped: '已停止', succeeded: '成功', missed: '已錯過' },
  deleteRoutine: '刪除例行任務',
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
  everyRunKept: '每次執行也都會保存在執行紀錄中',
  nothingSelected: '什麼都沒選？結果仍會保存在執行紀錄中。',
  agentChat: agent => `與 ${agent} 的對話`,
  needsHomeChannel: '請先設定主頻道',
  noChannels: '這個智慧體還沒有連接任何訊息管道。',
  instructionRequired: '請告訴智慧體要做什麼。',
  scheduleRequired: '請選擇執行時間。',
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
  askInChatPrompt: '幫我設定一個例行任務：',
  railAttention: count => `${count} 個例行任務需要處理`,
  frequency: '頻率',
  selectFrequency: '選擇頻率',
  lastRan: '上次執行',
  nextRunLabel: '下次執行',
  notScheduled: '未安排',
  missedRunsHint: 'Hermes 未執行期間錯過的執行如何處理。',
  aboutField: label => `關於「${label}」`,
  runColumns: { when: '時間', duration: '耗時', outcome: '結果' },
  loadingMoreRuns: '正在載入更多執行紀錄…',
  newestRuns: count => `顯示最近 ${count} 次執行。`,
  deleteConfirm: '刪除這個例行任務？',
  deleteConfirmHint: '它的執行紀錄也會一併刪除。',
  keepIt: '保留',
  deleted: name => `已刪除 ${name}。`,
  deleteFailed: name => `無法刪除 ${name}。它已暫停，仍在清單中。`,
  undoFailed: name => `${name} 已復原，但無法繼續執行。它處於暫停狀態。`,
  createTitle: '建立例行任務',
  editTitle: name => `編輯「${name}」`,
  startFrom: '起點',
  blank: '空白',
  blankHint: '自己建立例行任務',
  template: '範本',
  templateHint: '從範本中選擇',
  searchTemplates: '搜尋範本',
  noTemplates: '沒有符合的範本。',
  templatesFailed: '無法載入範本。',
  templateCategories: { daily: '每天', weekly: '每週', email: '郵件', general: '一般' },
  fromTemplate: title => `來自範本：${title}`,
  agentLocked: '例行任務建立後無法更改智慧體。',
  optional: '（選填）',
  firstRun: when => `首次執行：${when}`,
  runsOnce: when => `只執行一次：${when}`,
  createInChat: '在對話中建立',
  saveRoutine: '儲存例行任務',
  agentRoutinesHint: '例行任務在「例行任務」頁面建立和修改。',
  openRoutines: '開啟例行任務'
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
  groups: {
    attention: 'Требуют внимания',
    scheduled: 'По расписанию',
    paused: 'Приостановлены',
    finished: 'Завершены'
  },
  search: 'Поиск регулярных задач',
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
  noMatchesDescription: 'Ни одна регулярная задача не подходит под этот поиск или фильтр.',
  emptyDescription: 'Пусть агент сам выполняет работу в выбранное время и присылает вам результаты.',
  banner: {
    failed: {
      today: () => 'Сегодняшний запуск завершился ошибкой',
      yesterday: () => 'Вчерашний запуск завершился ошибкой',
      date: date => `Запуск ${date} завершился ошибкой`,
      last: 'Последний запуск завершился ошибкой'
    },
    notDelivered: {
      today: () => 'Результат сегодняшнего запуска не доставлен',
      yesterday: () => 'Результат вчерашнего запуска не доставлен',
      date: date => `Результат запуска ${date} не доставлен`,
      last: 'Результат последнего запуска не доставлен'
    },
    failedStreak: runs => `Последние запуски (${runs}) завершились ошибкой`,
    overdue: 'Задача не запустилась в назначенное время',
    overdueHint: 'Возможно, Hermes не запущен на этом компьютере.',
    retryAt: when => `Следующая попытка — ${when}.`,
    agentNotRunning: agent => `${agent} сейчас не запущен на этом компьютере.`,
    details: 'Подробнее'
  },
  actions: {
    runNow: 'Запустить сейчас',
    retry: 'Повторить',
    change: 'Изменить',
    view: 'Открыть',
    delete: 'Удалить',
    runRoutine: 'Запустить задачу',
    runNowHint: 'Запустить один раз сейчас. Расписание не изменится.',
    viewError: 'Открыть ошибку',
    undo: 'Отменить'
  },
  sendResultsTo: 'Куда отправлять результаты',
  keptInRuns: 'Только в истории запусков',
  scriptRoutine: 'Выполняет скрипт сам по себе, без агента.',
  advanced: 'Дополнительно',
  model: 'Модель',
  modelDefault: 'Модель по умолчанию',
  runLimit: 'Лимит запусков',
  runLimitHint: 'Полностью останавливает регулярную задачу после такого общего числа запусков за всё время её работы — это не ограничение повторов внутри одного запуска.',
  stopAfter: 'Остановить после',
  runsUnit: 'запусков',
  rememberLastRun: 'Помнить прошлый запуск',
  rememberHint: 'Каждый запуск видит отчёт предыдущего и продолжает с того же места.',
  routineId: 'ID регулярной задачи',
  runs: 'История запусков',
  noRuns: 'Запусков пока не было',
  startingRun: 'Запуск начинается…',
  runChips: { running: 'Выполняется', failed: 'Ошибка', delivery: 'Не доставлено', stopped: 'Остановлен', succeeded: 'Успешно', missed: 'Пропущен' },
  deleteRoutine: 'Удалить регулярную задачу',
  agent: 'Агент',
  name: 'Название',
  namePlaceholder: 'Утренняя сводка',
  instruction: 'Инструкция',
  instructionPlaceholder: 'Что агенту делать каждый раз? Например: кратко пересказать непрочитанную почту и перечислить срочное.',
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
  everyRunKept: 'Каждый запуск сохраняется в истории',
  nothingSelected: 'Ничего не выбрано? Результаты всё равно сохранятся в истории запусков.',
  agentChat: agent => `Чат с ${agent}`,
  needsHomeChannel: 'Сначала задайте основной канал',
  noChannels: 'К этому агенту пока не подключён ни один мессенджер.',
  instructionRequired: 'Напишите, что должен сделать агент.',
  scheduleRequired: 'Выберите, когда запускать.',
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
  askInChatPrompt: 'Настрой регулярную задачу, которая ',
  railAttention: count =>
    `${count} ${RU_PLURAL(count, 'регулярная задача требует', 'регулярные задачи требуют', 'регулярных задач требуют')} внимания`,
  frequency: 'Частота',
  selectFrequency: 'Выберите частоту',
  lastRan: 'Последний запуск',
  nextRunLabel: 'Следующий запуск',
  notScheduled: 'Не запланирован',
  missedRunsHint: 'Что делать с запусками, пропущенными, пока Hermes не работал.',
  aboutField: label => `Подробнее: ${label}`,
  runColumns: { when: 'Когда', duration: 'Длительность', outcome: 'Результат' },
  loadingMoreRuns: 'Загрузка других запусков…',
  newestRuns: count => `Показаны последние запуски: ${count}.`,
  deleteConfirm: 'Удалить эту регулярную задачу?',
  deleteConfirmHint: 'Её история запусков тоже будет удалена.',
  keepIt: 'Оставить',
  deleted: name => `${name}: удалено.`,
  deleteFailed: name => `Не удалось удалить ${name}. Задача приостановлена и осталась в списке.`,
  undoFailed: name => `${name}: задача восстановлена, но не возобновлена. Она приостановлена.`,
  createTitle: 'Новая регулярная задача',
  editTitle: name => `Изменить «${name}»`,
  startFrom: 'Начать с',
  blank: 'С нуля',
  blankHint: 'Создайте свою задачу',
  template: 'Шаблон',
  templateHint: 'Выберите из шаблонов',
  searchTemplates: 'Поиск шаблонов',
  noTemplates: 'Подходящих шаблонов нет.',
  templatesFailed: 'Не удалось загрузить шаблоны.',
  templateCategories: { daily: 'Ежедневные', weekly: 'Еженедельные', email: 'Почта', general: 'Общие' },
  fromTemplate: title => `По шаблону: ${title}`,
  agentLocked: 'После создания задачи агента изменить нельзя.',
  optional: '(необязательно)',
  firstRun: when => `Первый запуск: ${when}`,
  runsOnce: when => `Один запуск: ${when}`,
  createInChat: 'Создать в чате',
  saveRoutine: 'Сохранить задачу',
  agentRoutinesHint: 'Регулярные задачи создаются и меняются на странице «Регулярные задачи».',
  openRoutines: 'Открыть регулярные задачи'
}

const ar: RoutinesCopy = {
  groups: { attention: 'تحتاج إلى انتباه', scheduled: 'مجدولة', paused: 'متوقفة مؤقتًا', finished: 'منتهية' },
  search: 'البحث في المهام الروتينية',
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
  noMatchesDescription: 'لا توجد مهمة روتينية تطابق هذا البحث أو عامل التصفية.',
  emptyDescription: 'دع وكيلًا ينجز عملًا بنفسه في الأوقات التي تختارها ويرسل إليك النتائج.',
  banner: {
    failed: {
      today: () => 'فشل تشغيل اليوم',
      yesterday: () => 'فشل تشغيل الأمس',
      date: date => `فشل تشغيل ${date}`,
      last: 'فشل آخر تشغيل'
    },
    notDelivered: {
      today: () => 'لم تُرسل نتائج تشغيل اليوم',
      yesterday: () => 'لم تُرسل نتائج تشغيل الأمس',
      date: date => `لم تُرسل نتائج تشغيل ${date}`,
      last: 'لم تُرسل نتائج آخر تشغيل'
    },
    failedStreak: runs => `فشلت آخر ${runs} مرات تشغيل`,
    overdue: 'لم تُشغَّل منذ موعدها',
    overdueHint: 'ربما لا يعمل Hermes على هذا الجهاز.',
    retryAt: when => `ستُعاد المحاولة ${when}.`,
    agentNotRunning: agent => `${agent} لا يعمل على هذا الجهاز الآن.`,
    details: 'التفاصيل'
  },
  actions: {
    runNow: 'تشغيل الآن',
    retry: 'إعادة المحاولة',
    change: 'تغيير',
    view: 'عرض',
    delete: 'حذف',
    runRoutine: 'تشغيل المهمة الروتينية',
    runNowHint: 'تشغيل مرة واحدة الآن. يبقى جدولك كما هو.',
    viewError: 'عرض الخطأ',
    undo: 'تراجع'
  },
  sendResultsTo: 'إرسال النتائج إلى',
  keptInRuns: 'تُحفظ في سجل التشغيل فقط',
  scriptRoutine: 'تشغّل نصًا برمجيًا وحده، دون الوكيل.',
  advanced: 'متقدم',
  model: 'النموذج',
  modelDefault: 'النموذج الافتراضي',
  runLimit: 'حد التشغيل',
  runLimitHint: 'يوقف المهمة الروتينية نهائيًا بعد هذا العدد الإجمالي من مرات التشغيل طوال عمرها، وليس حدًّا لإعادة المحاولة داخل التشغيل الواحد.',
  stopAfter: 'الإيقاف بعد',
  runsUnit: 'مرات تشغيل',
  rememberLastRun: 'تذكّر آخر تشغيل',
  rememberHint: 'يرى كل تشغيل ما أبلغ عنه التشغيل السابق ويكمل من حيث توقف.',
  routineId: 'معرّف المهمة الروتينية',
  runs: 'سجل التشغيل',
  noRuns: 'لا توجد مرات تشغيل بعد',
  startingRun: 'جارٍ بدء التشغيل…',
  runChips: { running: 'قيد التشغيل', failed: 'فشل', delivery: 'لم يُرسل', stopped: 'أُوقف', succeeded: 'نجح', missed: 'فات' },
  deleteRoutine: 'حذف المهمة الروتينية',
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
  everyRunKept: 'يُحفظ كل تشغيل في السجل أيضًا',
  nothingSelected: 'لم تختر شيئًا؟ تبقى النتائج محفوظة في سجل التشغيل.',
  agentChat: agent => `المحادثة مع ${agent}`,
  needsHomeChannel: 'عيّن قناة رئيسية أولًا',
  noChannels: 'لا توجد قناة مراسلة متصلة بهذا الوكيل بعد.',
  instructionRequired: 'أخبر الوكيل بما يفعله.',
  scheduleRequired: 'اختر وقت التشغيل.',
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
  askInChatPrompt: 'أعدّ لي مهمة روتينية ',
  railAttention: count => `مهام روتينية تحتاج إلى انتباه: ${count}`,
  frequency: 'التكرار',
  selectFrequency: 'اختر التكرار',
  lastRan: 'آخر تشغيل',
  nextRunLabel: 'التشغيل التالي',
  notScheduled: 'غير مجدول',
  missedRunsHint: 'ما يحدث لمرات التشغيل الفائتة أثناء توقف Hermes.',
  aboutField: label => `حول ${label}`,
  runColumns: { when: 'الوقت', duration: 'المدة', outcome: 'النتيجة' },
  loadingMoreRuns: 'جارٍ تحميل المزيد…',
  newestRuns: count => `يُعرض أحدث ${count} مرات تشغيل.`,
  deleteConfirm: 'حذف هذه المهمة الروتينية؟',
  deleteConfirmHint: 'سيُحذف سجل تشغيلها أيضًا.',
  keepIt: 'الإبقاء عليها',
  deleted: name => `حُذفت ${name}.`,
  deleteFailed: name => `تعذّر حذف ${name}. إنها متوقفة مؤقتًا وما زالت في القائمة.`,
  undoFailed: name => `عادت ${name}، لكن تعذّر استئنافها. إنها متوقفة مؤقتًا.`,
  createTitle: 'إنشاء مهمة روتينية',
  editTitle: name => `تعديل «${name}»`,
  startFrom: 'البدء من',
  blank: 'فارغة',
  blankHint: 'أنشئ مهمتك الروتينية',
  template: 'قالب',
  templateHint: 'اختر من القوالب',
  searchTemplates: 'البحث في القوالب',
  noTemplates: 'لا يوجد قالب مطابق.',
  templatesFailed: 'تعذّر تحميل القوالب.',
  templateCategories: { daily: 'يومية', weekly: 'أسبوعية', email: 'البريد', general: 'عامة' },
  fromTemplate: title => `من القالب: ${title}`,
  agentLocked: 'لا يمكن تغيير الوكيل بعد إنشاء المهمة الروتينية.',
  optional: '(اختياري)',
  firstRun: when => `أول تشغيل: ${when}`,
  runsOnce: when => `تشغيل مرة واحدة: ${when}`,
  createInChat: 'الإنشاء في محادثة',
  saveRoutine: 'حفظ المهمة الروتينية',
  agentRoutinesHint: 'تُنشأ المهام الروتينية وتُعدَّل في صفحة المهام الروتينية.',
  openRoutines: 'فتح المهام الروتينية'
}

const COPY: Record<BundledLocale, RoutinesCopy> = { en, fr, de, es, ja, zh, 'zh-hant': zhHant, ru, ar }

export function routinesCopy(locale: string): RoutinesCopy {
  return COPY[isBundledLocale(locale) ? locale : 'en']
}

export function useRoutinesCopy(): RoutinesCopy {
  return routinesCopy(useI18n().locale)
}
