import type { TranslationOverride } from '@hermes/shared/i18n'

import type { Translations } from './types'

// The in-app browser pane's copy (`preview.web`), composed by de.ts.
export const dePreviewWeb: TranslationOverride<Translations['preview']['web']> = {
  embeddedPreviewHint:
    'Manche Websites blockieren eingebettete Vorschauen. Öffne die Originalseite in einem Browser-Tab.',
  appFailedToBoot: 'Die Vorschau-App konnte nicht gestartet werden',
  serverNotFound: 'Server nicht gefunden',
  remoteLoopback:
    'Diese Adresse verweist auf den Rechner, auf dem Ihr Agent läuft – nicht auf diesen. Das Browserfenster lädt Seiten lokal, daher braucht ein entfernter Entwicklungsserver eine Portweiterleitung oder einen erreichbaren Hostnamen.',
  failedToLoad: 'Vorschau konnte nicht geladen werden',
  tryAgain: 'Nochmal versuchen',
  restarting: 'Hermes wird neu gestartet …',
  askRestart: 'Hermes bitten, den Server neu zu starten',
  lookingRestart: taskId => `Hermes sucht nach einem Vorschau-Server zum Neustarten (${taskId})`,
  restartingTitle: 'Vorschau-Server wird neu gestartet',
  restartingMessage: 'Hermes arbeitet im Hintergrund. Beobachte im Fortschritt die Vorschau-Konsole.',
  startRestartFailed: message => `Server-Neustart konnte nicht gestartet werden: ${message}`,
  restartFailed: 'Server-Neustart fehlgeschlagen',
  hideConsole: 'Vorschau-Konsole ausblenden',
  showConsole: 'Vorschau-Konsole anzeigen',
  hideDevTools: 'Vorschau-DevTools ausblenden',
  openDevTools: 'Vorschau-DevTools öffnen',
  goBack: 'Zurück',
  goForward: 'Vor',
  reload: 'Seite neu laden',
  address: 'Adresse',
  addressPlaceholder: 'Adresse eingeben',
  blankPageBody: 'Geben Sie oben eine Adresse ein, um zu browsen, oder bitten Sie Hermes, eine Seite zu öffnen.',
  finishedRestarting: message => `Hermes hat den Vorschau-Server neu gestartet${message ? `: ${message}` : ''}`,
  failedRestarting: message => `Server-Neustart fehlgeschlagen: ${message}`,
  unknownError: 'unbekannter Fehler',
  restartedTitle: 'Vorschau-Server neu gestartet',
  reloadingNow: 'Die Vorschau wird jetzt neu geladen.',
  restartFailedTitle: 'Vorschau-Neustart fehlgeschlagen',
  restartFailedMessage: 'Hermes konnte den Server nicht neu starten.',
  stillWorking:
    'Hermes arbeitet noch, aber es ist noch kein Ergebnis des Neustarts eingetroffen. Der Server-Befehl läuft möglicherweise im Vordergrund.',
  workspaceReloading: 'Arbeitsbereich geändert, Vorschau wird neu geladen',
  fileChanged: url => `Datei geändert, Vorschau wird neu geladen: ${url}`,
  filesChanged: (count, url) => `${count} Dateiänderungen, Vorschau wird neu geladen: ${url}`,
  watchFailed: message => `Vorschau-Datei konnte nicht überwacht werden: ${message}`,
  moduleMimeDescription:
    'Modul-Skripte werden mit dem falschen MIME-Typ ausgeliefert. Das bedeutet meist, dass ein statischer Datei-Server eine Vite/React-App ausliefert statt des Projekt-Entwicklungs-Servers.',
  loadFailedConsole: (code, message) => `Laden fehlgeschlagen${code ? ` (${code})` : ''}: ${message}`,
  unreachableDescription: 'Die Vorschau-Seite konnte nicht erreicht werden.',
  openTarget: url => `${url} öffnen`,
  fallbackTitle: 'Vorschau',
  annotate: 'Annotieren',
  annotateOn: 'Annotation beenden',
  annotateNeedPage: 'Öffnen Sie zuerst eine Seite im In-App-Browser.',
  annotateFailed: 'Annotationsmodus konnte nicht gestartet werden',
  commenting: 'Kommentieren',
  addComments: count => (count === 1 ? '1 Kommentar hinzufügen' : `${count} Kommentare hinzufügen`),
  commentPlaceholder: 'Kommentar hinzufügen …',
  commentTitle: n => `Kommentar ${n}`,
  saveComment: 'Speichern',
  cancelComment: 'Kommentar abbrechen'
}
