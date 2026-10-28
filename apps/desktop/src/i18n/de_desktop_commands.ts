// Desktop-side /rollback /agents /config /history /save /export /import /timestamps lines
// (en_desktop_commands.ts is the source).
export const deDesktopCommands = {
  error: (message: string) => `Fehler: ${message}`,
  rollbackUsage: 'Verwendung: /rollback [list] | diff <n|hash> | <n|hash> [file] [--all]',
  checkpointsDisabled:
    'Checkpoints sind für diese Sitzung nicht aktiviert (checkpoints.enabled in config.yaml setzen).',
  checkpointsEmpty: 'Für dieses Arbeitsverzeichnis gibt es noch keine Checkpoints.',
  checkpointsHeader: 'Checkpoints (neueste zuerst):',
  checkpointsHint:
    'Wiederherstellen mit /rollback <n> [file] (mit --all werden auch eigene Änderungen überschrieben); Vorschau mit /rollback diff <n>.',
  noDiff: 'Seit diesem Checkpoint gibt es keine Änderungen.',
  rollbackFailed: (error: string) => `Rollback fehlgeschlagen: ${error}`,
  unknownError: 'unbekannter Fehler',
  restored: (hash: string) => (hash ? `Checkpoint ${hash} wiederhergestellt.` : 'Checkpoint wiederhergestellt.'),
  rewound: (count: string) => `${count} Nachricht(en) aus dieser Unterhaltung zurückgenommen.`,
  reopenForRewind: 'Öffne diesen Chat erneut, um die zurückgesetzte Unterhaltung zu sehen.',
  agentsEmpty: 'Keine aktiven Subagents oder Hintergrundprozesse in diesem Chat.',
  subagentsHeader: (count: string) => `Subagents (${count}):`,
  subagentFallback: 'Subagent',
  statusRunning: 'läuft',
  toolCalls: (count: string) => `${count} Tool-Aufrufe`,
  processesHeader: (count: string) => `Hintergrundprozesse (${count}):`,
  configHeader: 'Konfiguration (Anzeige, Freigaben, Einfügen und Sprache; alles andere wird nicht angezeigt):',
  configDefaults: '(Standardwerte)',
  configFooter:
    'Ändere Einstellungen unter Einstellungen oder bearbeite config.yaml mit `hermes config edit` auf dem Gateway-Host.',
  historyHeader: (count: string) => `Gesprächsverlauf (${count} Nachrichten):`,
  historyEmpty: 'Noch keine Nachrichten.',
  historyYou: 'Du',
  historyHermes: 'Hermes',
  saveUsage: 'Verwendung: /save [json|md|html] [filename] [redact]',
  saveLegacyRefused:
    'Nicht gespeichert: Dieses Gateway kann keine geschwärzte, benannte oder Nicht-JSON-Kopie exportieren. Aktualisiere das Gateway oder führe /save ohne Optionen aus, um rohes JSON auf dem Gateway-Host zu schreiben.',
  savedOnHost: (path: string) => `Auf dem Gateway-Host gespeichert: ${path}`,
  unknownPath: '(unbekannter Pfad)',
  saveUnsupported: 'Fehler: Diese Desktop-Version kann keine Dateien speichern',
  saveDialogTitle: 'Unterhaltung speichern',
  savedTo: (path: string) => `Unterhaltung gespeichert unter ${path}`,
  saveCancelled: 'Nicht gespeichert (abgebrochen).',
  exportRemote:
    '/export packt ein Profil auf dem Gateway-Host, und das ist nicht dieser Rechner. Führe `hermes profile export` auf diesem Host aus.',
  importRemote:
    '/import installiert ein Profil auf dem Gateway-Host, und das ist nicht dieser Rechner. Führe `hermes profile import <archive>` auf diesem Host aus.',
  timestampsUsage: 'Verwendung: /timestamps [on|off|status]',
  timestampsOn: 'Zeitstempel: an (dieses Fenster; display.timestamps in config.yaml setzen, um es zu behalten)',
  timestampsOff: 'Zeitstempel: aus (dieses Fenster; display.timestamps in config.yaml setzen, um es zu behalten)'
}
