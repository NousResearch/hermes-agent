import type { TranslationOverrides } from './define-locale'

export const deFileMenu = {
    revealFinder: 'Im Finder anzeigen',
    revealExplorer: 'Im Datei-Explorer anzeigen',
    newFile: 'Neue Datei…',
    newFolder: 'Neuer Ordner…',
    createFailed: 'Erstellen fehlgeschlagen',
    deleteRemoteBody: 'Die Datei wird dauerhaft auf dem entfernten Computer gelöscht — dieser Vorgang kann nicht rückgängig gemacht werden.',
    revealFileManager: 'Enthaltenden Ordner öffnen',
    revealInSidebar: 'In Dateibaum anzeigen',
    copyPath: 'Pfad kopieren',
    copyRelativePath: 'Relativen Pfad kopieren',
    download: 'Herunterladen',
    downloadSaved: 'Gespeichert',
    downloadFailed: 'Download fehlgeschlagen',
    rename: 'Umbenennen…',
    delete: 'Löschen',
    renameTitle: 'Umbenennen',
    renameLabel: 'Neuer Name',
    deleteTitle: name => `„${name}“ löschen?`,
    deleteBody: 'Es wird in den Papierkorb verschoben – von dort können Sie es wiederherstellen.',
    pathCopied: 'Pfad kopiert',
    revealMissing: 'Dieser Ordner befindet sich nicht auf diesem Computer',
    revealUnavailable:
      'Dieser Pfad befindet sich nicht auf diesem Computer, sondern auf dem Backend-Rechner. Verwenden Sie „Im Dateibaum anzeigen“.'
  } satisfies NonNullable<TranslationOverrides['fileMenu']>
