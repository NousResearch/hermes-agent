import type { TranslationOverrides } from './define-locale'

export const deChrome = {
  fileMenu: {
    openWithApp: 'Mit Standard-App öffnen',
    revealFinder: 'Im Finder anzeigen',
    revealExplorer: 'Im Datei-Explorer anzeigen',
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
    deleteTitle: (name: string) => `„${name}“ löschen?`,
    deleteBody: 'Es wird in den Papierkorb verschoben – von dort können Sie es wiederherstellen.',
    pathCopied: 'Pfad kopiert',
    revealMissing: 'Dieser Ordner befindet sich nicht auf diesem Computer',
    revealUnavailable:
      'Dieser Pfad befindet sich nicht auf diesem Computer, sondern auf dem Backend-Rechner. Verwenden Sie „Im Dateibaum anzeigen“.'
  }
} satisfies TranslationOverrides
