import type { FileMenuTranslations } from './types_chrome'

export const enChrome = {
  fileMenu: {
    openWithApp: 'Open with default app',
    revealFinder: 'Reveal in Finder',
    revealExplorer: 'Reveal in File Explorer',
    revealFileManager: 'Open containing folder',
    revealInSidebar: 'Reveal in filetree',
    copyPath: 'Copy path',
    copyRelativePath: 'Copy relative path',
    download: 'Download',
    downloadSaved: 'Saved',
    downloadFailed: 'Download failed',
    rename: 'Rename…',
    delete: 'Delete',
    renameTitle: 'Rename',
    renameLabel: 'New name',
    deleteTitle: (name: string) => `Delete ${name}?`,
    deleteBody: 'It will be moved to the Trash — you can restore it from there.',
    pathCopied: 'Path copied',
    revealMissing: 'That folder is not on this computer',
    revealUnavailable: 'That path is not on this computer — it lives on the backend machine. Use “Reveal in filetree”.'
  }
} satisfies { fileMenu: FileMenuTranslations }
