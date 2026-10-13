import type { TranslationOverrides } from './define-locale'

export const enFileMenu = {
    revealFinder: 'Reveal in Finder',
    revealExplorer: 'Reveal in File Explorer',
    revealFileManager: 'Open containing folder',
    revealInSidebar: 'Reveal in filetree',
    newFile: 'New File…',
    newFolder: 'New Folder…',
    createFailed: 'Create failed',
    deleteRemoteBody: 'It will be permanently deleted on the remote machine — this cannot be undone.',
    copyPath: 'Copy path',
    copyRelativePath: 'Copy relative path',
    download: 'Download',
    downloadSaved: 'Saved',
    downloadFailed: 'Download failed',
    rename: 'Rename…',
    delete: 'Delete',
    renameTitle: 'Rename',
    renameLabel: 'New name',
    deleteTitle: name => `Delete ${name}?`,
    deleteBody: 'It will be moved to the Trash — you can restore it from there.',
    pathCopied: 'Path copied',
    revealMissing: 'That folder is not on this computer',
    revealUnavailable: 'That path is not on this computer — it lives on the backend machine. Use “Reveal in filetree”.'
  } satisfies NonNullable<TranslationOverrides['fileMenu']>
