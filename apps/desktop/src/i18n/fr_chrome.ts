import type { TranslationOverrides } from './define-locale'

export const frChrome = {
  fileMenu: {
    openWithApp: "Ouvrir avec l'application par défaut",
    revealFinder: 'Afficher dans le Finder',
    revealExplorer: "Afficher dans l'Explorateur de fichiers",
    revealFileManager: 'Ouvrir le dossier parent',
    revealInSidebar: "Afficher dans l'arborescence",
    copyPath: 'Copier le chemin',
    copyRelativePath: 'Copier le chemin relatif',
    download: 'Télécharger',
    downloadSaved: 'Enregistré',
    downloadFailed: 'Échec du téléchargement',
    rename: 'Renommer…',
    delete: 'Supprimer',
    renameTitle: 'Renommer',
    renameLabel: 'Nouveau nom',
    deleteTitle: (name: string) => `Supprimer ${name} ?`,
    deleteBody: 'Il sera déplacé dans la Corbeille — vous pourrez le restaurer depuis là.',
    pathCopied: 'Chemin copié',
    revealMissing: "Ce dossier n'est pas sur cet ordinateur",
    revealUnavailable:
      "Ce chemin n'est pas sur cet ordinateur : il se trouve sur la machine du backend. Utilisez « Afficher dans l'arborescence »."
  }
} satisfies TranslationOverrides
