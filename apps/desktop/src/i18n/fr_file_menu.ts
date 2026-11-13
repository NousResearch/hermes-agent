import type { TranslationOverrides } from './define-locale'

export const frFileMenu = {
    revealFinder: 'Afficher dans le Finder',
    revealExplorer: "Afficher dans l'Explorateur de fichiers",
    revealFileManager: 'Ouvrir le dossier parent',
    revealInSidebar: "Afficher dans l'arborescence",
    newFile: 'Nouveau fichier…',
    newFolder: 'Nouveau dossier…',
    createFailed: 'Échec de la création',
    deleteRemoteBody: 'Il sera définitivement supprimé sur la machine distante — cette action est irréversible.',
    copyPath: 'Copier le chemin',
    copyRelativePath: 'Copier le chemin relatif',
    download: 'Télécharger',
    downloadSaved: 'Enregistré',
    downloadFailed: 'Échec du téléchargement',
    rename: 'Renommer…',
    delete: 'Supprimer',
    renameTitle: 'Renommer',
    renameLabel: 'Nouveau nom',
    deleteTitle: name => `Supprimer ${name} ?`,
    deleteBody: 'Il sera déplacé dans la Corbeille — vous pourrez le restaurer depuis là.',
    pathCopied: 'Chemin copié',
    revealMissing: "Ce dossier n'est pas sur cet ordinateur",
    revealUnavailable:
      "Ce chemin n'est pas sur cet ordinateur : il se trouve sur la machine du backend. Utilisez « Afficher dans l'arborescence »."
  } satisfies NonNullable<TranslationOverrides['fileMenu']>
