// The repo-scan field copy (labels + descriptions), composed by fr.ts.
import type { FieldCopyTree } from '@/app/settings/field-copy'

export const frRepoScanLabels: FieldCopyTree = {
  desktop: {
    repoScanEnabled: 'Découverte automatique de dépôts',
    repoScanNested: 'Découvrir aussi les dépôts imbriqués',
    repoScanRoots: 'Racines de découverte de dépôts',
    repoScanExcludePaths: 'Chemins de dépôt exclus'
  }
}

export const frRepoScanDescriptions: FieldCopyTree = {
  desktop: {
    repoScanEnabled: 'Analyser les dossiers locaux à la recherche de dépôts Git à afficher dans Projets.',
    repoScanNested: 'Afficher les dépôts situés dans un dépôt existant comme sous-projets.',
    repoScanRoots: 'Dossiers supplémentaires à analyser. Vide analyse votre Répertoire de travail.',
    repoScanExcludePaths: 'Dossiers et leurs descendants à ignorer lors de la découverte de dépôts.'
  }
}
