// The repo-scan field copy (labels + descriptions), composed by de.ts.
import type { FieldCopyTree } from '@/app/settings/field-copy'

export const deRepoScanLabels: FieldCopyTree = {
  desktop: {
    repoScanEnabled: 'Automatische Repository-Erkennung',
    repoScanNested: 'Verschachtelte Repositories mitentdecken',
    repoScanRoots: 'Repository-Erkennungs-Wurzeln',
    repoScanExcludePaths: 'Ausgeschlossene Repository-Pfade'
  }
}

export const deRepoScanDescriptions: FieldCopyTree = {
  desktop: {
    repoScanEnabled: 'Lokale Ordner nach Git-Repositories durchsuchen, die in Projekten angezeigt werden.',
    repoScanNested: 'Repositories innerhalb eines bereits vorhandenen als Unterprojekte anzeigen.',
    repoScanRoots: 'Zusätzliche zu durchsuchende Ordner. Leer durchsucht Ihr Arbeitsverzeichnis.',
    repoScanExcludePaths: 'Ordner und deren Unterordner, die bei der Repository-Erkennung übersprungen werden.'
  }
}
