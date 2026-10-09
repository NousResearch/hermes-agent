// The repo-scan field copy (labels + descriptions), composed by es.ts.
import type { FieldCopyTree } from '@/app/settings/field-copy'

export const esRepoScanLabels: FieldCopyTree = {
  desktop: {
    repoScanEnabled: 'Detección automática de repositorios',
    repoScanNested: 'Detectar también repositorios anidados',
    repoScanRoots: 'Carpetas de búsqueda de repositorios',
    repoScanExcludePaths: 'Rutas de repositorio excluidas'
  }
}

export const esRepoScanDescriptions: FieldCopyTree = {
  desktop: {
    repoScanEnabled: 'Busca repositorios Git en carpetas locales para mostrarlos en Proyectos.',
    repoScanNested: 'Mostrar los repositorios dentro de otro que ya tengas como subproyectos.',
    repoScanRoots: 'Carpetas adicionales que buscar. Si está vacío se busca tu Directorio de trabajo.',
    repoScanExcludePaths:
      'Carpetas que se omitirán, junto con todos sus subdirectorios, durante la detección de repositorios.'
  }
}
