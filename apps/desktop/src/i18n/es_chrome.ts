import type { TranslationOverrides } from './define-locale'

export const esChrome = {
  fileMenu: {
    openWithApp: 'Abrir con la aplicación predeterminada',
    revealFinder: 'Mostrar en Finder',
    revealExplorer: 'Mostrar en el Explorador de archivos',
    revealFileManager: 'Abrir carpeta contenedora',
    revealInSidebar: 'Mostrar en el árbol de archivos',
    copyPath: 'Copiar ruta',
    copyRelativePath: 'Copiar ruta relativa',
    download: 'Descargar',
    downloadSaved: 'Guardado',
    downloadFailed: 'Error al descargar',
    rename: 'Cambiar nombre…',
    delete: 'Eliminar',
    renameTitle: 'Cambiar nombre',
    renameLabel: 'Nuevo nombre',
    deleteTitle: (name: string) => `¿Eliminar ${name}?`,
    deleteBody: 'Se moverá a la Papelera; podrás restaurarlo desde allí.',
    pathCopied: 'Ruta copiada',
    revealMissing: 'Esa carpeta no está en este equipo',
    revealUnavailable:
      'Esa ruta no está en este equipo: está en la máquina del backend. Usa “Mostrar en el árbol de archivos”.'
  }
} satisfies TranslationOverrides
