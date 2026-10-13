import type { TranslationOverrides } from './define-locale'

export const esFileMenu = {
    revealFinder: 'Mostrar en Finder',
    revealExplorer: 'Mostrar en el Explorador de archivos',
    newFile: 'Archivo nuevo…',
    newFolder: 'Carpeta nueva…',
    createFailed: 'No se pudo crear',
    deleteRemoteBody: 'Se eliminará permanentemente en el equipo remoto; esta acción no se puede deshacer.',
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
    deleteTitle: name => `¿Eliminar ${name}?`,
    deleteBody: 'Se moverá a la Papelera; podrás restaurarlo desde allí.',
    pathCopied: 'Ruta copiada',
    revealMissing: 'Esa carpeta no está en este equipo',
    revealUnavailable:
      'Esa ruta no está en este equipo: está en la máquina del backend. Usa “Mostrar en el árbol de archivos”.'
  } satisfies NonNullable<TranslationOverrides['fileMenu']>
