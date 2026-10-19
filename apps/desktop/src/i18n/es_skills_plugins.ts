import type { TranslationOverrides } from './define-locale'

/** Spanish copy for the Skills ▸ Plugins tab; `skills.plugins`, composed by es.ts. */
export const esSkillsPlugins: NonNullable<NonNullable<TranslationOverrides['skills']>['plugins']> = {
  agentTitle: 'Plugins del agente',
  agentBlurb:
    'Amplían el agente del perfil seleccionado: herramientas, hooks, proveedores. Se aplican tras reiniciar el gateway.',
  pageBlurb: 'Un plugin puede ampliar esta app, el agente o ambos; cada mitad tiene su propio interruptor.',
  halfDesktop: 'Escritorio',
  halfDesktopHint: 'esta app, igual para todos los perfiles',
  halfAgent: 'Agente',
  halfAgentIn: (profile: string) => `Agente en ${profile}`,
  defaultProfile: 'Hermes (predeterminado)',
  kindAgent: 'Agente',
  kindDesktop: 'Escritorio',
  kindBoth: 'Agente + Escritorio',
  installAgentHere: 'Instalar aquí',
  installAgentHereTip: (profile: string) =>
    `La mitad de escritorio está cargada en esta app, pero la mitad del agente no está instalada en ${profile}. Instálala allí.`,
  installAgentHereNoOrigin:
    'La mitad del agente no está instalada en este perfil y este paquete se copió a mano (sin entrada de catálogo ni remoto de git), así que no se puede instalar desde aquí. Copia su carpeta en el perfil o reinstálalo desde Git.',
  desktopHalfPending: 'copiando…',
  desktopHalfPendingTip:
    'Este paquete incluye una mitad de escritorio que todavía no se ha copiado en la app. Usa Volver a escanear o reinicia la app.',
  desktopHalfRemote: 'no disponible (backend remoto)',
  desktopHalfRemoteTip:
    'La mitad de escritorio de este paquete está en el disco del backend remoto, que esta app no puede leer. Para usarla aquí, ejecuta Instalar desde Git con la URL del repositorio del paquete y el destino Escritorio marcado; eso clona la mitad de escritorio en este equipo.',
  emptyAll: 'Todavía no hay plugins.',
  empty: 'No hay plugins del agente instalados para este perfil.',
  emptyHint: 'Explora el catálogo de abajo e instala un plugin revisado con un clic.',
  loadFailed: 'No se pudieron cargar los plugins del agente',
  toggleFailed: (name: string) => `No se pudo cambiar ${name}`,
  toolsetOn: (name: string, profile: string) => `Herramientas de agente de ${name} activadas para ${profile}`,
  toolsetOff: (name: string, profile: string) => `Herramientas de agente de ${name} desactivadas para ${profile}`,
  toolsetToggleFailed: (name: string) =>
    `No se pudieron cambiar las herramientas de agente de ${name}; el panel de Escritorio no se modificó`,
  legacyBackend:
    'Este backend es anterior a los interruptores de plugins por clave: actualiza Hermes para gestionarlo aquí.',
  portableBadge: 'portátil',
  serverStates: {
    connected: 'conectado',
    app_not_running: 'la app no se está ejecutando',
    hermes_not_connected: 'falta la conexión MCP',
    endpoint_unavailable: 'endpoint no disponible',
    no_interactive_session: 'sin sesión interactiva',
    version_too_old: 'versión demasiado antigua',
    missing_app: 'falta la app',
    unsupported_gpu: 'GPU no compatible',
    unknown: 'estado desconocido'
  },
  catalogTitle: 'Catálogo de plugins',
  catalogBrowse: 'Explorar',
  catalogHide: 'Ocultar el explorador del catálogo',
  catalogHint:
    'Pulsa "+ Añadir a este agente" en cualquier plugin: las entradas revisadas se instalan en su commit fijado en el perfil seleccionado. Los plugins agente+escritorio incluidos ofrecen ambas mitades.',
  alreadyInstalled: (name: string) => `${name} ya está instalado en este perfil.`,
  catalogProvenance: (sha: string) =>
    `Instalado desde el catálogo de Hermes${sha ? ` en el commit fijado ${sha}` : ''}.`,
  pinnedProvenance: (sha: string) =>
    `Fijado al commit ${sha}. Las actualizaciones se rechazan hasta que se reinstale con un nuevo commit fijado.`,
  pinnedBadge: (sha: string) => `fijado @ ${sha}`,
  update: 'Actualizar',
  updateToPin: (sha: string) => `Actualizar a ${sha}`,
  updateFailed: (name: string) => `No se pudo actualizar ${name}`,
  updated: (name: string) => `${name} se actualizó. Reinicia el gateway para aplicarlo.`,
  upToDate: (name: string) => `${name} ya está actualizado.`,
  updateConsentTitle: (name: string) => `${name} necesita tu aprobación`,
  updateConsentBody: (name: string) =>
    `Actualizar ${name} trae cambios que deberías revisar. Aplícalo solo si confías en la fuente:`,
  updateConsentConfirm: 'Aplicar actualización',
  uninstall: 'Desinstalar',
  uninstallTip: (name: string, profile: string) => `Desinstalar ${name} de ${profile}`,
  uninstallConfirmTitle: (name: string) => `¿Desinstalar ${name}?`,
  uninstallConfirmBody: (name: string, profile: string) =>
    `Esto elimina los archivos del plugin del perfil ${profile}. Cualquier mitad de escritorio que incluyera se quita con él. Puedes reinstalarlo desde el catálogo o desde Git cuando quieras.`,
  uninstallFailed: (name: string) => `No se pudo desinstalar ${name}`,
  uninstalled: (name: string) => `${name} desinstalado. Reinicia el gateway para descargarlo.`,
  uninstallDesktopTip: (name: string) => `Desinstalar ${name} de esta app`,
  uninstallDesktopConfirmBody: (name: string) =>
    `Esto elimina ${name} de la carpeta desktop-plugins de este equipo y lo descarga ahora. Puedes reinstalarlo desde Git o volver a poner la carpeta cuando quieras.`,
  uninstalledDesktop: (name: string) => `${name} desinstalado.`,
  deepLinkErrorTitle: 'Enlace de instalación de plugin rechazado',
  deepLinkCatalogInvalidName: 'Falta el nombre del catálogo del enlace o no es válido.',
  deepLinkCatalogUnknown: (name: string) =>
    `\u201C${name}\u201D no está en el catálogo de plugins de Hermes. No se instaló nada.`,
  deepLinkCatalogUnavailable:
    'No se pudo cargar el catálogo de plugins de Hermes. Comprueba tu conexión y vuelve a abrir el enlace.',
  settingsToggle: (name: string) => `Configuración: ${name}`,
  settingsForm: {
    save: 'Guardar configuración',
    saved: (name: string) => `Configuración de ${name} guardada.`,
    saveFailed: (name: string) => `No se pudo guardar la configuración de ${name}`,
    required: 'Obligatorio',
    secretSet: '•••••••• (configurado)',
    secretStoredAs: (env: string) =>
      `Se guarda en el .env del perfil como ${env}, nunca en config.yaml; déjalo en blanco para conservar el valor actual.`
  }
}
