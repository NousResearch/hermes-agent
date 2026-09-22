import { defineLocale } from './define-locale'

/**
 * Spanish desktop locale.
 *
 * The locale intentionally uses the shared English fallback for keys that
 * have not been translated yet. This keeps the picker and persistence safe
 * while allowing the catalog to grow incrementally without blank UI copy.
 */
export const es = defineLocale({
  common: {
    apply: 'Aplicar',
    back: 'Atrás',
    save: 'Guardar',
    saving: 'Guardando…',
    cancel: 'Cancelar',
    change: 'Cambiar',
    choose: 'Elegir',
    clear: 'Limpiar',
    close: 'Cerrar',
    collapse: 'Contraer',
    confirm: 'Confirmar',
    connect: 'Conectar',
    connecting: 'Conectando',
    continue: 'Continuar',
    bots: 'Bots',
    copied: 'Copiado',
    copy: 'Copiar',
    copyFailed: 'No se pudo copiar',
    delete: 'Eliminar',
    docs: 'Documentación',
    done: 'Listo',
    error: 'Error',
    expand: 'Expandir',
    failed: 'Falló',
    formatJson: 'Formatear JSON',
    free: 'Gratis',
    notSet: 'No configurado',
    refresh: 'Actualizar',
    remove: 'Quitar',
    replace: 'Reemplazar',
    retry: 'Reintentar',
    run: 'Ejecutar',
    send: 'Enviar',
    set: 'Establecer',
    skip: 'Omitir',
    update: 'Actualizar',
    tryHint: term => `Prueba con «${term}»`,
    on: 'Activado',
    off: 'Desactivado'
  },
  sessionImport: {
    title: 'Continuar desde otra aplicación',
    subtitle: 'Importa una conversación a Hermes y continúa donde la dejaste.',
    action: 'Importar sesión',
    search: 'Buscar sesiones cargadas',
    scanning: 'Buscando conversaciones',
    empty: 'Todavía no hay conversaciones',
    noMatches: 'No se encontraron coincidencias',
    choose: 'Conversación que quieres continuar',
    importing: 'Importando…',
    open: 'Abrir en Hermes',
    continue: 'Continuar en Hermes'
  },
  connectors: {
    title: 'Conectores',
    connect: 'Conectar',
    skip: 'Omitir',
    cancel: 'Cancelar',
    retry: 'Reintentar',
    grant: 'Conceder acceso',
    connected: 'Conectado',
    checking: 'Comprobando…',
    notConnected: 'No conectado',
    failed: 'Falló la conexión',
    search: 'Buscar conectores',
    empty: 'No hay conectores disponibles'
  },
  notifications: {
    region: 'Notificaciones',
    hide: 'Ocultar',
    show: 'Mostrar',
    clearAll: 'Borrar todo',
    dismiss: 'Descartar',
    details: 'Detalles',
    errors: {
      diskFull: 'No hay espacio suficiente en el disco.',
      storageFailure: 'No se pudo guardar la información.',
      microphonePermission: 'Hermes no tiene permiso para usar el micrófono.'
    }
  },
  onboarding: {
    headerTitle: 'Configura Hermes Agent',
    headerDesc: 'Conecta un proveedor de modelos para comenzar a conversar.',
    starting: 'Iniciando Hermes…',
    collapse: 'Contraer',
    otherProviders: 'Otros proveedores',
    haveApiKey: 'Tengo una clave API',
    chooseLater: 'Elegiré un proveedor después',
    recommended: 'Recomendado',
    connected: 'Conectado',
    backToSignIn: 'Volver al inicio de sesión',
    getKey: 'Obtener una clave',
    replaceCurrent: 'Reemplazar valor actual',
    pasteApiKey: 'Pegar clave API',
    couldNotSave: 'No se pudo guardar la credencial.',
    connecting: 'Conectando',
    update: 'Actualizar'
  },
  desktop: {
    audioReadFailed: 'No se pudo leer el audio.',
    sessionUnavailable: 'La sesión no está disponible.',
    createSessionFailed: 'No se pudo crear la sesión.',
    promptFailed: 'No se pudo enviar el mensaje.',
    providerCredentialRequired: 'Se necesitan credenciales del proveedor.',
    emptySlashCommand: 'Escribe un comando.',
    desktopCommands: 'Comandos de escritorio'
  }
})
