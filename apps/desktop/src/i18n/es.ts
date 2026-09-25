import { defineFieldCopy } from '@/app/settings/field-copy'

import { defineLocale, type TranslationOverrides } from './define-locale'
import { introEs } from './intro-es'

export const esOverrides = {
  intro: introEs,
  connectors: {
    title: 'Conecta tus apps',
    connect: 'Conectar',
    skip: 'Ahora no',
    cancel: 'Dejar de esperar',
    retry: 'Reintentar',
    grant: 'Volver a conectar',
    connected: 'Conectado',
    checking: 'Comprobando tus apps…',
    notConnected: 'No conectado',
    skipped: 'Omitido',
    disabled: 'No disponible',
    failed: 'No se pudo conectar',
    needsAuth: 'Acceso caducado',
    opening: 'Abriendo el inicio de sesión…',
    waiting: 'Esperando a tu navegador…',
    timeout: 'Seguimos esperando la autorización.',
    refresh: 'Actualizar estado',
    connectError: 'No se pudo iniciar la autorización. Inténtalo de nuevo.',
    connectErrorFor: (app: string) => `No se pudo iniciar la autorización para ${app}.`,
    unavailable: 'Los conectores no están disponibles en esta sesión.',
    ownerMissing: 'Vuelve a abrir esta conversación para gestionar sus conexiones.',
    search: 'Buscar una app',
    empty: 'No hay apps que coincidan',
    disclaimer: 'Conectar es opcional. Autoriza solo las apps que quieras que use Hermes.',
    execution: 'Herramientas de conectores',
    setup: server => `Configurar ${server}`,
    openInBrowser: 'Abrir en el navegador',
    setupCancel: 'Cancelar',
    authorizedToolsUnavailable: 'Autorizado. Herramientas no disponibles.',
    required: 'Obligatorio'
  },
  connectorsPage: {
    title: 'Conectores',
    searchPlaceholder: (count: number) => `Buscar entre ${count} apps`,
    filterCategory: 'Categoría',
    categoryAll: 'Todas las categorías',
    uncategorised: 'Sin categoría',
    residencyLocal: 'En este dispositivo',
    segment: {
      all: 'Todos',
      available: 'Disponibles',
      connected: 'Conectados',
      off: 'Desactivados'
    },
    group: {
      connected: 'Conectados',
      connectedNote: 'Primero las conexiones con fallos.',
      available: 'Disponibles',
      off: 'Desactivados',
      offNote: 'Los inicios de sesión se conservan.'
    },
    card: {
      kindManaged: 'Administrado',
      kindCatalog: 'MCP · Catálogo',
      kindCustom: 'MCP · Personalizado',
      kindPlugin: (plugin: string) => `MCP · Plugin ${plugin}`,
      inCatalog: 'En el catálogo de Hermes',
      hostedTwin: 'Versión administrada disponible',
      alsoLocal: 'También se ejecuta en este dispositivo',
      open: (name: string) => `Abrir ${name}`,
      turnServerOn: (name: string) => `Activar ${name}`,
      turnServerOff: (name: string) => `Desactivar ${name}`,
      state: {
        accessExpired: 'Acceso caducado',
        available: 'Disponible',
        connected: 'Conectado',
        connecting: 'Conectando',
        connectionUnknown: 'Estado desconocido',
        couldNotConnect: 'No se pudo conectar',
        offByYourOrganisation: 'Desactivado por tu organización',
        offForYou: 'Desactivado para ti',
        serverConnecting: 'Conectando…',
        serverError: 'Error',
        serverNeedsAuth: 'Requiere autenticación',
        serverOff: 'Desactivado',
        serverOn: 'Activado',
        serverOnUnused: 'Activado, sin usar'
      },
      fact: {
        tools: (count: number) => `${count} herramienta${count === 1 ? '' : 's'}`,
        toolsOff: (count: number) =>
          `${count} herramienta${count === 1 ? '' : 's'} desactivada${count === 1 ? '' : 's'}`,
        toolsOn: (count: number) => `${count} herramienta${count === 1 ? '' : 's'} activada${count === 1 ? '' : 's'}`,
        toolsSomeOn: (total: number, on: number) => `${total} herramientas, ${on} activada${on === 1 ? '' : 's'}`
      },
      verb: {
        authenticate: 'Autenticar',
        connect: 'Conectar',
        install: 'Instalar',
        openLogs: 'Abrir registros',
        reconnect: 'Volver a conectar',
        stopWaiting: 'Dejar de esperar',
        tryAgain: 'Reintentar',
        turnBackOn: 'Volver a activar'
      },
      reason: {
        finishSignIn: 'Termina de iniciar sesión en tu navegador.',
        reconnect: 'Vuelve a conectar para que esta app siga funcionando.',
        serverError: 'El servidor rechazó la conexión.',
        serverNeedsAuth: 'Inicia sesión para que este servidor pueda responder.'
      }
    },
    page: {
      loading: 'Leyendo el catálogo y los servidores de este equipo',
      emptyTitle: 'Todavía no hay apps. Añade un servidor en este equipo para empezar.',
      noMatchTitle: 'No hay apps que coincidan',
      noMatchBody: 'No hay coincidencias. Indica a Hermes tu propio servidor MCP para añadirlo.',
      clearSearch: 'Borrar la búsqueda',
      hostedFailedTitle: 'No se pudo acceder a las apps alojadas.',
      hostedFailedBody: 'Los servidores de este equipo no se ven afectados y siguen funcionando. No se desactivó nada.',
      retry: 'Reintentar',
      matchesElsewhere: (count: number) => `${count} coincidencia${count === 1 ? '' : 's'} más en otros grupos.`,
      showAllMatches: 'Mostrar todas las coincidencias',
      segmentNoMatch: (segment: string) => `No hay coincidencias en ${segment}, así que se muestran todas.`,
      freeTierNote: 'Las conexiones se quedan en este equipo hasta que inicies sesión.',
      signInLine: 'Inicia sesión en Nous para usar las apps administradas.',
      signIn: 'Iniciar sesión',
      managedUnavailable: 'Las apps administradas aún no están disponibles para esta cuenta.',
      writeFailed: 'No se guardó ese cambio.',
      refreshFailed: 'No se actualizó la lista de herramientas.',
      disconnectNoAccount:
        'Hermes no tiene ninguna cuenta que desconectar aquí. Actualiza la página e inténtalo de nuevo.',
      disconnectRefused:
        'Nous no pudo quitar este inicio de sesión ahora. Desactiva la app con el interruptor o inténtalo más tarde.'
    },
    add: {
      action: 'Añadir el tuyo',
      title: 'Conectar con un MCP personalizado',
      hint: 'una nueva entrada en mcp.json en este dispositivo',
      pasteLabel: 'Pega un comando o un fragmento',
      pastePlaceholder: 'npx -y @modelcontextprotocol/server-filesystem /ruta/a/carpeta',
      pasteNoMatch: 'Nada de esto parece un servidor. Rellena los campos de abajo.',
      name: 'Nombre',
      nameTaken: 'Ese nombre ya está en uso.',
      type: 'Tipo',
      typeStdio: 'STDIO',
      typeHttp: 'HTTP transmitible',
      command: 'Comando de inicio',
      args: 'Argumentos',
      addArg: '+ Añadir argumento',
      envVars: 'Variables de entorno',
      addEnvVar: '+ Añadir variable de entorno',
      passthrough: 'Paso de variables de entorno',
      addPassthrough: '+ Añadir variable',
      cwd: 'Directorio de trabajo',
      url: 'URL',
      headers: 'Encabezados',
      addHeader: '+ Añadir encabezado',
      auth: 'Autenticación',
      authNone: 'Ninguna',
      authOauth: 'OAuth',
      authBearer: 'Token Bearer',
      keyPlaceholder: 'CLAVE',
      valuePlaceholder: 'valor',
      removeRow: 'Quitar esta fila',
      editJson: 'Editar mcp.json',
      saveFailed: 'No se guardó ese servidor.'
    },
    dialog: {
      disconnect: 'Desconectar',
      disconnectTitle: (name: string) => `¿Desconectar ${name}?`,
      disconnectBody: 'Hermes deja de actuar con esta cuenta. Puedes volver a conectarla cuando quieras.',
      menuRefreshTools: 'Actualizar herramientas',
      moreActions: 'Más acciones',
      removeServerTitle: (name: string) => `¿Quitar ${name}?`,
      removeServerBody: 'La entrada se quita de mcp.json en este equipo. No se elimina nada más.',
      appSwitch: (name: string) => `Hermes puede usar ${name}`,
      waysTitle: (name: string) => `Dónde se ejecuta ${name}`,
      wayNotConnected: (name: string) => `Aún no está conectado. Inicia sesión en ${name} desde tu navegador.`,
      wayHosted: 'Administrado',
      bothOn: (name: string) => `Ambos están activados, así que Hermes ve cada herramienta de ${name} dos veces.`,
      turnOffLocal: 'Desactivar el servidor local',
      providedByPlugin: (plugin: string) => `Proporcionado por el plugin ${plugin}`,
      openPlugins: 'Abrir la pestaña Plugins',
      nousLine: 'Las apps de Nous siguen a tu cuenta, no al perfil.',
      rulesReadOnly: 'Las reglas no se pueden cambiar ahora.',
      rulesAppOff: (name: string) => `Activa ${name} para cambiar sus herramientas.`,
      rulesSignIn: 'Inicia sesión para cambiar lo que Hermes puede hacer aquí.',
      orgNote: (count: number) => `Tu organización desactivó ${count} herramienta${count === 1 ? '' : 's'}.`,
      orgLink: 'Abrir la administración de conectores',
      connectEnded: 'El inicio de sesión no terminó.',
      connectOpenAgain: 'Abrir el enlace de nuevo',
      tokensPerCall: 'tokens por llamada',
      usesPerMonth: 'usos en 30 días',
      advanced: 'Avanzado',
      advancedHint: 'la entrada de mcp.json y los registros'
    },
    tools: {
      title: 'Herramientas',
      notInstalledBody: 'Instálalo en este dispositivo para ver las herramientas que incluye.',
      summaryTitle: (name: string) => `Lo que Hermes puede hacer con ${name}`,
      summaryPreviewTitle: (name: string) => `Lo que Hermes podría hacer con ${name} cuando lo conectes`,
      summaryCount: (count: number) => `${count} herramienta${count === 1 ? '' : 's'}`,
      summaryAllTools: 'Todas las herramientas',
      summaryOther: 'Otras',
      allToolsSwitch: 'Activar o desactivar todas las herramientas',
      summaryAllOn: 'todas activadas',
      summarySomeOn: (on: number, total: number) => `${on} de ${total} activadas`,
      summaryOff: 'desactivadas',
      showAllTools: (count: number) =>
        count === 1 ? `Mostrar ${count} herramienta` : `Mostrar las ${count} herramientas`,
      showSummary: 'Mostrar resumen',
      facetSwitch: (facet: string) => `Activar o desactivar las herramientas de ${facet}`,
      moreHints: (count: number) => `+${count}`,
      staleSignIn: 'Inicia sesión para leer la lista de herramientas más reciente.',
      searchCountPlaceholder: (count: number) => `Buscar entre ${count} herramientas`,
      toolList: (name: string) => `Herramientas de ${name}`,
      categorySelect: (count: number) => `${count} categoría${count === 1 ? '' : 's'}`,
      showDeprecated: (count: number) => `Mostrar ${count} obsoleta${count === 1 ? '' : 's'}`,
      hideDeprecated: (count: number) => `Ocultar ${count} obsoleta${count === 1 ? '' : 's'}`,
      quickReadOnly: 'Solo lectura',
      quickNoDestructive: 'Desactivar las destructivas',
      quickEverythingOn: 'Todo activado',
      lockedHint: 'desactivada por tu organización',
      turnToolOn: (tool: string) => `Activar ${tool}`,
      turnToolOff: (tool: string) => `Desactivar ${tool}`,
      showDetails: (tool: string) => `Mostrar qué hace ${tool}`,
      hideDetails: (tool: string) => `Ocultar qué hace ${tool}`,
      noMatch: 'Ninguna herramienta coincide con estos filtros.',
      loading: 'Leyendo la lista de herramientas',
      unavailableLine: 'Lista de herramientas no disponible.',
      needsAuthTitle: (name: string) => `Inicia sesión en ${name} para leer sus herramientas.`,
      needsAuthBody: 'El inicio de sesión se queda en este equipo. Nada sale de él.',
      retry: 'Reintentar',
      goneTitle: (name: string) => `${name} salió del catálogo.`,
      goneBody: 'Hermes ya no puede llamarlo. La fila se queda hasta que la quites, así que nada desaparece.',
      remove: 'Quitar',
      offTitle: (name: string) => `${name} está desactivado.`,
      offBody: 'Actívalo con el interruptor de arriba para leer las herramientas que incluye.',
      signedOutTitle: 'Inicia sesión en Nous para leer la lista de herramientas.',
      signedOutBody: 'Tus servidores en este equipo no se ven afectados.',
      conflictTitle: 'Alguien cambió esta regla mientras la editabas.',
      conflictBody: (theyOff: number, theyOn: number) => {
        const they = [
          theyOff > 0
            ? `desactivó ${theyOff} herramienta${theyOff === 1 ? '' : 's'} que tienes activada${theyOff === 1 ? '' : 's'}`
            : '',
          theyOn > 0
            ? `dejó activada${theyOn === 1 ? '' : 's'} ${theyOn} herramienta${theyOn === 1 ? '' : 's'} que desactivaste`
            : ''
        ].filter(Boolean)

        return `${they.length > 0 ? `Esa persona ${they.join(' y ')}. ` : ''}Tus cambios siguen en pantalla; no se escribió nada.`
      },
      conflictReload: 'Recargar su versión',
      conflictSave: 'Guardar sobre su versión',
      saveFailed: 'No se guardaron esas reglas de herramientas.',
      footerDirty: (off: number, backOn: number) =>
        `${off} herramienta${off === 1 ? '' : 's'} desactivada${off === 1 ? '' : 's'}, ${backOn === 0 ? 'ninguna' : backOn} reactivada${backOn === 1 ? '' : 's'}`,
      discard: 'Descartar',
      save: 'Guardar cambios',
      saving: 'Guardando...'
    },
    vocabulary: {
      facetRead: {
        label: 'Lectura',
        long: 'Lee datos de esta app. No cambia nada.'
      },
      facetWrite: {
        label: 'Escritura',
        long: 'Crea o cambia algo en esta app.'
      },
      facetDestructive: {
        label: 'Destructiva',
        long: 'Puede eliminar algo de esta app de forma definitiva.'
      },
      facetUnclassified: {
        label: 'Efecto desconocido',
        long: 'La app nunca indicó qué hace esta herramienta.'
      },
      hintReadOnly: {
        label: 'Solo lectura',
        long: 'La herramienta declara que solo lee.'
      },
      hintCreate: {
        label: 'Crea',
        long: 'Crea algo nuevo.'
      },
      hintUpdate: {
        label: 'Actualiza',
        long: 'Cambia algo que ya existe.'
      },
      hintDelete: {
        label: 'Elimina',
        long: 'Quita algo.'
      },
      hintDestructive: {
        label: 'Destructiva',
        long: 'El cambio que hace no se puede deshacer aquí.'
      },
      hintIdempotent: {
        label: 'Repetible',
        long: 'Ejecutarla dos veces hace lo mismo que ejecutarla una vez.'
      },
      hintOpenWorld: {
        label: 'Externa',
        long: 'Llega a algo fuera de esta app.'
      }
    }
  },
  sessionImport: {
    title: 'Continuar desde otra app',
    subtitle: 'Trae una conversación a Hermes y retómala donde la dejaste.',
    action: 'Importar sesión',
    readingFrom: 'Leyendo desde',
    connectedComputer: 'el equipo conectado',
    destination: 'Importar a',
    all: 'Todas',
    search: 'Buscar en las sesiones cargadas',
    scanning: 'Buscando conversaciones',
    scanError: 'No se pudieron encontrar sesiones',
    scanHelp:
      'Comprueba la conexión con el backend e inténtalo de nuevo. Los backends antiguos pueden necesitar una actualización.',
    empty: 'No se encontraron conversaciones',
    emptyHelp: 'Aquí aparecerán las sesiones de Claude Code y Codex de este backend.',
    noMatches: 'No hay conversaciones que coincidan',
    searchHelp: 'Prueba con otro título o carpeta, o carga más sesiones.',
    skipped: 'Algunos registros estaban vacíos, no se podían leer o eran demasiado grandes para previsualizarlos.',
    more: 'Cargar más sesiones',
    messages: 'mensajes',
    choose: 'Una conversación que vale la pena continuar',
    chooseHelp: 'Elige una sesión para leer su historial antes de traerla a Hermes.',
    previewLoading: 'Abriendo la vista previa',
    previewError: 'Vista previa no disponible',
    previewHelp: 'Es posible que el origen se haya movido o cambiado. Actualiza la lista e inténtalo de nuevo.',
    previewLimit: 'Vista previa acortada para facilitar la lectura. Se importa la conversación completa.',
    you: 'Tú',
    snapshot: 'Esta conversación ya está en Hermes. Abre tu copia existente para continuar.',
    copyNotice:
      'Copia el texto de la conversación. Los archivos de origen no cambian. La salida de herramientas y el razonamiento no se trasladan.',
    importing: 'Importando…',
    open: 'Abrir en Hermes',
    continue: 'Continuar en Hermes',
    importError: 'No se pudo importar esta conversación.'
  },
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
    docs: 'Docs',
    done: 'Listo',
    error: 'Error',
    expand: 'Expandir',
    failed: 'Falló',
    formatJson: 'Formatear JSON',
    free: 'Gratis',
    loading: 'Cargando…',
    notSet: 'Sin definir',
    refresh: 'Actualizar',
    remove: 'Quitar',
    replace: 'Reemplazar',
    retry: 'Reintentar',
    run: 'Ejecutar',
    send: 'Enviar',
    set: 'Definir',
    skip: 'Omitir',
    update: 'Actualizar',
    tryHint: term => `Prueba “${term}”`,
    on: 'Activado',
    off: 'Desactivado'
  },
  fileMenu: {
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
    deleteTitle: name => `¿Eliminar ${name}?`,
    deleteBody: 'Se moverá a la Papelera; podrás restaurarlo desde allí.',
    pathCopied: 'Ruta copiada',
    revealMissing: 'Esa carpeta no está en este equipo',
    revealUnavailable:
      'Esa ruta no está en este equipo: está en la máquina del backend. Usa “Mostrar en el árbol de archivos”.'
  },
  boot: {
    ready: 'Hermes Desktop está listo',
    desktopBootFailedWithMessage: message => `Falló el arranque del escritorio: ${message}`,
    steps: {
      connectingGateway: 'Conectando el gateway de escritorio en vivo',
      loadingSettings: 'Cargando la configuración de Hermes',
      loadingSessions: 'Cargando sesiones recientes',
      retryingRemoteBackend: 'Reconectando al backend remoto de Hermes…',
      startingDesktopConnection: 'Iniciando la conexión de escritorio',
      startingHermesDesktop: 'Iniciando Hermes Desktop…'
    },
    errors: {
      backgroundExited:
        'El servicio que ejecuta tus chats se cerró de forma inesperada. Reinícialo para continuar; tus chats y ajustes están a salvo.',
      backgroundExitedDuringStartup: 'Hermes se detuvo justo después de iniciarse.',
      backendStopped: 'Hermes dejó de funcionar en segundo plano',
      restartHermes: 'Reiniciar Hermes',
      openLogs: 'Abrir registros',
      desktopBootFailed: 'Hermes no pudo iniciarse',
      gatewayConnectionLost: 'Hermes perdió la conexión',
      gatewayConnectionLostDetail:
        'Seguimos intentando reconectar. Puedes seguir leyendo y escribiendo borradores. Si continúa, reconecta ahora o revisa los ajustes de conexión.',
      reconnectNow: 'Reconectar ahora',
      connectionSettings: 'Configuración de conexión',
      gatewaySignInRequired: 'Tu Hermes remoto cerró tu sesión',
      gatewaySignInRequiredDetail: 'Vuelve a iniciar sesión para reconectar. Tus chats y ajustes están a salvo.',
      signInAgain: 'Volver a iniciar sesión',
      ipcBridgeUnavailable: 'Hermes Desktop no pudo comunicarse con su propia capa en segundo plano. Reinicia la app.'
    },
    causes: {
      exitedEarly: 'El servicio en segundo plano de Hermes se detuvo justo después de iniciarse.',
      timedOut: 'El servicio en segundo plano de Hermes no respondió a tiempo.',
      permission: 'Hermes no pudo escribir en su carpeta de datos (problema de permisos).',
      diskFull: 'El disco está lleno, así que Hermes no pudo iniciarse.',
      portInUse: 'Otro programa está usando el puerto de red que necesita Hermes.',
      installMissing: 'Falta parte de la instalación de Hermes. Elige Reparar instalación para restaurarla.'
    },
    failure: {
      title: 'Hermes no pudo iniciarse',
      description:
        'El servicio en segundo plano de Hermes no arrancó. Prueba uno de los pasos de recuperación de abajo. Nada de esto elimina tus chats ni tus ajustes.',
      details: 'Detalles',
      remoteTitle: 'Se requiere iniciar sesión en el gateway remoto',
      remoteDescription:
        'Tu sesión del gateway remoto caducó. Inicia sesión de nuevo para reconectar. Esto no elimina tus chats ni tu configuración.',
      retry: 'Reintentar',
      repairInstall: 'Reparar instalación',
      useLocalGateway: 'Usar gateway local',
      gatewaySettings: 'Configuración del gateway',
      back: 'Atrás',
      openLogs: 'Abrir registros',
      repairHint: 'La reparación vuelve a ejecutar el instalador y puede tardar unos minutos en una máquina nueva.',
      remoteSignInHint: signInLabel =>
        `Cierra la sesión guardada del navegador remoto y abre ${signInLabel}. Usa el gateway local para cambiar al backend incluido.`,
      signOutAndSignIn: 'Cerrar sesión e iniciar sesión',
      remoteFailureHint: 'Revisa la URL e inicia sesión en Configuración del gateway, o cambia al gateway local.',
      cloudDownTitle: 'El agente de Nous Cloud no está disponible',
      cloudDownDescription:
        'El agente en la nube administrado por Nous al que se conecta este gateway devuelve un error de servidor. No se puede reiniciar desde aquí: revisa su estado, cambia al gateway local o pide ayuda.',
      cloudDownHint:
        'Los botones de abajo abren el Nous Portal (estado y controles de la instancia) y nuestro Discord para obtener ayuda.',
      cloudDownCheckPortal: 'Ver el estado en el Portal',
      cloudDownDiscord: 'Pedir ayuda en Discord',
      hideRecentLogs: 'Ocultar registros recientes',
      showRecentLogs: 'Mostrar registros recientes',
      signedInTitle: 'Sesión iniciada',
      signedInMessage: 'Reconectando con el gateway remoto…',
      signInIncompleteTitle: 'Inicio de sesión incompleto',
      signInIncompleteMessage: 'La ventana de inicio de sesión se cerró antes de que terminara la autenticación.',
      signInFailed: 'No se pudo iniciar sesión',
      signInToRemoteGateway: 'Iniciar sesión en el gateway remoto',
      signInWithProvider: provider => `Iniciar sesión con ${provider}`,
      identityProvider: 'tu proveedor de identidad'
    }
  },
  notifications: {
    region: 'Notificaciones',
    hide: 'Ocultar',
    show: 'Mostrar',
    more: count => `${count} ${count === 1 ? 'notificación más' : 'notificaciones más'}`,
    clearAll: 'Borrar todo',
    dismiss: 'Descartar notificación',
    details: 'Detalles',
    copyDetail: 'Copiar detalle',
    copyDetailFailed: 'No se pudo copiar el detalle de la notificación',
    backendOutOfDateTitle: 'Backend desactualizado',
    backendOutOfDateMessage:
      'Tu backend de Hermes es más antiguo que esta compilación de escritorio y puede no funcionar correctamente. Actualízalo para alinearlos.',
    installMethodUnsupportedTitle: 'Método de instalación no compatible',
    updateHermes: 'Actualizar Hermes',
    updateReadyTitle: 'Actualización lista',
    updateReadyMessage: count => `${count} ${count === 1 ? 'cambio nuevo disponible' : 'cambios nuevos disponibles'}.`,
    updateReadyMessageUnknown: 'Hay una nueva actualización disponible.',
    seeWhatsNew: 'Ver novedades',
    mcp: {
      needsAuthTitle: 'El servidor MCP necesita re-autenticación',
      needsAuthMessage: name => `${name} MCP necesita re-autenticación.`,
      errorTitle: 'Servidor MCP inalcanzable',
      errorMessage: name => `${name} MCP falló su verificación de salud.`,
      signIn: 'Iniciar sesión',
      view: 'Ver',
      disable: 'Desactivar',
      disabledMessage: name => `MCP ${name} desactivado. Vuelve a activarlo cuando quieras desde Capacidades → MCP.`,
      disableFailed: name => `No se pudo desactivar el MCP ${name}.`
    },
    errors: {
      elevenLabsNeedsKey: 'La entrada de voz necesita una clave de ElevenLabs. Añade una en Configuración → Claves.',
      elevenLabsRejectedKey:
        'ElevenLabs no aceptó tu clave API. Actualízala en Configuración → Claves e inténtalo de nuevo.',
      diskFull: 'Disco lleno — libera espacio y vuelve a intentarlo.',
      storageFailure: 'Hermes no pudo guardar en su carpeta de datos. Abre Mantenimiento para comprobarla y repararla.',
      gatewayAuthFailed:
        'Este Hermes ya no acepta tu inicio de sesión guardado. Abre Gateways y vuelve a iniciar sesión (o pega un nuevo token de acceso) e inténtalo otra vez.',
      methodNotAllowed:
        'El servicio en segundo plano de Hermes no está sincronizado con la app, probablemente tras una actualización. Reinícialo para solucionarlo.',
      microphonePermission: 'Se denegó el permiso del micrófono.',
      openaiRejectedApiKey:
        'OpenAI no aceptó tu clave API. Actualízala en Configuración → Claves e inténtalo de nuevo.',
      openaiTtsNeedsKey: 'La voz necesita una clave de OpenAI. Añade una en Configuración → Claves.',
      codeSkewRestartRequired:
        'Hermes se actualizó, pero sigue ejecutando la versión anterior. Reinícialo para terminar la actualización.',
      rpcOutOfSync: 'La app y el backend están en versiones distintas. Actualiza ambos.',
      restartHermesFailed: 'No se pudo reiniciar Hermes'
    },
    actions: {
      restartHermes: 'Reiniciar Hermes',
      openKeys: 'Abrir Claves',
      openGateways: 'Abrir Gateways',
      openMaintenance: 'Abrir Mantenimiento'
    },
    voice: {
      configureSpeechToText: 'Configura voz a texto para usar el modo de voz.',
      couldNotStartSession: 'No se pudo iniciar la sesión de voz',
      microphoneAccessDenied: 'Acceso al micrófono denegado.',
      microphoneConstraintsUnsupported: 'Este dispositivo no admite las restricciones del micrófono.',
      microphoneFailed: 'Falló el micrófono',
      microphoneInUse: 'El micrófono ya está en uso por otra app.',
      microphonePermissionDenied: 'Se denegó el permiso del micrófono.',
      microphoneStartFailed: 'No se pudo iniciar la grabación del micrófono.',
      microphoneUnsupported: 'Este runtime no admite grabación de micrófono.',
      noMicrophone: 'No se encontró ningún micrófono.',
      noSpeechDetected: 'No se detectó voz',
      playbackFailed: 'Falló la reproducción de voz',
      recordingFailed: 'Falló la grabación de voz',
      sayStopToEnd: phrase => `Di "${phrase}" para terminar el chat de voz.`,
      transcriptionFailed: 'Falló la transcripción de voz',
      transcriptionUnavailable: 'La transcripción de voz aún no está disponible.',
      tryRecordingAgain: 'Prueba a grabar de nuevo.',
      unavailable: 'Voz no disponible',
      liveEnded: 'Sesión de voz en vivo finalizada',
      liveEndedConnectionLost: 'La sesión de voz en vivo perdió la conexión.',
      liveEndedClosed: 'El servicio cerró la sesión de voz en vivo.',
      liveError: 'Voz en vivo',
      liveDelegationFailed: 'No se pudo pasar la solicitud a Hermes',
      liveUnavailable: reason =>
        `El chat de voz GPT-Live no está disponible: ${reason}. Se usará voz a texto en su lugar.`
    },
    native: {
      approvalTitle: 'Se necesita aprobación',
      approvalTitleNamed: session => `Se necesita aprobación — ${session}`,
      approveAction: 'Aprobar',
      rejectAction: 'Rechazar',
      inputTitle: 'Se necesita información',
      inputTitleNamed: session => `Se necesita una respuesta — ${session}`,
      inputBody: 'Hermes espera tu respuesta.',
      turnDoneTitle: 'Hermes terminó',
      turnDoneBody: '',
      turnErrorTitle: 'El turno falló',
      backgroundDoneTitle: 'Tarea en segundo plano finalizada',
      backgroundFailedTitle: 'La tarea en segundo plano falló',
      creditsTitle: 'Créditos'
    }
  },
  remoteDisplayBanner: {
    message: reason =>
      `Renderizado por software activo — se detectó una pantalla remota (${reason}). Se desactivó la aceleración por GPU para evitar parpadeos.`
  },
  billingBlock: {
    titleNous: 'Sin créditos de Nous',
    titleProvider: provider => `Sin créditos — ${provider}`,
    fallbackMessage: 'Tu cuenta se quedó sin créditos. Añade créditos para continuar.',
    openBilling: 'Abrir facturación',
    addCredits: 'Añadir créditos',
    dismiss: 'Descartar'
  },
  sendDiagnostics: {
    title: 'Enviar diagnóstico a Nous',
    privacyNotice:
      'Esto sube un paquete de depuración a un almacenamiento interno de Nous (no a un sitio público). Incluye información del sistema (SO, versiones, proveedor y qué claves API están configuradas, nunca las claves en sí) y los registros completos del agente, del gateway y de la app de escritorio (hasta 512 KB cada uno), que probablemente contengan contenido de conversaciones, salidas de herramientas y rutas de archivos. Los secretos se ocultan antes de subirlo. Solo el personal de Nous y los moderadores autorizados de Discord pueden ver el paquete, que se elimina automáticamente a los 14 días.',
    upload: 'Subir',
    uploading: 'Subiendo…',
    cancel: 'Cancelar',
    close: 'Cerrar',
    copyLink: 'Copiar enlace',
    uploadIdFallback: id => `No se devolvió un enlace para verlo: indica el ID de subida ${id} al equipo de soporte`,
    doneTitle: 'Diagnóstico enviado',
    doneDescription:
      'Tu paquete se subió de forma privada. Comparte el enlace de abajo en tu hilo de soporte para que el equipo pueda ver tus registros.',
    failedTitle: 'Error al subir',
    failedHint:
      'También puedes ejecutar `hermes debug share --nous` desde una terminal, o `hermes debug share --local` para mostrar el informe sin subirlo.',
    handoffLead: 'Continúa la conversación en:',
    links: {
      github: 'Issues de GitHub',
      portal: 'Soporte de Nous Portal',
      discord: 'Discord'
    }
  },
  titlebar: {
    hideSidebar: 'Ocultar barra lateral',
    showSidebar: 'Mostrar barra lateral',
    search: 'Buscar',
    searchTitle: 'Buscar sesiones, vistas y acciones',
    swapSidebarSides: 'Intercambiar lados de las barras laterales',
    hideRightSidebar: 'Ocultar barra lateral derecha',
    showRightSidebar: 'Mostrar barra lateral derecha',
    unreadSessions: count => (count === 1 ? '1 sesión sin leer' : `${count} sesiones sin leer`),
    muteHaptics: 'Silenciar háptica',
    unmuteHaptics: 'Activar háptica',
    openSettings: 'Abrir configuración',
    openStarmap: 'Abrir grafo de memoria',
    enterHud: 'Modo HUD',
    exitHud: 'Salir del modo HUD',
    resetHudLayout: 'Restablecer el tamaño y la posición del HUD',
    layoutEditor: 'Editor de diseño',
    layoutEditorTitle: mod => `Editor de diseño — clic ${mod} restablece el diseño`
  },
  keybinds: {
    title: 'Atajos de teclado',
    subtitle: open => `Haz clic en un atajo para reasignarlo · ${open} vuelve a abrir este panel.`,
    search: 'Buscar atajos…',
    rebind: 'Reasignar',
    reset: 'Restablecer predeterminado',
    resetAll: 'Restablecer todo',
    pressKey: 'Pulsa una tecla…',
    set: 'definido',
    conflictWith: label => `También asignado a “${label}”`,
    categories: {
      composer: 'Compositor',
      profiles: 'Perfiles',
      session: 'Sesión',
      navigation: 'Navegación',
      view: 'Vista'
    },
    actions: {
      'keybinds.openPanel': 'Abrir atajos de teclado',
      'nav.commandPalette': 'Abrir paleta de comandos',
      'nav.commandCenter': 'Abrir Centro de comandos',
      'nav.settings': 'Abrir configuración',
      'nav.profiles': 'Abrir perfiles',
      'nav.capabilities': 'Abrir skills',
      'nav.messaging': 'Abrir mensajería',
      'nav.artifacts': 'Abrir artefactos',
      'nav.cron': 'Abrir tareas programadas',
      'nav.agents': 'Abrir agentes',
      'session.new': 'Nueva sesión',
      'session.newTab': 'Nueva pestaña de sesión',
      'session.newWindow': 'Nueva ventana',
      'session.next': 'Siguiente sesión',
      'session.prev': 'Sesión anterior',
      'session.slot.1': 'Cambiar a la sesión reciente 1',
      'session.slot.2': 'Cambiar a la sesión reciente 2',
      'session.slot.3': 'Cambiar a la sesión reciente 3',
      'session.slot.4': 'Cambiar a la sesión reciente 4',
      'session.slot.5': 'Cambiar a la sesión reciente 5',
      'session.slot.6': 'Cambiar a la sesión reciente 6',
      'session.slot.7': 'Cambiar a la sesión reciente 7',
      'session.slot.8': 'Cambiar a la sesión reciente 8',
      'session.slot.9': 'Cambiar a la sesión reciente 9',
      'session.focusSearch': 'Buscar sesiones',
      'session.togglePin': 'Fijar / desfijar sesión actual',
      'session.archive': 'Archivar la sesión actual',
      'workspace.newWorktree': 'Nuevo worktree',
      'workspace.openFolder': 'Abrir carpeta como proyecto',
      'composer.focus': 'Enfocar compositor',
      'composer.modelPicker': 'Abrir selector de modelo',
      'composer.voice': 'Iniciar / detener conversación por voz',
      'view.toggleSidebar': 'Alternar barra lateral de sesiones',
      'view.cycleSidebarGrouping': 'Cambiar la agrupación de sesiones',
      'view.toggleRightSidebar': 'Alternar explorador de archivos',
      'view.toggleReview': 'Alternar panel de revisión',
      'view.toggleStatusbar': 'Alternar barra de estado',
      'view.toggleTabStrip': 'Mostrar u ocultar pestañas',
      'view.toggleProfileRail': 'Mostrar u ocultar la barra de perfiles',
      'view.toggleSimpleMode': 'Activar o desactivar el modo simple',
      'view.showFiles': 'Mostrar explorador de archivos',
      'view.showBrowser': 'Abrir el navegador',
      'view.toggleHud': 'Alternar modo HUD',
      'hud.snapToPointer': 'Mover HUD al puntero (global, mientras el HUD esté abierto)',
      'view.showTerminal': 'Mostrar terminal',
      'view.newTerminal': 'Nuevo terminal',
      'view.nextTerminal': 'Siguiente terminal',
      'view.prevTerminal': 'Terminal anterior',
      'view.closeTerminal': 'Cerrar terminal',
      'view.selectionToComposer': 'Enviar la selección al compositor',
      'view.terminalCopy': 'Copiar selección de terminal',
      'view.terminalPaste': 'Pegar en terminal',
      'view.closeTab': 'Cerrar pestaña',
      'view.reopenTab': 'Reabrir pestaña cerrada',
      'view.flipPanes': 'Intercambiar lados de las barras laterales',
      'view.findInPage': 'Buscar en la página',
      'view.findNext': 'Siguiente coincidencia',
      'view.findPrevious': 'Coincidencia anterior',
      'appearance.toggleMode': 'Alternar claro / oscuro',
      'profile.default': 'Cambiar al perfil predeterminado',
      'profile.switch.1': 'Cambiar al perfil 1',
      'profile.switch.2': 'Cambiar al perfil 2',
      'profile.switch.3': 'Cambiar al perfil 3',
      'profile.switch.4': 'Cambiar al perfil 4',
      'profile.switch.5': 'Cambiar al perfil 5',
      'profile.switch.6': 'Cambiar al perfil 6',
      'profile.switch.7': 'Cambiar al perfil 7',
      'profile.switch.8': 'Cambiar al perfil 8',
      'profile.switch.9': 'Cambiar al perfil 9',
      'profile.switch.10': 'Cambiar al perfil 10',
      'profile.switch.11': 'Cambiar al perfil 11',
      'profile.switch.12': 'Cambiar al perfil 12',
      'profile.switch.13': 'Cambiar al perfil 13',
      'profile.switch.14': 'Cambiar al perfil 14',
      'profile.switch.15': 'Cambiar al perfil 15',
      'profile.switch.16': 'Cambiar al perfil 16',
      'profile.switch.17': 'Cambiar al perfil 17',
      'profile.switch.18': 'Cambiar al perfil 18',
      'profile.next': 'Siguiente perfil',
      'profile.prev': 'Perfil anterior',
      'profile.toggleAll': 'Alternar vista de todos los perfiles',
      'profile.create': 'Crear perfil',
      'composer.send': 'Enviar mensaje',
      'composer.newline': 'Insertar salto de línea',
      'composer.steer': 'Guiar el turno en ejecución',
      'composer.queue': 'Poner mensaje en cola',
      'composer.sendQueued': 'Enviar el siguiente turno en cola',
      'composer.mention': 'Referenciar archivos, carpetas y URL',
      'composer.slash': 'Paleta de comandos slash',
      'composer.help': 'Ayuda rápida',
      'composer.history': 'Recorrer popover / historial',
      'composer.cancel': 'Cerrar popover · cancelar ejecución'
    }
  },
  findInPage: {
    next: 'Siguiente coincidencia',
    previous: 'Coincidencia anterior'
  },
  language: {
    label: 'Idioma',
    description: 'Elige el idioma de la interfaz de escritorio.',
    saving: 'Guardando idioma…',
    saveError: 'No se pudo actualizar el idioma',
    switchTo: 'Cambiar idioma',
    searchPlaceholder: 'Buscar idiomas…',
    noResults: 'No se encontraron idiomas'
  },
  settings: {
    subpages: {
      appearanceTheme: 'Tema',
      appearanceTypography: 'Tipografía',
      appearanceWindowLayout: 'Ventana y diseño',
      appearanceChatDisplay: 'Visualización del chat',
      appearancePet: 'Mascota',
      appearanceGeneral: 'General',
      modelMain: 'Modelo principal',
      modelAuxiliary: 'Modelos auxiliares',
      modelMoa: 'Mixture of Agents',
      modelFallbacks: 'Modelos de respaldo',
      chatBehavior: 'Comportamiento',
      chatAttachments: 'Adjuntos',
      workspaceProjects: 'Proyectos y detección',
      workspaceShell: 'Entorno de shell',
      workspaceFiles: 'Archivos y ejecución',
      safetyApprovals: 'Aprobaciones',
      safetyPrivacy: 'Privacidad y red',
      safetyCheckpoints: 'Puntos de control',
      browserProfile: 'Perfil del navegador',
      browserNetwork: 'URL locales y privadas',
      memoryPersistent: 'Memoria persistente',
      memoryContext: 'Contexto y compresión',
      voiceConversation: 'Conversación por voz',
      voiceTranscription: 'Voz a texto',
      voiceSpeech: 'Texto a voz',
      advancedRuntime: 'Límites del agente',
      advancedTools: 'Acceso a herramientas',
      advancedTerminal: 'Backend del terminal',
      advancedOutput: 'Límites de salida',
      advancedDelegation: 'Subagentes',
      advancedDesktop: 'Escritorio e inicio',
      gatewayConnection: 'Esta ventana',
      gatewayDevices: 'Conexiones guardadas',
      gatewayManagedUpdates: 'Actualizaciones remotas',
      gatewayManagedUpdatesUnavailable:
        'Las actualizaciones remotas requieren una versión de escritorio compatible con actualizaciones SSH administradas.',
      gatewayManagedUpdatesEmpty:
        'Añade una conexión SSH en Conexiones guardadas para gestionar sus actualizaciones aquí.',
      keyboardShortcuts: 'Atajos de teclado',
      hudGesture: 'Gesto del HUD',
      screenCapture: 'Captura de pantalla',
      notificationAlerts: 'Alertas de escritorio',
      notificationSounds: 'Sonidos',
      archivedSessions: 'Archivo y retención',
      defaultDirectory: 'Carpeta de proyecto predeterminada',
      vaultCredentials: 'Credenciales guardadas',
      vaultSources: 'Gestores de contraseñas',
      appUpdates: 'Versión y actualizaciones',
      uninstall: 'Desinstalar',
      billingOverview: 'Resumen',
      billingPlans: 'Planes'
    },
    closeSettings: 'Cerrar configuración',
    exportConfig: 'Exportar configuración',
    importConfig: 'Importar configuración',
    resetToDefaults: 'Restablecer valores predeterminados',
    resetConfirm: '¿Restablecer toda la configuración a los valores predeterminados de Hermes?',
    exportFailed: 'Falló la exportación',
    resetFailed: 'Falló el restablecimiento',
    nav: {
      providers: 'Proveedores',
      providerAccounts: 'Cuentas',
      providerApiKeys: 'Claves API',
      providerCustomEndpoints: 'Endpoints personalizados',
      providerLocalModels: 'Modelos locales',
      gateway: 'Gateway',
      apiKeys: 'Herramientas y claves',
      keybinds: 'Atajos de teclado',
      keysTools: 'Herramientas',
      keysSettings: 'Configuración',
      mcp: 'MCP',
      archivedChats: 'Chats archivados',
      sessions: 'Sesiones',
      about: 'Acerca de',
      billing: 'Facturación',
      notifications: 'Notificaciones',
      vault: 'Contraseñas e inicios de sesión'
    },
    plugins: {
      title: 'Plugins de escritorio',
      blurb:
        'Amplía esta app, no un agente: se instala una sola vez para toda la app, sea cual sea el perfil, gateway o equipo al que te conectes. Incluidos o copiados en la carpeta desktop-plugins; los interruptores se aplican al instante.',
      count: n => `${n} instalados`,
      openFolder: 'Abrir la carpeta de plugins de escritorio',
      rescan: 'Volver a buscar',
      reveal: 'Mostrar en el gestor de archivos',
      enable: 'Activar',
      disable: 'Desactivar',
      failed: 'falló',
      empty: 'Aún no hay plugins de escritorio instalados.',
      kinds: {
        bundled: 'incluido',
        disk: 'en disco',
        runtime: 'en ejecución'
      },
      agentHalfMissing: 'falta la parte del agente aquí',
      agentHalfMissingTip:
        'Esta es la parte de escritorio de un plugin incluido, pero su parte del agente no está instalada en el backend o perfil conectado. Instálala desde Capacidades → Plugins.',
      installModal: {
        installFromGit: 'Instalar desde Git',
        reviewRepository: 'Revisar repositorio',
        repoPlaceholder: 'https://github.com/propietario/repo',
        title: 'Instalar plugin',
        description: 'Revisa qué contiene este repositorio antes de instalar nada.',
        repoLabel: 'Repositorio',
        includesHeading: 'Este paquete incluye',
        agentLabel: 'Plugin del agente',
        desktopLabel: 'Interfaz de escritorio',
        profileLabel: 'Instalar para el perfil',
        agentTargetLocal: (profile, dir) => `Se instala en el backend ${profile} (${dir})`,
        agentTargetRemote: profile => `Se instala en el backend ${profile} conectado`,
        catalogPinned: (name, sha) =>
          `Entrada del catálogo de Hermes “${name}”: el componente del agente se instala en la versión fijada revisada${sha ? ` ${sha}` : ''}, no en la punta de la rama.`,
        reviewedHeading: 'Entrada del catálogo revisada',
        reviewedIntro:
          'Una persona revisó esta entrada en su commit fijado. Aun así, puedes inspeccionar el código exacto abajo.',
        toolsConnected: n => (n === 1 ? '1 herramienta conectada' : `${n} herramientas conectadas`),
        skillsReady: names => (names.length === 1 ? `skill ${names[0]} lista` : `${names.length} skills listas`),
        nextChat: 'más herramientas disponibles en tu próximo chat',
        serverNotConnected: (server, reason) =>
          `El servidor MCP ${server} no está conectado${reason ? `: ${reason}` : '.'}`,
        missingEnvAction: 'Configurarlo',
        alreadyInstalled: (name: string) => `${name} ya está instalado.`,
        desktopTarget: 'Se instala en la carpeta local desktop-plugins de esta app',
        desktopTargetFromPackage: 'Se carga en esta app desde el paquete de arriba; es igual para todos los perfiles',
        desktopOnlyNote: 'Los paquetes solo de escritorio no instalan un plugin del agente en el backend.',
        insecureWarning:
          'Esta URL usa un esquema inseguro o local. Para instalaciones de producción, usa https:// o git@.',
        securityHeading: 'Antes de instalar',
        securityIntro:
          'Instala solo desde fuentes de confianza; revisa el repositorio de abajo si quieres ver qué se añadirá.',
        sourceHeading: 'Código fuente',
        viewRepository: 'Ver repositorio',
        viewPluginFiles: 'Ver archivos del plugin',
        gitCloneLabel: 'URL de git clone',
        enableAgent: 'Activar el plugin del agente tras instalarlo',
        forceReinstall: 'Forzar reinstalación (reemplazar si ya está instalado)',
        pinToCommit: 'Fijar a un commit (opcional)',
        pinToCommitPlaceholder: 'SHA de commit completo de 40 caracteres',
        pinToCommitHint:
          'Todos los que instalen este SHA obtienen el mismo código; después, el plugin rechaza actualizaciones hasta que se vuelva a fijar. Déjalo vacío para usar el último commit.',
        pinToCommitInvalid: 'Debe ser un SHA de commit completo de 40 caracteres (no se aceptan ramas ni etiquetas).',
        install: 'Instalar',
        installing: 'Instalando…',
        probing: 'Inspeccionando el repositorio…',
        probeUnavailable: 'La inspección de plugins no está disponible en este entorno.',
        desktopUnavailable: 'La instalación de plugins de escritorio no está disponible en este entorno.',
        selectComponent: 'Selecciona al menos un componente para instalar.',
        agentSuccess: name => `Plugin del agente ${name} instalado`,
        desktopSuccess: name => `Plugin de escritorio ${name} instalado`,
        agentFailed: 'Error al instalar el plugin del agente',
        installUncertain:
          'Hermes dejó de esperar el resultado de la instalación, pero es posible que el plugin siga instalándose. Cierra este cuadro y actualiza la lista de plugins antes de volver a instalarlo.',
        desktopFailed: 'Error al instalar el plugin de escritorio',
        missingEnv: (name, vars) =>
          `${name} está instalado, pero necesita una clave para funcionar: ${vars}. Añádela ahora o las herramientas del plugin fallarán.`
      }
    },
    vault: {
      title: 'Contraseñas e inicios de sesión',
      blurb:
        'Di “inicia sesión en GitHub” y el agente lo hará por ti. La primera vez que encuentre una página de inicio de sesión, te pedirá los datos ahí mismo; después, simplemente funcionará. Las contraseñas se cifran en este equipo y se introducen directamente en la página: el modelo nunca las ve.',
      count: n => `${n} guardado${n === 1 ? '' : 's'}`,
      loadFailed: 'No se pudieron cargar los elementos guardados',
      empty: 'Todavía no hay nada guardado',
      emptyDesc:
        'No necesitas añadir nada aquí. Pide al agente que inicie sesión en un sitio y te pedirá los datos una vez, en ese momento. Usa Añadir si prefieres introducirlos de antemano.',
      add: 'Añadir',
      addTitle: 'Añadir un inicio de sesión, una tarjeta o una dirección',
      addDescription: 'Se guarda cifrado en este equipo. El agente nunca ve la contraseña.',
      added: 'Guardado.',
      adding: 'Guardando…',
      addConfirm: 'Guardar',
      kindField: 'Tipo',
      kinds: {
        login: 'Inicio de sesión',
        payment: 'Tarjeta de pago',
        address: 'Dirección'
      },
      labelField: 'Etiqueta',
      labelPlaceholder: 'p. ej., cuenta de trabajo de GitHub',
      labelRequired: 'La etiqueta es obligatoria.',
      originField: 'Origen del sitio',
      originPlaceholder: 'https://github.com',
      originPlaceholderCheckout: 'https://tienda.example.com',
      originInvalid: 'Introduce una URL válida, como https://example.com.',
      identifierTypeField: 'Tipo de identificador',
      identifierTypes: {
        email: 'Correo electrónico',
        phone: 'Teléfono',
        username: 'Nombre de usuario'
      },
      identifierField: 'Identificador',
      identifierShown: identifier => identifier,
      passwordField: 'Contraseña',
      loginFieldsRequired: 'El identificador y la contraseña son obligatorios.',
      cardNumberField: 'Número de tarjeta',
      cardNameField: 'Nombre en la tarjeta',
      expMonthField: 'Mes de venc.',
      expYearField: 'Año de venc.',
      cvcField: 'CVC',
      postalField: 'Código postal',
      addressLine1Field: 'Dirección, línea 1',
      addressLine2Field: 'Dirección, línea 2',
      cityField: 'Ciudad',
      stateField: 'Estado / región',
      countryField: 'País',
      optional: '(opcional)',
      createdOn: date => `Añadido el ${date}`,
      deleteAction: 'Quitar elemento guardado',
      otpField: 'Clave del autenticador',
      otpPlaceholder: 'Secreto Base32 o enlace otpauth://',
      otpHint:
        'La “clave de configuración” que muestra el sitio al activar la 2FA. Si la guardas, Hermes genera los códigos por sí mismo.',
      twoFactorBadge: '2FA automática',
      deleteTitle: '¿Eliminar este elemento?',
      deleteDescription: label => `Se quitará “${label}”. Esto no se puede deshacer.`,
      deleteConfirm: 'Eliminar',
      sources: {
        title: 'Gestores de contraseñas',
        blurb:
          'Los gestores de contraseñas instalados se detectan automáticamente. El agente te pide desbloquear uno la primera vez que necesita un inicio de sesión de él (una vez por sesión); solo se guarda en memoria un token de sesión, y el agente nunca ve tu contraseña maestra ni ningún inicio de sesión.',
        toggleFailed: 'No se pudo actualizar el gestor de contraseñas',
        notInstalled: name =>
          `No detectado. Instala la herramienta de línea de comandos de ${name} e inicia sesión en ella; Hermes la detectará automáticamente.`,
        disabledDesc: 'Detectado, pero desactivado para Hermes.',
        lockedDesc:
          'Detectado. El agente te pedirá desbloquearlo cuando necesite un inicio de sesión, o puedes desbloquearlo ahora.',
        unlockedDesc:
          'Desbloqueado para esta sesión. Se bloquea automáticamente tras 30 minutos de inactividad o al cerrar Hermes.',
        statusLocked: 'Bloqueado',
        statusNotDetected: 'No detectado',
        statusOff: 'Desactivado',
        statusUnlocked: 'Desbloqueado',
        unlock: 'Desbloquear',
        unlocking: 'Desbloqueando…',
        lock: 'Bloquear',
        unlocked: name => `${name} desbloqueado para esta sesión.`,
        unlockTitle: name => `Desbloquear ${name}`,
        unlockDescription:
          'Introduce tu contraseña maestra. Se entrega al gestor de contraseñas de este equipo y se descarta: nunca se guarda, se registra ni se muestra al agente.',
        masterPasswordPlaceholder: 'Contraseña maestra'
      }
    },
    notifications: {
      title: 'Notificaciones',
      intro: 'Notificaciones del sistema operativo (no avisos dentro de la app). Por dispositivo.',
      enableAll: 'Activar notificaciones',
      enableAllDesc: 'Desactivado silencia todas las notificaciones siguientes.',
      focusedHint: 'Los avisos de finalización solo se activan cuando Hermes está en segundo plano.',
      kinds: {
        approval: {
          label: 'Se necesita aprobación',
          description: 'Un comando espera que lo apruebes o rechaces.'
        },
        input: {
          label: 'Se necesita información',
          description: 'Hermes hizo una pregunta o necesita una contraseña o un secreto.'
        },
        turnDone: {
          label: 'Respuesta lista',
          description: 'Terminó un turno mientras Hermes estaba en segundo plano.'
        },
        turnError: {
          label: 'El turno falló',
          description: 'Errores de turnos en segundo plano.'
        },
        backgroundDone: {
          label: 'Tarea en segundo plano finalizada',
          description: 'Se completó un comando de terminal en segundo plano.'
        },
        credits: {
          label: 'Avisos de crédito',
          description: 'El acceso a los créditos se pausa o se restablece.'
        },
        plugin: {
          label: 'Notificaciones de complementos',
          description: 'Un complemento de escritorio envió una notificación mientras Hermes estaba en segundo plano.'
        }
      },
      test: 'Enviar notificación de prueba',
      testTitle: 'Hermes',
      testBody: 'Las notificaciones funcionan.',
      testSent:
        'Prueba enviada. Si no aparece nada, revisa los permisos de notificaciones del sistema operativo y el modo Concentración o No molestar.',
      testUnsupported: 'Este sistema no admite notificaciones nativas.',
      completionSoundTitle: 'Sonido de finalización',
      completionSoundDesc:
        'Reproduce un sonido cuando termina el turno de un agente. Elige un ajuste predefinido y pruébalo aquí.',
      completionSoundPreview: 'Vista previa'
    },
    sections: {
      model: 'Modelo',
      chat: 'Chat',
      appearance: 'Apariencia',
      workspace: 'Espacio de trabajo',
      safety: 'Seguridad',
      memory: 'Memoria y contexto',
      voice: 'Voz',
      advanced: 'Avanzado'
    },
    searchPlaceholder: {
      about: 'Acerca de Hermes Desktop',
      config: 'Buscar configuración...',
      gateway: 'Conexión del gateway...',
      keys: 'Buscar claves API...',
      mcp: 'Buscar servidores MCP...',
      sessions: 'Buscar sesiones archivadas...'
    },
    modeOptions: {
      light: {
        label: 'Claro',
        description: 'Superficies de escritorio luminosas'
      },
      dark: {
        label: 'Oscuro',
        description: 'Espacio de trabajo con menos brillo'
      },
      system: {
        label: 'Sistema',
        description: 'Seguir la apariencia del SO'
      }
    },
    appearance: {
      title: 'Apariencia',
      intro: 'Solo escritorio. El modo es el brillo; el tema es la paleta y el marco del chat.',
      colorMode: 'Modo de color',
      colorModeDesc: 'Elige un modo fijo o deja que Hermes siga la configuración del sistema.',
      toolViewTitle: 'Visualización de llamadas a herramientas',
      toolViewDesc: 'Producto oculta las cargas útiles sin procesar; Técnico muestra entrada/salida completas.',
      hideCodeDiffsTitle: 'Ocultar diffs de código',
      hideCodeDiffsDesc:
        'Muestra las ediciones de archivos como filas de herramienta con el recuento de líneas añadidas/eliminadas, sin el código.',
      hideThreadTimelineTitle: 'Ocultar las barras de la línea de tiempo',
      hideThreadTimelineDesc: 'Oculta las barras de navegación del borde derecho de cada conversación.',
      reasoningCollapsedTitle: 'Contraer el razonamiento por defecto',
      reasoningCollapsedDesc: 'Mantiene disponible el razonamiento transmitido sin expandirlo hasta que lo abras.',
      uiScaleTitle: 'Escala de la interfaz',
      uiScaleDesc: percent =>
        `Escala el texto y los controles de toda la app. También funciona Cmd/Ctrl con +, - y 0. Actual: ${percent}%.`,
      sessionDensityTitle: 'Densidad de la lista de sesiones',
      sessionDensityDesc: 'Elige cuánto contexto aparece bajo los títulos de las sesiones en la barra lateral.',
      sessionDensityCompact: 'Compacta',
      sessionDensityComfortable: 'Cómoda',
      sessionDensityDetailed: 'Detallada',
      tabStripTitle: 'Barra de pestañas',
      tabStripDesc:
        'Muestra pestañas encima de una zona. Auto las oculta si hay un solo panel, salvo que haya otra zona de chat o de mosaico abierta.',
      tabStripAuto: 'Auto',
      tabStripAlways: 'Siempre',
      tabStripNever: 'Nunca',
      appActionsTitle: 'Acciones de la app',
      appActionsDesc:
        'Dónde se colocan Configuración, Diseño y HUD en la barra de título. A la derecha deja espacio para pestañas a la izquierda.',
      appActionsLeft: 'Izquierda',
      appActionsRight: 'Derecha',
      terminalFontTitle: 'Fuente de terminal',
      terminalFontDesc:
        'Elige una fuente instalada para las terminales del escritorio. Las Nerd Fonts renderizan Powerlevel10k y los iconos de shell; déjalo vacío para usar la JetBrains Mono incluida.',
      terminalFontPlaceholder: 'MesloLGS NF o una pila de fuentes CSS',
      terminalFontPreview: 'Vista previa de glifos',
      terminalFontReset: 'Usar la predeterminada',
      chatFontTitle: 'Fuente del chat',
      chatFontDesc:
        'Elige una fuente instalada para el chat y el resto de la app. Útil para fuentes de lectura como OpenDyslexic; déjalo en blanco para usar la fuente del tema.',
      chatFontPlaceholder: 'OpenDyslexic o una pila de fuentes CSS',
      chatFontPreview: 'Vista previa',
      chatFontSample: 'El veloz murciélago hindú comía feliz cardillo y kiwi. 0123456789',
      chatFontReset: 'Usar la fuente del tema',
      translucencyTitle: 'Translucidez de la ventana',
      translucencyDesc: 'Verás tu escritorio a través de toda la ventana. Solo macOS y Windows.',
      translucencyGlassDesc:
        'Vidrio mate: el escritorio se ve a través con un desenfoque suave mientras el texto se mantiene nítido. Ajustado por separado para claro y oscuro.',
      translucencyModeClear: 'Transparente',
      translucencyModeGlass: 'Vidrio',
      translucencyTintTitle: 'Tinte',
      translucencyFadeTitle: 'Atenuación',
      translucencyFrostTitle: 'Escarcha',
      translucencyFrost: {
        'under-window': 'Profunda',
        popover: 'Suave',
        titlebar: 'Brillante',
        header: 'Resplandor'
      },
      translucencyScopeTitle: 'Área',
      translucencyScope: {
        window: 'Toda la ventana',
        sidebar: 'Solo la barra lateral'
      },
      backdropTitle: 'Fondo del chat',
      backdropDesc: 'La tenue imagen de la estatua detrás de la conversación.',
      userBubbleTitle: 'Burbuja de mensaje',
      userBubbleDesc: 'Cuánta transparencia tienen tus propios mensajes. Opaca en 0; en 100 solo queda el contorno.',
      textDirectionTitle: 'Dirección del texto',
      textDirectionDesc:
        'Cómo eligen su dirección los mensajes del chat y el campo de escritura. Auto sigue la primera letra de cada párrafo; elige una dirección cuando un texto mixto se alinee mal. El código siempre va de izquierda a derecha.',
      textDirection: { auto: 'Auto', rtl: 'De derecha a izquierda', ltr: 'De izquierda a derecha' },
      introSplashTitle: 'Pantalla de bienvenida',
      introSplashDesc: 'El logotipo y la indicación que se muestran en un chat vacío.',
      reactionsTitle: 'Reacciones a mensajes',
      reactionsDesc:
        'Reacciones emoji estilo iMessage — reacciona a los mensajes, y Hermes puede reaccionar a los tuyos.',
      tipsTitle: 'Consejos en la app',
      tipsDesc:
        'Sugerencias ocasionales de la app y de Hermes. Cada consejo aparece una vez. Se desactiva automáticamente tras tus primeros 30 días; puedes volver a activarlo.',
      tipsReset: (count: number) => `Volver a mostrar ${count} ${count === 1 ? 'consejo' : 'consejos'}`,
      toursTitle: 'Recorridos guiados',
      toursDesc:
        'Deja que Hermes resalte cada paso mientras te guía por la app. Se desactiva automáticamente tras tus primeros 30 días; puedes volver a activarlo.',
      composerPopoutTitle: 'Compositor flotante',
      composerPopoutDesc:
        'Permite arrastrar el compositor fuera de su posición fija. Desactívalo para mantenerlo anclado abajo.',
      vibeHeartsTitle: 'Corazones de vibra',
      vibeHeartsDesc:
        'Corazones flotantes cuando dices gracias, te quiero, buen bot o envías un corazón. Independiente de las reacciones a mensajes de arriba.',
      embedsTitle: 'Contenido incrustado',
      embedsDesc:
        'Las vistas previas enriquecidas se cargan desde sitios de terceros (YouTube, X, …). Preguntar muestra un marcador de posición hasta que permitas cada una; Siempre las carga automáticamente; Desactivado conserva los enlaces simples.',
      embedsAsk: 'Preguntar',
      embedsAlways: 'Siempre',
      embedsOff: 'Desactivado',
      embedsReset: count => `Restablecer ${count} ${count === 1 ? 'servicio permitido' : 'servicios permitidos'}`,
      resumeLastSessionTitle: 'Reabrir el último chat al iniciar',
      resumeLastSessionDesc:
        'Si está activado, la app reabre tu chat más reciente al iniciarse en frío. Desactívalo para empezar siempre con un chat nuevo.',
      product: 'Producto',
      productDesc: 'Actividad de herramientas legible con resúmenes concisos.',
      technical: 'Técnico',
      technicalDesc: 'Incluye argumentos/resultados sin procesar y detalles de bajo nivel.',
      themeTitle: 'Tema',
      themeDesc: 'Paletas solo para escritorio. Se aplican sobre el modo seleccionado.',
      themeSearchPlaceholder: 'Busca en tus temas o en el VS Code Marketplace…',
      themeProfileNote: profile => `Guardado para el perfil ${profile}; cada perfil conserva su propio tema.`,
      installTitle: 'Instalar desde VS Code',
      installDesc:
        'Pega un ID de extensión de Marketplace (por ejemplo, dracula-theme.theme-dracula) para convertir su tema de color en una paleta de escritorio.',
      installPlaceholder: 'publisher.extension',
      installButton: 'Instalar',
      installing: 'Instalando…',
      installError: 'No se pudo instalar ese tema.',
      installed: name => `Instalado “${name}”.`,
      removeTheme: 'Quitar tema',
      importedBadge: 'Importado',
      pet: {
        title: 'Mascota',
        intro:
          'Adopta una mascota animada de petdex que flota sobre la app y reacciona a lo que hace Hermes: corre mientras se ejecutan herramientas, celebra los éxitos y se entristece con los errores.',
        restartHint:
          'Las mascotas necesitan un reinicio rápido: la aplicación en ejecución se inició antes de que se añadiera esta función. Cierra y vuelve a abrir Hermes y luego vuelve aquí.',
        scaleTitle: 'Tamaño',
        scaleDesc: 'Cambia el tamaño de la mascota flotante. Se aplica al instante en todas partes.',
        roamTitle: 'Moverse libremente',
        roamDesc: 'Permite que la mascota recorra la ventana por su cuenta cuando esté inactiva.',
        chooseTitle: 'Elige una mascota',
        chooseDesc: 'Al elegir una, se instala si es necesario y queda activa.',
        searchPlaceholder: 'Buscar mascotas…',
        unreachable: 'No se pudo acceder a la galería de petdex. Revisa tu conexión y vuelve a abrir esta página.',
        noMatch: query => `No hay mascotas que coincidan con “${query}”.`,
        installedTag: 'instalada',
        generatedTag: 'Generada',
        countCapped: (cap, total) => `Se muestran ${cap} de ${total}; escribe para acotar la lista.`,
        count: n => `${n} ${n === 1 ? 'mascota' : 'mascotas'}.`,
        uninstall: name => `Desinstalar ${name}`,
        delete: name => `Eliminar ${name}`,
        deleteTitle: name => `¿Eliminar ${name}?`,
        deleteBody: 'Esto elimina la mascota permanentemente; no se podrá reinstalar.',
        deleteConfirm: 'Eliminar',
        rename: name => `Cambiar nombre de ${name}`,
        renameTitle: 'Cambiar nombre de mascota',
        renamePlaceholder: 'Nombra a tu mascota',
        renameSave: 'Guardar',
        exportPet: name => `Exportar ${name}`,
        adoptFailed: slug => `No se pudo adoptar ${slug}`,
        uninstallFailed: slug => `No se pudo desinstalar ${slug}`,
        renameFailed: slug => `No se pudo cambiar el nombre de ${slug}`,
        exportFailed: slug => `No se pudo exportar ${slug}`,
        noneAvailable: 'No hay mascotas disponibles para activar ahora mismo.',
        turnOnFailed: 'No se pudo activar la mascota.',
        turnOffFailed: 'No se pudo desactivar la mascota.'
      }
    },
    fieldLabels: defineFieldCopy({
      model: 'Modelo predeterminado',
      modelContextLength: 'Ventana de contexto del modelo principal (forzada)',
      fallbackProviders: 'Modelos de respaldo',
      toolsets: 'Conjuntos de herramientas activados',
      timezone: 'Zona horaria',
      display: {
        personality: 'Personalidad',
        showReasoning: 'Bloques de razonamiento'
      },
      desktop: {
        repoScanEnabled: 'Detección automática de repositorios',
        repoScanRoots: 'Carpetas de búsqueda de repositorios',
        repoScanExcludePaths: 'Rutas de repositorio excluidas'
      },
      agent: {
        maxTurns: 'Pasos máximos del agente',
        imageInputMode: 'Adjuntos de imagen',
        apiMaxRetries: 'Reintentos de API',
        serviceTier: 'Nivel de servicio',
        toolUseEnforcement: 'Aplicación de uso de herramientas'
      },
      terminal: {
        cwd: 'Directorio de trabajo',
        backend: 'Backend de ejecución',
        timeout: 'Tiempo límite de comandos',
        persistentShell: 'Shell persistente',
        envPassthrough: 'Paso de variables de entorno',
        dockerImage: 'Imagen de Docker',
        singularityImage: 'Imagen de Singularity',
        modalImage: 'Imagen de Modal',
        daytonaImage: 'Imagen de Daytona'
      },
      fileReadMaxChars: 'Límite de lectura de archivos',
      toolOutput: {
        maxBytes: 'Límite de salida del terminal',
        maxLines: 'Límite de páginas de archivo',
        maxLineLength: 'Límite de longitud de línea'
      },
      codeExecution: {
        mode: 'Modo de ejecución de código'
      },
      approvals: {
        mode: 'Modo de aprobación',
        timeout: 'Tiempo límite de aprobación',
        mcpReloadConfirm: 'Confirmar recargas de MCP'
      },
      commandAllowlist: 'Lista de comandos permitidos',
      security: {
        redactSecrets: 'Redactar secretos',
        allowPrivateUrls: 'Permitir URL privadas'
      },
      browser: {
        allowPrivateUrls: 'URL privadas del navegador',
        autoLocalForPrivateUrls: 'Navegador local para URL privadas',
        useRealProfile: 'Usar mi perfil real del navegador'
      },
      checkpoints: {
        enabled: 'Checkpoints de archivos',
        maxSnapshots: 'Límite de checkpoints'
      },
      voice: {
        maxRecordingSeconds: 'Duración máxima de grabación',
        autoTts: 'Leer respuestas en voz alta',
        voiceChatMode: 'Modo de chat de voz',
        gptLive: {
          voice: 'Voz de GPT-Live',
          instructions: 'Personalidad de GPT-Live'
        }
      },
      stt: {
        enabled: 'Voz a texto',
        echoTranscripts: 'Eco de transcripciones',
        provider: 'Proveedor de voz a texto',
        local: {
          model: 'Modelo local de transcripción',
          language: 'Idioma de transcripción'
        },
        openai: {
          model: 'Modelo STT de OpenAI'
        },
        groq: {
          model: 'Modelo STT de Groq'
        },
        mistral: {
          model: 'Modelo STT de Mistral'
        },
        elevenlabs: {
          modelId: 'Modelo STT de ElevenLabs',
          languageCode: 'Idioma de ElevenLabs',
          tagAudioEvents: 'Etiquetar eventos de audio',
          diarize: 'Diarización de hablantes'
        }
      },
      tts: {
        provider: 'Proveedor de texto a voz',
        edge: {
          voice: 'Voz de Edge'
        },
        openai: {
          model: 'Modelo TTS de OpenAI',
          voice: 'Voz de OpenAI'
        },
        elevenlabs: {
          voiceId: 'Voz de ElevenLabs',
          modelId: 'Modelo de ElevenLabs'
        },
        xai: {
          voiceId: 'Voz de xAI (Grok)',
          language: 'Idioma de xAI',
          speed: 'Velocidad de reproducción de xAI',
          autoSpeechTags: 'Etiquetas de voz automáticas de xAI',
          optimizeStreamingLatency: 'Optimización de latencia de streaming de xAI',
          sampleRate: 'Frecuencia de muestreo de xAI',
          bitRate: 'Tasa de bits de xAI'
        },
        minimax: {
          model: 'Modelo TTS de MiniMax',
          voiceId: 'Voz de MiniMax'
        },
        mistral: {
          model: 'Modelo TTS de Mistral',
          voiceId: 'Voz de Mistral'
        },
        gemini: {
          model: 'Modelo TTS de Gemini',
          voice: 'Voz de Gemini'
        },
        neutts: {
          model: 'Modelo de NeuTTS',
          device: 'Dispositivo de NeuTTS'
        },
        kittentts: {
          model: 'Modelo de KittenTTS',
          voice: 'Voz de KittenTTS'
        },
        piper: {
          voice: 'Voz de Piper'
        },
        deepinfra: {
          model: 'Modelo TTS de DeepInfra',
          voice: 'Voz de DeepInfra'
        }
      },
      memory: {
        memoryEnabled: 'Memoria persistente',
        userProfileEnabled: 'Perfil de usuario',
        memoryCharLimit: 'Presupuesto de memoria',
        userCharLimit: 'Presupuesto de perfil',
        provider: 'Proveedor de memoria'
      },
      context: {
        engine: 'Motor de contexto'
      },
      compression: {
        enabled: 'Compresión automática',
        threshold: 'Umbral de compresión',
        codexGpt55Autoraise: 'Aumento automático de compresión de Codex',
        targetRatio: 'Objetivo de compresión',
        protectLastN: 'Mensajes recientes protegidos'
      },
      auxiliary: {
        compression: {
          timeout: 'Tiempo de espera del modelo de compresión (s)'
        }
      },
      delegation: {
        model: 'Modelo de subagente',
        provider: 'Proveedor de subagente',
        maxIterations: 'Límite de turnos del subagente',
        maxConcurrentChildren: 'Subagentes paralelos',
        childTimeoutSeconds: 'Tiempo límite del subagente',
        reasoningEffort: 'Esfuerzo de razonamiento del subagente'
      },
      updates: {
        nonInteractiveLocalChanges: 'Cambios locales en actualización desde la app'
      }
    }),
    fieldDescriptions: defineFieldCopy({
      model: 'Se usa para chats nuevos salvo que elijas otro modelo en el compositor.',
      modelContextLength:
        'Sustituye la ventana de contexto detectada solo del modelo de chat PRINCIPAL (en tokens). Déjalo en 0 para usar el valor detectado del modelo seleccionado. No afecta a los modelos auxiliares/MoA.',
      fallbackProviders: 'Entradas proveedor:model de respaldo para probar si falla el modelo predeterminado.',
      display: {
        personality: 'Estilo predeterminado del asistente para sesiones nuevas.',
        showReasoning: 'Muestra secciones de razonamiento cuando el backend las proporcione.'
      },
      desktop: {
        repoScanEnabled: 'Busca repositorios Git en carpetas locales para mostrarlos en Proyectos.',
        repoScanRoots: 'Carpetas que se buscarán. Déjalo vacío para buscar en tu directorio de inicio.',
        repoScanExcludePaths:
          'Carpetas que se omitirán, junto con todos sus subdirectorios, durante la detección de repositorios.'
      },
      timezone: 'Identificador de zona horaria IANA. Vacío usa la zona horaria del sistema.',
      browser: {
        useRealProfile:
          'La navegación local usa tus inicios de sesión reales. Hermes copia el perfil de tu navegador predeterminado (cookies, inicios de sesión, preferencias) en una instantánea gestionada y lo controla con su Chromium integrado: tu perfil activo nunca se abre directamente y la copia se actualiza a partir de él en cada ejecución. También permite que el agente abra bajo petición una sesión local con tu perfil real, incluso si hay un backend de navegador en la nube configurado. Solo se admiten navegadores Chromium (Chrome, Edge, Brave, Brave Origin, Chromium); un navegador predeterminado que no sea Chromium falla con un mensaje claro. Desactivado de forma predeterminada.'
      },
      agent: {
        imageInputMode: 'Controla cómo se envían los adjuntos de imagen al modelo.',
        maxTurns: 'Límite superior de turnos con llamadas a herramientas antes de que Hermes detenga una ejecución.'
      },
      terminal: {
        cwd: 'Carpeta de proyecto predeterminada para herramientas y terminal.',
        persistentShell: 'Mantiene el estado de la shell entre comandos cuando el backend lo admite.',
        envPassthrough: 'Variables de entorno que se pasan a la ejecución de herramientas.',
        dockerImage: 'Imagen de contenedor usada cuando el backend de ejecución es Docker.',
        singularityImage: 'Imagen usada cuando el backend de ejecución es Singularity.',
        modalImage: 'Imagen usada cuando el backend de ejecución es Modal.',
        daytonaImage: 'Imagen usada cuando el backend de ejecución es Daytona.'
      },
      codeExecution: {
        mode: 'Qué tan estrictamente se limita la ejecución de código al proyecto actual.'
      },
      fileReadMaxChars: 'Máximo de caracteres que Hermes puede leer en una solicitud de archivo.',
      approvals: {
        mode: 'Cómo maneja Hermes los comandos que necesitan aprobación explícita.',
        timeout: 'Cuánto esperan los prompts de aprobación antes de vencer.'
      },
      security: {
        redactSecrets: 'Oculta secretos detectados del contenido visible para el modelo cuando sea posible.'
      },
      checkpoints: {
        enabled: 'Crea snapshots de reversión antes de editar archivos.'
      },
      memory: {
        memoryEnabled: 'Guarda memorias duraderas que pueden ayudar en sesiones futuras.',
        userProfileEnabled: 'Mantiene un perfil compacto de preferencias del usuario.'
      },
      context: {
        engine: 'Estrategia para gestionar conversaciones largas cerca del límite de contexto.'
      },
      compression: {
        enabled: 'Resume contexto antiguo cuando las conversaciones crecen.',
        codexGpt55Autoraise: 'Sube la compresión al 85 % para los modelos compatibles de ChatGPT Codex OAuth.'
      },
      auxiliary: {
        compression: {
          timeout:
            'Segundos de espera del modelo auxiliar de compresión por llamada (120 por defecto). Súbelo para modelos locales lentos.'
        }
      },
      voice: {
        autoTts: 'Lee automáticamente en voz alta las respuestas del asistente.',
        voiceChatMode:
          'chained: voz a texto → Hermes → texto a voz con los proveedores de abajo. gpt-live: un modelo de voz full-duplex de OpenAI (gpt-live-1) escucha y habla, y pasa cada solicitud real a Hermes; el modelo que hayas seleccionado responde con todas las herramientas. Requiere una clave API de OpenAI; la capa de voz cuesta 0,05 US$ por minuto.',
        gptLive: {
          voice: 'Voz del modo GPT-Live. Se aceptan ID de voz personalizados.',
          instructions:
            'Frases adicionales para la personalidad de voz en vivo (tono, ritmo, idioma). Hermes mantiene su propio prompt de sistema.'
        }
      },
      tts: {
        xai: {
          voiceId: 'ID de voz de xAI (por ejemplo, eve) o un ID de voz personalizado.',
          language: 'Código del idioma hablado (p. ej., en, pt-BR) o "auto" para detectarlo automáticamente.',
          speed: 'Velocidad de reproducción. 0.7 = más lento, 1.0 = normal, 1.5 = más rápido.',
          autoSpeechTags:
            'Permite que un LLM inserte etiquetas de audio expresivas ([laughing], [sighs]) en el guion antes de sintetizarlo.',
          optimizeStreamingLatency: 'Equilibrio entre latencia y calidad. 0 = mejor calidad, 2 = menor latencia.',
          sampleRate:
            'Frecuencia de muestreo del audio en Hz. Un valor mayor ofrece más calidad y archivos más grandes.',
          bitRate: 'Tasa de bits de MP3 en bps. Solo se aplica cuando el códec es mp3.'
        },
        neutts: {
          device: 'Dispositivo de inferencia local para NeuTTS.'
        }
      },
      stt: {
        enabled: 'Activa transcripción de voz local o respaldada por proveedor.',
        echoTranscripts: 'Publica la transcripción sin procesar 🎙️ de los mensajes de voz en el chat.',
        elevenlabs: {
          languageCode: 'Código de idioma ISO-639-3 opcional. En blanco deja que ElevenLabs lo detecte automáticamente.'
        }
      },
      updates: {
        nonInteractiveLocalChanges:
          'Cuando Hermes se actualiza desde la app sin prompt de terminal, conserva los cambios locales de código fuente (stash) o descártalos. Las actualizaciones desde terminal siempre preguntan.'
      }
    }),
    uninstallSection: {
      dangerZone: 'Zona de peligro',
      checkingInstalled: 'Comprobando lo que está instalado…',
      uninstallHermes: 'Desinstalar Hermes',
      chooseHowMuch:
        'Elige cuánto quieres quitar. La app se cierra para terminar; vuelve a abrir el instalador cuando quieras para volver.',
      confirmUninstall: 'Confirmar desinstalación',
      confirmBody: what => `Esto quita ${what}. No se puede deshacer.`,
      appLabel: 'App:',
      couldNotStart: 'No se pudo iniciar la desinstalación.',
      uninstalling: 'Desinstalando…',
      yesUninstall: 'Sí, desinstalar',
      options: {
        gui: {
          title: 'Desinstalar solo la interfaz de chat',
          description: 'Quita esta app de escritorio. El agente de Hermes, tu configuración y tus chats se conservan.',
          consequence: 'la interfaz de chat de escritorio (esta app y sus datos)'
        },
        lite: {
          title: 'Desinstalar la interfaz y el agente, conservar mis datos',
          description:
            'Quita la app y el agente de Hermes, pero conserva la configuración, los chats y los secretos para una futura reinstalación.',
          consequence:
            'la interfaz de chat y el agente de Hermes (se conservan la configuración, los chats y los secretos)'
        },
        full: {
          title: 'Desinstalar todo',
          description:
            'Quita la app, el agente y todos los datos de usuario: configuración, chats, tareas programadas, secretos y registros.',
          consequence:
            'TODO: la interfaz de chat, el agente de Hermes y toda tu configuración, chats, secretos y registros'
        }
      }
    },
    poolLimits: {
      warmBotBackendsAria: 'Backends de bots en caliente',
      warmBotBackendsTitle: 'Backends de bots en caliente',
      backendIdleTimeoutAria: 'Tiempo de inactividad del backend en milisegundos',
      backendIdleTimeoutTitle: 'Tiempo de inactividad del backend'
    },
    customEndpoints: {
      active: 'Activo',
      apiKeySet: 'Clave API configurada',
      use: 'Usar',
      editTitle: 'Editar endpoint',
      addTitle: 'Añadir endpoint',
      fields: {
        name: 'Nombre',
        providerId: 'ID del proveedor',
        endpointUrl: 'URL del endpoint',
        defaultModel: 'Modelo predeterminado',
        context: 'Contexto',
        apiKey: 'Clave API',
        apiKeyNewPlaceholder: 'Déjalo en blanco para conservar la clave actual',
        apiKeyPlaceholder: 'Opcional',
        useNewChats: 'Usar en los chats nuevos',
        discoverModels: 'Descubrir modelos'
      },
      test: 'Probar',
      save: 'Guardar',
      newEndpoint: 'Nuevo endpoint',
      apiMode: 'Modo de API',
      autoDetect: 'Detección automática',
      couldNotLoad: 'No se pudieron cargar los endpoints personalizados',
      endpointSaved: 'Endpoint personalizado guardado.',
      saveFailed: 'Error al guardar',
      endpointReachable: 'El endpoint es accesible.',
      endpointReachableTransport: transport => `El endpoint es accesible (ruta ${transport} atendida).`,
      endpointReachableModels: (reachable, count) => `${reachable} Se encontraron ${count} modelos.`,
      endpointValidationFailed: 'Falló la validación del endpoint.',
      validationFailed: 'Falló la validación',
      activationFailed: 'Falló la activación',
      deleteConfirm: name => `¿Eliminar ${name}?`,
      deleteFailed: 'Error al eliminar',
      title: 'Endpoints personalizados',
      deleteEndpoint: 'Eliminar endpoint',
      emptyDescription: 'Añade abajo un endpoint compatible con OpenAI.',
      emptyTitle: 'No hay endpoints personalizados',
      namePlaceholder: 'Axet Proxy',
      contextPlaceholder: 'Auto'
    },
    computerUse: {
      accessibility: 'Accesibilidad',
      screenRecording: 'Grabación de pantalla',
      driverHealth: 'Estado del controlador'
    },
    about: {
      updates: 'Actualizaciones'
    },
    config: {
      minimizeToTrayTitle: 'Minimizar a la bandeja',
      minimizeToTrayDesc:
        'Al minimizar las ventanas o cerrar la ventana principal, se ocultan en la bandeja del sistema (barra de menús en macOS) y Hermes sigue ejecutándose. Usa Salir de Hermes en el menú de la bandeja o Cmd+Q para salir. Desactivado por defecto; se aplica solo a este dispositivo.',
      minimizeToTrayUnavailable:
        'La bandeja del sistema no está disponible. Las ventanas se minimizarán y cerrarán con normalidad. Desactiva y vuelve a activar esta opción para reintentarlo.',
      none: 'Ninguno',
      noneParen: '(ninguno)',
      builtinOnly: 'Solo integradas',
      notSet: 'Sin definir',
      commaSeparated: 'valores separados por comas',
      searchPlaceholder: 'Buscar…',
      noResults: 'No se encontraron resultados',
      systemDefault: 'Valor del sistema',
      loading: 'Cargando configuración de Hermes...',
      emptyTitle: 'Nada que configurar',
      emptyDesc: 'Esta sección no tiene ajustes configurables.',
      failedLoad: 'No se pudo cargar la configuración',
      autosaveFailed: 'Falló el autoguardado',
      imported: 'Configuración importada',
      invalidJson: 'JSON de configuración no válido',
      toolsetsWipeConfirm:
        '¿Quitar todos los conjuntos de herramientas activados? Esto desactiva la memoria, el terminal, la búsqueda web, la delegación y la mayoría de las demás herramientas hasta que los vuelvas a activar.',
      keepAwakeTitle: 'Mantener el equipo activo',
      keepAwakeDesc:
        'Impide que este equipo entre en reposo para que las ejecuciones largas o nocturnas continúen. La pantalla puede seguir atenuándose.',
      disableF12Title: 'Desactivar DevTools con F12',
      disableF12Desc:
        'Impide que F12 abra las herramientas para desarrolladores. Ctrl+Shift+I (o Cmd+Opt+I en Mac) sigue funcionando.',
      attachmentSizeTitle: 'Tamaño máximo de vista previa / carga de imagen',
      attachmentSizeDesc:
        'Tamaño máximo de archivo local que el escritorio cargará para vistas previas y adjuntos de imagen, en MB. El valor por defecto es 16. Los adjuntos remotos no-imagen usan un límite separado de 256 MB. Un valor muy alto carga el archivo completo en memoria y puede congelar o bloquear la app.',
      attachmentSizeUnit: 'MB',
      attachmentSizeLabel: 'Tamaño máximo de vista previa / carga de imagen en megabytes',
      showOptions: 'Mostrar opciones'
    },
    hudModifier: {
      title: 'Pulsar para mostrar el HUD',
      description:
        'Pulsa y suelta ⌘ + Opción en Mac, o Ctrl + Alt en Windows/Linux, para traer el HUD al frente desde cualquier app. Desactivado por defecto; se aplica solo a este dispositivo.',
      permission:
        'Permite Hermes en Ajustes del Sistema → Privacidad y seguridad → Monitorización de entrada y vuelve a intentarlo. Este gesto no registra pulsaciones de teclas ni captura tu pantalla.',
      unavailable:
        'El asistente del gesto del HUD no pudo iniciarse o se detuvo de forma inesperada. Reinténtalo o reinicia Hermes. El atajo del HUD existente sigue funcionando dentro de Hermes.',
      missingHelper:
        'A esta instalación de Hermes le falta el asistente del gesto del HUD. Actualiza o reinstala Hermes y vuelve a intentarlo.',
      unsupportedSession:
        'Esta sesión de escritorio no admite pulsaciones globales de teclas modificadoras. Linux requiere X11; Wayland no es compatible.'
    },
    screenshot: {
      enabledTitle: 'Atajo de captura de pantalla',
      enabledDesc:
        'Pulsa las dos teclas Comando a la vez desde cualquier app para capturar su ventana frontal y adjuntarla a tu borrador actual de Hermes. Nunca se envía automáticamente. Desactivado por defecto; se aplica solo a este Mac. El contenido de la ventana puede ser confidencial: revisa el adjunto antes de enviarlo.',
      statusTitle: 'Estado del atajo de captura',
      checking: 'Comprobando el atajo de captura…',
      disabled: 'El atajo de captura está desactivado.',
      starting: 'Iniciando la escucha del atajo. Todavía no está listo.',
      ready: 'El atajo está listo. Las capturas se adjuntan a tu borrador actual sin enviarse.',
      inputPermission:
        'El permiso de Monitorización de entrada permite a Hermes detectar las dos teclas Comando mientras otra app está activa. Permite Hermes en Ajustes del Sistema → Privacidad y seguridad → Monitorización de entrada, vuelve aquí y reinténtalo.',
      screenPermission:
        'El permiso de Grabación de pantalla permite a Hermes capturar la ventana frontal cuando usas este atajo. Permite Hermes en Ajustes del Sistema → Privacidad y seguridad → Grabación de pantalla, vuelve aquí y reinténtalo. Reinicia Hermes si macOS te lo pide.',
      openSettings: 'Abrir Ajustes del Sistema',
      retry: 'Reintentar',
      unavailable: 'El atajo de captura no está disponible. Reinténtalo o desactívalo.',
      errorTitle: 'Error del atajo de captura',
      loadFailed: 'No se pudo leer el estado del atajo. Reinténtalo para comprobar su ajuste actual.',
      saveFailed: 'No se pudo confirmar el cambio del atajo. Reinténtalo para comprobar su ajuste actual.',
      permissionFailed:
        'No se pudieron abrir los Ajustes del Sistema. Abre Privacidad y seguridad manualmente y reinténtalo.',
      captureFailed: 'No se pudo capturar la ventana frontal. No se adjuntó ni se envió nada.',
      contextChanged: 'El borrador actual cambió durante la captura. La captura no se adjuntó ni se envió.'
    },
    quickEntry: {
      enabledTitle: 'Entrada rápida',
      enabledDesc:
        'Invoca un pequeño compositor desde cualquier lugar con un atajo global y envía un prompt sin abrir Hermes.',
      shortcutTitle: 'Atajo de entrada rápida',
      shortcutDesc: 'Necesita al menos un modificador, p. ej. CommandOrControl+Shift+Espacio.',
      active: 'El atajo está activo.',
      takenBy: 'Otra app ya usa este atajo — elige uno diferente.',
      invalidShortcut: 'No es un atajo válido. Incluye al menos una tecla modificadora.'
    },
    credentials: {
      pasteKey: 'Pegar clave',
      pasteLabelKey: label => `Pegar clave de ${label}`,
      optional: 'Opcional',
      enterValueFirst: 'Introduce primero un valor.',
      couldNotSave: 'No se pudo guardar la credencial.',
      remove: 'Quitar',
      getKey: 'Obtener una clave',
      saving: 'Guardando'
    },
    envActions: {
      actions: 'Acciones',
      manageInKeys: 'Gestionar en Claves API',
      docs: 'Docs',
      hideValue: 'Ocultar valor',
      revealValue: 'Mostrar valor',
      replace: 'Reemplazar',
      set: 'Definir',
      clear: 'Limpiar'
    },
    connections: {
      title: 'Gateways registrados',
      intro:
        'Gestiona este dispositivo y cada gateway de Hermes al que puede llegar mediante conexiones remotas, SSH o Cloud.',
      stagedNote:
        'Cambia de gateway desde Sesiones. Los perfiles, chats, mensajería y tareas cron se quedan con su gateway; el trabajo en otros gateways sigue ejecutándose.',
      launchModeTitle: 'Al iniciar, volver a Sesiones en el último gateway usado',
      launchModeDesc: 'Si está desactivado, Sesiones se abre en el gateway principal.',
      searchPlaceholder: 'Buscar gateways…',
      noSearchResults: 'Ningún gateway coincide con tu búsqueda.',
      loadFailed: 'No se pudieron cargar las conexiones',
      currentPill: 'Actual',
      primaryPill: 'Principal',
      managedPill: 'Administrado por la app',
      addConnection: 'Añadir conexión',
      editConnection: 'Editar',
      removeConnection: 'Quitar',
      removeConfirmTitle: '¿Quitar esta conexión?',
      removeConfirmDesc: (label: string) =>
        `“${label}” se quitará de esta app. La instancia en sí no se toca; puedes volver a añadirla cuando quieras.`,
      makePrimary: 'Hacer principal',
      testConnection: 'Probar',
      testOk: 'Accesible',
      testFailed: 'Falló la prueba de conexión',
      saveFailed: 'No se pudo guardar la conexión',
      removeFailed: 'No se pudo quitar la conexión',
      updateAll: 'Actualizar todas las instancias',
      updateAllRunning: 'Actualizando todas las instancias…',
      updateAllDone: 'Actualizaciones enviadas',
      updateAllFailed: 'Falló el envío de actualizaciones',
      updateSkippedCloud: 'Administrado por Hermes Cloud',
      kindLocal: 'Local',
      kindRemote: 'Gateway remoto',
      kindCloud: 'Hermes Cloud',
      kindSsh: 'SSH',
      kindLocalDesc: 'El entorno de ejecución de Hermes administrado por esta app.',
      kindRemoteDesc: 'Un gateway de Hermes accesible por HTTP(S): LAN, Tailscale o internet.',
      kindCloudDesc: 'Una instancia alojada detectada a través de tu cuenta de Hermes Cloud.',
      kindSshDesc: 'Una instalación de Hermes accesible por SSH.',
      labelTitle: 'Nombre',
      labelDesc:
        'Obligatorio. Se muestra en todos los lugares donde aparece esta instancia; debe ser único (p. ej., “Homelab”, “Portátil del trabajo”).',
      labelPlaceholder: 'Homelab',
      urlTitle: 'URL del gateway',
      sshHostTitle: 'Host SSH',
      headersTitle: 'Encabezados adicionales del gateway',
      headersDesc:
        'Se envían con cada solicitud HTTP y WebSocket a este gateway, para proxies de acceso como Cloudflare Access (CF-Access-Client-Id / CF-Access-Client-Secret). Los valores se guardan cifrados. Se ignoran los encabezados que gestiona Hermes (Authorization, Cookie, Host…).',
      headerValuePlaceholder: 'Valor',
      headerValueSaved: 'Guardado: déjalo en blanco para conservarlo',
      headerAdd: 'Añadir encabezado',
      headerRemove: 'Quitar',
      duplicateLocal: 'Esta app ya administra una conexión local; solo puede haber una.',
      duplicateUrl: (label: string) => `Ya existe una conexión a la URL de este gateway (“${label}”).`,
      duplicateSsh: (label: string) => `Ya existe una conexión a este host SSH (“${label}”).`,
      sameBackendHint: (label: string) => `Mismo backend que “${label}”`,
      localAddHint: 'Local no está disponible: la conexión local administrada ya existe (solo puede haber una).',
      cloudAddHint:
        'Consejo: al iniciar sesión en Hermes Cloud arriba, tus agentes se detectan automáticamente; usa este formulario solo para registrar a mano la URL de una instancia conocida.',
      save: 'Guardar conexión',
      saving: 'Guardando…',
      cancel: 'Cancelar',
      empty: 'Todavía no hay conexiones registradas.'
    },
    managedUpdates: {
      title: 'Actualizaciones administradas',
      intro:
        'Actualiza de forma transaccional las instalaciones SSH administradas por la app: las sesiones se vacían, la copia remota se actualiza y cada perfil se restaura con un comprobante correlacionado.',
      sshConnection: 'Instalación SSH administrada por la app',
      update: 'Actualizar',
      updating: 'Actualizando…',
      progress: 'Vaciando sesiones, actualizando la instalación remota y restaurando perfiles…',
      updated: 'Actualizado',
      partial: 'Actualizado, pero falló la restauración',
      refused: 'Rechazado',
      failed: 'Falló la actualización',
      alreadyRunning: 'Ya hay una actualización en curso',
      receipt: (id: string, outcome: string) => `Comprobante ${id} · ${outcome}`,
      receiptVersions: (pre: string, post: string) => `${pre} → ${post}`,
      scopesRestored: (profiles: string) => `Perfiles restaurados: ${profiles}`,
      scopeNotRestored: (profile: string, error: string) => `No se restauró el perfil “${profile}”: ${error}`
    },
    gateway: {
      loading: 'Cargando ajustes del gateway...',
      unavailableTitle: 'Ajustes del gateway no disponibles',
      unavailableDesc:
        'Los ajustes de conexión solo se pueden cambiar desde la app Hermes Desktop en el equipo que la ejecuta.',
      title: 'Conexión del gateway',
      envOverride: 'anulación de entorno',
      intro:
        'Local por predeterminado. Usa remoto cuando esta app deba controlar un backend de Hermes en otro lugar. Anulaciones por perfil a continuación.',
      envOverrideTitle: 'Esta conexión quedó fijada por la forma en que se inició Hermes.',
      envOverrideDesc:
        'Un ajuste de inicio externo a la app eligió esta conexión, así que las opciones de abajo son de solo lectura. Reinicia Hermes sin ese ajuste (o pregunta a quien lo configuró) para cambiarla aquí.',
      modeTitle: 'Modo de conexión',
      localTitle: 'Gateway local',
      localDesc:
        'Inicia un backend privado de Hermes en localhost. Es el valor predeterminado y funciona sin conexión.',
      remoteTitle: 'Gateway remoto',
      remoteDesc: 'Conecta esta shell de escritorio a un backend remoto de Hermes.',
      remoteAuthHint:
        'Los gateways alojados usan OAuth o usuario y contraseña; los autohospedados pueden usar un token de sesión.',
      cloudTitle: 'Hermes Cloud',
      cloudDesc:
        'Inicia sesión una vez en Hermes Cloud y elige uno de los agentes de tu cuenta; no tienes que pegar ninguna URL.',
      cloudSignInTitle: 'Hermes Cloud',
      cloudSignIn: 'Iniciar sesión en Hermes Cloud',
      cloudSignedIn: 'Sesión iniciada en Hermes Cloud',
      cloudNeedsSignIn: 'Inicia sesión en Hermes Cloud para descubrir los agentes de tu cuenta.',
      cloudSignedInDesc: 'Has iniciado sesión. Elige un agente de abajo; la sesión se actualiza automáticamente.',
      cloudAgentsTitle: 'Tus agentes',
      cloudOrgPickerTitle: 'Elige una organización',
      cloudOrgSelect: 'Seleccionar',
      cloudOrgChange: 'Cambiar organización',
      cloudOrgRole: role => `Rol: ${role}`,
      cloudLoadingAgents: 'Cargando tus agentes…',
      cloudNoAgents: {
        before: 'No se encontraron agentes en esta cuenta. Crea uno en el ',
        linkText: 'Portal de Nous',
        after: ', y luego actualiza.'
      },
      cloudRefresh: 'Actualizar',
      cloudConnect: 'Conectar',
      cloudSavedTitle: 'Gateways de Cloud guardados',
      cloudSavedDesc:
        'Usa un gateway guardado sin cambiar el predeterminado. Inicia sesión abajo para añadir instancias. Gestiona los nombres y el inicio de sesión en la lista de conexiones guardadas.',
      cloudUseSaved: 'Usar gateway',
      cloudActive: 'Activo en esta ventana',
      cloudConnecting: 'Conectando…',
      cloudDiscoverFailed: 'No se pudieron cargar tus agentes de Hermes Cloud',
      cloudConnectFailed: 'No se pudo conectar con ese agente',
      cloudSignInFailed: 'Falló el inicio de sesión en Hermes Cloud',
      cloudSignedOutTitle: 'Sesión cerrada en Hermes Cloud',
      cloudSignedOutMessage: 'Se borró la sesión de Hermes Cloud.',
      cloudConnectedTitle: 'Conectado',
      cloudConnectedPill: 'Conectado',
      cloudConnectedTo: name => `Conectado a ${name}.`,
      cloudAgentProvisioning: 'Preparando…',
      cloudStatusLabel: status => `Estado: ${status}`,
      remoteUrlTitle: 'URL remota',
      remoteUrlDesc: 'URL base del backend del dashboard remoto. Se admiten prefijos de ruta, por ejemplo /hermes.',
      probing: 'Comprobando cómo se autentica este gateway…',
      probeError:
        'Hermes no puede llegar a esa dirección. Comprueba la URL y que el otro equipo esté ejecutando Hermes; las opciones de inicio de sesión aparecen cuando responde.',
      signedIn: 'Sesión iniciada',
      signIn: 'Iniciar sesión',
      signOut: 'Cerrar sesión',
      signInWith: provider => `Iniciar sesión con ${provider}`,
      authTitle: 'Autenticación',
      authSignedInPassword:
        'Este gateway usa usuario y contraseña. Ya iniciaste sesión; la sesión se actualiza automáticamente.',
      authSignedInOauth: 'Este gateway usa OAuth. Ya iniciaste sesión; la sesión se actualiza automáticamente.',
      authNeedsPassword: 'Este gateway usa usuario y contraseña. Inicia sesión para autorizar esta app de escritorio.',
      authNeedsOauth: provider => `Este gateway usa OAuth. Inicia sesión con ${provider} para autorizar esta app.`,
      tokenTitle: 'Token de sesión',
      tokenDesc: 'Token de sesión del dashboard usado para REST y WebSocket. Déjalo vacío para conservar el guardado.',
      existingToken: value => `Token existente ${value}`,
      savedToken: 'guardado',
      pasteSessionToken: 'Pegar token de sesión',
      plainTextConfirmTitle: '¿Guardar el token de la puerta de enlace en texto plano?',
      plainTextConfirmDesc:
        'No se encontró ningún servicio de llavero del sistema en este equipo, así que el token se guardaría sin cifrar en el archivo de ajustes de conexión de la app, legible por cualquier proceso que se ejecute con este usuario. Instala o activa el llavero del sistema (GNOME Keyring o KWallet en Linux) para guardarlo cifrado.',
      plainTextConfirmAction: 'Guardar como texto plano',
      plainTextStoredTitle: 'Token guardado en texto plano',
      plainTextStoredDesc:
        'El almacenamiento seguro no está disponible, así que el token guardado está sin cifrar en el archivo de ajustes de conexión de la app en este equipo. Instala o activa el llavero del sistema (GNOME Keyring o KWallet en Linux) para cifrarlo.',
      keychainEncryptionTitle: 'Cifrar los secretos guardados con el llavero del sistema',
      keychainEncryptionDesc:
        'Desactivado por defecto. Si está activado, los tokens del gateway y las credenciales de inicio de sesión se cifran con el llavero del sistema (Acceso a Llaveros, GNOME Keyring o DPAPI de Windows); es posible que el sistema te pida permiso o una contraseña. Si está desactivado, se guardan como archivos normales que solo tu cuenta de usuario puede leer.',
      keychainEncryptionFailed: 'No se pudo cambiar el cifrado de secretos',
      testRemote: 'Probar remoto',
      saveForRestart: 'Guardar para el próximo reinicio',
      saveAndReconnect: 'Guardar y reconectar',
      diagnostics: 'Diagnóstico',
      diagnosticsDesc: 'Muestra desktop.log en tu gestor de archivos; útil cuando el gateway no arranca.',
      openLogs: 'Abrir registros',
      incompleteTitle: 'Gateway remoto incompleto',
      incompleteSignIn: 'Introduce una URL remota e inicia sesión antes de cambiar a remoto.',
      incompleteToken: 'Introduce una URL remota y un token de sesión antes de cambiar a remoto.',
      incompleteSignInTest: 'Introduce una URL remota e inicia sesión antes de probar.',
      incompleteTokenTest: 'Introduce una URL remota y un token de sesión antes de probar.',
      enterUrlFirst: 'Introduce primero una URL remota.',
      restartingTitle: 'Reiniciando conexión del gateway',
      savedTitle: 'Ajustes del gateway guardados',
      restartingMessage: 'Hermes Desktop se reconectará con los ajustes guardados.',
      savedMessage: 'Guardado para el próximo reinicio.',
      connectedTo: (baseUrl, version) => `Conectado a ${baseUrl}${version ? ` · Hermes ${version}` : ''}`,
      reachableTitle: 'Gateway remoto accesible',
      signedOutTitle: 'Sesión cerrada',
      signedOutMessage: 'Se borró la sesión del gateway remoto.',
      failedLoad: 'No se pudieron cargar los ajustes del gateway',
      signInFailed: 'No se pudo iniciar sesión',
      signOutFailed: 'No se pudo cerrar sesión',
      testFailed: 'Falló la prueba del gateway remoto',
      applyFailed: 'No se pudieron aplicar los ajustes del gateway',
      saveFailed: 'No se pudieron guardar los ajustes del gateway',
      sshTitle: 'Conectar por SSH',
      sshDesc:
        'Hermes se inicia en el equipo remoto mediante SSH y se conecta a esta app a través de un túnel; no tienes que iniciar ni exponer nada por tu cuenta. Requiere acceso SSH mediante claves que ya funcione con el host.',
      sshTrustHint:
        'La primera clave de host presentada se acepta y se fija; si cambia después, la conexión se rechaza.',
      sshHostTitle: 'Host',
      sshHostDesc: 'usuario@host o un alias Host de ~/.ssh/config.',
      sshHostPick: 'Seleccionar un host…',
      sshHostPickTitle: 'Host',
      sshHostPickDesc: 'Un alias Host de ~/.ssh/config, o Personalizado para escribir uno.',
      sshHostCustom: 'Personalizado (introducir manualmente)…',
      sshUserTitle: 'Usuario',
      sshUserDesc: 'En blanco = ~/.ssh/config o tu usuario actual.',
      sshUserPlaceholder: 'desde ~/.ssh/config',
      sshPortTitle: 'Puerto',
      sshPortDesc: 'En blanco = 22 o el puerto de ~/.ssh/config.',
      sshKeyTitle: 'Archivo de identidad',
      sshKeyDesc: 'Ruta de la clave privada. En blanco = ssh-agent o ~/.ssh/config.',
      sshHermesPathTitle: 'Ruta de Hermes (opcional)',
      sshHermesPathDesc: 'Ruta completa al binario remoto de Hermes. En blanco = detección automática.',
      sshHermesPathPlaceholder: 'detección automática',
      sshTestConnection: 'Probar SSH',
      sshConnect: 'Conectar',
      sshButtonsHint: 'Guardar se aplica en el próximo inicio. Conectar vuelve a conectar ahora.',
      sshReachable: (host, platform) => `Accesible: ${host} (${platform}) — se encontró Hermes`,
      sshIncompleteHost: 'Introduce un host SSH antes de conectar.',
      sshErrUnreachable: 'No se pudo acceder a ese host por SSH. Revisa el host, el puerto y tu red.',
      sshErrAuth:
        'Falló la autenticación SSH. Carga tu clave en ssh-agent (ssh-add) o configura un IdentityFile en ~/.ssh/config; Hermes ejecuta SSH de forma no interactiva.',
      sshErrHostKey:
        'La clave del host CAMBIÓ desde la última conexión. Confirma que sea un cambio esperado, ejecuta ssh-keygen -R <host> y vuelve a conectar.',
      sshErrNotInstalled:
        'Hermes no está instalado en el host remoto. Instálalo allí (curl -fsSL https://hermes-agent.nousresearch.com/install.sh | sh) o indica la ruta de Hermes.',
      sshErrPlatform:
        'Plataforma remota no compatible. El modo SSH de Hermes Desktop admite hosts remotos Linux, macOS y Windows.',
      sshErrTimeout: 'La conexión SSH agotó el tiempo de espera. Es posible que el host no responda o esté en reposo.',
      sshErrUpdateRequired: 'Actualiza Hermes en el host remoto antes de conectarte con Desktop SSH.',
      sshErrUnknown: 'Falló la conexión SSH.'
    },
    keys: {
      loading: 'Cargando claves API y credenciales...',
      failedLoad: 'No se pudieron cargar las claves API',
      empty: 'Aún no hay nada configurado en esta categoría.'
    },
    search: {
      placeholder: 'Buscar en todos los ajustes…',
      pill: 'Buscar'
    },
    profileScope: {
      appliesTo: 'Se aplica a',
      editsProfile: profile => `Los cambios de esta página se aplican al perfil “${profile}”.`
    },
    mcp: {
      loading: 'Cargando servidores MCP...',
      invalidJson: 'JSON MCP no válido',
      saveFailed: 'No se pudo guardar',
      removeFailed: 'No se pudo quitar',
      reloadFailed: 'Falló la recarga de MCP',
      savedTitle: 'Servidor MCP guardado',
      savedMessage: name => `${name} se aplica después de recargar MCP.`,
      disabled: 'deshabilitado',
      name: 'Nombre',
      serverJson: 'JSON del servidor',
      remove: 'Quitar',
      test: 'Probar conexión',
      catalogLoading: 'Cargando catálogo MCP...',
      catalogInstallFailed: name => `No se pudo instalar ${name}`,
      catalogEnvRequired: 'Completa los valores obligatorios antes de instalar.',
      capabilitySummary: (tools, prompts, resources) =>
        `Habilitado: ${[`${tools} herramientas`, ...(prompts ? [`${prompts} prompts`] : []), ...(resources ? [`${resources} recursos`] : [])].join(', ')}`,
      costTokens: tokens => `~${tokens} tok/llamada`,
      usage30d: uses => `${uses} usos/30 d`,
      statusConnecting: 'Conectando…',
      statusNeedsAuth: 'Necesita autenticación',
      statusError: 'Error',
      statusOff: 'Desactivado',
      allServers: 'Todos los servidores',
      authenticatedTitle: 'Autenticado',
      authenticatedMessage: (server, count) => `${server}: ${count} herramientas`,
      authenticate: 'Autenticar',
      noOutput: 'Aún no hay salida.',
      deepLinkTitle: '¿Añadir servidor MCP?',
      deepLinkDescription:
        'Un enlace pidió añadir este servidor MCP a Hermes. Revisa la configuración exacta de abajo: viene del enlace, no de Hermes.',
      deepLinkStdioWarning:
        'Este servidor ejecuta un proceso local en tu equipo con el comando que se muestra abajo. Continúa solo si confías en su origen.',
      deepLinkConfirm: 'Añadir servidor',
      deepLinkNameInvalid: 'Los nombres usan de 1 a 64 letras, dígitos, puntos, guiones o guiones bajos.',
      deepLinkNameConflict: name => `Ya existe un servidor llamado ${name}: elige otro nombre o cancela.`,
      deepLinkErrorTitle: 'Enlace de instalación de MCP rechazado',
      deepLinkErrorName: 'Falta el nombre del servidor del enlace o no es válido.',
      deepLinkErrorConfig: 'La configuración del enlace no es JSON válido codificado en base64.',
      deepLinkErrorShape: 'La configuración debe ser un objeto JSON con un campo `url` o `command` de tipo cadena.',
      deepLinkErrorUrl: 'Solo se permiten URL de servidor http:// y https://.',
      deepLinkErrorTooLarge: 'La configuración supera el límite de 32 KB.'
    },
    model: {
      setupProviderFallback: 'proveedor',
      setUpProvider: name => `Configurar ${name}`,
      staleAuxBefore: (count, names) =>
        count === 1
          ? `${count} tarea auxiliar (${names}) todavía se ejecuta en `
          : `${count} tareas auxiliares (${names}) todavía se ejecutan en `,
      staleAuxAfter: ', no en tu modelo principal.',
      staleAuxOtherProviders: 'otros proveedores',
      moaEnabled: 'Activado',
      moaSetDefault: 'Establecer como predeterminado',
      moaNewPresetPlaceholder: 'nuevo preajuste',
      moaAddPreset: 'Añadir preajuste',
      customModel: 'Modelo personalizado…',
      customModelPlaceholder: 'ID del modelo',
      chooseFromList: 'Elegir de la lista',
      moaDefault: 'Predeterminado:',
      moaReferenceToggle: (enabled, index) => `${enabled ? 'Desactivar' : 'Activar'} la referencia ${index}`,
      moaReferenceTitle: index => `Referencia ${index}`,
      moaAddReference: 'Añadir modelo de referencia',
      loading: 'Cargando configuración de modelo...',
      appliesDesc:
        'Se aplica a sesiones nuevas. Usa el selector de modelo en el compositor para cambiar el chat activo.',
      provider: 'Proveedor',
      model: 'Modelo',
      applying: 'Aplicando...',
      defaultsLabel: 'Valores predeterminados',
      reasoning: 'Razonamiento',
      reasoningOff: 'Desactivado',
      defaultsFailed: 'No se pudieron guardar los valores predeterminados del modelo',
      loadFailed: 'No se pudieron cargar los modelos',
      restartRequired:
        'Este backend ejecuta código antiguo tras una actualización. Reinícialo para cargar el código nuevo.',
      restartBackend: 'Reiniciar backend',
      restartingBackend: 'Reiniciando backend...',
      restartFailed: 'No se pudo reiniciar el backend',
      auxiliaryTitle: 'Modelos auxiliares',
      resetAllToMain: 'Restablecer todos al principal',
      auxiliaryDesc:
        'Las tareas auxiliares usan el modelo principal de forma predeterminada. Asigna un modelo dedicado a cualquier tarea para anularlo.',
      setToMain: 'Usar principal',
      change: 'Cambiar',
      autoUseMain: 'auto · usar modelo principal',
      inheritMainEffort: 'heredar · esfuerzo del modelo principal',
      providerDefault: '(predeterminado del proveedor)',
      fallbackAdd: 'Añadir respaldo',
      fallbackEmpty: 'No hay modelos de respaldo; se usa el modelo predeterminado salvo que falle.',
      notInCatalog: 'no está en la lista de modelos de este proveedor; las llamadas pueden recurrir a un respaldo.',
      moaTitle: 'Mixture of Agents',
      moaPreset: 'Preajuste',
      moaDescription:
        'Configura preajustes con nombre que aparecen como modelos del proveedor Mixture of Agents. El agregador es el modelo que actúa: ejecuta cada paso del bucle de herramientas, y casi todo el coste de la ejecución se factura a su proveedor. Por defecto, las referencias solo asesoran una vez por turno del usuario.',
      moaAggregator: 'Agregador',
      moaAggregatorBilled: 'modelo que actúa · se factura por la ejecución',
      moaReferenceHint: 'asesora una vez por turno por defecto',
      tasks: {
        vision: {
          label: 'Visión',
          hint: 'Análisis de imágenes'
        },
        compression: {
          label: 'Compresión',
          hint: 'Compactación de contexto'
        },
        skills_hub: {
          label: 'Hub de skills',
          hint: 'Búsqueda de skills'
        },
        approval: {
          label: 'Aprobación',
          hint: 'Aprobación automática inteligente'
        },
        mcp: {
          label: 'MCP',
          hint: 'Enrutamiento de herramientas MCP'
        },
        title_generation: {
          label: 'Generación de títulos',
          hint: 'Títulos de sesión'
        },
        review: {
          label: 'Revisión',
          hint: 'subagente revisor de /review'
        },
        triage_specifier: {
          label: 'Especificador de triaje',
          hint: 'Detalle de especificaciones de Kanban'
        },
        kanban_decomposer: {
          label: 'Descomponedor de Kanban',
          hint: 'Descomposición de tareas'
        },
        profile_describer: {
          label: 'Descriptor de perfiles',
          hint: 'Descripciones automáticas de perfiles'
        },
        curator: {
          label: 'Curador',
          hint: 'Revisión de uso de skills'
        }
      }
    },
    localModels: {
      connectionChanged: 'Cambió la conexión de los modelos locales',
      title: 'Modelos locales',
      runtimeTitle: 'Entorno local',
      runtimeReady: backend => `Listo · ${backend}`,
      serverRunning: 'En ejecución',
      runtimeInstalled: 'Entorno llama.cpp instalado',
      runtimeInstalledDetail: (tag, backend) =>
        `Compilación ${tag}, backend ${backend}. Hermes inicia y gestiona el servidor por ti.`,
      installTitle: 'Instalar el entorno local',
      installDetail:
        'Descarga el motor de inferencia llama.cpp (unos cientos de MB). Los modelos que descargues se ejecutan por completo en este equipo: sin cuenta y sin que nada salga de tu computadora.',
      installAction: 'Instalar entorno',
      installing: 'Instalando el entorno…',
      installFailed: 'Falló la instalación del entorno',
      hardwareTitle: 'Este equipo',
      hardwareLoading: 'Comprobando tu hardware…',
      vram: label => `${label} de memoria de GPU`,
      ram: label => `${label} de RAM`,
      unifiedMemory: 'Memoria unificada',
      modelsTitle: 'Modelos',
      recommended: 'Recomendado',
      recommendedReason: {
        'best-quality-resident':
          'El modelo de mayor calidad que se ejecuta por completo en tu GPU a máxima velocidad. La selección equilibra la calidad con la velocidad prevista en este hardware.',
        'speed-gated-quality':
          'Cabe un modelo de mayor calidad en este equipo, pero respondería demasiado lento por el ancho de banda de su memoria; este es el mejor modelo que se mantiene rápido.',
        'fastest-resident':
          'Ningún modelo alcanza la velocidad máxima en este hardware; este es el que más se acerca ejecutándose por completo en la memoria de la GPU.'
      },
      noRecommendationTitle: 'No hay recomendación automática para este equipo',
      noRecommendationDetail:
        'La configuración automática requiere un modelo seleccionado que quepa por completo en la memoria de la GPU o unificada. Aun así, puedes elegir un modelo abajo o explorar más modelos.',
      noRecommendationAction: 'Explorar modelos',
      downloaded: 'Descargado',
      downloadAction: size => `Descargar · ${size}`,
      downloadProgress: (done, total) => `Descargando ${done} de ${total}`,
      downloadDoneToast: model => `${model} está listo.`,
      installDoneToast: 'Entorno local instalado y listo.',
      quickstartTitle: 'Ejecutar un modelo en este equipo',
      quickstartDetail: (model, size) =>
        `Un clic lo configura todo: el motor local, ${model} (${size} de descarga) y tu modelo predeterminado para chats nuevos. Nada sale de este equipo.`,
      quickstartDetailReady: model =>
        `Un clic convierte ${model} en tu modelo predeterminado para chats nuevos. Todo se ejecuta en este equipo.`,
      quickstartAction: 'Configurarlo por mí',
      quickstartConfigure: 'Prefiero elegir',
      quickstartDoneToast: model => `${model} está configurado: los chats nuevos se ejecutan en este equipo.`,
      quickstartFailed: 'Falló la configuración del modelo local',
      quickstartStageEngine: 'Motor',
      quickstartStageModel: 'Modelo',
      quickstartStageFinish: 'Finalizar',
      useAction: 'Usar',
      activePill: 'Predeterminado',
      updateTitle: 'Hay una actualización del motor',
      updateDetail: (next, current) =>
        `Hay una compilación más reciente de llama.cpp (${next}) lista para instalar; tienes ${current}. Los modelos siguen funcionando durante la descarga.`,
      updateAction: 'Actualizar motor',
      updating: 'Actualizando el motor…',
      upToDateTitle: 'Motor actualizado',
      upToDateDetail: (tag, backend) => `Ejecutando llama.cpp ${tag} (${backend}), la compilación configurada.`,
      activeDetail: 'Los chats nuevos usan este modelo; se carga cuando envías tu primer mensaje',
      activeNotLoaded: 'Se carga con tu primer mensaje',
      loadedPill: 'En memoria',
      placementResident: 'todo en GPU',
      placementSpilled: 'parte en RAM',
      placementResidentTip:
        'Se ejecuta por completo en la memoria de la GPU con esta ventana de contexto: máxima velocidad.',
      placementSpilledTip:
        'Parte de este modelo se ejecuta desde la RAM del sistema: funciona, pero más lento. Una compilación más compacta o un contexto menor cabría por completo.',
      loadingPill: 'Cargando…',
      ejectTip: 'Liberar memoria de GPU (se vuelve a cargar con el siguiente mensaje)',
      ejected: 'Modelo descargado de memoria: memoria de GPU liberada.',
      ejectFailed: 'No se pudo descargar el modelo de memoria',
      stopServer: 'Apagar',
      startServer: 'Encender',
      runtimeRunningDetail:
        'El servidor local está en ejecución. Apagarlo libera toda la memoria de GPU e impide que los chats nuevos usen modelos locales hasta que lo vuelvas a encender.',
      serverStopped: 'Servidor local detenido: memoria de GPU liberada.',
      serverStarted: 'Servidor local en ejecución.',
      serverStopFailed: 'No se pudo detener el servidor local',
      serverStartFailed: 'No se pudo iniciar el servidor local',
      activating: 'Iniciando…',
      activateFailed: model => `No se pudo cambiar a ${model}`,
      activateDoneToast: model => `Los chats nuevos usan ${model}.`,
      downloadFailed: model => `Falló la descarga de ${model}`,
      pillFitsGpu: 'Cabe en tu GPU',
      pillUsesRam: 'Usa RAM del sistema',
      pillTooBig: 'Demasiado grande para este equipo',
      browseTitle: 'Buscar más modelos',
      browseHint:
        'Busca en todo Hugging Face. Los modelos que descargues aquí se ajustan automáticamente a tu equipo, pero no los hemos probado.',
      browsePlaceholder: 'Busca modelos por nombre o autor…',
      browseSearching: 'Buscando en Hugging Face',
      browseListing: 'Leyendo los archivos del modelo',
      browseShowFiles: 'Mostrar archivos',
      browseRefresh: 'Actualizar',
      browseDownloads: 'descargas',
      browseLikes: 'me gusta',
      browseGated: 'requiere iniciar sesión en Hugging Face',
      browseNoGguf: 'No se encontraron archivos de modelo compatibles.',
      browseFitUnknown: 'Ajuste desconocido',
      browseAlreadyDownloaded: 'Ya descargado.',
      addedByYou: 'Añadido por ti',
      browseDownloadStarted: 'Descargando {name}',
      browseDownloadAria: 'Descargar {name}',
      sideloadButton: 'Añadir archivo de modelo',
      sideloadTitle: 'Elige un archivo de modelo GGUF',
      sideloadDone: 'Se añadió {name}.',
      sideloadAlreadyPresent: 'Ya está en tu biblioteca.',
      pillFullContext: (max: string) => `Contexto completo de ${max}`,
      pillFullContextTip: 'Se ejecuta con la ventana de contexto completa del modelo desde el principio',
      pillUpTo: (max: string) => `Hasta ${max} de contexto`,
      pillGrowsTip: 'Crece automáticamente cuando tu conversación necesita más espacio',
      pillVision: 'Ve imágenes',
      deleteAction: 'Eliminar modelo',
      deleteConfirm: (model: string) => `¿Eliminar ${model} del disco?`,
      deleted: (model: string) => `${model} eliminado.`,
      deleteFailed: 'Error al eliminar'
    },
    billing: {
      perMonth: (amount: string) => `${amount}/mes`,
      creditsPerMonth: (amount: string) => `${amount} créditos/mes`,
      usageLabel: (label: string) => `Uso de ${label}`,
      freeTier: {
        signIn: 'Iniciar sesión',
        title: 'Estás en el plan gratuito de Nous',
        message: 'Inicia sesión con una cuenta de Nous para desbloquear más modelos y herramientas.',
        caption:
          'Funciona con nous/welcome, con conectores incluidos. Al iniciar sesión conservas tus conectores y se añaden las herramientas que requieren cuenta y todos los demás modelos.',
        name: 'Nous · plan gratuito',
        footnote:
          'El plan gratuito no tiene saldo ni nada que pagar. El pago y el uso aparecen al iniciar sesión con una cuenta de Nous.',
        plan: 'Plan gratuito',
        model: 'Modelo',
        connectors: 'Conectores',
        included: 'Incluidos'
      },
      amountValidation: {
        reloadTo: 'Recargar hasta',
        greaterThanThreshold: 'El importe de recarga debe ser mayor que el umbral.',
        decimal: (label: string) => `${label}: introduce un importe en dólares con 2 decimales como máximo.`,
        positive: (label: string) => `${label}: el importe debe ser mayor que 0 $.`,
        minimum: (label: string, amount: string) => `${label}: el mínimo es ${amount}.`,
        maximum: (label: string, amount: string) => `${label}: el máximo es ${amount}.`
      },
      stepUp: {
        openVerification: 'Abrir la página de verificación',
        dismiss: 'Descartar',
        waiting: 'Esperando el enlace de verificación…',
        verify: 'Verifica para continuar',
        deniedTitle: 'La verificación no se aprobó',
        deniedBody: 'La verificación terminó sin permitir el gasto remoto para este terminal.',
        successTitle: 'Verificación completada',
        successBody: 'El gasto remoto está permitido para este terminal.'
      },
      charge: {
        added: (amount?: string) => (amount ? `Se añadieron ${amount} $.` : 'Créditos añadidos.'),
        failedTitle: 'Falló el cargo',
        unconfirmedTitle: 'Resultado del cargo sin confirmar',
        unconfirmedBody: (message: string) =>
          `${message} El resultado de tu último cargo no está confirmado: revisa tu saldo/historial antes de reintentarlo.`,
        checkTitle: 'No se pudo comprobar el cargo',
        checkBody: 'No se pudo comprobar el cargo.',
        untrackedTitle: 'No se pudo seguir el cargo',
        untrackedBody: 'El servicio de facturación aceptó la solicitud, pero no devolvió un ID de cargo.',
        timeoutTitle: 'Sigue procesándose tras 5 minutos',
        timeoutBody: 'Es posible que el cargo todavía se complete. Revisa el portal antes de reintentarlo.',
        authenticationRequired:
          'Tu banco requiere verificación (3DS). Complétala en el portal para terminar esta compra.',
        expired: 'Tu tarjeta ha caducado. Actualízala en el portal.',
        declined: 'Tu tarjeta fue rechazada. Prueba otra tarjeta en el portal.',
        failedBody: (reason: string) => `El cargo no se completó (${reason}).`
      },
      title: 'Facturación',
      preview: 'vista previa',
      summary: {
        balance: 'Saldo',
        plan: 'Plan',
        autoRefill: 'Recarga automática'
      },
      sections: {
        invoices: 'Facturas',
        plan: 'Plan',
        paymentAndCredits: 'Pago y créditos',
        usage: 'Uso'
      },
      usage: {
        title: 'Uso'
      },
      buyCredits: {
        customAmount: 'Importe de créditos personalizado',
        title: 'Comprar créditos ahora',
        buyButton: 'Comprar',
        processing: 'Procesando… comprobando la liquidación',
        added: (amount: string) => `Se añadieron ${amount}. El saldo se está actualizando.`,
        retry: 'Reintentar',
        openPortal: 'Abrir el portal'
      },
      plan: {
        title: 'Planes',
        changePlan: 'Cambiar de plan',
        viewPlans: 'Ver planes',
        backAria: 'Volver a facturación',
        current: 'Plan actual',
        scheduled: 'Programado',
        empty: 'Ahora mismo no hay planes disponibles a los que cambiar.',
        undo: 'Deshacer',
        undoing: 'Deshaciendo…',
        downgrade: 'Bajar de plan',
        confirmDowngrade: 'Confirmar el cambio a un plan inferior',
        tryAgain: 'Volver a intentarlo',
        checkingChange: 'Comprobando este cambio…',
        cannotChange: 'Ese cambio no se puede hacer aquí.',
        alreadyOn: (name: string) => `Ya tienes el plan ${name}; no hay nada que cambiar.`,
        notScheduleable: 'Este cambio no se puede programar aquí.',
        scheduling: 'Programando…',
        cancel: 'Cancelar',
        effectScheduled: (targetName: string, effectiveAt: string, creditsDelta?: string) =>
          `Cambio a ${targetName}: entra en vigor ${effectiveAt}. No se cobra nada ahora; conservas tu plan actual hasta entonces.${creditsDelta ? ` Cambio de créditos mensuales: ${creditsDelta}.` : ''}`
      },
      autoReload: {
        threshold: 'Umbral',
        thresholdAria: 'Umbral de recarga automática',
        reloadTo: 'Recargar hasta',
        reloadToAria: 'Importe objetivo de la recarga automática',
        turnOffConfirm: '¿Desactivar la recarga automática?',
        turnOff: 'Desactivar',
        disable: 'Desactivar',
        updated: 'Recarga automática actualizada.',
        turnedOff: 'Recarga automática desactivada.',
        manage: 'Gestionar',
        save: 'Guardar',
        saving: 'Guardando…',
        cancel: 'Cancelar'
      },
      state: {
        notice: {
          loggedOut: {
            title: 'Conecta tu cuenta de Nous',
            message: 'Inicia sesión con tu cuenta de Nous para ver aquí tu saldo, plan y uso.',
            action: 'Iniciar sesión'
          },
          openPortal: 'Abrir el portal ↗',
          noCard: {
            title: 'No hay ningún método de pago registrado',
            message:
              'La compra de créditos y la recarga automática siguen desactivadas hasta que registres una tarjeta. Añade una en el portal.',
            action: 'Añadir tarjeta ↗'
          }
        },
        paymentMethod: {
          title: 'Método de pago',
          description: 'Gestiona la tarjeta que se usa para recargas y renovaciones de la suscripción.',
          addAction: 'Añadir método de pago',
          updateAction: 'Actualizar',
          provenance: {
            autoRefill: 'tarjeta de recarga automática',
            customerDefault: 'predeterminada del cliente',
            subPin: 'tarjeta de la suscripción',
            suffix: (label: string) => ` - ${label}`
          }
        },
        buyCredits: {
          description: 'Un único cargo en tu tarjeta que se añade hoy a tu saldo.'
        },
        autoRefill: {
          title: 'Recargar cuando quede poco saldo',
          genericDescription: 'Mantén tu saldo recargado cuando baje de tu umbral.',
          offPill: 'Desactivada',
          enabledPill: 'Activada',
          notAvailablePill: '—',
          manageCaption: 'Gestiona la recarga automática desde el portal.',
          turnOnCaption: 'Activa la recarga automática desde el portal',
          chargesDescription: (reloadTo: string, threshold: string) =>
            `Cobra ${reloadTo} automáticamente cuando tu saldo baja de ${threshold}.`,
          distinctCardCaption: (cardLabel: string) =>
            `La recarga automática cobra a ${cardLabel}: concílialo en el portal`,
          distinctCardFallback: 'otra tarjeta',
          reconcileAction: 'Conciliar ↗'
        },
        usage: {
          subscriptionCredits: {
            title: 'Créditos de la suscripción',
            barLabel: 'Créditos de la suscripción restantes',
            captionResets: (date: string) => `Se restablece ${date}`,
            valueOf: (remaining: string, monthly: string) => `Quedan ${remaining} de ${monthly}`,
            valueOver: (remaining: string, monthly: string, over: string) =>
              `Quedan ${remaining} de ${monthly} · ${over} de exceso`
          },
          topupCredits: {
            title: 'Créditos de recarga',
            caption: 'No caducan'
          },
          monthlyCap: {
            title: 'Límite de gasto mensual',
            barLabel: 'Límite de gasto mensual usado',
            captionDefault: 'Límite predeterminado',
            captionSpending: 'Gasto remoto mensual',
            valueUsed: (spent: string, limit: string) => `${spent} de ${limit} usados`
          }
        },
        planCard: {
          freeTier: 'Gratis',
          chooseAction: 'Elegir ↗',
          adjustPlanAction: 'Ajustar plan ↗',
          unavailableCaption: 'Los detalles de la suscripción no están disponibles; aún puedes abrir el portal.',
          downgradeCaption: (tierName: string, when: string) => `Cambia a ${tierName} el ${when}.`,
          cancellationCaption: (when: string) => `Se cancela el ${when}.`,
          renewsCaption: (date: string) => `Se renueva ${date}`,
          noSubscriptionCaption: 'No hay suscripción activa: los modelos de pago consumen créditos de recarga.'
        }
      },
      errors: {
        consentRequired: {
          title: 'Hace falta confirmar la tarjeta',
          message: 'Confirma esta tarjeta para los cargos del terminal en el portal'
        },
        insufficientScope: {
          title: 'El gasto remoto necesita aprobación',
          message: 'Esto requiere permitir el gasto remoto. Inicia una recarga para permitirlo y vuelve a intentarlo.'
        },
        remoteSpendingRevoked: {
          title: 'Se detuvo el gasto remoto',
          messageByAdmin: 'Un administrador detuvo el gasto remoto para este terminal.',
          messageBySelf: 'Detuviste el gasto remoto para este terminal.'
        },
        remoteSpendingReconnect: (who: string) =>
          `${who} Vuelve a conectarte desde Configuración -> Gateway para autorizar de nuevo este dispositivo.`,
        sessionRevoked: {
          title: 'Sesión cerrada',
          message: 'Se cerró tu sesión. Vuelve a iniciar sesión desde Configuración → Gateway.'
        },
        cliBillingDisabled: {
          title: 'El gasto remoto está desactivado',
          message:
            'El gasto remoto está desactivado para esta cuenta; un administrador de facturación puede activarlo desde la página de Hermes Agent del portal.'
        },
        roleRequired: {
          title: 'Se requiere rol de administrador',
          message:
            'Para añadir fondos hace falta un administrador o propietario de la organización. Pide ayuda a un administrador o gestiónalo en el portal.'
        },
        idempotencyConflict: {
          title: 'Inicia una nueva recarga',
          message: '🔴 Esa clave de cargo ya se usó para otro importe. Inicia una nueva recarga.'
        },
        noPaymentMethod: {
          title: 'No hay tarjeta guardada',
          message:
            '💳 Todavía no hay una tarjeta guardada para cargos del terminal. Configura una en el portal ' +
            '(las compras únicas de créditos no guardan una tarjeta reutilizable).'
        },
        orgAccessDenied: {
          title: 'Acceso a la organización denegado',
          message: 'Este token no está vinculado a una organización que puedas gestionar'
        },
        monthlyCapExceeded: {
          title: 'Se alcanzó el límite de gasto mensual',
          messageReached: '🔴 Se alcanzó el límite de gasto mensual.',
          messageHeadroom: (remaining: string) =>
            `🔴 Se alcanzó el límite de gasto mensual: quedan ${remaining} $ de margen.`
        },
        rateLimited: {
          title: 'Demasiados cargos ahora mismo',
          message: (mins: number) =>
            mins > 0
              ? `🟡 Demasiados cargos ahora mismo (reinténtalo en ~${mins} min). No es un fallo de pago.`
              : '🟡 Demasiados cargos ahora mismo. No es un fallo de pago.'
        },
        stripeUnavailable: {
          title: 'Stripe tiene problemas',
          message: (mins: number) =>
            mins > 0
              ? `Stripe tiene problemas: reinténtalo en ~${mins} min`
              : 'Stripe tiene problemas: reinténtalo en breve'
        },
        upgradeCapExceeded: {
          title: 'Se alcanzó el límite diario de cambios de plan',
          message: 'Se alcanzó el límite diario de cambios de plan: reinténtalo mañana'
        },
        endpointUnavailable: {
          title: 'Endpoint de facturación no disponible',
          message:
            'El endpoint de facturación devolvió una respuesta que no es JSON (puede que no esté disponible en este despliegue).'
        },
        timeout: {
          title: 'Se agotó el tiempo de la solicitud de facturación',
          message: 'Se agotó el tiempo de la solicitud de facturación.'
        },
        transport: {
          title: 'Falló la conexión de facturación',
          message: 'La solicitud de facturación falló antes de llegar al gateway.'
        },
        default: {
          title: 'Falló la solicitud de facturación',
          message: 'Falló la solicitud de facturación.'
        }
      }
    },
    providers: {
      connectAccount: 'Conectar una cuenta',
      haveApiKey: '¿Tienes una clave API?',
      intro:
        'Inicia sesión con una suscripción, sin copiar claves API. Hermes ejecuta el inicio de sesión del navegador por ti, aquí mismo en la app.',
      connected: 'Conectado',
      collapse: 'Contraer',
      connectAnother: 'Conectar otro proveedor',
      otherProviders: 'Otros proveedores',
      disconnect: 'Desconectar',
      disconnectInTerminal: 'Desconectar (ejecuta el comando de eliminación en el terminal)',
      removeConfirm: provider => `¿Eliminar ${provider}?`,
      removeExternalGeneric: provider => `${provider} se gestiona con su propia CLI; elimínalo allí.`,
      removeKeyManaged: provider => `${provider} se configura con una clave API. Quítalo en Claves API.`,
      removeTerminalConfirm: (provider, command) =>
        `¿Desconectar ${provider}? Esto ejecutará “${command}” en el terminal para borrar la credencial.`,
      removeTerminalRunning: provider => `Ejecutando la desconexión de ${provider} en el terminal…`,
      removedTitle: 'Cuenta eliminada',
      removedMessage: provider => `Se eliminó ${provider}.`,
      failedRemove: provider => `No se pudo eliminar ${provider}`,
      noProviderKeys: 'No hay claves API de proveedores disponibles.',
      searchKeys: 'Buscar proveedores…',
      noKeysMatch: 'Ningún proveedor coincide con tu búsqueda.',
      localEndpoint: {
        title: 'Endpoint local o personalizado',
        description:
          'Conecta Hermes con cualquier endpoint compatible con OpenAI (Zyphra, vLLM, llama.cpp, Ollama, etc.).'
      },
      loading: 'Cargando proveedores...'
    },
    sessions: {
      loading: 'Cargando sesiones archivadas…',
      archivedTitle: 'Sesiones archivadas',
      archivedIntro:
        'Los chats archivados se ocultan de la barra lateral, pero conservan todos sus mensajes. Haz Alt/⌥+Mayús/⇧-clic en un chat de la barra lateral para archivarlo.',
      emptyArchivedTitle: 'Nada archivado',
      emptyArchivedDesc: 'Archiva un chat para ocultarlo aquí.',
      unarchive: 'Desarchivar',
      deletePermanently: 'Eliminar permanentemente',
      messages: count => `${count} ${count === 1 ? 'mensaje' : 'mensajes'}`,
      restored: 'Restaurado',
      deleteConfirm: title => `¿Eliminar permanentemente "${title}"? Esto no se puede deshacer.`,
      autoArchiveTitle: 'Archivar automáticamente los chats inactivos',
      autoArchiveDesc:
        'Archiva automáticamente los chats que no has usado durante un tiempo. Los chats fijados nunca se archivan y no se elimina nada; los chats archivados solo se mueven aquí.',
      autoArchiveDaysLabel: 'Archivar después de',
      autoArchiveDaysUnit: 'días de inactividad',
      autoArchiveFailed: 'No se pudo actualizar el archivado automático',
      defaultDirTitle: 'Directorio de proyecto predeterminado',
      defaultDirDesc:
        'Las sesiones nuevas empiezan en esta carpeta salvo que elijas otra. Déjala sin definir para usar tu directorio de inicio.',
      defaultDirUpdated:
        'Directorio de proyecto predeterminado actualizado; inicia un chat nuevo (Ctrl/⌘+N) para que el cambio surta efecto.',
      defaultsTo: label => `Predeterminado: ${label}.`,
      change: 'Cambiar',
      choose: 'Elegir',
      clear: 'Limpiar',
      notSet: 'Sin definir',
      failedLoad: 'No se pudieron cargar las sesiones archivadas',
      unarchiveFailed: 'No se pudo desarchivar',
      deleteFailed: 'No se pudo eliminar',
      updateDirFailed: 'No se pudo actualizar el directorio predeterminado',
      clearDirFailed: 'No se pudo limpiar el directorio predeterminado'
    },
    toolsets: {
      loadingConfig: 'Cargando configuración',
      savedTitle: 'Credencial guardada',
      savedMessage: key => `${key} actualizada.`,
      removedTitle: 'Credencial quitada',
      removedMessage: key => `${key} quitada.`,
      failedSave: key => `No se pudo guardar ${key}`,
      failedRemove: key => `No se pudo quitar ${key}`,
      failedReveal: key => `No se pudo mostrar ${key}`,
      removeConfirm: key => `¿Quitar ${key} de .env?`,
      set: 'Definir',
      notSet: 'Sin definir',
      selectedTitle: 'Proveedor seleccionado',
      selectedMessage: provider => `${provider} está activo.`,
      failedSelect: provider => `No se pudo seleccionar ${provider}`,
      failedLoad: 'No se pudo cargar la configuración de herramientas',
      noProviderOptions:
        'Este conjunto de herramientas no tiene opciones de proveedor; actívalo y funcionará con tu configuración actual.',
      noProviders: 'No hay proveedores disponibles para este conjunto de herramientas ahora mismo.',
      ready: 'Listo',
      needsSignIn: 'Necesita iniciar sesión',
      needsSetup: 'Configuración requerida',
      activeBackend: 'Activo',
      activeBackendHint: 'Este es tu backend activo',
      useBackend: 'Usar este backend',
      nousIncluded: 'Incluido con una suscripción de Nous: inicia sesión con tu cuenta de Nous para activarlo.',
      nousAuthNeededTitle: 'Inicia sesión con tu cuenta de Nous',
      nousAuthNeededMessage: (provider: string) =>
        `${provider} está guardado, pero solo funcionará cuando inicies sesión con tu cuenta de Nous.`,
      nousAuthSignIn: 'Iniciar sesión',
      nousAuthDoneTitle: 'Cuenta de Nous conectada',
      nousAuthDoneMessage: 'Los backends de tu suscripción ya están activos.',
      nousAuthFailed: 'No se completó el inicio de sesión en Nous',
      nousAuthFailedMessage: 'Vuelve a intentarlo.',
      nousAuthTryAgain: 'Reintentar',
      noApiKeyRequired: 'No se requiere clave API.',
      postSetupHint: step =>
        `Este backend necesita una instalación única (${step}). Se ejecuta en esta máquina y puede tardar unos minutos.`,
      postSetupInstalledHint: 'Instalado. Repite la configuración solo si algo no funciona.',
      postSetupRun: 'Ejecutar instalación',
      postSetupRerun: 'Repetir instalación',
      postSetupInstalled: 'Instalado',
      postSetupRunning: 'Instalando…',
      postSetupStarting: 'Iniciando…',
      postSetupCompleteTitle: 'Instalación completa',
      postSetupCompleteMessage: step => `${step} instalado.`,
      postSetupErrorTitle: 'La instalación terminó con errores',
      postSetupErrorMessage: (step: string) =>
        `La configuración de ${step} no terminó. Abre los registros para ver por qué y vuelve a ejecutar la configuración.`,
      postSetupOpenLogs: 'Abrir registros',
      postSetupRunAgain: 'Ejecutar de nuevo',
      postSetupFailed: step => `No se pudo ejecutar la instalación de ${step}`,
      webSearchActive: backend => `Búsqueda: ${backend}`,
      webExtractActive: backend => `Extracción: ${backend}`,
      webCapabilityUnset: 'sin definir',
      webUseForSearch: 'Usar para búsqueda',
      webUseForExtract: 'Usar para extracción',
      webUsedForSearch: 'Backend de búsqueda',
      webUsedForExtract: 'Backend de extracción',
      webCapabilitySelectedMessage: (provider, capability) =>
        `${provider} ahora se encarga de ${capability === 'search' ? 'las búsquedas web' : 'la extracción de contenido web'}.`,
      failedSelectCapability: provider => `No se pudo configurar ${provider}`,
      loadingModels: 'Cargando catálogo de modelos...',
      modelSectionTitle: 'Modelo',
      modelCount: count => `${count} ${count === 1 ? 'modelo' : 'modelos'}`,
      modelInUse: 'En uso',
      modelDefault: 'predeterminado',
      modelInactiveHint: 'Selecciona primero este backend para cambiar su modelo.',
      modelSelectedTitle: 'Modelo seleccionado',
      modelSelectedMessage: model => `${model} se aplica a las sesiones nuevas.`,
      failedSelectModel: model => `No se pudo seleccionar ${model}`,
      terminalBackend: {
        sectionTitle: 'Backend de ejecución',
        loading: 'Comprobando backends de ejecución…',
        failedLoad: 'No se pudieron cargar los backends del terminal',
        ready: 'Listo',
        needsSetup: 'Necesita configuración',
        unavailable: 'No disponible',
        inUse: 'En uso',
        selectedTitle: 'Backend seleccionado',
        selectedMessage: backend =>
          `Los comandos del terminal ahora se ejecutan mediante ${backend}. Se aplica a las sesiones nuevas.`,
        failedSelect: backend => `No se pudo seleccionar ${backend}`,
        needsSetupHint:
          'Este backend está seleccionado sin configuración completa: los comandos fallarán hasta que termine la configuración.',
        needsSetupConfirmTitle: (backend: string) => `¿Seleccionar ${backend} de todos modos?`,
        needsSetupConfirmDescription: (detail: string) =>
          `${detail} Las sesiones que empiecen tras este cambio no tendrán herramientas de terminal ni de archivos hasta que termine la configuración.`,
        needsSetupConfirmDescriptionGeneric:
          'Este backend todavía no está configurado. Las sesiones que empiecen tras este cambio no tendrán herramientas de terminal ni de archivos hasta que termine la configuración.',
        needsSetupConfirmAction: 'Seleccionar de todos modos',
        unavailableTitle: 'Los comandos de terminal no están disponibles',
        unavailableMessage: (backend: string) =>
          `Hermes no puede ejecutar comandos de shell ahora mismo: ${backend} no está listo. Cambia a Local o termina de configurar ${backend} y vuelve a intentarlo.`,
        openBackendSettings: 'Abrir ajustes del terminal',
        useLocal: 'Usar Local',
        switchedToLocal: 'Los comandos de terminal ahora se ejecutan localmente. Se aplica a las sesiones nuevas.'
      },
      browserRealProfile: {
        label: 'Usar mi perfil real del navegador',
        description:
          'Copia los inicios de sesión y las cookies de tu navegador predeterminado en una instantánea administrada con la que navega el agente. Tu perfil en uso nunca se abre directamente. Se aplica a las sesiones nuevas.',
        enabledTitle: 'Navegación con perfil real activada',
        enabledMessage: 'Las sesiones nuevas navegarán con una instantánea de tu perfil predeterminado del navegador.',
        disabledTitle: 'Navegación con perfil real desactivada',
        disabledMessage: 'Se eliminará la instantánea del perfil; las sesiones nuevas usan un navegador limpio.',
        failedSave: 'No se pudo guardar el ajuste del perfil real',
        prompt: {
          title: 'Mantén la sesión iniciada en tus sitios',
          body: 'Deja que Hermes navegue con una instantánea de tu perfil predeterminado del navegador, para que los sitios se abran con la sesión ya iniciada.',
          bulletSnapshot: 'Las cookies y los inicios de sesión se copian en una instantánea administrada.',
          bulletLiveProfile: 'Tu perfil del navegador en uso nunca se abre directamente.',
          bulletLocal: 'Nada sale de este equipo.',
          dontShowAgain: 'No volver a mostrar',
          notNow: 'Ahora no',
          enable: 'Usar mi perfil'
        }
      }
    }
  },
  skills: {
    tabSkills: 'Skills',
    tabToolsets: 'Conjuntos de herramientas',
    configuringProfile: 'Configurando:',
    all: 'Todo',
    searchSkills: 'Buscar skills...',
    searchToolsets: 'Buscar conjuntos de herramientas...',
    refresh: 'Actualizar skills',
    refreshing: 'Actualizando skills',
    loading: 'Cargando capacidades...',
    noSkillsTitle: 'No se encontraron skills',
    noSkillsDesc: 'Prueba una búsqueda más amplia u otra categoría.',
    noToolsetsTitle: 'No se encontraron conjuntos de herramientas',
    noToolsetsDesc: 'Prueba una búsqueda más amplia.',
    noDescription: 'Sin descripción.',
    configured: 'Configurado',
    needsKeys: 'Necesita claves',
    visionModelHint:
      'Visión usa la configuración de tu modelo auxiliar; el modelo compatible con imágenes se elige allí, no aquí para cada proveedor.',
    visionModelLink: 'Elegir modelo de visión en Configuración → Modelos',
    toolsetsEnabled: (enabled, total) => `${enabled}/${total} conjuntos activados`,
    configureToolset: label => `Configurar ${label}`,
    toggleToolset: (label, enabled) =>
      `Activar/desactivar conjunto de herramientas ${label} ${enabled ? 'activado' : 'desactivado'}`,
    skillsLoadFailed: 'No se pudieron cargar los skills',
    toolsetsRefreshFailed: 'No se pudieron actualizar los conjuntos de herramientas',
    skillEnabled: 'Skill activado',
    skillDisabled: 'Skill desactivado',
    toolsetEnabled: 'Conjunto activado',
    toolsetDisabled: 'Conjunto desactivado',
    appliesToNewSessions: name => `${name} se aplica a sesiones nuevas.`,
    failedToUpdate: name => `No se pudo actualizar ${name}`,
    sortMostUsed: 'Más usadas',
    sortAlpha: 'A–Z',
    sortMostUsedDesc: '↓ Más usadas',
    sortLeastUsedAsc: '↑ Menos usadas',
    enableAll: 'Activar todo',
    disableAll: 'Desactivar todo',
    disableUnused: 'Desactivar los que no se usan',
    bulkUpdated: count =>
      `${count === 1 ? 'Se actualizó' : 'Se actualizaron'} ${count} ${count === 1 ? 'elemento' : 'elementos'} para las sesiones nuevas.`,
    bulkNoChange: 'No hay nada que cambiar.',
    usageCount: count => `usado ${count}×`,
    provenance: {
      agent: 'Aprendido',
      bundled: 'Integrado',
      hub: 'Hub'
    },
    emptyNoneFound: noun => `No se encontraron ${noun}`,
    emptyNothingMatches: query => `No hay coincidencias para “${query}”.`,
    emptyNoneAvailable: noun => `Aún no hay ${noun} disponibles.`,
    changesApplyNewSessions: 'Los cambios se aplican a las sesiones nuevas.',
    skillUpdated: 'Skill actualizado',
    edit: 'Editar',
    archive: 'Archivar',
    skillArchivedTitle: 'Skill archivado',
    skillArchivedMessage: 'Puedes restaurarlo con hermes curator restore.',
    tabPlugins: 'Plugins',
    plugins: {
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
      legacyBackend:
        'Este backend es anterior a los interruptores de plugins por clave: actualiza Hermes para gestionarlo aquí.',
      portableBadge: 'portátil',
      serverStates: {
        connected: 'conectado',
        app_not_running: 'la app no se está ejecutando',
        endpoint_unavailable: 'endpoint no disponible',
        no_interactive_session: 'sin sesión interactiva',
        version_too_old: 'versión demasiado antigua',
        missing_app: 'falta la app',
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
      tierOfficial: 'oficial',
      tierCommunity: 'comunidad',
      updateToPin: (sha: string) => `Actualizar a ${sha}`,
      updateFailed: (name: string) => `No se pudo actualizar ${name}`,
      updated: (name: string) =>
        `${name} se actualizó al commit fijado actual del catálogo. Reinicia el gateway para aplicarlo.`,
      updateConsentTitle: (name: string) => `${name} pide más`,
      updateConsentBody: (name: string, sha: string) =>
        `El nuevo commit fijado de ${name} en el catálogo (${sha}) añade superficies que la versión instalada no tiene. Aplícalo solo si confías en ellas:`,
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
        optional: '(opcional)',
        secretSet: '•••••••• (configurado)',
        secretStoredAs: (env: string) =>
          `Se guarda en el .env del perfil como ${env}, nunca en config.yaml; déjalo en blanco para conservar el valor actual.`
      }
    },
    officialCatalog: 'Disponibles para instalar',
    officialPill: 'Oficial',
    hub: {
      searchPlaceholder: 'Buscar en el hub de skills',
      search: 'Buscar',
      searching: 'Buscando…',
      connectingHubs: 'Conectando con los hubs de skills…',
      connectedHubs: 'Hubs conectados:',
      featured: 'Skills destacadas',
      landingHint:
        'Busca en el hub para explorar skills instalables del índice oficial, GitHub y fuentes de la comunidad.',
      noResults: 'No se encontraron skills coincidentes en el hub.',
      resultCount: (count, ms) =>
        `${count} ${count === 1 ? 'resultado' : 'resultados'}${ms !== null ? ` en ${ms} ms` : ''}`,
      timedOut: sources => `Se agotó el tiempo de espera: ${sources}`,
      installed: 'Instalado',
      install: 'Instalar',
      installing: 'Instalando…',
      uninstall: 'Desinstalar',
      uninstalling: 'Desinstalando…',
      updateAll: 'Actualizar skills instaladas',
      updating: 'Actualizando…',
      preview: 'Vista previa',
      scan: 'Analizar',
      scanning: 'Analizando…',
      close: 'Cerrar',
      files: 'Archivos',
      noReadme: 'Esta skill no tiene vista previa de SKILL.md.',
      trust: {
        builtin: 'integrado',
        trusted: 'de confianza',
        community: 'comunidad'
      },
      verdictSafe: 'Seguro',
      verdictCaution: 'Precaución',
      verdictDangerous: 'Peligroso',
      policyAllow: 'Instalación permitida',
      policyAsk: 'Revisar antes de instalar',
      policyBlock: 'Instalación bloqueada por la política',
      findings: count => `${count} ${count === 1 ? 'hallazgo' : 'hallazgos'}`,
      noFindings: 'No se encontraron hallazgos de seguridad.',
      installStarted: name => `Instalando ${name}…`,
      uninstallStarted: name => `Desinstalando ${name}…`,
      updateStarted: 'Actualizando las skills instaladas…',
      actionFailed: 'Falló la acción de la skill',
      installBlockedTitle: (name: string) => `No se pudo instalar ${name}`,
      installBlockedMessage: (findings: number, unverified: boolean) =>
        `El análisis de seguridad marcó ${findings > 0 ? `${findings} ${findings === 1 ? 'elemento' : 'elementos'}` : 'patrones de riesgo'} para revisar${unverified ? ' y la skill proviene de una fuente no verificada' : ''}. Lee el análisis antes de decidir si confías en el autor.`,
      viewScan: 'Ver análisis',
      openLog: 'Abrir registro',
      actionLog: 'Registro de acciones',
      alreadyInstalled: (name: string) => `"${name}" ya está instalada`,
      pickerTitle: 'Skills Hub',
      pickerBrowse: 'Explorar todo el hub',
      pickerHide: 'Ocultar el explorador del hub',
      pickerHint: 'Pulsa "+ Añadir a este agente" en cualquier skill: se instala y aparece en la lista de arriba.',
      loadFailed: 'No se pudo cargar el hub de skills',
      previewFailed: 'No se pudo cargar la vista previa de la skill',
      scanFailed: 'Falló el análisis de seguridad',
      searchFailed: 'Falló la búsqueda en el hub'
    }
  },
  starmap: {
    title: 'Grafo de memoria',
    subtitle: (nodes, clusters) => `${nodes} skills en ${clusters} categorías`,
    close: 'Cerrar grafo de memoria',
    refresh: 'Actualizar',
    memory: 'Memoria',
    filterAll: 'Todo',
    filterUsed: 'Usado',
    filterLearned: 'Aprendido',
    viewGraph: 'Grafo',
    loadFailed: 'No se pudo cargar el grafo de memoria',
    loading: 'Cargando…',
    emptyTitle: 'Aún no se ha aprendido nada',
    emptyDesc: 'A medida que Hermes crea skills y memorias para tu trabajo, aparecerán aquí.',
    share: 'Compartir mapa',
    shareHint:
      'Copia el código para compartir este mapa o pega uno para cargarlo. Solo incluye el diseño, no el texto de tus memorias ni skills.',
    shareTitle: 'Importar / exportar mapa',
    sharePlaceholder: 'Pega un código de mapa…',
    copy: 'Copiar código del mapa',
    copied: '¡Copiado!',
    importMap: 'Importar un mapa',
    importBtn: 'Cargar',
    importEmpty: 'Pega un código de mapa para cargarlo.',
    importSuccess: nodes => `Se cargó un mapa con ${nodes} ${nodes === 1 ? 'nodo' : 'nodos'}.`,
    importedBadge: 'mapa importado',
    resetToMine: 'Volver a mi mapa'
  },
  agents: {
    extendedTranscript: 'Transcripción ampliada',
    transcriptTruncated: 'Mostrando los últimos 16 KiB',
    transcriptUnavailable: 'Transcripción en vivo no disponible',
    close: 'Cerrar agentes',
    title: 'Árbol de generación',
    subtitle: 'Actividad de subagentes en vivo para el turno actual.',
    emptyTitle: 'No hay subagentes activos',
    emptyDesc: 'Cuando un turno delegue trabajo, los agentes hijos mostrarán su progreso aquí.',
    running: 'En ejecución',
    failed: 'Falló',
    done: 'Listo',
    streaming: 'Transmitiendo',
    files: 'Archivos',
    moreFiles: count => `+${count} archivos más`,
    moreAgents: (count: number) => `+${count} agentes más`,
    queued: 'En cola',
    waitingActivity: 'Esperando actividad',
    steer: 'Orientar',
    steerPlaceholder: 'Instrucciones para este subagente',
    steerQueued: 'En cola para el siguiente punto de control',
    stopRequested: 'Detención solicitada',
    requestRejected: 'El subagente no aceptó la solicitud',
    delegation: index => `Delegación ${index}`,
    workers: count => `${count} ${count === 1 ? 'trabajador' : 'trabajadores'}`,
    workersActive: count => `${count} activos`,
    agentsCount: count => `${count} ${count === 1 ? 'agente' : 'agentes'}`,
    activeCount: count => `${count} activos`,
    failedCount: count => `${count} fallidos`,
    toolsCount: count => `${count} herramientas`,
    filesCount: count => `${count} archivos`,
    updatedAgo: age => `actualizado ${age}`,
    ageNow: 'ahora',
    ageSeconds: seconds => `hace ${seconds}s`,
    ageMinutes: minutes => `hace ${minutes}m`,
    ageHours: hours => `hace ${hours}h`,
    ageDays: days => `hace ${days}d`,
    durationSeconds: seconds => `${seconds}s`,
    durationMinutes: (minutes, seconds) => `${minutes}m ${seconds}s`,
    tokens: value => `${value} tok`
  },
  commandCenter: {
    close: 'Cerrar Centro de comandos',
    paletteTitle: 'Paleta de comandos',
    back: 'Atrás',
    searchPlaceholder: 'Buscar sesiones, vistas y acciones',
    goTo: 'Ir a',
    goToSession: 'Ir a la sesión',
    branches: 'Ramas',
    projects: 'Proyectos',
    openFolder: 'Abrir carpeta como proyecto…',
    openFolderAt: path => `Abrir carpeta como proyecto — ${path}`,
    newSessionInProject: project => `Nueva sesión en ${project}`,
    commands: 'Comandos',
    startInBranch: branch => `Nueva conversación en ${branch}`,
    commandCenter: 'Centro de comandos',
    appearance: 'Apariencia',
    settings: 'Configuración',
    changeTheme: 'Cambiar tema...',
    changeColorMode: 'Cambiar modo de color...',
    pets: {
      title: 'Mascotas',
      placeholder: 'Buscar mascotas…',
      loading: 'Cargando la galería petdex…',
      error: 'No se pudo acceder a la galería petdex.',
      staleBackend: 'Reinicia Hermes para usar mascotas; el backend es anterior a esta función.',
      empty: 'No hay mascotas coincidentes.',
      turnOff: 'Desactivar',
      turnOn: 'Activar',
      installed: 'Instalada',
      generatedTag: 'Generada',
      adoptFailed: 'No se pudo adoptar esa mascota.',
      toggleFailed: enabled => `No se pudo ${enabled ? 'encender' : 'apagar'} la mascota.`,
      noneAvailable: 'No hay mascotas disponibles. Elige una de abajo para instalarla.'
    },
    generatePet: {
      title: 'Generar una mascota',
      placeholder: 'Describe una mascota para generar…',
      promptHint: 'Escribe una descripción y pulsa Intro para crear cuatro versiones.',
      readyHint: 'Pulsa Intro para crear cuatro versiones a partir de tu descripción.',
      generate: 'Generar',
      generating: 'Generando…',
      retry: 'Reintentar',
      hatch: 'Eclosionar',
      spawning: 'Creando…',
      hatching: 'Eclosionando tu mascota…',
      hatchingSub: 'Dándole vida…',
      hatched: '¡Ha eclosionado!',
      hatchRow: (_state, done, total) => `Bocetando el fotograma ${done} de ${total}…`,
      hatchComposing: 'Uniendo las piezas…',
      hatchSaving: 'Ya casi…',
      namePlaceholder: 'Ponle nombre a tu mascota',
      staleBackend: 'Actualiza Hermes para generar mascotas.',
      backgroundHint: 'Puedes cerrar esto; Hermes te avisará cuando termine.',
      slowProviderHint: 'Esto puede tardar varios minutos',
      remix: 'Remixar',
      remixConfirmTitle: '¿Remixar este aspecto?',
      remixConfirmBody:
        'Esto genera un nuevo conjunto de borradores usando este como punto de partida. Puede tardar varios minutos.',
      genericError: 'La generación falló. Inténtalo de nuevo o elige una sugerencia.',
      referenceImageTooLarge: 'La imagen de referencia es demasiado grande. Usa una de menos de 16 MB.',
      referenceImageInvalid: 'No se pudo leer esa imagen de referencia. Prueba con un PNG, JPG, WebP o GIF.',
      adopt: 'Adoptar',
      startOver: 'Empezar de nuevo'
    },
    installTheme: {
      title: 'Instalar tema…',
      pageTitle: 'Instalar tema',
      placeholder: 'Buscar en VS Code Marketplace…',
      loading: 'Buscando en Marketplace…',
      error: 'No se pudo acceder al Marketplace.',
      empty: 'No hay temas coincidentes.',
      install: 'Instalar',
      installing: 'Instalando…',
      installed: 'Instalado',
      installs: count => `${count} instalaciones`
    },
    settingsFields: 'Campos de configuración',
    mcpServers: 'Servidores MCP',
    archivedChats: 'Chats archivados',
    sections: {
      maintenance: 'Mantenimiento',
      sessions: 'Sesiones',
      system: 'Sistema',
      usage: 'Uso'
    },
    sectionDescriptions: {
      maintenance: 'Diagnóstico, copias de seguridad, curador y datos de memoria',
      sessions: 'Buscar y gestionar sesiones',
      system: 'Estado, registros y acciones del sistema',
      usage: 'Actividad de tokens, coste y skills a lo largo del tiempo'
    },
    nav: {
      newChat: {
        title: 'Nueva sesión',
        detail: 'Inicia una sesión nueva'
      },
      settings: {
        title: 'Configuración',
        detail: 'Configura Hermes Desktop'
      },
      capabilities: {
        title: 'Capacidades',
        detail: 'Skills, herramientas, servidores MCP y plugins'
      },
      messaging: {
        title: 'Mensajería',
        detail: 'Configura Telegram, Slack, Discord y más'
      },
      artifacts: {
        title: 'Artefactos',
        detail: 'Explora salidas generadas'
      }
    },
    sectionEntries: {
      sessions: {
        title: 'Panel de sesiones',
        detail: 'Busca, fija y gestiona sesiones'
      },
      system: {
        title: 'Panel del sistema',
        detail: 'Estado del gateway, registros, reinicio/actualización'
      },
      usage: {
        title: 'Panel de uso',
        detail: 'Actividad de tokens, coste y skills'
      }
    },
    providerNavigate: 'Navegar',
    providerSessions: 'Sesiones',
    refresh: 'Actualizar',
    refreshing: 'Actualizando...',
    noResults: 'No se encontraron resultados.',
    pinSession: 'Fijar sesión',
    unpinSession: 'Desfijar sesión',
    exportSession: 'Exportar sesión',
    deleteSession: 'Eliminar sesión',
    noSessions: 'Aún no hay sesiones.',
    gatewayRunning: 'Gateway de mensajería en ejecución',
    gatewayStopped: 'Gateway de mensajería detenido',
    hermesActiveSessions: (version, count) => `Hermes ${version} · Sesiones activas ${count}`,
    restartGateway: 'Reiniciar gateway',
    openBrowser: 'Abrir navegador',
    gatewayRestartFailed: 'No se pudo reiniciar el gateway.',
    sharedGatewayRestartTitle: '¿Reiniciar el gateway compartido?',
    sharedGatewayRestartDescription: (bots: string) => `Todos los bots de este dispositivo se reconectan: ${bots}`,
    sharedGatewayRestartConfirm: 'Reiniciar todo',
    sharedGatewayRestarted: (count: number) =>
      `Gateway compartido reiniciado (${count} ${count === 1 ? 'bot' : 'bots'})`,
    updateHermes: 'Actualizar Hermes',
    reloadWindow: 'Recargar ventana',
    actionRunning: 'en ejecución',
    actionDone: 'listo',
    actionFailed: 'falló',
    actionStartedWaiting: 'Acción iniciada, esperando estado...',
    loadingStatus: 'Cargando estado...',
    recentLogs: 'Registros recientes',
    noLogs: 'Aún no hay registros cargados.',
    days: count => `${count}d`,
    statSessions: 'Sesiones',
    statApiCalls: 'Llamadas API',
    statTokens: 'Tokens entrada/salida',
    statCost: 'Coste est.',
    actualCost: cost => `real ${cost}`,
    loadingUsage: 'Cargando uso...',
    noUsage: period => `Sin uso en los últimos ${period} días.`,
    retry: 'Reintentar',
    dailyTokens: 'Tokens diarios',
    input: 'entrada',
    output: 'salida',
    noDailyActivity: 'Sin actividad diaria.',
    topModels: 'Modelos principales',
    noModelUsage: 'Aún no hay uso de modelos.',
    topSkills: 'Skills principales',
    noSkillActivity: 'Aún no hay actividad de skills.',
    actions: count => `${count} acciones`,
    logFile: 'Archivo de registro',
    logLevel: 'Nivel',
    logSearchPlaceholder: 'Filtrar líneas de registro…',
    maintenance: {
      runOps: 'Diagnóstico',
      doctor: 'Ejecutar diagnóstico',
      doctorDesc: 'Comprobar el estado de la instalación, la configuración y los proveedores',
      securityAudit: 'Auditoría de seguridad',
      securityAuditDesc: 'Analizar la configuración y las skills en busca de ajustes riesgosos',
      backup: 'Crear copia de seguridad',
      backupDesc: 'Comprimir la configuración, memorias, skills y sesiones en un archivo ZIP',
      debugShare: 'Compartir datos de depuración',
      debugShareDesc:
        'Sube un informe y registros con datos sensibles ocultos y obtén enlaces para compartir (se eliminan automáticamente en 6 h)',
      debugShareRunning: 'Subiendo informe de depuración…',
      debugShareLinks: 'Enlaces para compartir',
      debugShareFailed: 'No se pudieron compartir los datos de depuración',
      copyLink: 'Copiar enlace',
      linkCopied: 'Enlace copiado',
      curator: 'Curador de skills',
      curatorDesc: 'Revisión en segundo plano que archiva skills creados por agentes que ya no se usan',
      curatorPaused: 'En pausa',
      curatorActive: 'Activo',
      curatorDisabled: 'Desactivado',
      curatorLastRun: when => `Última ejecución: ${when}`,
      curatorNeverRan: 'Aún no se ha ejecutado',
      pause: 'Pausar',
      resume: 'Reanudar',
      runNow: 'Ejecutar ahora',
      memoryData: 'Datos de memoria',
      memoryDataDesc: 'Archivos de memoria integrados que se incluyen en cada sesión',
      memoryProvider: name => `Proveedor activo: ${name}`,
      builtinMemory: 'integrada',
      memoryFile: 'Memoria del agente (MEMORY.md)',
      userFile: 'Perfil de usuario (USER.md)',
      bytes: size => size,
      empty: 'vacío',
      resetMemory: 'Restablecer memoria',
      resetUser: 'Restablecer perfil',
      resetAll: 'Restablecer ambos',
      resetConfirm: target => `¿Eliminar ${target}? Esta acción no se puede deshacer.`,
      resetDone: files => `Eliminado: ${files}.`,
      resetFailed: 'No se pudo restablecer la memoria',
      actionStarted: name => `${name} se inició; siguiendo el registro…`,
      actionFailed: name => `No se pudo iniciar ${name}`,
      running: 'En ejecución…',
      viewLog: 'Registro de acciones'
    }
  },
  messaging: {
    search: 'Buscar mensajería...',
    loading: 'Cargando plataformas de mensajería...',
    loadFailed: 'No se pudieron cargar las plataformas de mensajería',
    states: {
      connected: 'Conectado',
      connecting: 'Conectando',
      disabled: 'Deshabilitado',
      fatal: 'Error',
      gateway_stopped: 'Gateway de mensajería detenido',
      not_configured: 'Necesita configuración',
      pending_restart: 'Reinicio necesario',
      retrying: 'Reintentando',
      startup_failed: 'Falló el inicio'
    },
    unknown: 'Desconocido',
    hintPendingRestart: 'Reinicia el gateway desde la barra de estado para aplicar este cambio.',
    sharedListenerUrl: 'Servido en el listener del gateway compartido en',
    hintGatewayStopped: 'Inicia el gateway desde la barra de estado para conectar.',
    credentialsSet: 'Credenciales definidas',
    needsSetup: 'Necesita configuración',
    gatewayStopped: 'Gateway de mensajería detenido',
    getCredentials: 'Obtener credenciales',
    openSetupGuide: 'Abrir guía de configuración',
    required: 'Obligatorio',
    recommended: 'Recomendado',
    advanced: count => `Avanzado (${count})`,
    noTokenNeeded:
      'Esta plataforma no necesita un token aquí. Usa la guía de configuración de arriba y actívala abajo.',
    enabled: 'Activado',
    disabled: 'Desactivado',
    unsavedChanges: 'Cambios sin guardar',
    saving: 'Guardando...',
    saveChanges: 'Guardar cambios',
    saved: 'Guardado',
    replaceValue: 'Reemplazar valor actual',
    openDocs: 'Abrir docs',
    clearField: key => `Limpiar ${key}`,
    enableAria: name => `Activar ${name}`,
    disableAria: name => `Desactivar ${name}`,
    platformEnabled: name => `${name} activado`,
    platformDisabled: name => `${name} desactivado`,
    restartToApply: 'Reinicia el gateway para que este cambio surta efecto.',
    setupSaved: name => `Configuración de ${name} guardada`,
    restartToReconnect: 'Reinicia el gateway para reconectar con las credenciales nuevas.',
    appliedLive: 'Aplicado al gateway en ejecución.',
    connectingLive: 'El gateway en ejecución se está conectando con las credenciales nuevas.',
    keyCleared: key => `${key} limpiado`,
    setupUpdated: name => `La configuración de ${name} se actualizó.`,
    failedUpdate: name => `No se pudo actualizar ${name}`,
    failedSave: name => `No se pudo guardar ${name}`,
    failedClear: key => `No se pudo limpiar ${key}`,
    pendingRequests: count => `Solicitudes pendientes (${count})`,
    pendingAria: count =>
      `${count} solicitud${count === 1 ? '' : 'es'} de emparejamiento pendiente${count === 1 ? '' : 's'}`,
    approvedUsers: count => `Usuarios aprobados (${count})`,
    approve: 'Aprobar',
    approving: 'Aprobando...',
    revoke: 'Revocar',
    revoking: 'Revocando...',
    revokeAria: name => `Revocar ${name}`,
    revokeTitle: 'Revocar acceso',
    revokeDesc: name => `${name} perderá el acceso y dejará de ser reconocido en su próximo mensaje.`,
    approvedUser: name => `${name} aprobado`,
    approvedHint: 'Se le reconoce automáticamente en su próximo mensaje.',
    revokedUser: name => `${name} revocado`,
    failedApprove: name => `No se pudo aprobar a ${name}`,
    failedRevoke: name => `No se pudo revocar a ${name}`,
    pairingLockedOut: 'Demasiados fallos de aprobación — esta plataforma está bloqueada. Inténtalo de nuevo más tarde.',
    waitingSince: minutes => (minutes < 1 ? 'justo ahora' : `hace ${minutes}m`),
    restartNeeded: 'Guardado. Reinicia el gateway de mensajería para que la nueva configuración surta efecto.',
    restartNow: 'Reiniciar ahora',
    restarting: 'Reiniciando…',
    restartFailedManual: 'Hermes no pudo reiniciarse para aplicar tu configuración de mensajería',
    restartFailedManualDetail:
      'Vuelve a pulsar Reiniciar; si sigue fallando, abre los registros y envía un diagnóstico.',
    restartAgain: 'Reiniciar de nuevo',
    openLogs: 'Abrir registros',
    telegramQr: {
      title: 'Elige cómo conectar tu bot de Telegram',
      subtitle:
        'Ambas opciones conectan un bot que controlas y guardan sus credenciales solo en esta instalación de Hermes.',
      quickSetup: 'Configuración rápida',
      recommended: 'Recomendado',
      quickHelp:
        'Escanea un código QR y confirma en Telegram. Hermes crea el bot y detecta tu ID de usuario de Telegram automáticamente.',
      createWithQr: 'Crear con QR',
      starting: 'Iniciando…',
      replaceWarning:
        'Ya hay credenciales de Telegram configuradas. Una nueva configuración por QR o un nuevo token de bot sustituirá al bot actual cuando guardes.',
      scanHint: 'Escanéalo con la app de Telegram en tu teléfono o abre el enlace en este equipo.',
      waiting: 'Esperando a Telegram…',
      expiresIn: (remaining: string) => `Caduca en ${remaining}`,
      expired: 'Caducado',
      openTelegram: 'Abrir Telegram',
      ready: 'Bot creado',
      allowedUsers: 'Usuarios permitidos',
      ownerDetected: 'Propietario detectado',
      addAtLeastOne: 'Añade al menos un ID de usuario de Telegram.',
      userIdPlaceholder: 'ID de usuario de Telegram',
      add: 'Añadir',
      numericOnly: 'Los ID de usuario de Telegram permitidos deben ser numéricos.',
      saveAndRestart: 'Guardar y reiniciar',
      applying: 'Guardando…',
      pairingExpired:
        'La vinculación con Telegram caducó. Inicia una nueva configuración por QR para volver a intentarlo.',
      stillWaiting: (detail: string) => `Seguimos esperando a Telegram. Reintentando tras: ${detail}`,
      savedRestarting: 'Telegram guardado; reiniciando el gateway…',
      savedRestartFailed: (detail: string) => `Telegram guardado; falló el reinicio del gateway${detail}`
    },
    fieldCopy: {
      TELEGRAM_BOT_TOKEN: {
        label: 'Token del bot',
        help: 'Crea un bot con @BotFather y pega el token que te dé.',
        placeholder: 'Pegar token del bot de Telegram'
      },
      TELEGRAM_ALLOWED_USERS: {
        label: 'IDs de usuarios de Telegram permitidos',
        help: 'Recomendado. IDs numéricos separados por comas desde @userinfobot. Sin esto, cualquiera puede enviar DM a tu bot.'
      },
      TELEGRAM_PROXY: {
        label: 'URL de proxy',
        help: 'Solo necesario en redes donde Telegram está bloqueado.'
      },
      DISCORD_BOT_TOKEN: {
        label: 'Token del bot',
        help: 'Crea una aplicación en Discord Developer Portal, añade un bot y pega su token.'
      },
      DISCORD_ALLOWED_USERS: {
        label: 'IDs de usuarios de Discord permitidos',
        help: 'Recomendado. IDs de usuarios de Discord separados por comas.'
      },
      DISCORD_REPLY_TO_MODE: {
        label: 'Estilo de respuesta',
        help: '`first`, `all` u `off`.'
      },
      DISCORD_ALLOW_ALL_USERS: {
        label: 'Permitir todos los usuarios de Discord',
        help: 'Solo desarrollo. Si es true, cualquiera puede enviar DM al bot sin allowlist.'
      },
      DISCORD_HOME_CHANNEL: {
        label: 'ID del canal principal',
        help: 'Canal donde el bot envía mensajes proactivos (salida de cron, recordatorios).'
      },
      DISCORD_HOME_CHANNEL_NAME: {
        label: 'Nombre del canal principal',
        help: 'Nombre visible del canal principal en registros y salida de estado.'
      },
      BLUEBUBBLES_ALLOW_ALL_USERS: {
        label: 'Permitir todos los usuarios de iMessage',
        help: 'Si es true, omite la allowlist de BlueBubbles.'
      },
      MATTERMOST_ALLOW_ALL_USERS: {
        label: 'Permitir todos los usuarios de Mattermost'
      },
      MATTERMOST_HOME_CHANNEL: {
        label: 'Canal principal'
      },
      QQ_ALLOW_ALL_USERS: {
        label: 'Permitir todos los usuarios de QQ'
      },
      QQBOT_HOME_CHANNEL: {
        label: 'Canal principal de QQ',
        help: 'Canal o grupo predeterminado para entrega cron.'
      },
      QQBOT_HOME_CHANNEL_NAME: {
        label: 'Nombre del canal principal de QQ'
      },
      SLACK_BOT_TOKEN: {
        label: 'Token del bot de Slack',
        help: 'Usa el token del bot de OAuth & Permissions después de instalar tu app de Slack.',
        placeholder: 'Pegar token del bot de Slack'
      },
      SLACK_APP_TOKEN: {
        label: 'Token de app de Slack',
        help: 'Usa el token de nivel de app requerido para Socket Mode.',
        placeholder: 'Pegar token de app de Slack'
      },
      SLACK_ALLOWED_USERS: {
        label: 'IDs de usuarios de Slack permitidos',
        help: 'Recomendado. IDs de Slack separados por comas.'
      },
      MATTERMOST_URL: {
        label: 'URL del servidor',
        placeholder: 'https://mattermost.example.com'
      },
      MATTERMOST_TOKEN: {
        label: 'Token del bot'
      },
      MATTERMOST_ALLOWED_USERS: {
        label: 'IDs de usuarios permitidos',
        help: 'Recomendado. IDs de Mattermost separados por comas.'
      },
      MATRIX_HOMESERVER: {
        label: 'URL del homeserver',
        placeholder: 'https://matrix.org'
      },
      MATRIX_ACCESS_TOKEN: {
        label: 'Token de acceso'
      },
      MATRIX_USER_ID: {
        label: 'ID de usuario del bot',
        placeholder: '@hermes:example.org'
      },
      MATRIX_ALLOWED_USERS: {
        label: 'IDs de usuarios de Matrix permitidos',
        help: 'Recomendado. IDs separados por comas en formato @usuario:servidor.'
      },
      SIGNAL_HTTP_URL: {
        label: 'URL del puente Signal',
        placeholder: 'http://127.0.0.1:8080',
        help: 'URL de un puente REST signal-cli en ejecución.'
      },
      SIGNAL_ACCOUNT: {
        label: 'Número de teléfono',
        help: 'El número registrado con tu puente signal-cli.'
      },
      SIGNAL_ALLOWED_USERS: {
        label: 'Usuarios de Signal permitidos',
        help: 'Recomendado. Identificadores de Signal separados por comas.'
      },
      WHATSAPP_ENABLED: {
        label: 'Activar puente de WhatsApp',
        help: 'Se define automáticamente con el interruptor de abajo. No lo toques salvo que sepas que lo necesitas.'
      },
      WHATSAPP_MODE: {
        label: 'Modo del puente'
      },
      WHATSAPP_ALLOWED_USERS: {
        label: 'Usuarios de WhatsApp permitidos',
        help: 'Recomendado. Números de teléfono o IDs de WhatsApp separados por comas.'
      }
    },
    platformIntro: {}
  },
  webhooks: {
    search: 'Buscar webhooks…',
    loading: 'Cargando webhooks…',
    loadFailed: 'No se pudieron cargar los webhooks',
    subscriptions: count => `Suscripciones (${count})`,
    hint: 'Los cambios en las suscripciones se recargan al instante cuando el receptor está en ejecución. Las suscripciones desactivadas rechazan los eventos entrantes.',
    empty: 'Aún no hay suscripciones de webhook.',
    disabledTitle: 'Receptor de webhooks desactivado',
    disabledBody:
      'Los webhooks tienen su propia plataforma de gateway. Actívalos aquí para aceptar eventos HTTP entrantes; los canales de chat solo son necesarios cuando una suscripción entrega contenido a Telegram, Discord, Slack u otro canal.',
    enable: 'Activar webhooks',
    enabling: 'Activando…',
    enabled: name => `Activado: "${name}"`,
    disabled: name => `Desactivado: "${name}"`,
    enableRow: 'Activar',
    disableRow: 'Desactivar',
    delete: 'Eliminar',
    deleting: 'Eliminando…',
    deleted: 'Webhook eliminado',
    deleteTitle: 'Eliminar webhook',
    deleteDescPrefix: 'Esto eliminará permanentemente ',
    deleteDescSuffix: '. No se puede deshacer.',
    deleteFailed: name => `No se pudo eliminar "${name}"`,
    toggleFailed: (name, enabled) => `No se pudo ${enabled ? 'activar' : 'desactivar'} "${name}"`,
    newSubscription: 'Nueva suscripción',
    restarting: 'Reiniciando el gateway…',
    restartNeeded:
      'Los webhooks están activados, pero el gateway todavía debe reiniciarse para que el receptor pueda ponerse en línea.',
    restartGateway: 'Reiniciar gateway',
    restartingGateway: 'Reiniciando…',
    restartFailed: detail => `Falló el reinicio del gateway${detail}`,
    enabledRestarting: 'Webhooks activados; reiniciando el gateway…',
    all: '(todos)',
    deliverOnly: 'solo entrega',
    createdTitle: 'Suscripción creada',
    createdSecretHint: 'Copia el secreto ahora; solo se muestra una vez.',
    webhookUrl: 'URL del webhook',
    secretOnce: 'Secreto (se muestra una vez)',
    done: 'Listo',
    fieldName: 'Nombre',
    fieldNamePlaceholder: 'p. ej., github-push',
    fieldDescription: 'Descripción',
    fieldDescriptionPlaceholder: 'Qué hace este webhook (opcional)',
    fieldEvents: 'Eventos',
    fieldEventsPlaceholder: 'separados por comas; en blanco para todos',
    fieldSkills: 'Skills',
    fieldSkillsPlaceholder: 'nombres de skills separados por comas (opcional)',
    fieldDeliver: 'Entregar a',
    fieldDeliverOnly: 'Entregar solo la carga útil',
    fieldPrompt: 'Prompt',
    fieldPromptPlaceholder: 'Instrucciones para el agente cuando se active este webhook (opcional)',
    nameRequired: 'Nombre obligatorio',
    create: 'Crear',
    creating: 'Creando…',
    created: 'Creado',
    createFailed: detail => `No se pudo crear: ${detail}`,
    copy: 'Copiar',
    deliverOptions: {
      log: 'Registro',
      telegram: 'Telegram',
      discord: 'Discord',
      slack: 'Slack',
      email: 'Correo electrónico',
      github_comment: 'Comentario de GitHub'
    }
  },
  profiles: {
    close: 'Cerrar perfiles',
    nameHint: 'Letras minúsculas, dígitos, guiones y guiones bajos. Debe empezar con una letra o dígito.',
    title: 'Perfiles',
    count: count => `${count} ${count === 1 ? 'perfil' : 'perfiles'}`,
    search: 'Buscar perfiles…',
    loading: 'Cargando perfiles...',
    newProfile: 'Nuevo perfil',
    importProfile: 'Importar perfil…',
    exportProfile: 'Exportar perfil…',
    imported: 'Perfil importado',
    exported: 'Perfil exportado',
    failedImport: 'No se pudo importar el perfil',
    failedExport: 'No se pudo exportar el perfil',
    allProfiles: 'Todos los perfiles',
    showAllProfiles: 'Mostrar todos los perfiles',
    switchToProfile: name => `Cambiar a ${name}`,
    switchToConnection: name => `Cambiar a ${name}`,
    switchConnectionFailed: name => `No se pudo conectar a ${name}`,
    manageProfiles: 'Gestionar perfiles...',
    connectGateway: 'Gestionar gateways…',
    fleet: {
      allOnGateway: 'Todos los perfiles de este gateway',
      gateway: (gateway: string) => `Perfiles en ${gateway}`,
      gatewayUnreachable: (gateway: string) => `${gateway} · inaccesible`,
      onGateway: (name: string, gateway: string) => `${name} · ${gateway}`,
      switchTo: (name: string, gateway: string) => `Cambiar a ${name} en ${gateway}`,
      deleteOn: (gateway: string) => ` en ${gateway}`,
      localDevice: 'Este dispositivo (backend local: instala Hermes si falta; si no, abre una sesión nueva)',
      switchDeviceTitle: '¿Cambiar a este dispositivo?',
      switchDeviceDesc:
        'Esto abre una sesión nueva en este equipo. La conversación actual permanece en el otro gateway.',
      switchDeviceConfirm: 'Cambiar',
      installDeviceTitle: '¿Cambiar a este dispositivo?',
      installDeviceDesc:
        'Esto instalará Hermes localmente y luego abrirá una sesión nueva en este equipo. No se instala nada hasta que confirmes.',
      installDeviceConfirm: 'Instalar localmente',
      connectExistingInstead: 'Conectar uno existente en su lugar'
    },
    status: {
      unread: (count: number) => (count === 1 ? '1 sesión sin leer' : `${count} sesiones sin leer`),
      needsInput: (count: number) =>
        count === 1 ? '1 sesión espera tu respuesta' : `${count} sesiones esperan tu respuesta`,
      working: (count: number) => (count === 1 ? '1 sesión en ejecución' : `${count} sesiones en ejecución`)
    },
    remoteOverride: {
      menuItem: 'Conectar a un host remoto…',
      badge: (host: string) => `Se ejecuta en ${host}`,
      title: (profile: string) => `Conectar ${profile} a un host remoto`,
      description:
        'Las sesiones de este perfil se ejecutarán en el Hermes remoto que indiques, en lugar de en este equipo.',
      urlLabel: 'Dirección remota',
      urlPlaceholder: 'https://hermes.example.com',
      urlInvalid: 'Introduce una dirección completa que empiece por http:// o https://',
      tokenLabel: 'Token de acceso',
      tokenPlaceholder: 'Pega el token de sesión remota',
      tokenSavedHint: 'Ya hay un token guardado. Déjalo en blanco para conservarlo.',
      plainTextOptIn:
        'Este equipo no tiene almacenamiento seguro de claves, así que el token se guardaría sin cifrar en el disco. Guardarlo de todos modos.',
      collisionWarning: (label: string) =>
        `Ya existe un gateway llamado “${label}” en Configuración. Esta conexión del perfil es independiente y no lo cambiará.`,
      confirmTitle: '¿Conectar este perfil a un host remoto?',
      confirmNote: (profile: string, host: string) =>
        `Los chats nuevos de ${profile} se ejecutarán en ${host}. Ese equipo ejecutará comandos y leerá archivos allí, no en este. Conéctate solo a un host de confianza.`,
      confirmBack: 'Atrás',
      connect: 'Conectar',
      connecting: 'Conectando…',
      disconnect: 'Quitar conexión remota',
      savedTitle: 'Perfil conectado',
      savedMessage: (profile: string, host: string) => `${profile} ahora se ejecuta en ${host}`,
      removedTitle: 'Conexión remota quitada',
      removedMessage: (profile: string) => `${profile} ahora se ejecuta en este equipo`,
      removeFailed: 'No se pudo quitar la conexión remota',
      authFailedTitle: 'El host remoto rechazó el token guardado',
      authFailedMessage: (profile: string, host: string) =>
        `${host} rechazó el token guardado para ${profile}. Puede que se haya cambiado en el lado remoto.`,
      updateToken: 'Introducir nuevo token…'
    },
    actions: 'Acciones',
    color: 'Color...',
    colorFor: 'Color',
    openInNewWindow: 'Abrir en una ventana nueva',
    setAsDefault: 'Establecer como predeterminado',
    defaultProfile: 'Perfil predeterminado',
    defaultSet: (name: string) => `${name} es ahora el predeterminado`,
    defaultDescription:
      'Se usa al abrir Hermes y para los chats nuevos. Las sesiones existentes se quedan en sus perfiles.',
    failedSetDefault: 'No se pudo establecer el perfil predeterminado',
    setColor: color => `Definir color ${color}`,
    autoColor: 'Auto',
    noProfiles: 'Aún no hay perfiles.',
    selectPrompt: 'Selecciona un perfil para ver sus detalles.',
    refresh: 'Actualizar perfiles',
    refreshing: 'Actualizando perfiles',
    default: 'predeterminado',
    skills: count => `${count} ${count === 1 ? 'skill' : 'skills'}`,
    env: 'env',
    defaultBadge: 'Predeterminado',
    rename: 'Renombrar',
    renameMenu: 'Renombrar…',
    exportMenu: 'Exportar…',
    editSoul: 'Editar SOUL.md…',
    copySetup: 'Copiar configuración',
    copying: 'Copiando...',
    modelLabel: 'Modelo',
    skillsLabel: 'Skills',
    notSet: 'Sin definir',
    soulDesc: 'El prompt de sistema y las instrucciones de persona integrados en este perfil.',
    soulOptional: 'opcional',
    soulPlaceholder: mode =>
      `El prompt de sistema / persona para este perfil.\nDéjalo en blanco para conservar el ${mode} predeterminado.`,
    soulPlaceholderCloned: 'clonado',
    soulPlaceholderEmpty: 'vacío',
    unsavedChanges: 'Cambios sin guardar',
    loadingSoul: 'Cargando SOUL.md...',
    emptySoul: 'SOUL.md vacío; empieza a escribir la persona...',
    saving: 'Guardando...',
    saveSoul: 'Guardar SOUL.md',
    deleteTitle: '¿Eliminar perfil?',
    deleteDescPrefix: 'Esto eliminará ',
    deleteDescMid: ' y quitará su directorio ',
    deleteDescSuffix: '. Esto no se puede deshacer.',
    deleting: 'Eliminando...',
    createDesc: 'Los perfiles son entornos independientes de Hermes: configuración, skills y SOUL.md separados.',
    nameLabel: 'Nombre',
    cloneFrom: 'Clonar desde',
    cloneFromNone: 'Ninguno (vacío)',
    cloneFromDesc: 'Copia la configuración, las skills y SOUL.md del perfil de origen seleccionado.',
    cloneFromDefault: 'Clonar desde el predeterminado',
    cloneFromDefaultDesc: 'Copia configuración, skills y SOUL.md desde tu perfil predeterminado.',
    invalidName: hint => `Nombre no válido. ${hint}`,
    nameRequired: 'El nombre es obligatorio.',
    creating: 'Creando...',
    createAction: 'Crear perfil',
    renameTitle: 'Renombrar perfil',
    renameDescPrefix: 'Renombrar actualiza el directorio del perfil y cualquier script wrapper en ',
    renameDescSuffix: '.',
    displayNameTitle: 'Ponle nombre a este agente',
    displayNameDesc:
      'Define un nombre visible que se muestra en toda la app. El ID interno del perfil sigue siendo "default".',
    displayNameLabel: 'Nombre visible',
    newNameLabel: 'Nombre nuevo',
    renaming: 'Renombrando...',
    created: 'Perfil creado',
    renamed: 'Perfil renombrado',
    deleted: 'Perfil eliminado',
    setupCopied: 'Comando de configuración copiado',
    soulSaved: 'SOUL.md guardado',
    failedLoad: 'No se pudieron cargar los perfiles',
    failedDelete: 'No se pudo eliminar el perfil',
    failedCopy: 'No se pudo copiar el comando de configuración',
    failedLoadSoul: 'No se pudo cargar SOUL.md',
    failedSaveSoul: 'No se pudo guardar SOUL.md',
    failedCreate: 'No se pudo crear el perfil',
    failedRename: 'No se pudo renombrar el perfil'
  },
  modelAssignment: {
    saveFailed: 'Hermes no guardó ese cambio de modelo.',
    confirmTitle: 'Aviso sobre la selección de modelo',
    confirmDetail: 'Confirma solo si aceptas esta contrapartida.',
    confirmAction: 'Confirmar',
    declined: 'Cambio de modelo cancelado: rechazaste el aviso del nivel con entrenamiento de datos.'
  },
  cron: {
    close: 'Cerrar cron',
    title: 'Tareas programadas',
    count: count => `${count} ${count === 1 ? 'tarea' : 'tareas'}`,
    search: 'Buscar tareas cron...',
    loading: 'Cargando tareas cron...',
    states: {
      enabled: 'activada',
      scheduled: 'programada',
      running: 'en ejecución',
      paused: 'pausada',
      disabled: 'deshabilitada',
      error: 'falló la última ejecución',
      completed: 'completada'
    },
    lastRunFailed: 'Falló la última ejecución:',
    editJob: 'Editar tarea',
    runAgain: 'Ejecutar de nuevo',
    deliveryLabels: {
      local: 'Este escritorio',
      telegram: 'Telegram',
      discord: 'Discord',
      slack: 'Slack',
      email: 'Email'
    },
    scheduleLabels: {
      daily: 'Diario',
      weekdays: 'Días laborables',
      weekly: 'Semanal',
      monthly: 'Mensual',
      hourly: 'Cada hora',
      'every-15-minutes': 'Cada 15 minutos',
      custom: 'Personalizado'
    },
    scheduleHints: {
      daily: 'Todos los días a las 9:00',
      weekdays: 'De lunes a viernes a las 9:00',
      weekly: 'Cada lunes a las 9:00',
      monthly: 'El primer día de cada mes a las 9:00',
      hourly: 'Al inicio de cada hora',
      'every-15-minutes': 'Cada 15 minutos',
      custom: 'Sintaxis cron o lenguaje natural'
    },
    days: {
      '0': 'Domingo',
      '1': 'Lunes',
      '2': 'Martes',
      '3': 'Miércoles',
      '4': 'Jueves',
      '5': 'Viernes',
      '6': 'Sábado',
      '7': 'Domingo'
    },
    dayFallback: value => `día ${value}`,
    everyDayAt: time => `Todos los días a las ${time}`,
    weekdaysAt: time => `Días laborables a las ${time}`,
    everyDayOfWeekAt: (day, time) => `Cada ${day} a las ${time}`,
    monthlyOnDayAt: (dayOfMonth, time) => `Mensual el día ${dayOfMonth} a las ${time}`,
    topOfHour: 'Al inicio de cada hora',
    everyHourAt: minute => `Cada hora en :${minute}`,
    newCron: 'Nueva tarea cron',
    emptyDescNew:
      'Programa un prompt para ejecutarlo con una expresión cron. Hermes lo ejecutará y entregará los resultados al destino que elijas.',
    emptyDescSearch: 'Prueba una búsqueda más amplia.',
    emptyTitleNew: 'Aún no hay tareas programadas',
    emptyTitleSearch: 'Sin coincidencias',
    last: 'Última:',
    next: 'Siguiente:',
    overdueSince: 'Atrasada desde:',
    noRuns: 'Sin ejecuciones aún',
    manage: 'Gestionar',
    showRuns: 'Mostrar ejecuciones',
    hideRuns: 'Ocultar ejecuciones',
    runHistory: 'Historial de ejecuciones',
    actionsTitle: 'Acciones de tarea cron',
    resume: 'Reanudar cron',
    pause: 'Pausar cron',
    resumeTitle: 'Reanudar',
    pauseTitle: 'Pausar',
    triggerNow: 'Ejecutar ahora',
    edit: 'Editar cron',
    deleteTitle: '¿Eliminar tarea cron?',
    deleteDescPrefix: 'Esto eliminará ',
    deleteDescSuffix: ' permanentemente. Dejará de ejecutarse de inmediato.',
    deleting: 'Eliminando...',
    resumed: 'Cron reanudado',
    paused: 'Cron pausado',
    triggered: 'Cron ejecutado',
    deleted: 'Cron eliminado',
    created: 'Cron creado',
    updated: 'Cron actualizado',
    failedLoad: 'No se pudieron cargar las tareas cron',
    failedUpdate: 'No se pudo actualizar la tarea cron',
    failedTrigger: 'No se pudo ejecutar la tarea cron',
    failedDelete: 'No se pudo eliminar la tarea cron',
    failedSave: 'No se pudo guardar la tarea cron',
    editTitle: 'Editar tarea cron',
    createTitle: 'Nueva tarea cron',
    editDesc: 'Actualiza la programación, el prompt o el destino. Los cambios se aplican en la próxima ejecución.',
    createDesc:
      'Programa un prompt para ejecutarlo automáticamente. Usa sintaxis cron o una frase como "cada 15 minutos".',
    nameLabel: 'Nombre',
    namePlaceholder: 'Resumen matutino',
    promptLabel: 'Prompt',
    promptPlaceholder: 'Resume mis hilos de Slack sin leer y envíame por email los 5 principales...',
    frequencyLabel: 'Frecuencia',
    deliverLabel: 'Entregar a',
    deliverNeedsHomeChannel: 'configura primero un canal principal',
    modelLabel: 'Modelo',
    modelDefault: 'Predeterminado (modelo global)',
    customScheduleLabel: 'Programación personalizada',
    customPlaceholder: '0 9 * * * o días laborables a las 9',
    customHint: 'Expresión cron, o frases como "cada hora" o "días laborables a las 9".',
    optional: 'Opcional',
    promptRequired: 'El prompt es obligatorio.',
    promptScheduleRequired: 'El prompt y la programación son obligatorios.',
    scheduleRequired: 'La programación es obligatoria.',
    scriptOnlyEditHint: 'Tarea solo con script (sin prompt de IA). ID de la tarea:',
    saveChanges: 'Guardar cambios',
    createAction: 'Crear cron',
    tabs: {
      jobs: 'Tareas',
      blueprints: 'Plantillas'
    },
    blueprints: {
      tab: 'Plantillas',
      startFrom: 'Empezar con',
      custom: 'Personalizada',
      subtitle: 'Automatizaciones listas para usar',
      dialogDesc: 'Completa los detalles y prográmala.',
      scheduleIt: 'Programar',
      scheduling: 'Programando…',
      scheduled: 'Plantilla programada',
      loading: 'Cargando plantillas…',
      failedLoad: 'No se pudieron cargar las plantillas',
      emptyTitle: 'No hay plantillas disponibles',
      emptyDesc: 'Este backend no ofrece plantillas de automatización.'
    }
  },
  artifacts: {
    search: 'Buscar artefactos...',
    refresh: 'Actualizar artefactos',
    refreshing: 'Actualizando artefactos',
    indexing: 'Indexando artefactos recientes de sesiones',
    tabAll: 'Todo',
    tabImages: 'Imágenes',
    tabFiles: 'Archivos',
    tabLinks: 'Enlaces',
    noArtifactsTitle: 'No se encontraron artefactos',
    noArtifactsDesc:
      'Las imágenes generadas y las salidas de archivo aparecerán aquí cuando las sesiones las produzcan.',
    failedLoad: 'No se pudieron cargar los artefactos',
    openFailed: 'No se pudo abrir',
    itemsImage: 'imágenes',
    itemsLink: 'enlaces',
    itemsFile: 'archivos',
    itemsGeneric: 'elementos',
    zero: '0',
    rangeOf: (start, end, total) => `${start}-${end} de ${total}`,
    goToPage: (itemLabel, page) => `Ir a la página ${page} de ${itemLabel}`,
    colTitleLink: 'Título del enlace',
    colTitleFile: 'Nombre',
    colTitleDefault: 'Título / nombre',
    colLocationLink: 'URL',
    colLocationFile: 'Ruta',
    colLocationDefault: 'Ubicación',
    colSession: 'Sesión',
    kindImage: 'imagen',
    kindFile: 'archivo',
    kindLink: 'enlace',
    chat: 'Chat',
    copyUrl: 'Copiar URL',
    copyPath: 'Copiar ruta'
  },
  artifactCard: {
    kind: {
      code: 'Código',
      html: 'Página interactiva',
      svg: 'Gráfico'
    },
    generating: lines => `Generando… ${lines} líneas`,
    versionBadge: count => `${count} versiones`,
    open: 'Abrir'
  },
  artifactPreview: {
    versionOf: (current, total) => `v${current} de ${total}`,
    olderVersion: 'Versión anterior',
    newerVersion: 'Versión más reciente',
    latest: 'Última',
    copyContent: 'Copiar contenido',
    download: 'Descargar',
    openInBrowser: 'Abrir en el navegador',
    openInBrowserFailed: 'No se pudo abrir en el navegador',
    missingTitle: 'Artefacto no disponible',
    missingBody: 'Este artefacto ya no está en el registro local.'
  },
  sidebar: {
    filter: {
      grouping: 'Agrupación',
      ordering: 'Orden',
      show: 'Mostrar',
      filters: 'Filtros',
      status: 'Estado',
      pullRequest: 'Pull request',
      profile: 'Perfil',
      project: 'Proyecto',
      archived: 'Archivados',
      resetToDefaults: 'Restablecer valores predeterminados',
      expandAll: 'Expandir todo',
      collapseAll: 'Contraer todo',
      inboxStyle: 'Estilo bandeja de entrada',
      updated: 'Actualizado',
      created: 'Creado',
      tokens: 'Tokens',
      cost: 'Coste',
      manual: 'Manual',
      preview: 'Vista previa',
      pr: 'PR',
      needsInput: 'Requiere respuesta',
      working: 'Trabajando',
      unread: 'No leídos',
      draft: 'Borrador',
      idle: 'Inactivo',
      open: 'Abierta',
      merged: 'Fusionada',
      closed: 'Cerrada',
      noPR: 'Sin PR'
    },
    gatewayGroups: {
      grouping: 'Gateway y perfil',
      rename: 'Renombrar grupo',
      aliasLabel: 'Nombre visible',
      aliasHint: 'Solo el nombre visible; los nombres del gateway y del perfil no cambian.',
      resetName: 'Restablecer nombre',
      moveUp: 'Subir',
      moveDown: 'Bajar',
      reorder: 'Reordenar grupo',
      actions: 'Acciones del grupo'
    },
    profileRail: 'Barra de perfiles',
    nav: {
      'new-session': 'Nueva sesión',
      capabilities: 'Capacidades',
      messaging: 'Mensajería',
      artifacts: 'Artefactos',
      cron: 'Tareas programadas'
    },
    searchAria: 'Buscar sesiones',
    searchPlaceholder: 'Buscar sesiones…',
    clearSearch: 'Limpiar búsqueda',
    noMatch: query => `Ninguna sesión coincide con “${query}”.`,
    results: 'Resultados',
    pinned: 'Fijadas',
    sessions: 'Sesiones',
    terminal: 'Terminal',
    files: 'Archivos',
    review: 'Revisión',
    logs: 'Registros',
    cronJobs: 'Tareas cron',
    groupAriaGrouped: 'Mostrar sesiones como una sola lista',
    groupAriaUngrouped: 'Agrupar sesiones por espacio de trabajo',
    showProjects: 'Mostrar proyectos',
    showSessions: 'Mostrar sesiones',
    groupTitleGrouped: 'Desagrupar sesiones',
    groupTitleUngrouped: 'Agrupar por espacio de trabajo',
    allPinned: 'Todo aquí está fijado. Desfija un chat para mostrarlo en recientes.',
    shiftClickHint: 'Mayús-clic en un chat para fijarlo',
    noWorkspace: 'Sin espacio de trabajo',
    projectEmpty: 'Aún no hay sesiones',
    projectLoadFailed: 'No se pudieron cargar las sesiones',
    noSessions: 'Aún no hay sesiones',
    storageCorrupt: {
      title: 'La base de datos de sesiones está dañada',
      body: (profiles: string) =>
        `Hermes no puede leer todo el historial de sesiones de ${profiles}. Los chats que faltan en esta lista no se eliminaron; el archivo donde se guardan está dañado.`,
      action:
        'Sal de Hermes en este perfil y luego inspecciona el archivo sin modificarlo, o restaura una instantánea:',
      guide: 'Guía de recuperación'
    },
    noFilterMatches: 'Ninguna sesión coincide con estos filtros',
    projects: {
      showAllSessions: 'Mostrar todas las sesiones',
      sectionLabel: 'Proyectos',
      home: 'Inicio',
      autoDiscovered: 'Detectado automáticamente',
      newButton: 'Nuevo proyecto',
      createTitle: 'Nuevo proyecto',
      createDesc: 'Asigne un nombre a un espacio de trabajo y agregue una o más carpetas.',
      renameTitle: 'Cambiar nombre del proyecto',
      addFolderTitle: 'Agregar carpeta',
      namePlaceholder: 'p.ej. Skunkworks',
      foldersLabel: 'Carpetas',
      ideaLabel: 'Idea',
      ideaPlaceholder: '¿De qué se trata este proyecto? (guardado en IDEA.md)',
      ideaGenerate: 'Generar idea',
      ideaGenerating: 'Generando…',
      ideaShuffle: 'Plantillas aleatorias',
      noFolders: 'Aún no se han agregado carpetas.',
      addFolder: 'Agregar carpeta',
      primaryBadge: 'principal',
      removeFolder: 'Eliminar',
      create: 'Crear',
      menu: 'Acciones',
      menuRename: 'Renombrar',
      menuAppearance: 'Apariencia',
      noColor: 'Sin color',
      menuAddFolder: 'Agregar carpeta',
      menuSetActive: 'Establecer activo',
      menuDelete: 'Borrar',
      moveToProject: 'Mover a proyecto',
      movedTo: name => `Movido a ${name}`,
      moveFailed: 'No se pudo mover la sesión',
      moveNoFolder: 'Ese proyecto no tiene carpeta a la que mover',
      moveNoProjects: 'No hay otros proyectos',
      reveal: 'Revelar en carpeta',
      copyPath: 'Copiar ruta',
      removeFromSidebar: 'Ocultar de la barra lateral',
      createFailed: 'No se pudo crear el proyecto',
      staleBackend:
        'Actualiza el backend de Hermes para crear proyectos: tu backend es más antiguo que esta aplicación de escritorio (Configuración → Actualizaciones → Backend).',
      deleteConfirm:
        'Esto elimina el proyecto guardado de Hermes. Los archivos, los repositorios de git y los árboles de trabajo permanecen intactos.',
      startWork: 'Nuevo worktree',
      newWorktreeTitle: 'Nuevo worktree',
      newWorktreeDesc: 'Asigna un nombre a la rama de este worktree.',
      branchPlaceholder: 'p.ej. mi-característica',
      branchOff: () => ({ after: '', before: 'ramificar desde ' }),
      baseBranchPlaceholder: 'Buscar ramas…',
      baseBranchNone: 'No se encontraron ramas',
      startWorkFailed: 'No se pudo crear el worktree',
      worktreeStaleBackend:
        'Actualiza el backend de Hermes para crear worktrees por esta conexión remota: es anterior a la API de git worktree.',
      worktreeProjectLabel: 'Proyecto',
      worktreeProjectPlaceholder: 'Buscar proyectos…',
      worktreeProjectNone: 'Ningún proyecto con carpeta',
      convertBranch: 'Convertir una rama…',
      convertBranchTitle: 'Convertir una rama',
      convertBranchDesc: 'Abre ramas ya activas o crea un worktree para una rama disponible.',
      convertBranchPlaceholder: 'Buscar ramas…',
      convertBranchInstead: 'Convertir una rama existente',
      branchOpenExisting: 'abrir',
      branchSwitchHome: 'cambiar al principal',
      branchCreateWorktree: 'nuevo worktree',
      branchTrackRemote: 'seguir remota',
      branchesLoading: 'Cargando ramas…',
      noBranches: 'No se encontraron ramas',
      removeWorktree: 'Eliminar worktree',
      removeWorktreeFailed: 'No se pudo eliminar el worktree (¿hay cambios sin confirmar?)',
      removeWorktreeConfirm:
        'Elimínalo de Git (se borra el directorio del worktree; la rama se conserva) o simplemente oculta el carril de la barra lateral y deja el worktree en disco.',
      removeWorktreeDirty:
        'Este worktree tiene cambios sin confirmar. Fuerza la eliminación (se descartarán esos cambios) o simplemente oculta el carril y consérvalo en disco.',
      forceRemove: 'Forzar eliminación',
      enter: label => `Abrir ${label}`,
      reorder: label => `Reordenar ${label}`,
      toggle: (label, open) => `${open ? 'Mostrar' : 'Ocultar'} sesiones de ${label}`,
      showAllCount: (count: number) => `Mostrar las ${count} sesiones`,
      back: 'Todos los proyectos'
    },
    newSessionIn: label => `Nueva sesión en ${label}`,
    showMoreIn: (count, label) => `Mostrar ${count} más en ${label}`,
    loading: 'Cargando…',
    loadMore: 'Cargar más',
    loadCount: step => `Cargar ${step} más`,
    messageCount: (count: number) => `${count} ${count === 1 ? 'mensaje' : 'mensajes'}`,
    toolCallCount: (count: number) => `${count} ${count === 1 ? 'llamada a herramienta' : 'llamadas a herramientas'}`,
    row: {
      pin: 'Fijar',
      unpin: 'Desfijar',
      markUnread: 'Marcar como no leído',
      markRead: 'Marcar como leído',
      unreadFailed: 'No se pudo actualizar el estado de no leído',
      copyId: 'Copiar ID',
      export: 'Exportar',
      branchFrom: 'Rama',
      rename: 'Renombrar',
      archive: 'Archivar',
      unarchive: 'Desarchivar',
      newWindow: 'Nueva ventana',
      openInTerminal: 'Abrir en el terminal',
      hideTabBar: 'Ocultar barra de pestañas',
      openInNewTab: 'Abrir en una pestaña nueva',
      openInSplit: 'Abrir en vista dividida',
      copyIdFailed: 'No se pudo copiar el ID de sesión',
      sessionActions: 'Acciones de sesión',
      sessionRunning: 'Sesión en ejecución',
      needsInput: 'Necesita tu respuesta',
      waitingForAnswer: 'Esperando tu respuesta',
      finishedUnread: 'Finalizada — sin leer',
      backgroundRunning: 'Tarea en segundo plano en ejecución',
      draftSession: 'Borrador — aún no se ha enviado nada',
      handoffOrigin: platform => `Transferido desde ${platform}`,
      ownedByProfile: profile => `Perfil: ${profile}`,
      renamed: 'Renombrada',
      renameFailed: 'No se pudo renombrar',
      renameTitle: 'Renombrar sesión',
      renameDesc: 'Déjalo vacío para limpiarlo.',
      untitledPlaceholder: 'Sesión sin título',
      deleteTitle: '¿Eliminar la sesión?',
      deleteDesc: (title: string) => `Se eliminará “${title}” de forma permanente. No se puede deshacer.`,
      deleting: 'Eliminando…',
      deleted: 'Sesión eliminada',
      untitledChat: id => `Chat ${id}`,
      messageCount: count => `${count} ${count === 1 ? 'mensaje' : 'mensajes'}`,
      todoProgress: 'Tareas completadas',
      ageNow: 'ahora',
      ageDay: 'd',
      ageHour: 'h',
      ageMin: 'm'
    },
    dateDivider: {
      today: 'Hoy, más temprano',
      yesterday: 'Ayer',
      thisWeek: 'Esta semana',
      lastWeek: 'La semana pasada',
      thisMonth: 'Este mes'
    },
    statusDivider: {
      working: 'En progreso',
      done: 'Completado'
    },
    markAllRead: 'Marcar todo como leído'
  },
  composer: {
    message: 'Mensaje',
    wakingProfile: profile => `Despertando ${profile}…`,
    placeholderStarting: 'Iniciando Hermes...',
    placeholderReconnecting: 'Reconectando con Hermes…',
    placeholderFollowUp: 'Enviar seguimiento',
    newSessionPlaceholders: [
      '¿Qué vamos a construir?',
      'Dale una tarea a Hermes',
      '¿Qué tienes en mente?',
      'Describe lo que necesitas',
      '¿Qué abordamos?',
      'Pregunta lo que quieras',
      'Empieza con un objetivo'
    ],
    followUpPlaceholders: [
      'Enviar seguimiento',
      'Añadir más contexto',
      'Refinar la solicitud',
      '¿Qué sigue?',
      'Sigamos',
      'Llevarlo más lejos',
      'Ajustar o continuar'
    ],
    startVoice: 'Iniciar conversación de voz',
    openDirective: 'Abrir',
    queueMessage: 'Poner mensaje en cola',
    steer: 'Guiar la ejecución actual',
    stop: 'Detener',
    send: 'Enviar',
    speaking: 'Hablando',
    transcribing: 'Transcribiendo',
    thinking: 'Pensando',
    muted: 'Silenciado',
    listening: 'Escuchando',
    muteMic: 'Silenciar micrófono',
    unmuteMic: 'Activar micrófono',
    stopListening: 'Dejar de escuchar y enviar',
    stopShort: 'Detener',
    endConversation: 'Terminar conversación de voz',
    endShort: 'Terminar',
    stopDictation: 'Detener dictado',
    transcribingDictation: 'Transcribiendo dictado',
    voiceControls: 'Voz',
    voiceEngine: 'Motor del chat de voz',
    voiceEngineChained: 'Voz a texto + voz de Hermes',
    voiceEngineLive: 'GPT-Live (full-duplex, delega en Hermes)',
    voiceEngineLiveNeedsKey: 'Requiere una clave API de OpenAI',
    voiceEngineChangeFailed: 'No se pudo cambiar el motor del chat de voz',
    voiceEngineChainedShort: 'voz a texto',
    voiceEngineLiveShort: 'GPT-Live',
    voiceDictation: 'Dictado por voz',
    speakReplies: 'Leer las respuestas en voz alta',
    stopSpeakingReplies: 'Dejar de leer las respuestas en voz alta',
    wakeWord: (phrase: string) => `Palabra de activación "${phrase}"`,
    wakeWordListening: phrase => `Palabra de activación: "${phrase}" — escuchando`,
    wakeWordOff: phrase => `Palabra de activación: "${phrase}" — desactivada`,
    wakeWordPausedVoice: phrase => `Palabra de activación: "${phrase}" — pausada durante el chat de voz`,
    lookupLoading: 'Buscando…',
    lookupNoMatches: 'Sin coincidencias.',
    lookupTry: 'Prueba',
    lookupOr: 'o',
    commonCommands: 'Comandos comunes',
    hotkeys: 'Atajos',
    helpFooter: 'abre el panel completo · retroceso descarta',
    commandDescs: {
      '/help': 'Mostrar los comandos de barra del escritorio',
      '/clear': 'iniciar una sesión nueva',
      '/resume': 'Reanudar una sesión guardada',
      '/details': 'controlar el nivel de detalle de la transcripción',
      '/copy': 'copiar selección o último mensaje del asistente',
      '/quit': 'salir de hermes',
      '/start': 'Confirmar los pings de inicio de la plataforma sin responder',
      '/new': 'Iniciar un chat nuevo de escritorio',
      '/topic': 'Activar o inspeccionar las sesiones por tema de los MD de Telegram',
      '/save': 'Guardar la transcripción actual en JSON',
      '/retry': 'Reintentar el último mensaje (reenviarlo al agente)',
      '/prompt': 'Redactar tu siguiente prompt en $EDITOR (markdown) y enviarlo',
      '/undo': 'Retroceder N turnos del usuario y volver a preguntar (1 por defecto)',
      '/title': 'Renombrar la sesión actual',
      '/handoff': 'Pasar esta sesión a una plataforma de mensajería',
      '/branch': 'Ramificar el último mensaje en un chat nuevo',
      '/worktree': 'Mostrar, listar, crear o podar worktrees de git aislados',
      '/compress': 'Comprimir el contexto de esta conversación',
      '/rollback':
        'Listar o restaurar puntos de control del sistema de archivos (las restauraciones conservan tus ediciones manuales; --all las sobrescribe)',
      '/export': 'Exportar un perfil (configuración, skills, tema) a un archivo compartible',
      '/import': 'Importar un archivo de perfil compartido como perfil nuevo',
      '/stop': 'Detener el turno activo y los procesos en segundo plano',
      '/pause': "Pausar globalmente el trabajo nuevo (parada de emergencia); '/pause off' lo reanuda",
      '/bg': 'Ejecutar un prompt en una sesión independiente en segundo plano',
      '/btw': 'Hacer una pregunta al margen sobre esta conversación sin interrumpirla',
      '/agents': 'Mostrar los agentes activos y las tareas en curso',
      '/journey': 'Abrir el grafo de memoria: skills y recuerdos a lo largo del tiempo',
      '/queue':
        'Poner un prompt en cola para el siguiente turno, o listar/editar/quitar/mover/vaciar los prompts en cola',
      '/steer': 'Insertar un mensaje tras la siguiente llamada a herramienta sin interrumpir',
      '/goal': 'Fijar un objetivo permanente en el que Hermes trabaja durante varios turnos hasta cumplirlo',
      '/heartbeat': 'Configurar un prompt recurrente que vuelve a esta sesión cuando está inactiva',
      '/refine': 'Revisar esta conversación ahora y guardar lo aprendido en memoria/skills',
      '/review':
        'Lanzar un subagente independiente para revisar el trabajo que se acaba de comentar (PR, código, docs)',
      '/loop': 'Volver a ejecutar un prompt a intervalos regulares en esta sesión',
      '/plan': 'Escribir un plan de implementación en markdown en .hermes/plans/ sin ejecutar nada',
      '/moa': 'Ejecutar un prompt con el preajuste predeterminado de Mixture of Agents y luego restaurar tu modelo',
      '/subgoal': 'Añadir o gestionar criterios adicionales del objetivo activo',
      '/status': 'Mostrar el estado de la sesión actual',
      '/egress': 'Mostrar el estado del proxy de salida de Docker',
      '/context':
        'Mostrar la vista detallada de la ventana de contexto con indicador de uso, desglose por categoría, estadísticas de compresión y rendimiento',
      '/whoami': 'Mostrar tu acceso a los comandos de barra (admin / usuario)',
      '/profile': 'Cambiar el perfil activo de Hermes',
      '/codex-runtime': 'Activar o desactivar el runtime codex app-server para modelos OpenAI/Codex',
      '/personality': 'Establecer una personalidad predefinida',
      '/battery': 'Mostrar u ocultar un indicador de batería por colores en la barra de estado',
      '/timestamps': 'Mostrar u ocultar las marcas de tiempo [HH:MM] en los mensajes y /history',
      '/diff': 'Mostrar los cambios de git en el directorio de trabajo',
      '/focus': 'Activar o desactivar la vista de enfoque: solo tu prompt y la respuesta final',
      '/yolo': 'Activar o desactivar YOLO: aprobar automáticamente los comandos peligrosos',
      '/approvals': 'Mostrar o establecer el modo persistente de aprobación de comandos peligrosos',
      '/reasoning': 'Esfuerzo o visualización del razonamiento [<level> [--global]|show|hide|full|clamp]',
      '/skin': 'Cambiar el tema de escritorio o pasar al siguiente',
      '/wake': 'Controlar la escucha de la palabra de activación del escritorio [on|off|status]',
      '/tools': 'Gestionar herramientas: /tools [list|disable|enable] [name...]',
      '/memory': 'Revisar las escrituras de memoria pendientes / activar o desactivar la aprobación',
      '/bundles': 'Listar los paquetes de skills (alias /<name> para varias skills)',
      '/pet': 'Mostrar u ocultar o adoptar una mascota de petdex (/pet, /pet list, /pet boba)',
      '/hatch': 'Generar una mascota nueva (abre el generador)',
      '/learn': 'Aprender una skill reutilizable a partir de lo que describas (carpetas, URL, este chat, notas)',
      '/init': 'Generar o actualizar las instrucciones de proyecto AGENTS.md a partir de un análisis del repositorio',
      '/suggestions': 'Revisar las automatizaciones sugeridas (aceptar/descartar)',
      '/blueprint': 'Configurar una automatización a partir de una plantilla',
      '/browser': 'Gestionar la conexión CDP del navegador [connect|disconnect|status] (solo gateway local)',
      '/palette': 'Abrir la paleta de comandos aproximada (también Ctrl+P)',
      '/usage':
        'Mostrar el uso de tokens y los límites de frecuencia; `reset` canjea un restablecimiento de límite de Codex acumulado',
      '/subscription': 'Ver tu plan de Nous y cambiarlo en el navegador',
      '/topup': 'Mostrar tu saldo de Nous y gestionar la facturación en el portal',
      '/platform': 'Pausar, reanudar o listar una plataforma del gateway que falla',
      '/version': 'Mostrar la versión de Hermes Agent',
      '/debug': 'Subir un informe de depuración (información del sistema + registros) y obtener enlaces para compartir',
      '/model': 'Cambiar el modelo de esta sesión'
    },
    hotkeyDescs: {
      'composer.mention': 'referenciar archivos, carpetas, URL y Git',
      'composer.slash': 'paleta de comandos con /',
      'composer.help': 'esta ayuda rápida (eliminar para descartarla)',
      'composer.sendNewline': 'enviar · Shift+Enter para insertar una línea nueva',
      'composer.sendQueued': 'enviar el siguiente turno en cola',
      'keybinds.openPanel': 'todos los atajos de teclado',
      'composer.cancel': 'cerrar el menú emergente · cancelar la ejecución',
      'composer.history': 'recorrer el menú emergente o el historial'
    },
    attachUrlTitle: 'Adjuntar una URL',
    attachUrlDesc: 'Hermes obtendrá la página y la incluirá como contexto para este turno.',
    urlPlaceholder: 'https://example.com/post',
    urlHintPre: 'Incluye la URL completa, p. ej. ',
    attach: 'Adjuntar',
    queued: count => `${count} en cola`,
    queuedPaused: count => `${count} en cola — en pausa`,
    attachmentOnly: 'Turno solo con adjunto',
    emptyTurn: 'Turno vacío',
    hiddenQueued: 'Nota de configuración',
    attachments: count => `${count} ${count === 1 ? 'adjunto' : 'adjuntos'}`,
    editingInComposer: 'Editando en el compositor',
    editingQueuedInComposer: 'Editando turno en cola en el compositor',
    restoredDraftNotice: 'Se restauró tu mensaje sin enviar',
    restoredDraftUndo: 'Deshacer',
    queueEdit: 'Editar',
    queueSendNext: 'Próximo',
    queueSteer: 'Redirigir — encauzar el turno en vivo ahora',
    queueSend: 'Enviar',
    queueDelete: 'Borrar',
    queueResume: 'Reanudar',
    queueResumeTip: 'La cola se pausó al detener; reanuda el envío de los turnos en cola',
    queueStuckTitle: 'Mensaje en cola no enviado',
    queueStuckBody: 'Un turno en cola no llegó a enviarse. Sigue en la cola; vuelve a intentarlo.',
    previewUnavailable: 'Vista previa no disponible',
    previewLabel: label => `Vista previa de ${label}`,
    couldNotPreview: label => `No se pudo previsualizar ${label}`,
    removeAttachment: label => `Quitar ${label}`,
    dictating: 'Dictando',
    preparingAudio: 'Preparando audio',
    speakingResponse: 'Respuesta hablada',
    readingAloud: 'Leyendo en voz alta',
    themeSuggestions: 'Sugerencias de tema de escritorio',
    noMatchingThemes: 'No hay temas coincidentes.',
    themeTryPre: 'Prueba ',
    themeTryPost: '.',
    attachLabel: 'Adjuntar',
    files: 'Archivos…',
    folder: 'Carpeta…',
    images: 'Imágenes…',
    pasteImage: 'Pegar imagen',
    url: 'URL…',
    promptSnippets: 'Fragmentos de prompt…',
    tipPre: 'Consejo: escribe ',
    tipPost: ' para referenciar archivos en línea.',
    snippetsTitle: 'Fragmentos de prompt',
    snippetsDesc: 'Elige un prompt inicial para insertarlo en el compositor.',
    dropFiles: 'Suelta archivos para adjuntarlos',
    dropSession: 'Suelta para enlazar este chat',
    mcpSuggestions: {
      label: server => `Añadir ${server}`,
      tip: keyword => `Sugerido porque mencionaste "${keyword}" — haz clic para conectar`,
      connecting: server => `Conectando ${server}…`,
      cancelTip: 'Haz clic para cancelar',
      added: server => `${server} añadido`,
      addedTip: 'Conectado — sus herramientas están listas en este chat',
      connectFailed: server => `No se pudo conectar ${server}`
    },
    skillSuggestions: {
      label: skill => `Usar habilidad: ${skill}`,
      tip: skill => `Mencionaste "${skill}" — haz clic para empezar con esa habilidad`,
      done: skill => `Añadido /${skill}`,
      doneTip: 'La habilidad se carga cuando envíes'
    },
    githubSuggestions: {
      label: 'Configurar GitHub',
      tip: 'GitHub funciona aquí mediante las skills de la CLI gh: haz clic para conectar tu cuenta',
      done: 'Se añadió /github-auth',
      doneTip: 'Envía el mensaje y el agente te guiará para iniciar sesión en GitHub'
    },
    repairSuggestions: {
      label: server => `Reconectar ${server}`,
      tip: server => `Una llamada a ${server} acaba de fallar con un error de conexión`,
      working: server => `Reconectando ${server}…`,
      workingTip: 'Haz clic para cancelar',
      done: server => `${server} reconectado`,
      doneTip: 'Las credenciales nuevas están activas en este chat',
      failed: server => `No se pudo reconectar ${server}`
    },
    cronSuggestions: {
      label: 'Programar esto',
      tip: phrase => `"${phrase}" suena recurrente — ejecútalo con un horario en su lugar`,
      prefix: 'Configúralo como tarea programada:',
      done: 'Marcado para programar',
      doneTip: 'Envíalo y el agente creará la tarea'
    },
    snippets: {
      codeReview: {
        label: 'Revisión de código',
        description: 'Audita el cambio actual en busca de regresiones, casos límite omitidos y pruebas faltantes.',
        text: 'Revisa esto para detectar bugs, regresiones y pruebas faltantes.'
      },
      implementationPlan: {
        label: 'Plan de implementación',
        description: 'Esboza un enfoque antes de tocar código para mantener el diff enfocado.',
        text: 'Haz un plan de implementación conciso antes de cambiar código.'
      },
      explainThis: {
        label: 'Explica esto',
        description: 'Recorre cómo funciona el código seleccionado y enlaza los archivos clave.',
        text: 'Explica cómo funciona esto y señala los archivos clave.'
      }
    }
  },
  statusStack: {
    hideStack: 'Ocultar la pila de estado',
    showStack: 'Mostrar la pila de estado',
    agents: 'Agentes',
    background: count => `${count} en segundo plano`,
    goalActive: 'Objetivo activo',
    goalBlocked: 'Objetivo bloqueado',
    goalDone: 'Objetivo completado',
    goalPaused: 'Objetivo en pausa',
    goalWaiting: 'Objetivo esperando',
    subagents: count => `${count} subagente${count === 1 ? '' : 's'}`,
    todos: (done, total) => `Tareas ${done}/${total}`,
    running: 'En ejecución',
    stop: 'Detener',
    dismiss: 'Descartar',
    exit: code => `salida ${code}`,
    control: {
      goalActiveTurns: (turn: number, maxTurns: number) => `Turno ${turn}/${maxTurns}`,
      goalDoneTurns: (turns: number) => `${turns} ${turns === 1 ? 'turno' : 'turnos'}`,
      goalTurn: (turn: number) => `Turno ${turn}`,
      goalActions: 'Acciones del objetivo',
      viewDetails: 'Ver detalles',
      addCriterion: 'Añadir criterio',
      addCriterionDialogTitle: 'Añadir criterio',
      addCriterionPlaceholder: 'Escribe el texto del criterio...',
      criterionLabel: 'Criterio',
      pauseGoal: 'Pausar objetivo',
      resumeGoal: 'Reanudar objetivo',
      resumeNow: 'Reanudar ahora',
      clearGoal: 'Borrar objetivo',
      clearGoalConfirmTitle: '¿Borrar el objetivo?',
      clearGoalConfirmBody: '¿Seguro que quieres borrar el objetivo activo? No se puede deshacer.',
      copyCriterion: (index: number) => `Copiar criterio ${index}`,
      removeCriterion: (index: number) => `Quitar criterio ${index}`,
      removeCriterionConfirmTitle: (index: number) => `¿Quitar el criterio ${index}?`,
      removeCriterionConfirmBody: (index: number) => `¿Seguro que quieres quitar el criterio ${index}?`,
      clearCriteria: 'Borrar todos los criterios',
      clearCriteriaConfirmTitle: '¿Borrar todos los criterios?',
      clearCriteriaConfirmBody: '¿Seguro que quieres quitar todos los criterios de este objetivo?',
      criteriaHeader: (count: number) => `Criterios · ${count}`,
      noCriteria: 'Sin criterios',
      goalDetailsTitle: 'Detalles del objetivo',
      objectiveLabel: 'Objetivo',
      contractOutcome: 'Resultado',
      contractVerification: 'Verificación',
      contractConstraints: 'Restricciones',
      contractBoundaries: 'Límites',
      contractStopWhen: 'Detener cuando',
      waitBarrierTitle: 'Condición de espera',
      waitUntil: (target: string) => `Esperando hasta ${target}`,
      waitSession: (target: string) => `Esperando a la sesión ${target}`,
      waitPid: (pid: number) => `Esperando al proceso ${pid}`,
      qualityGatesTitle: 'Controles de calidad',
      gateCommand: 'Comando',
      gateAttempts: (attempts: number, max: number) => `${attempts}/${max} intentos`,
      gateTimeout: (seconds: number) => `tiempo límite de ${seconds} s`,
      gateLastExit: (code: number | null) => (code === null ? 'Pendiente' : `Código de salida: ${code}`),
      loopActive: 'Bucle activo',
      loopPaused: 'Bucle en pausa',
      loopDeferred: 'Bucle aplazado',
      loopFinished: 'Bucle terminado',
      loopRuns: (runs: number) => `${runs} ${runs === 1 ? 'ejecución' : 'ejecuciones'}`,
      loopRunCount: (current: number, total: number) => `Ejecución ${current}/${total}`,
      loopNext: (time: string) => `próxima ${time}`,
      loopEverySeconds: (seconds: number) => `cada ${seconds} s`,
      loopEveryMinutes: (minutes: number) => `cada ${minutes} min`,
      loopEveryHours: (hours: number) => `cada ${hours} h`,
      loopSelfPaced: 'a su ritmo',
      loopActions: 'Acciones del bucle',
      pauseLoop: 'Pausar bucle',
      resumeLoop: 'Reanudar bucle',
      stopLoop: 'Detener bucle',
      stopLoopConfirmTitle: '¿Detener el bucle?',
      stopLoopConfirmBody: '¿Seguro que quieres detener este bucle?',
      dismissLoop: 'Descartar bucle',
      loopPromptLabel: 'Prompt',
      loopCadenceLabel: 'Frecuencia',
      loopUntilLabel: 'Condición de fin',
      loopDeferredNotice: 'Un objetivo activo controla la sesión en este momento.',
      loopAwaitingResponse: 'Esperando respuesta',
      heartbeatActive: 'Heartbeat activo',
      heartbeatPaused: 'Heartbeat en pausa',
      heartbeatEveryMinutes: (minutes: number) => `cada ${minutes} min`,
      heartbeatEveryHours: (hours: number) => `cada ${hours} h`,
      heartbeatEverySeconds: (seconds: number) => `cada ${seconds} s`,
      heartbeatNext: (time: string) => `próximo ${time}`,
      heartbeatDueWaitingForIdle: 'pendiente: esperando inactividad',
      heartbeatActions: 'Acciones del heartbeat',
      pauseHeartbeat: 'Pausar heartbeat',
      resumeHeartbeat: 'Reanudar heartbeat',
      clearHeartbeat: 'Borrar heartbeat',
      clearHeartbeatConfirmTitle: '¿Borrar el heartbeat?',
      clearHeartbeatConfirmBody: '¿Seguro que quieres borrar este heartbeat?',
      heartbeatFiredCount: (count: number) => `Se activó ${count} ${count === 1 ? 'vez' : 'veces'}`,
      actionFailed: (msg: string) => `La acción falló: ${msg}`,
      actionSucceeded: 'La acción se completó',
      copySuccess: 'Criterio copiado al portapapeles',
      copyFailure: 'No se pudo copiar el criterio al portapapeles',
      continuationFailed: 'No se pudo enviar la continuación del objetivo',
      continuationQueued: 'Objetivo reanudado: la continuación queda en cola hasta que termine el turno actual',
      continuationBusy: 'Objetivo reanudado: la sesión está ocupada; usa /interrupt en el turno actual para continuar',
      controlUnavailable: (msg: string) => `Controles de sesión no disponibles: ${msg}`,
      dismissError: 'Descartar error',
      add: 'Añadir'
    },
    coding: {
      title: 'Árbol de trabajo',
      noBranch: 'Sin rama',
      detached: 'separada',
      clean: 'Limpio',
      changed: count => `${count} cambio${count === 1 ? '' : 's'}`,
      ahead: count => `${count} por delante`,
      behind: count => `${count} por detrás`,
      review: 'Revisar',
      close: 'Cerrar',
      openChanges: 'Abrir cambios',
      openFile: 'Abrir archivo',
      stage: 'Preparar',
      unstage: 'Quitar de preparación',
      stageAll: 'Preparar todo',
      viewAsTree: 'Ver como árbol',
      viewAsList: 'Ver como lista',
      revert: 'Revertir',
      revertAll: 'Revertir todo',
      revertConfirm:
        '¿Descartar los cambios en este archivo y restaurarlo al estado comprometido? Esto no se puede deshacer.',
      revertAllConfirm:
        '¿Descartar todos los cambios y restaurar archivos al estado comprometido? Esto no se puede deshacer.',
      staged: 'Preparados',
      noChanges: 'Sin cambios',
      notRepo: 'No es un repositorio Git',
      noDiff: 'No hay diferencias que mostrar',
      scopeUncommitted: 'Sin confirmar',
      scopeBranch: 'Rama',
      scopeLastTurn: 'Último turno',
      commit: 'Hacer commit',
      commitAndPush: 'Hacer commit y enviar',
      commitPlaceholder: (shortcut: string) => `Mensaje (${shortcut} para hacer commit)`,
      generateCommitMessage: 'Generar mensaje de commit',
      stopGenerating: 'Dejar de generar',
      createPr: 'Crear PR',
      openPr: 'Abrir PR',
      ghMissing: 'Instala GitHub CLI (gh) e inicia sesión para abrir PR',
      agentShip: 'Pedir a Hermes que abra un PR',
      agentShipUnavailable: 'El chat al que pertenecen estos cambios no está en pantalla.',
      agentShipPrompt:
        'Revisa los cambios actuales, haz un commit con un mensaje convencional claro, envía la rama y abre un pull request.',
      newBranch: 'Nueva rama',
      branchOffFrom: base => `Nueva rama desde ${base}`,
      switchTo: branch => `Cambiar a ${branch}`,
      switchFailed: branch => `No se pudo cambiar a ${branch}`,
      worktrees: 'Worktrees'
    }
  },
  updates: {
    discontinuedTitle: 'Esta versión de Hermes ya no tiene soporte',
    discontinuedBody: 'Esta versión de Hermes ya no tiene soporte y podría dejar de funcionar; desinstálala. Tus datos permanecen en el disco.',
    channels: { stable: 'Estable', canary: 'Canary' },
    appName: 'Hermes',
    availableBodyRelease: tag => `La versión ${tag} está lista para instalarse.`,
    releaseAvailable: tag => `La versión ${tag} está disponible.`,
    checkingShort: 'Comprobando…',
    availableBodyAppInstaller: 'Hay una nueva versión de Hermes. Hermes se cerrará, Windows terminará la actualización y Hermes volverá a abrirse automáticamente.',
    applyingBodyAppInstaller: 'Hermes se cerrará y Windows terminará la actualización. Hermes volverá a abrirse al finalizar; no tienes que hacer nada.',
    applyingCloseAppInstaller: 'Esta ventana se cerrará; Windows terminará la actualización y Hermes volverá a abrirse automáticamente.',
    checkUnknownTitleAppInstaller: 'No se pudieron buscar actualizaciones',
    checkUnknownBodyAppInstaller: 'Windows no pudo buscar actualizaciones ahora. También se instalan automáticamente al reiniciar Hermes.',
    versionDetailsTitle: 'Detalles de la versión',
    versionDetailsBody: 'Esta instalación se administra fuera de la app. Actualízala de la misma forma en que la instalaste.',
    versionDetailsVersion: 'Versión',
    versionDetailsCommit: 'Commit',
    versionDetailsBuildOrigin: 'Origen de la compilación',
    versionDetailsDistribution: 'Distribución',
    versionDetailsDistributionDesktop: 'Aplicación de escritorio',
    versionDetailsDistributionDesktopMsix: 'Aplicación de escritorio (MSIX)',
    versionDetailsDistributionDesktopInstaller: 'Aplicación de escritorio (instalador)',
    versionDetailsDistributionSourceInstaller: 'Código fuente (script de instalación)',
    versionDetailsDistributionSourceInstallerDesktop: 'Código fuente (script de instalación) + hermes desktop',
    versionDetailsDistributionSource: 'Código fuente',
    versionDetailsDistributionSourceDesktop: 'Código fuente + hermes desktop',
    versionDetailsDistributionStore: 'Microsoft Store',
    versionDetailsRuntime: 'Entorno de ejecución',
    versionDetailsRuntimeEmbedded: 'Entorno de ejecución integrado',
    versionDetailsRuntimeExternal: 'Externo (usa el entorno de ejecución del equipo)',
    versionDetailsInstallId: 'ID de instalación',
    versionDetailsUncommittedChanges: 'cambios sin confirmar',
    version: value => `Versión ${value}`,
    versionUnavailable: 'Versión no disponible',
    bundleOutOfSync: 'La compilación de la app está desactualizada',
    bundleOutOfSyncDesc:
        'El entorno de ejecución de Hermes se actualizó, pero la app de escritorio sigue siendo una compilación anterior: faltarán funciones nuevas de la interfaz (como el modo Bot) hasta que se actualice. Ejecuta la actualización de abajo para recompilar la app. Si eso no elimina este aviso, reinstala desde el instalador de escritorio más reciente.',
    bundleOutOfSyncAction: 'Obtener el instalador',
    bundleSwapPending: 'Reinicia para terminar la actualización',
    bundleSwapPendingDesc:
        'La app actualizada ya está instalada; Hermes solo necesita reiniciarse para cargarla. Los chats y los ajustes no se tocan.',
    bundleSwapPendingAction: 'Reiniciar Hermes',
    checkNow: 'Comprobar ahora',
    seeWhatsNew: 'Ver novedades',
    releaseNotes: 'Notas de la versión',
    onLatest: 'Ya tienes la versión más reciente.',
    installing: 'Se está instalando una actualización.',
    cantReach: 'No pudimos contactar con el servidor de actualizaciones.',
    tapCheck: 'Pulsa "Comprobar ahora" para buscar actualizaciones.',
    updateReady: count =>
        `Hay una actualización lista (${count} ${count === 1 ? 'cambio incluido' : 'cambios incluidos'}).`,
    updateReadyUnknown: 'Hay una nueva actualización lista.',
    lastChecked: age => `Última comprobación ${age}`,
    justNowSuffix: ' · ahora mismo',
    never: 'nunca',
    justNow: 'ahora mismo',
    minAgo: count => `hace ${count} min`,
    hoursAgo: count => `hace ${count} h`,
    daysAgo: count => `hace ${count} d`,
    stages: {
      idle: 'Preparando…',
      prepare: 'Preparando…',
      fetch: 'Descargando…',
      pull: 'Casi listo…',
      pydeps: 'Terminando…',
      update: 'Actualizando Hermes…',
      rebuild: 'Reconstruyendo la aplicación de escritorio…',
      restart: 'Reiniciando Hermes…',
      done: 'Actualización completada',
      manual: 'Actualizar desde la terminal',
      guiSkew: 'Actualiza la aplicación de escritorio',
      error: 'Actualización pausada'
    },
    checking: 'Buscando actualizaciones…',
    checkFailedTitle: 'No se pudieron buscar actualizaciones',
    tryAgain: 'Intentar de nuevo',
    notAvailableTitle: 'Actualización no disponible',
    unsupportedMessage: 'Esta versión de Hermes no puede actualizarse desde la app.',
    connectionRetry:
      'Hermes no pudo llegar al servidor de actualizaciones. Comprueba tu conexión a internet y vuelve a intentarlo. Si usas un Hermes remoto, asegúrate de que esté en línea.',
    gitUnusable: 'Hermes no pudo ejecutar Git en este equipo, así que no pudo buscar actualizaciones.',
    connectionSettings: 'Configuración de conexión',
    openDownloadPage: 'Abrir la página de descarga',
    latestBody: 'Estás usando la versión más reciente.',
    latestBodyBackend: 'El backend está ejecutando la versión más reciente.',
    allSetTitle: 'Todo listo',
    availableTitle: 'Nueva actualización disponible',
    availableBody: 'Hay una nueva versión de Hermes lista para instalar.',
    availableTitleBackend: 'Actualización del backend disponible',
    availableBodyBackend:
      'Hay una versión más reciente del backend de Hermes al que estás conectado lista para instalar.',
    availableBodyNoChangelog:
      'Hay una versión más reciente lista. Las notas de la versión no están disponibles para este tipo de instalación.',
    updateNow: 'Actualizar ahora',
    maybeLater: 'Quizá más tarde',
    moreChanges: count => `+ ${count} ${count === 1 ? 'cambio incluido' : 'cambios incluidos'}.`,
    manualTitle: 'Actualizar desde la terminal',
    manualBody:
      'Instalaste Hermes desde la línea de comandos, así que las actualizaciones también se ejecutan ahí. Pega esto en tu terminal:',
    manualPickedUp: 'Hermes usará la nueva versión la próxima vez que lo abras.',
    guiSkewTitle: 'Actualiza la aplicación de escritorio',
    guiSkewBody:
      'El backend se actualizó, pero el paquete de esta aplicación de escritorio no cambió. Actualiza o reinstala la aplicación de escritorio de Hermes (tu AppImage / .deb / .rpm) para que coincidan.',
    copy: 'Copiar',
    copied: 'Copiado',
    done: 'Listo',
    applyingBody:
      'El actualizador de Hermes tomará el control en su propia ventana y volverá a abrir Hermes al terminar.',
    applyingBodyBackend:
      'El backend remoto está aplicando la actualización y se reiniciará. Hermes se reconectará automáticamente cuando vuelva a estar disponible.',
    applyingClose: 'Hermes se cerrará para aplicar la actualización.',
    errorTitle: 'La actualización no terminó',
    errorBody: 'No pasa nada: no se perdió nada. Puedes intentarlo de nuevo ahora.',
    blockerTitle: '¿Cerrar las vistas previas locales para actualizar Hermes?',
    blockerBody:
      'Hermes necesita detener estas vistas previas locales antes de actualizar. Esto no modifica ni elimina tus archivos.',
    foreignBlockerTitle: 'Cierra otros procesos para actualizar Hermes',
    foreignBlockerBody:
      'Hermes no puede cerrar estos procesos automáticamente de forma segura. Cierra la app, el terminal o el servicio al que pertenece cada uno y vuelve a intentar la actualización.',
    mixedBlockerBody:
      'Hermes puede cerrar las vistas previas locales que se indican abajo. Los demás procesos deben cerrarse manualmente antes de continuar con la actualización.',
    closePreviewsAndUpdate: 'Cerrar vistas previas y actualizar',
    closePreviewsAndCheckAgain: 'Cerrar vistas previas y volver a comprobar',
    localPreview: 'Vista previa local',
    portLabel: (port: number) => `Puerto ${port}`,
    pidLabel: (pid: number) => `PID ${pid}`,
    technicalDetails: 'Detalles técnicos',
    notNow: 'Ahora no',
    clientAlsoBehindTitle: 'La app de escritorio está desactualizada',
    clientAlsoBehindMessage:
      'El backend está actualizado, pero esta app de escritorio sigue en una versión anterior. Actualízala para obtener las últimas correcciones.',
    clientAlsoBehindAction: 'Actualizar la app de escritorio',
    everythingDispatched: 'Actualización enviada',
    everythingSkipped: 'Omitido',
    everythingRowFailed: 'Falló la actualización',
    everythingFanoutFailedTitle: 'No se pudieron actualizar las demás instancias',
    changeLogNew: 'Novedades',
    changeLogFixed: 'Corregido',
    changeLogFaster: 'Más rápido',
    changeLogImproved: 'Mejorado',
    changeLogOther: 'Otras mejoras',
    changeLogFallbackLabel: 'En esta actualización',
    changeLogFallbackItem: 'Mejoras y correcciones',
    applyStatus: {
      preparing: 'Actualizando backend…',
      pulling: 'Actualizando backend…',
      restarting: 'Reiniciando el backend para cargar la actualización…',
      notAvailable: 'La actualización no está disponible para este backend.',
      failed: 'Falló la actualización del backend.',
      noReturn:
        'El backend no volvió a estar disponible. Puede que la actualización no se haya completado; revisa el host del backend.'
    }
  },
  handoffTour: {
    profileTitle: 'Tu primera tarea se ejecuta en el perfil predeterminado',
    profileText:
      'Esta barra cambia de perfil. El que está iluminado ahora es el predeterminado, donde está la sesión de la tarea. El otro es el perfil de configuración, donde está el chat de bienvenida.',
    sessionsTitle: 'Cada perfil tiene sus propias sesiones',
    sessionsText:
      'Esta lista pertenece al perfil predeterminado. Nueva sesión crea una en el perfil que esté seleccionado. Cambia de perfil en la barra y la lista cambia con él.',
    stayTitle: 'Hermes está a un clic',
    stayText: 'Cambia al perfil de configuración y abre Bienvenida a Hermes siempre que necesites ayuda. Se queda ahí.'
  },
  guidedGreeting: {
    line: 'Hola, pasa. Soy Hermes. Dame dos minutos para prepararlo todo a tu medida y luego me pondremos a trabajar en algo que de verdad quieras hacer.\n\nPero antes, ¿cómo quieres que te llame?',
    nameSuggestion: (name: string) => `(También puedo llamarte simplemente ${name}, si lo prefieres.)`
  },
  install: {
    stageStates: {
      pending: 'Pendiente',
      running: 'Instalando',
      succeeded: 'Listo',
      skipped: 'Omitido',
      failed: 'Falló'
    },
    oneTimeTitle: 'Hermes necesita una instalación única',
    unsupportedDesc: platform =>
      `La instalación automática del primer inicio aún no está disponible en ${platform}. Abre Terminal y ejecuta el comando de abajo; luego vuelve a abrir la app. Los siguientes inicios omitirán este paso.`,
    installCommand: 'Comando de instalación',
    copyCommand: 'Copiar comando',
    viewDocs: 'Ver docs de instalación',
    installTo: 'Se instalará en',
    retryAfterRun: 'Ya lo ejecuté -- reintentar',
    setupChoiceTitle: 'Configurar Hermes Desktop',
    setupChoiceDesc:
      'Conecta esta app con un gateway de Hermes que ya esté en ejecución o instala Hermes localmente en este equipo.',
    connectExistingTitle: 'Conectar con un Hermes existente',
    connectExistingShort: 'Conectar existente',
    connectExistingDesc:
      'Usa un backend remoto con un token de sesión o inicio de sesión en el navegador. No se iniciará ninguna instalación local.',
    installLocalTitle: 'Instalar Hermes localmente',
    installLocalDesc: 'Descarga Hermes, crea su entorno de Python y ejecuta el backend en este equipo.',
    localStartUnavailable: 'No se pudo iniciar la instalación local. Reinicia Hermes Desktop e inténtalo de nuevo.',
    remoteSetupTitle: 'Conectar con un Hermes existente',
    remoteSetupDesc:
      'Introduce la URL de tu gateway. Hermes Desktop detectará si necesita un token o iniciar sesión en el navegador.',
    remoteUrlTitle: 'URL del gateway',
    remoteUrlDesc: 'Usa la URL base del gateway de Hermes e incluye https:// si es remoto.',
    remoteUrlPlaceholder: 'https://gateway.example.com/hermes',
    probing: 'Detectando la autenticación del gateway…',
    probeError:
      'Hermes no puede llegar a esa dirección. Comprueba la URL y que el otro equipo esté ejecutando Hermes; las opciones de inicio de sesión aparecen cuando responde.',
    probeErrorDetails: 'Detalles',
    identityProvider: 'tu proveedor de identidad',
    authTitle: 'Autenticación',
    authNeedsOauth: provider => `Inicia sesión con ${provider} antes de probar este gateway.`,
    authSignedIn: 'Inicio de sesión en el navegador completado.',
    connected: 'Conectado',
    signIn: 'Iniciar sesión',
    signInWith: provider => `Iniciar sesión con ${provider}`,
    enterUrlFirst: 'Introduce primero la URL de un gateway.',
    signInIncomplete: 'La ventana de inicio de sesión se cerró antes de completar la autenticación.',
    tokenTitle: 'Token de sesión',
    tokenDesc: 'Pega el token de sesión del archivo .env del gateway remoto.',
    pasteSessionToken: 'Pegar token de sesión',
    incompleteSignInTest: 'Inicia sesión antes de probar este gateway protegido con OAuth.',
    incompleteTokenTest: 'Introduce un token de sesión antes de probar este gateway.',
    testConnection: 'Probar conexión',
    testSucceeded: (baseUrl, version) => `Conectado a ${baseUrl}${version ? ` (${version})` : ''}.`,
    applyRemote: 'Aplicar y reconectar',
    backToSetup: 'Atrás',
    failedTitle: 'Falló la instalación',
    settingUpTitle: 'Configurando Hermes Agent',
    finishingTitle: 'Terminando',
    failedDesc:
      'Uno de los pasos de configuración no terminó. Puede ocurrir si hay otra copia de Hermes en ejecución, se cortó la conexión a internet o un antivirus bloqueó el instalador. Cierra las demás ventanas de Hermes y elige Recargar y reintentar. Si vuelve a fallar, abre los registros y envíalos al soporte.',
    activeDesc:
      'Esta configuración se realiza una sola vez. El instalador de Hermes está descargando dependencias y configurando tu máquina. Los siguientes inicios omitirán este paso.',
    progress: (completed, total) => `${completed} de ${total} pasos completados`,
    currentStage: stage => ` -- ahora: ${stage}`,
    fetchingManifest: 'Obteniendo manifiesto del instalador...',
    error: 'Error',
    hideOutput: 'Ocultar salida del instalador',
    showOutput: 'Mostrar salida del instalador',
    lines: count => `${count} ${count === 1 ? 'línea' : 'líneas'}`,
    noOutput: 'Aún no hay salida.',
    cancelling: 'Cancelando...',
    cancelInstall: 'Cancelar instalación',
    transcriptSaved: 'Transcripción completa guardada en',
    copiedOutput: '¡Copiado!',
    copyOutput: 'Copiar salida',
    reloadRetry: 'Recargar y reintentar',
    openLogs: 'Abrir registros'
  },
  onboarding: {
    headerTitle: 'Vamos a configurar Hermes Agent',
    headerDesc: 'Conecta un proveedor de modelo para empezar a chatear. La mayoría de opciones requieren un clic.',
    preparingInstall: 'Hermes está terminando la instalación. En el primer inicio suele tardar menos de un minuto.',
    starting: 'Iniciando Hermes…',
    lookingUpProviders: 'Buscando proveedores...',
    collapse: 'Contraer',
    otherProviders: 'Otros proveedores',
    haveApiKey: 'Tengo una clave API',
    chooseLater: 'Elegiré un proveedor más tarde',
    recommended: 'Recomendado',
    connected: 'Conectado',
    featuredPitch: 'Una suscripción, más de 300 modelos frontier: la forma recomendada de usar Hermes',
    fireworksPitch: 'API directa de modelos: modelos frontier alojados en Fireworks',
    localModelsTitle: 'Ejecutar modelos localmente',
    localModelsPitch: 'Sin cuenta: descarga un modelo y ejecútalo en este equipo',
    openRouterPitch: 'Una clave, cientos de modelos: un buen valor predeterminado',
    apiKeyOptions: {
      fireworks: {
        short: 'API de modelo directo',
        description: 'Acceso directo a modelos alojados en Fireworks AI.'
      },
      openrouter: {
        short: 'una clave, muchos modelos',
        description:
          'Aloja cientos de modelos detrás de una sola clave. Buen valor predeterminado para instalaciones nuevas.'
      },
      openai: {
        short: 'modelos tipo GPT',
        description: 'Acceso directo a modelos de OpenAI.'
      },
      gemini: {
        short: 'modelos Gemini',
        description: 'Acceso directo a modelos de Google Gemini.'
      },
      xai: {
        short: 'modelos Grok',
        description: 'Acceso directo a modelos Grok de xAI.'
      },
      local: {
        short: 'autohospedado',
        description:
          'Apunta Hermes a un endpoint local o autohospedado compatible con OpenAI (vLLM, llama.cpp, Ollama, etc.).'
      }
    },
    backToSignIn: 'Volver al inicio de sesión',
    getKey: 'Obtener una clave',
    replaceCurrent: 'Reemplazar valor actual',
    pasteApiKey: 'Pegar clave API',
    localApiKeyPlaceholder: 'Clave API (opcional; solo si tu endpoint la requiere)',
    couldNotSave: 'No se pudo guardar la credencial.',
    connecting: 'Conectando',
    update: 'Actualizar',
    flowSubtitles: {
      pkce: 'Abre tu navegador para iniciar sesión y luego continúa aquí',
      device_code: 'Abre una página de verificación en tu navegador; Hermes se conecta automáticamente',
      external: 'Inicia sesión una vez en tu terminal y vuelve para chatear'
    },
    startingSignIn: provider => `Iniciando sesión con ${provider}...`,
    verifyingCode: provider => `Verificando tu código con ${provider}...`,
    connectedProvider: provider => `${provider} conectado`,
    connectedPicking: provider => `${provider} conectado. Eligiendo un modelo predeterminado...`,
    signInFailed: 'No se pudo iniciar sesión. Inténtalo de nuevo.',
    signInExpired:
      'La página de inicio de sesión caducó antes de que terminaras. Vuelve a intentarlo y completa el paso del navegador en unos minutos, o usa una clave API.',
    signInDidNotFinish: (provider: string) =>
      `No se completó el inicio de sesión con ${provider}. Comprueba tu conexión a internet y vuelve a intentarlo, o elige otro proveedor.`,
    tryAgain: 'Reintentar',
    useApiKeyInstead: 'Usar una clave API',
    errorDetails: 'Detalles',
    pickDifferentProvider: 'Elegir otro proveedor',
    signInWith: provider => `Iniciar sesión con ${provider}`,
    openedBrowser: provider => `Abrimos ${provider} en tu navegador.`,
    authorizeThere: 'Autoriza Hermes allí.',
    copyAuthCode: 'Copia el código de autorización y pégalo abajo.',
    pasteAuthCode: 'Pegar código de autorización',
    reopenAuthPage: 'Volver a abrir página de autorización',
    autoBrowser: provider =>
      `Abrimos ${provider} en tu navegador. Autoriza Hermes allí y te conectarás automáticamente; no hay nada que copiar o pegar.`,
    reopenSignInPage: 'Volver a abrir página de inicio de sesión',
    waitingAuthorize: 'Esperando tu autorización...',
    externalPending: provider =>
      `${provider} inicia sesión con su propia CLI. Ejecuta este comando en una terminal y luego vuelve y elige "Ya inicié sesión":`,
    signedIn: 'Ya inicié sesión',
    deviceCodeOpened: provider => `Abrimos ${provider} en tu navegador. Introduce este código allí:`,
    reopenVerification: 'Volver a abrir página de verificación',
    copy: 'Copiar',
    defaultModel: 'Modelo predeterminado',
    freeTier: 'Nivel gratis',
    pro: 'Pro',
    free: 'Gratis',
    price: (input, output) => `${input} entrada / ${output} salida por Mtok`,
    change: 'Cambiar',
    startChatting: 'Empezar',
    docs: provider => `Docs de ${provider}`
  },
  freeTier: {
    providerRowTitle: 'Nous · plan gratuito',
    providerRowPitch: 'Inicia sesión con una cuenta de Nous para desbloquear más modelos y herramientas.',
    readyTitle: 'Hermes está listo.',
    readyCaption: 'Gratis · conectores incluidos',
    begin: 'Empezar',
    signInInstead: 'Iniciar sesión con una cuenta de Nous',
    otherProviders: 'Otros proveedores',
    stripTitle: 'Ya están disponibles la inferencia y los conectores gratuitos de Nous.',
    stripBody: 'Abre el selector de modelos para probarlos o inicia sesión con una cuenta de Nous.',
    openModelPicker: 'Abrir selector de modelos',
    dismiss: 'Descartar',
    providerName: 'Nous',
    statusLabel: (model: string) => `Nous · ${model}`,
    signIn: 'Iniciar sesión',
    signInHeading: 'Inicia sesión con una cuenta de Nous para desbloquear más modelos y herramientas.',
    settingUp: 'Configurando la inferencia gratuita…',
    codeBody: 'Introduce este código en tu navegador para terminar de iniciar sesión.',
    copyLink: 'Copiar enlace',
    doNotShare: 'No compartas este código.',
    waiting: 'Esperando el inicio de sesión…',
    finishingHeading: 'Terminando el inicio de sesión…',
    finishingBody: 'Aprobado en el navegador. Obteniendo los tokens de tu cuenta.',
    signedInAs: (email: string) => `Sesión iniciada como ${email}`,
    signedIn: 'Sesión iniciada.',
    completedBody: 'Tu cuenta ya incluye inferencia y herramientas.',
    defaultModel: 'Modelo predeterminado',
    change: 'Cambiar',
    done: 'Listo',
    notNow: 'Ahora no',
    tryAgain: 'Reintentar',
    startAgain: 'Empezar de nuevo',
    didNotComplete: 'No se completó el inicio de sesión',
    rejectedBody: 'No pasa nada, sigues en el servicio gratuito de Nous. Inicia sesión cuando quieras.',
    supersededBody:
      'Un código de inicio de sesión más reciente sustituyó a este. Usa el más reciente o empieza de nuevo.',
    timedOutHeading: 'Ese enlace de inicio de sesión caducó',
    timedOutBody: 'Empieza de nuevo cuando quieras. Sigues en el servicio gratuito de Nous.',
    retiredBody:
      'Tu sesión terminó antes de completar el inicio de sesión. Hermes iniciará una nueva; luego vuelve a iniciar sesión cuando quieras.',
    errorBody: 'No se completó el inicio de sesión. Vuelve a intentarlo cuando quieras.',
    busyHeading: 'Ya casi está',
    busyBody: (wait: string) =>
      `Hermes no pudo terminar de iniciar tu sesión porque el servicio de Nous está ocupado. Vuelve a intentarlo en ${wait}. Mientras tanto, tu sesión sigue aquí.`,
    unreachableBody:
      'Hermes no pudo llegar al servicio de Nous para terminar de iniciar tu sesión. Comprueba tu conexión a internet y vuelve a intentarlo. Tu sesión sigue aquí.',
    alreadySignedInHeading: 'Ya has iniciado sesión.',
    alreadySignedInBody: 'Este Hermes ya tiene la sesión iniciada en una cuenta de Nous.',
    setupFailed: {
      gateClosed:
        'Esta versión de Hermes no puede iniciarse sin una cuenta de Nous. Inicia sesión o crea una: es gratis y solo lleva un minuto.',
      paused:
        'El uso de Hermes sin iniciar sesión está en pausa por un momento. Hermes seguirá comprobándolo. Iniciar sesión es gratis y te permite empezar ahora mismo.',
      rateLimited: (wait: string) =>
        `Mucha gente está empezando ahora mismo, así que Hermes volverá a intentarlo en ${wait}. Iniciar sesión es gratis y te ahorra la espera.`,
      unreachable:
        'Hermes no pudo llegar al servicio de Nous. Comprueba tu conexión a internet y pulsa Reintentar. O conecta otro proveedor por ahora.',
      serverError:
        'El servicio de Nous tuvo un fallo. Pulsa Reintentar en un momento o conecta otro proveedor por ahora.',
      powRequired:
        'El servidor de Nous pidió una prueba de trabajo, pero tu agente todavía no la implementa. Inicia sesión o crea una cuenta gratuita de Nous para continuar.',
      locked:
        'Esta sesión no puede continuar sin iniciar sesión. Inicia sesión o crea una cuenta gratuita de Nous para seguir.',
      generic:
        'Hermes no pudo configurar el acceso gratuito sin iniciar sesión. Iniciar sesión es gratis; también puedes conectar otro proveedor.',
      signInBelow: 'Iniciar sesión es gratis. Elige Nous abajo.',
      tryAgain: 'Reintentar',
      retrying: 'Reintentando…'
    }
  },
  modelPicker: {
    title: 'Cambiar modelo',
    current: 'actual:',
    unknown: '(desconocido)',
    search: 'Filtrar proveedores y modelos...',
    noModels: 'No se encontraron modelos.',
    addProvider: 'Añadir proveedor',
    loadFailed: 'No se pudieron cargar los modelos',
    loadingIntoMemory: 'Cargando en memoria',
    downloading: 'Descargando',
    localDownloadsHeading: 'Local',
    noAuthenticatedProviders: 'No hay proveedores autenticados.',
    pro: 'Pro',
    proNeedsSubscription: 'Los modelos Pro necesitan una suscripción de Nous de pago.',
    free: 'Gratis',
    freeTier: 'Nivel gratis',
    priceTitle: 'Precio de entrada / salida por millón de tokens',
    wasPrice: 'antes',
    customModel: 'Modelo personalizado',
    addCustomModelAction: 'Añadir modelo personalizado…',
    customModelPlaceholder: 'Escribe un ID de modelo, p. ej., openai/gpt-5'
  },
  modelVisibility: {
    title: 'Modelos',
    search: 'Buscar modelos',
    noAuthenticatedProviders: 'No hay proveedores autenticados.',
    addProvider: 'Añadir proveedor…',
    addCustomModel: 'Añadir modelo personalizado',
    removeCustomModel: 'Quitar modelo personalizado'
  },
  shell: {
    windowControls: 'Controles de ventana',
    paneControls: 'Controles de panel',
    appControls: 'Controles de app',
    modelMenu: {
      search: 'Buscar modelos',
      noModels: 'No se encontraron modelos',
      editModels: 'Editar modelos…',
      followDefault: 'Usar el predeterminado de Ajustes',
      refreshModels: 'Actualizar modelos',
      fast: 'Rápido'
    },
    modelOptions: {
      noOptions: 'No hay opciones para este modelo',
      options: 'Opciones',
      thinking: 'Razonamiento',
      fast: 'Rápido',
      effort: 'Esfuerzo',
      minimal: 'Mínimo',
      low: 'Bajo',
      medium: 'Medio',
      high: 'Alto',
      xhigh: 'Extra alto',
      max: 'Máximo',
      ultra: 'Ultra',
      sendsOnRoute: (level: string) => `envía ${level} en esta ruta`,
      updateFailed: 'No se pudo actualizar la opción del modelo',
      fastFailed: 'No se pudo actualizar el modo rápido'
    },
    gatewayMenu: {
      gateway: 'Gateway',
      connected: 'Conectado',
      connecting: 'Conectando',
      offline: 'Sin conexión',
      inferenceReady: 'Inferencia lista',
      inferenceNotReady: 'Inferencia no lista',
      checkingInference: 'Comprobando inferencia',
      disconnected: 'Desconectado',
      reconnectGateway: 'Reconectar gateway',
      openSystem: 'Abrir panel del sistema',
      connection: label => `Conexión: ${label}`,
      recentActivity: 'Actividad reciente',
      viewAllLogs: 'Ver todos los registros →',
      messagingPlatforms: 'Plataformas de mensajería'
    },
    approvalMode: {
      title: 'Modo de aprobación',
      ariaLabel: mode => `Modo de aprobación: ${mode}`,
      manual: 'Manual',
      manualDescription: 'Preguntar antes de acciones que requieran aprobación',
      smart: 'Inteligente',
      smartDescription: 'Evaluar automáticamente las acciones y preguntar cuando sea necesario',
      off: 'Desactivado',
      offDescription: 'Ejecutar sin solicitudes de aprobación'
    },
    statusbar: {
      unknown: 'desconocido',
      restart: 'reiniciar',
      update: 'actualizar',
      updateInProgress: 'Actualización en curso',
      commitsBehind: (count, branch) => `${count} ${count === 1 ? 'commit' : 'commits'} detrás de ${branch}`,
      desktopVersion: version => `Hermes Desktop v${version}`,
      backendVersion: version => `backend v${version}`,
      clientLabel: version => `cliente v${version}`,
      connectionSsh: host => `SSH: ${host}`,
      connectionRemote: host => `Remoto: ${host}`,
      connectionCloud: host => `Nube: ${host}`,
      connectionCloudTooltip: host => `Hermes Cloud · ${host}`,
      connectionSshTooltip: host => `SSH · ${host}`,
      connectionRemoteTooltip: host => `Remoto · ${host}`,
      backendLabel: version => `backend v${version}`,
      commit: sha => `commit ${sha}`,
      branch: branch => `rama ${branch}`,
      closeCommandCenter: 'Cerrar Centro de comandos',
      openCommandCenter: 'Abrir Centro de comandos',
      showTerminal: 'Mostrar terminal',
      hideTerminal: 'Ocultar terminal',
      gateway: 'Gateway',
      gatewayReady: 'listo',
      gatewayNeedsSetup: 'necesita configuración',
      gatewayUnavailable: 'inferencia no disponible',
      gatewayChecking: 'comprobando',
      gatewayConnecting: 'conectando',
      gatewayOffline: 'sin conexión',
      gatewayRestarting: 'reiniciando…',
      gatewayTitle: 'Puerta de enlace',
      customizeTitle: 'Mostrar en la barra de estado',
      hideStatusbar: 'Ocultar barra de estado',
      resetStatusbar: 'Restablecer a valores predeterminados',
      toggleApprovalMode: 'Aprobaciones',
      toggleBackendVersion: 'Versión del backend',
      toggleCacheHitRate: 'Tasa de aciertos de caché',
      toggleCommandCenter: 'Centro de comandos',
      toggleContextUsage: 'Medidor de contexto',
      toggleRunningTimer: 'Temporizador de turno',
      toggleSessionTimer: 'Temporizador de sesión',
      toggleTerminal: 'Terminal',
      toggleTokensPerSecond: 'Tokens por segundo',
      toggleVersion: 'Versión y actualizaciones',
      toggleFreeTier: 'Plan gratuito',
      toggleWorkspace: 'Espacio de trabajo',
      cacheHitRateTitle:
        'Tasa de aciertos de la caché de prompts en esta sesión: los tokens en caché cuestan menos, así que cuanto más alta, más barato',
      tokensPerSecondTitle: 'Tokens de salida por segundo, promediados en las últimas 10 llamadas al modelo',
      agents: 'Agentes',
      closeAgents: 'Cerrar agentes',
      openAgents: 'Abrir agentes',
      subagents: count => `${count} ${count === 1 ? 'subagente' : 'subagentes'}`,
      failed: count => `${count} fallidos`,
      running: count => `${count} en ejecución`,
      cron: 'Cron',
      openCron: 'Abrir tareas cron',
      webhooks: 'Webhooks',
      openWebhooks: 'Abrir webhooks',
      starmap: 'Grafo de memoria',
      openStarmap: 'Abrir grafo de memoria',
      turnRunning: 'En ejecución',
      contextUsage: 'Uso de contexto',
      systemResources: {
        title: 'Recursos del sistema',
        loading: 'Recursos…',
        gpuUtilization: 'Uso de GPU',
        gpuMemory: 'Memoria de GPU',
        ram: 'RAM',
        unifiedNote: 'Memoria unificada: la GPU y el sistema comparten este espacio.',
        toggle: 'Recursos del sistema'
      },
      contextUsagePanel: {
        categories: {
          conversation: 'Conversación',
          mcp: 'MCP',
          memory: 'Memoria',
          rules: 'Reglas',
          skills: 'Skills',
          subagent_definitions: 'Definiciones de subagente',
          system_prompt: 'Prompt del sistema',
          tool_definitions: 'Definiciones de herramientas'
        },
        empty: 'Aún no hay datos de contexto',
        loading: 'Cargando desglose…',
        percentFull: percent => `${percent}% lleno`,
        title: 'Uso del contexto',
        tokenSummary: (used, max) => `${used} / ${max} tokens`
      },
      focusedSince: 'Enfocado desde',
      focusedSinceTitle: 'Tiempo desde que este chat está enfocado — no cuánto lleva un turno',
      yoloOn: 'YOLO activado — autoaprobando comandos peligrosos. Shift+clic lo alterna globalmente.',
      yoloOff: 'YOLO desactivado. Shift+clic lo alterna globalmente.',
      modelNone: 'ninguno',
      noModel: 'sin modelo',
      switchModel: 'Cambiar modelo',
      openModelPicker: 'Abrir selector de modelo',
      modelPinned: 'fijado por ti; los chats nuevos lo usan en lugar del predeterminado de Configuración',
      modelTitle: (provider, model) => `Modelo · ${provider}: ${model}`,
      providerModelTitle: (provider, model) => `${provider} · ${model}`
    }
  },
  rightSidebar: {
    aria: 'Barra lateral derecha',
    panelsAria: 'Paneles de la barra lateral derecha',
    files: 'Sistema de archivos',
    terminal: 'Terminal',
    noFolderSelected: 'No hay carpeta seleccionada',
    changeCwdTitle: 'Cambiar directorio de trabajo',
    remotePickerTitle: 'Elige una carpeta remota',
    remotePickerDescription: 'Explora carpetas en el backend conectado.',
    remotePickerSelect: 'Seleccionar carpeta',
    remotePickerNewFolder: 'Nueva carpeta',
    remotePickerFolderName: 'Nombre de la carpeta',
    remotePickerCreateFolder: 'Crear carpeta',
    remotePickerInvalidFolderName: 'Escribe un solo nombre de carpeta, sin barras.',
    remotePickerCreateFolderFailed: error => `No se pudo crear la carpeta (${error}).`,
    folderTip: cwd => cwd,
    openFolder: 'Abrir carpeta',
    refreshTree: 'Actualizar árbol',
    collapseAll: 'Contraer todas las carpetas',
    showIgnored: 'Mostrar archivos ignorados por git',
    hideIgnored: 'Ocultar archivos ignorados por git',
    previewUnavailable: 'Vista previa no disponible',
    couldNotPreview: path => `No se pudo previsualizar ${path}`,
    noProjectTitle: 'Sin proyecto',
    noProjectBody: 'Define un directorio de trabajo desde la barra de estado para explorar archivos.',
    noProjectOpen: 'Ningún proyecto abierto',
    noDiffs: 'Sin diferencias',
    unreadableTitle: 'No legible',
    unreadableBody: error => `No se pudo leer esta carpeta (${error}).`,
    emptyTitle: 'Vacío',
    emptyBody: 'Esta carpeta está vacía.',
    treeErrorTitle: 'Error del árbol',
    treeErrorBody: 'El árbol de archivos encontró un error al renderizar esta carpeta.',
    tryAgain: 'Intentar de nuevo',
    loadingTree: 'Cargando árbol de archivos',
    loadingFiles: 'Cargando archivos',
    terminalHide: 'Ocultar terminal',
    terminalsAria: 'Terminales',
    terminalNew: 'Nueva terminal',
    terminalCloseOthers: 'Cerrar las demás',
    terminalCloseAll: 'Cerrar todo',
    addToChat: 'Añadir al chat'
  },
  preview: {
    tab: 'Vista previa',
    closePane: 'Cerrar panel de vista previa',
    loading: 'Cargando vista previa',
    unavailable: 'Vista previa no disponible',
    opening: 'Abriendo...',
    hide: 'Ocultar',
    openPreview: 'Abrir vista previa',
    openInBrowser: 'Abrir en el navegador',
    openInExternal: 'Abrir externamente',
    popIn: 'Acoplar',
    popOut: 'Desacoplar',
    linkHint: '⌘/Ctrl-clic para acceder al panel de vista previa',
    sourceLineTitle: 'Haz clic para seleccionar · Mayús-clic para ampliar · arrastra al compositor',
    source: 'FUENTE',
    renderedPreview: 'VISTA PREVIA',
    diff: 'Diferencias',
    unknownSize: 'tamaño desconocido',
    binaryTitle: 'Esto parece un archivo binario',
    binaryBody: label => `Previsualizar ${label} puede mostrar texto ilegible.`,
    largeTitle: 'Este archivo es grande',
    largeBody: (label, size) => `${label} pesa ${size}. Hermes solo mostrará los primeros 512 KB.`,
    previewAnyway: 'Previsualizar de todos modos',
    truncated: 'Mostrando los primeros 512 KB.',
    noInlineTitle: 'Sin vista previa inline',
    noInlineBody: mimeType => `${mimeType || 'Este tipo de archivo'} aún puede adjuntarse como contexto.`,
    edit: 'Editar',
    editing: 'Edición',
    unsavedChanges: 'Cambios no guardados',
    saveFailed: message => `No se pudo guardar: ${message}`,
    diskChangedTitle: 'Archivo cambiado en el disco',
    diskChangedBody:
      'Este archivo cambió desde que lo abriste. ¿Quieres sobrescribirlo con tu versión o descartar tus cambios y recargar?',
    overwrite: 'Sobrescribir',
    discardReload: 'Descartar y recargar',
    console: {
      deselect: 'Deseleccionar entrada',
      select: 'Seleccionar entrada',
      copyFailed: 'No se pudo copiar la salida de consola',
      copyEntry: 'Copiar esta entrada',
      sendEntry: 'Enviar esta entrada al chat',
      messages: count => `${count} ${count === 1 ? 'mensaje de consola' : 'mensajes de consola'}`,
      resize: 'Redimensionar consola de vista previa',
      title: 'Consola de vista previa',
      selected: count => `${count} seleccionados`,
      sendToChat: 'Enviar al chat',
      copySelected: 'Copiar selección al portapapeles',
      copyAll: 'Copiar todo al portapapeles',
      copy: 'Copiar',
      clear: 'Limpiar',
      empty: 'Aún no hay mensajes de consola.',
      promptHeader: 'Consola de vista previa:',
      sentTitle: 'Enviado al chat',
      sentMessage: count =>
        `${count} ${count === 1 ? 'entrada de registro añadida' : 'entradas de registro añadidas'} al compositor`
    },
    web: {
      appFailedToBoot: 'La app de vista previa no arrancó',
      serverNotFound: 'Servidor no encontrado',
      remoteLoopback:
        'Esta dirección apunta al equipo que ejecuta tu agente, no a este. El panel del navegador carga las páginas localmente, así que un servidor de desarrollo remoto necesita un reenvío de puertos o un nombre de host accesible.',
      failedToLoad: 'No se pudo cargar la vista previa',
      tryAgain: 'Intentar de nuevo',
      restarting: 'Hermes se está reiniciando...',
      askRestart: 'Pedir a Hermes que reinicie el servidor',
      lookingRestart: taskId => `Hermes está buscando un servidor de vista previa para reiniciar (${taskId})`,
      restartingTitle: 'Reiniciando servidor de vista previa',
      restartingMessage:
        'Hermes está trabajando en segundo plano. Mira la consola de vista previa para ver el progreso.',
      startRestartFailed: message => `No se pudo iniciar el reinicio del servidor: ${message}`,
      restartFailed: 'Falló el reinicio del servidor',
      hideConsole: 'Ocultar consola de vista previa',
      showConsole: 'Mostrar consola de vista previa',
      hideDevTools: 'Ocultar DevTools de vista previa',
      openDevTools: 'Abrir DevTools de vista previa',
      goBack: 'Atrás',
      goForward: 'Adelante',
      reload: 'Recargar página',
      address: 'Dirección',
      addressPlaceholder: 'Introduce una dirección',
      blankPageBody: 'Escribe una dirección arriba para navegar o pide a Hermes que abra una página.',
      finishedRestarting: message =>
        `Hermes terminó de reiniciar el servidor de vista previa${message ? `: ${message}` : ''}`,
      failedRestarting: message => `Falló el reinicio del servidor: ${message}`,
      unknownError: 'error desconocido',
      restartedTitle: 'Servidor de vista previa reiniciado',
      reloadingNow: 'Recargando la vista previa ahora.',
      restartFailedTitle: 'Falló el reinicio de la vista previa',
      restartFailedMessage: 'Hermes no pudo reiniciar el servidor.',
      stillWorking:
        'Hermes sigue trabajando, pero aún no llegó ningún resultado de reinicio. Puede que el comando del servidor siga en primer plano.',
      workspaceReloading: 'El espacio de trabajo cambió, recargando vista previa',
      fileChanged: url => `Archivo cambiado, recargando vista previa: ${url}`,
      filesChanged: (count, url) => `${count} cambios de archivo, recargando vista previa: ${url}`,
      watchFailed: message => `No se pudo vigilar el archivo de vista previa: ${message}`,
      moduleMimeDescription:
        'Los scripts de módulo se están sirviendo con el tipo MIME incorrecto. Normalmente significa que un servidor de archivos estáticos sirve una app Vite/React en lugar del dev server del proyecto.',
      loadFailedConsole: (code, message) => `Carga fallida${code ? ` (${code})` : ''}: ${message}`,
      unreachableDescription: 'No se pudo acceder a la página de vista previa.',
      openTarget: url => `Abrir ${url}`,
      fallbackTitle: 'Vista previa',
      annotate: 'Anotar',
      annotateOn: 'Dejar de anotar',
      annotateNeedPage: 'Primero abre una página en el navegador integrado.',
      annotateFailed: 'No se pudo iniciar el modo de anotación',
      commenting: 'Comentando',
      addComments: (count: number) => (count === 1 ? 'Añadir 1 comentario' : `Añadir ${count} comentarios`),
      commentPlaceholder: 'Añade un comentario...',
      commentTitle: (n: number) => `Comentario ${n}`,
      saveComment: 'Guardar',
      cancelComment: 'Cancelar comentario'
    }
  },
  interfaceMode: {
    title: 'Modo de interfaz',
    hint: 'Cambia lo que se muestra, no lo que Hermes puede hacer.',
    sessionNote:
      'Definido por el modo Simple. Un cambio aquí dura esta sesión; cambia a Avanzado para que sea permanente.',
    simple: {
      label: 'Simple',
      description: 'Para hablar con Hermes. Barra lateral y chat; sin paneles de terminal, archivos ni diferencias.'
    },
    advanced: {
      label: 'Avanzado',
      description:
        'Para desarrolladores. Terminal, archivos, diferencias, barra de estado y diseños, como los configures.'
    }
  },
  zones: {
    showTabStrip: 'Mostrar pestañas',
    hideTabStrip: 'Ocultar pestañas',
    showStripTab: title => `Mostrar ${title}`,
    hideStripTab: title => `Ocultar ${title}`,
    lastTabKeptTitle: 'La última pestaña permanece',
    lastTabKeptBody:
      'Esta zona necesita al menos una pestaña visible. Muestra otra pestaña primero, o colapsa toda la barra lateral.',
    toggleStripTab: title => `Alternar pestaña ${title}`,
    minimize: 'Minimizar',
    restore: 'Restaurar',
    closeRunningTitle: '¿Cerrar la pestaña en ejecución?',
    closeRunningBody:
      'Este chat sigue trabajando o espera tu respuesta. Cerrar la pestaña lo oculta; la sesión conserva su progreso y puede volver a abrirse desde la barra lateral.',
    closeRunningConfirm: 'Cerrar pestaña',
    reload: 'Recargar',
    closeOthers: 'Cerrar las demás',
    closeToRight: 'Cerrar las de la derecha',
    closeAll: 'Cerrar todo',
    newSessionTab: 'Nueva pestaña de sesión',
    newTab: 'Nueva pestaña',
    pluginDisabled: pluginId => `Plugin "${pluginId}" desactivado`,
    pluginDisabledBody: 'Vuelve a activarlo en Capacidades → Plugins para recuperar el panel.',
    missingPane: paneId => `Falta el panel: ${paneId}`,
    editTitle: 'Diseños',
    editHint: 'Elige una disposición, o arrastra paneles entre zonas.',
    reset: 'Restablecer',
    templates: 'Plantillas',
    custom: 'Personalizado',
    newGridLayout: 'Nuevo diseño de cuadrícula',
    saveCurrentAs: 'Guardar la disposición actual como plantilla',
    nameLayoutPlaceholder: 'Nombre del diseño…',
    deletePreset: name => `Eliminar ${name}`,
    zoneEditorTitle: 'Editor de zonas',
    editorHintPre: 'clic para dividir · ',
    editorHintPost:
      ' invierte la línea · arrastra entre zonas para unirlas · arrastra los bordes compartidos para redimensionar',
    templateColumns: 'Columnas',
    templateRows: 'Filas',
    templateGrid: 'Cuadrícula',
    templatePriority: 'Prioridad',
    zoneTag: index => `zona ${index}`,
    mergeZones: count => `Unir ${count} zonas`,
    customZoneName: count => `Diseño personalizado de ${count} zonas`,
    layoutNamePlaceholder: fallback => `Nombre del diseño (${fallback})`,
    saveApply: 'Guardar y aplicar',
    notExpressible: 'esta disposición se entrelaza (en molinete) y aún no puede expresarse como divisiones anidadas',
    zoneCount: count => `${count} zonas`,
    tabCount: count => `${count} pestañas`
  },
  contextMenu: {
    link: {
      openInApp: 'Abrir en el navegador integrado',
      openExternal: 'Abrir en navegador externo',
      copyUrl: 'Copiar URL',
      copyResolvedUrl: 'Copiar URL resuelta'
    },
    image: {
      copyImage: 'Copiar imagen',
      copyImageAddress: 'Copiar dirección de imagen',
      saveImageAs: 'Guardar imagen como…'
    },
    edit: {
      cut: 'Cortar',
      paste: 'Pegar',
      selectAll: 'Seleccionar todo',
      addToDictionary: 'Agregar al diccionario'
    },
    page: {
      copyPageUrl: 'Copiar URL de la página',
      inspectElement: 'Inspeccionar elemento'
    }
  },
  assistant: {
    thread: {
      loadingSession: 'Cargando sesión',
      showEarlier: 'Mostrar mensajes anteriores',
      loadingResponse: 'Hermes está cargando una respuesta',
      loadingLocalModel: (model: string) => `Cargando ${model} en memoria`,
      processingPrompt: 'Procesando el prompt',
      resumeWhenBackgroundDone: count =>
        count === 1
          ? 'Se reanudará cuando termine la tarea en segundo plano.'
          : `Se reanudará cuando terminen las ${count} tareas en segundo plano.`,
      thinking: 'Pensando',
      thought: 'Pensamiento',
      thoughtBriefly: 'Pensó brevemente',
      thoughtFor: duration => `Pensó durante ${duration}`,
      turnDuration: duration => `Este turno tomó ${duration}`,
      today: time => `Hoy, ${time}`,
      yesterday: time => `Ayer, ${time}`,
      copy: 'Copiar',
      refresh: 'Actualizar',
      moreActions: 'Más acciones',
      branchNewChat: 'Ramificar en chat nuevo',
      react: 'Reaccionar',
      dismissError: 'Descartar error',
      errorLayers: {
        auth: 'Problema de inicio de sesión',
        billing: 'Créditos agotados',
        disk: 'Disco lleno',
        endpoint: 'No se puede conectar con tu servidor de modelos',
        gateway: 'Hermes tuvo un problema',
        generic: 'Hermes no pudo terminar esta respuesta',
        provider: 'El servicio de IA devolvió un error',
        runtime: 'Hermes tuvo un problema',
        streaming: 'La respuesta se cortó'
      },
      errorLayerBodies: {
        auth: 'El servicio de IA rechazó tu inicio de sesión. Revisa las credenciales de este proveedor y vuelve a enviar el mensaje.',
        billing: 'Tu cuenta no tiene créditos para este proveedor. Recarga o cambia de proveedor y vuelve a enviarlo.',
        disk: 'Tu disco está lleno, así que Hermes no pudo guardar esta conversación. Libera espacio y reinténtalo.',
        endpoint:
          'Hermes no puede conectar con tu servidor de modelos personalizado. Comprueba que esté en ejecución y vuelve a enviar el mensaje.',
        gateway:
          'Hermes tuvo un problema interno al iniciar esta respuesta. Vuelve a enviar el mensaje; si sigue ocurriendo, envía un diagnóstico.',
        generic: 'Algo salió mal mientras Hermes respondía. Reinténtalo o copia los detalles si sigue ocurriendo.',
        provider:
          'El servicio de IA no pudo completar esta solicitud. Reinténtalo en un momento o cambia de proveedor.',
        runtime:
          'Hermes tuvo un problema interno al iniciar esta respuesta. Vuelve a enviar el mensaje; si sigue ocurriendo, envía un diagnóstico.',
        streaming: 'La conexión se cortó antes de que terminara la respuesta. Reinténtalo para enviarla de nuevo.'
      },
      errorCodes: {
        auth: {
          title: (provider: string) => `${provider} rechazó tu inicio de sesión`,
          body: (provider: string) =>
            `No se aceptaron las credenciales guardadas para ${provider}. Corrígelas en Configuración o cambia de proveedor y vuelve a enviar el mensaje.`
        },
        auth_permanent: {
          title: (provider: string) => `${provider} rechazó tu inicio de sesión`,
          body: (provider: string) =>
            `Las credenciales guardadas para ${provider} no son válidas o se revocaron. Actualízalas o cambia de proveedor y vuelve a enviar el mensaje.`
        },
        billing: {
          title: 'Sin créditos',
          body: (provider: string) =>
            `Tu cuenta de ${provider} no tiene créditos. Recarga o cambia de proveedor y vuelve a enviarlo.`
        },
        rate_limit: {
          title: 'El servicio de IA está ocupado',
          body: (provider: string) =>
            `${provider} está limitando las solicitudes ahora mismo. Espera un minuto y reinténtalo.`
        },
        upstream_rate_limit: {
          title: 'El servicio de IA está ocupado',
          body: (provider: string) =>
            `${provider} está limitando las solicitudes ahora mismo. Espera un minuto y reinténtalo.`
        },
        overloaded: {
          title: 'El servicio de IA está sobrecargado',
          body: (provider: string) =>
            `${provider} tiene problemas ahora mismo. Reinténtalo en un momento o cambia de proveedor.`
        },
        server_error: {
          title: 'El servicio de IA tuvo un problema',
          body: (provider: string) =>
            `${provider} devolvió un error del servidor. Reinténtalo en un momento o cambia de proveedor.`
        },
        timeout: {
          title: 'Se agotó el tiempo de la respuesta',
          body: (provider: string) => `${provider} no respondió a tiempo. Reinténtalo para enviarlo de nuevo.`
        },
        stream_drop: {
          title: 'La respuesta se cortó',
          body: 'La conexión se cortó antes de que terminara la respuesta. Reinténtalo para enviarla de nuevo.'
        },
        upstream_blocked: {
          title: 'Un firewall bloqueó la solicitud',
          body: (provider: string) =>
            `Un firewall o CDN delante de ${provider} bloqueó la solicitud antes de que llegara al modelo; probablemente tu clave está bien. Define un encabezado User-Agent mediante los extra_headers del proveedor en Configuración o cambia de proveedor y vuelve a enviar el mensaje.`
        },
        ssl_cert_verification: {
          title: 'Falló la conexión segura',
          body: (provider: string) =>
            `Hermes no pudo verificar la conexión segura con ${provider}. Revisa la configuración de red o del proxy, o cambia de proveedor, y vuelve a enviar el mensaje.`
        },
        context_overflow: {
          title: 'Esta conversación es demasiado larga',
          body: 'La conversación ya no cabe en el modelo. Comprímela o empieza un chat nuevo y vuelve a enviarlo.'
        },
        payload_too_large: {
          title: 'Este mensaje es demasiado grande',
          body: 'La solicitud era demasiado grande para el modelo. Comprime la conversación o empieza un chat nuevo y vuelve a enviarlo.'
        },
        model_not_found: {
          title: 'Este modelo no está disponible',
          body: (provider: string) =>
            `${provider} no ofrece este modelo en tu cuenta. Elige otro modelo y vuelve a enviar el mensaje.`
        },
        provider_policy_blocked: {
          title: 'La configuración de tu cuenta bloquea este modelo',
          body: (provider: string) =>
            `${provider} no enrutaría esta solicitud con la configuración de datos o privacidad de tu cuenta. Elige otro modelo o cambia de proveedor.`
        },
        content_policy_blocked: {
          title: 'El servicio de IA rechazó esta solicitud',
          body: (provider: string) => `${provider} no respondería a este mensaje. Edítalo y vuelve a enviarlo.`
        },
        format_error: {
          title: 'El servicio de IA rechazó la solicitud',
          body: (provider: string) =>
            `${provider} no aceptó cómo se construyó esta solicitud. Cambia de proveedor o envía un diagnóstico para que lo revisemos.`
        },
        truncated: {
          title: 'La respuesta quedó incompleta',
          body: 'El modelo se detuvo antes de terminar. Reinténtalo para obtener una respuesta completa.'
        },
        invalid_response: {
          title: 'El servicio de IA envió una respuesta ilegible',
          body: (provider: string) => `${provider} devolvió algo que Hermes no pudo leer. Reinténtalo en un momento.`
        },
        empty_response: {
          title: 'El servicio de IA envió una respuesta vacía',
          body: (provider: string) => `${provider} no devolvió nada para este mensaje. Reinténtalo en un momento.`
        },
        loop_error: {
          title: 'Hermes se quedó atascado en un bucle',
          body: 'La respuesta repetía los mismos pasos, así que Hermes la detuvo. Reinténtalo o empieza un chat nuevo si vuelve a ocurrir.'
        },
        SESSION_NOT_OWNED: {
          title: 'Este chat está abierto en otro sitio',
          body: 'Este chat está abierto en otra ventana de Hermes o en un terminal. Ciérralo allí y vuelve a enviar el mensaje, o empieza un chat nuevo aquí.'
        },
        disk_full: {
          title: 'Disco lleno',
          body: 'Tu disco está lleno, así que Hermes no pudo guardar esta conversación. Libera espacio y reinténtalo.'
        },
        free_tier_disabled: {
          title: 'El uso de Hermes sin iniciar sesión está desactivado ahora mismo',
          body: 'Inicia sesión con una cuenta de Nous para seguir chateando; es gratis.'
        },
        free_tier_rate_limited: {
          title: 'Agotaste el cupo para chatear sin iniciar sesión',
          body: 'Se renueva en breve. Inicia sesión con una cuenta de Nous para tener un cupo mayor; es gratis.'
        },
        free_tier_at_capacity: {
          title: 'Chatear sin iniciar sesión está muy solicitado ahora mismo',
          body: 'Inicia sesión para saltarte la cola (es gratis) o vuelve a intentarlo dentro de un rato.'
        },
        free_tier_model_not_free: {
          title: 'Ese modelo no está disponible sin iniciar sesión',
          body: 'Por ahora Hermes usa el modelo gratuito. Inicia sesión con una cuenta de Nous para tener más modelos; es gratis.'
        },
        free_tier_route: {
          title: 'Hermes no pudo llegar al modelo gratuito por esta ruta',
          body: 'Inicia sesión con una cuenta de Nous (es gratis) o revisa el ajuste NOUS_INFERENCE_BASE_URL.'
        },
        free_tier_outage: {
          title: 'El modelo gratuito tiene problemas para responder ahora mismo',
          body: 'Vuelve a enviar el mensaje dentro de un minuto.'
        },
        free_tier_refused: {
          title: 'Hermes no pudo enviarlo sin iniciar sesión',
          body: 'Iniciar sesión con una cuenta de Nous es gratis.'
        }
      },
      errorAuthKinds: {
        api_key: {
          title: (provider: string) => `${provider} rechazó tu clave API`,
          body: (provider: string) =>
            `La clave guardada para ${provider} no es válida o se revocó. Actualízala y reinténtalo.`
        },
        oauth: {
          title: (provider: string) => `Tu sesión de ${provider} caducó`
        }
      },
      errorDetails: 'Detalles',
      errorGenericProvider: 'El servicio de IA',
      errorToastTitle: 'Hermes no pudo terminar la respuesta',
      errorRetry: 'Reintentar',
      errorLimitResets: (time: string) => `El límite se restablece a las ${time}`,
      errorRetryAtReset: (time: string) => `Reintentar cuando se restablezca el límite (${time})`,
      errorRetryScheduled: (time: string, wait: string) => `Reintentando a las ${time}, dentro de ${wait}`,
      errorRetryScheduledCancel: 'Cancelar',
      errorStartNewSession: 'Iniciar sesión nueva',
      errorSwitchProvider: 'Cambiar proveedor',
      errorChooseModel: 'Elegir un modelo',
      errorCompressConversation: 'Comprimir conversación',
      errorCompressFailed: 'No se pudo comprimir la conversación',
      errorOpenHermesFolder: 'Abrir la carpeta de Hermes',
      errorOpenHermesFolderFailed: 'No se pudo abrir la carpeta de Hermes',
      errorUpdateApiKey: 'Actualizar clave API',
      errorSignInAgain: (provider: string) => `Volver a iniciar sesión en ${provider}`,
      errorSignInFreeTier: 'Iniciar sesión con una cuenta de Nous',
      errorOauthExpired: (provider: string) =>
        `Tu sesión de ${provider} caducó o se revocó. Vuelve a iniciar sesión para seguir chateando.`,
      errorOpenLogs: 'Abrir registros',
      errorOpenLogsFailed: 'No se pudo abrir la carpeta de registros',
      errorOpenDesktopLogs: 'Abrir registros de Desktop',
      errorCopyDiagnostics: 'Copiar detalles del error',
      errorSendDiagnostics: 'Enviar diagnóstico',
      filesChanged: count => (count === 1 ? '1 archivo cambiado' : `${count} archivos cambiados`),
      reviewChanges: 'Revisar',
      readAloudFailed: 'Falló la lectura en voz alta',
      preparingAudio: 'Preparando audio...',
      stopReading: 'Detener lectura',
      readAloud: 'Leer en voz alta',
      editMessage: 'Editar mensaje',
      expandMessage: 'Expandir mensaje',
      scrollToBottom: 'Desplazarse hacia abajo',
      stop: 'Detener',
      restorePrevious: 'Restaurar checkpoint anterior',
      restoreCheckpoint: 'Restaurar checkpoint',
      restoreFromHere: 'Restaurar punto de control y ejecutar de nuevo desde este mensaje',
      restoreTitle: '¿Restaurar a este punto de control?',
      restoreBody:
        'Todo lo que sigue a este mensaje se elimina de la conversación y el mensaje se ejecuta nuevamente desde aquí.',
      restoreConfirm: 'Restaurar y volver a ejecutar',
      restoreNext: 'Restaurar checkpoint siguiente',
      goForward: 'Avanzar',
      sendEdited: 'Enviar edición',
      attachingFile: 'Adjuntando…'
    },
    approval: {
      gatewayDisconnected:
        'Hermes está sin conexión ahora mismo. El comando sigue esperando tu respuesta (hasta que se agote el tiempo de aprobación). Reconéctate y vuelve a enviarla.',
      sendFailed: 'No se pudo enviar tu respuesta',
      reconnect: 'Reconectar',
      timedOutSystemLine:
        'Se agotó el tiempo de aprobación: el comando no se ejecutó. Pide a Hermes que lo intente de nuevo o sube el límite en Configuración → Seguridad → Tiempo de aprobación.',
      openSafetySettings: 'Abrir configuración de seguridad',
      run: 'Ejecutar',
      command: 'Comando',
      moreOptions: 'Más opciones de aprobación',
      allowSession: 'Permitir esta sesión',
      alwaysAllowMenu: 'Permitir siempre…',
      jumpToApproval: 'Aprobación necesaria',
      reject: 'Rechazar',
      alwaysTitle: '¿Permitir siempre este comando?',
      alwaysDescription: pattern =>
        `Esto añade el patrón “${pattern}” a tu allowlist permanente (~/.hermes/config.yaml). Hermes no volverá a preguntar por comandos como este, ni en esta sesión ni en futuras.`,
      alwaysAllow: 'Permitir siempre'
    },
    clarify: {
      notReady: 'La solicitud de aclaración aún no está lista',
      gatewayDisconnected: 'Hermes está sin conexión ahora mismo. Reconéctate y vuelve a enviarlo.',
      sendFailed: 'No se pudo enviar la respuesta de aclaración',
      loadingQuestion: 'Cargando pregunta…',
      other: 'Otro (escribe tu respuesta)',
      placeholder: 'Escribe tu respuesta…',
      skip: 'Omitir',
      skipped: 'Omitido',
      continueLabel: 'Continuar',
      confirmAndContinueLabel: 'Confirmar y continuar',
      answeredBadge: 'Respondido',
      questionProgress: (answered, total) => `${answered} de ${total} respondidas`,
      lateAnswer: (question, choice) => `Con respecto a “${question}”: mi respuesta es ${choice}`,
      lateAnswerTip: 'Redactar esta respuesta como mensaje de seguimiento',
      lateAnswerHint:
        'Este prompt ya no espera una respuesta. Elige una opción para redactarla como mensaje de seguimiento.'
    },
    catalogInstall: {
      preparing: 'Preparando la instalación…',
      install: 'Instalar',
      advanced: 'Avanzado',
      skip: 'Omitir',
      installing: 'Instalando…',
      installed: 'Instalado',
      notInstalled: 'No instalado',
      failed: 'Falló',
      showNames: 'mostrar nombres',
      hideNames: 'ocultar nombres',
      skill: (name: string) => `skill ${name}`,
      kind: {
        plugin: 'plugin',
        skill: 'skill'
      },
      tier: {
        official: 'oficial',
        community: 'comunidad'
      },
      targetProfile: (profile: string) => `Se instala en tu perfil ${profile}`,
      sendFailed: 'No se pudo enviar tu respuesta. Vuelve a intentarlo.',
      commitLabel: 'Commit',
      subdirLabel: 'Carpeta',
      securityHeading: 'Seguridad',
      scan: {
        passed: 'Análisis superado',
        warnings: 'El análisis encontró advertencias',
        failed: 'Falló el análisis'
      },
      requirementsLabel: 'Requiere',
      credentialsHeading: 'Credenciales'
    },
    mcpSetup: {
      installTitle: 'Añadir servidores MCP',
      enableTitle: 'Activar servidores MCP',
      authorizeTitle: 'Autorizar servidores MCP',
      installAction: 'Instalar',
      enableAction: 'Activar',
      authorizeAction: 'Autorizar',
      installed: server => `${server} instalado`,
      enabled: server => `${server} activado`,
      authorized: server => `${server} autorizado`,
      failed: server => `La configuración falló para ${server}`,
      toolCount: count => (count === 1 ? '1 herramienta' : `${count} herramientas`),
      envRequired: 'Rellena primero las credenciales obligatorias',
      sendFailed: 'No se pudo enviar la respuesta de configuración MCP',
      reloadFailed: 'Servidor guardado, pero falló la recarga de herramientas MCP — se cargarán en la próxima sesión',
      gatewayDisconnected: 'Hermes está sin conexión ahora mismo. Reconéctate y vuelve a enviarlo.'
    },
    tool: {
      copyCode: 'Copiar código',
      renderingImage: 'Renderizando imagen',
      copyOutput: 'Copiar salida',
      copyCommand: 'Copiar comando',
      copyContent: 'Copiar contenido',
      copyUrl: 'Copiar URL',
      copyResults: 'Copiar resultados',
      copyQuery: 'Copiar consulta',
      copyFile: 'Copiar archivo',
      copyPath: 'Copiar ruta',
      failedCalls: (count: number) =>
        `${count} ${count === 1 ? 'llamada a herramienta falló' : 'llamadas a herramientas fallaron'}`,
      skillActivity: {
        loading: 'Cargando skill',
        loaded: 'Skill cargada',
        loadFailed: 'No se pudo cargar la skill',
        readingResource: 'Leyendo recurso de la skill',
        readResource: 'Recurso de la skill leído',
        resourceFailed: 'No se pudo leer el recurso de la skill',
        listing: 'Listando skills',
        listed: 'Skills listadas',
        listFailed: 'No se pudieron listar las skills',
        unavailable: 'Resultado de la skill no disponible'
      },
      outputAlt: 'Salida de herramienta',
      rawResponse: 'Respuesta sin procesar',
      copyActivity: 'Copiar actividad',
      recoveredOne: 'Recuperado después de 1 paso fallido',
      recoveredMany: count => `Recuperado después de ${count} pasos fallidos`,
      failedOne: '1 paso falló',
      failedMany: count => `${count} pasos fallaron`,
      statusRunning: 'En ejecución',
      statusError: 'Error',
      statusRecovered: 'Recuperado',
      statusDone: 'Listo',
      resultUnavailable: 'Resultado no disponible',
      resultInterrupted: 'Interrumpido',
      memoryWriteNoted: 'Escritura de memoria anotada',
      actions: {
        read: 'Leyó',
        reading: 'Leyendo',
        opened: 'Abrió',
        opening: 'Abriendo',
        failedToOpen: 'No se pudo abrir',
        searched: 'Buscó',
        searching: 'Buscando',
        ran: 'Ejecutó',
        running: 'Ejecutando',
        ranCode: 'Ejecutó código',
        runningCode: 'Ejecutando código'
      },
      prefixes: {
        browser: 'Navegador',
        web: 'Web'
      },
      titleTemplates: {
        actionCommand: (action, command) => `${action} ${command}`,
        actionQuoted: (action, value) => `${action} “${value}”`,
        actionTarget: (action, target) => `${action} ${target}`,
        prefixedDone: (prefix, action) => `${prefix} ${action}`,
        runningPrefixedTool: (prefix, action) => `Ejecutando ${prefix.toLowerCase()} ${action.toLowerCase()}`,
        runningTool: action => `Ejecutando ${action.toLowerCase()}`
      },
      titles: {
        browser_click: {
          done: 'Se hizo clic en un elemento de la página',
          pending: 'Hacer clic en un elemento de la página',
          pendingAction: 'Haciendo clic'
        },
        browser_fill: {
          done: 'Se completó un campo del formulario',
          pending: 'Completar un campo del formulario',
          pendingAction: 'Completando'
        },
        browser_navigate: {
          done: 'Página abierta',
          pending: 'Abrir página',
          pendingAction: 'Abriendo'
        },
        browser_snapshot: {
          done: 'Se capturó una instantánea de la página',
          pending: 'Capturar una instantánea de la página',
          pendingAction: 'Capturando'
        },
        browser_take_screenshot: {
          done: 'Se capturó una captura de pantalla',
          pending: 'Capturar una captura de pantalla',
          pendingAction: 'Capturando'
        },
        browser_type: {
          done: 'Se escribió en la página',
          pending: 'Escribir en la página',
          pendingAction: 'Escribiendo'
        },
        clarify: {
          done: 'Se hizo una pregunta',
          pending: 'Hacer una pregunta',
          pendingAction: 'Preguntando'
        },
        cronjob: {
          done: 'Tarea cron completada',
          pending: 'Programar una tarea cron',
          pendingAction: 'Programando'
        },
        edit_file: {
          done: 'Archivo editado',
          pending: 'Editar archivo',
          pendingAction: 'Editando'
        },
        execute_code: {
          done: 'Código ejecutado',
          pending: 'Ejecutar código',
          pendingAction: 'Ejecutando'
        },
        image_generate: {
          done: 'Imagen generada',
          pending: 'Generar imagen',
          pendingAction: 'Generando'
        },
        list_files: {
          done: 'Lista de archivos obtenida',
          pending: 'Listar archivos',
          pendingAction: 'Listando'
        },
        memory: {
          done: 'Guardado en la memoria',
          pending: 'Guardar en la memoria',
          pendingAction: 'Guardando'
        },
        patch: {
          done: 'Archivo parcheado',
          pending: 'Aplicar un parche al archivo',
          pendingAction: 'Aplicando un parche'
        },
        read_file: {
          done: 'Archivo leído',
          pending: 'Leer archivo',
          pendingAction: 'Leyendo'
        },
        search_files: {
          done: 'Búsqueda de archivos completada',
          pending: 'Buscar archivos',
          pendingAction: 'Buscando'
        },
        session_search_recall: {
          done: 'Se buscó en el historial de sesiones',
          pending: 'Buscar en el historial de sesiones',
          pendingAction: 'Buscando'
        },
        terminal: {
          done: 'Comando ejecutado',
          pending: 'Ejecutar comando',
          pendingAction: 'Ejecutando'
        },
        todo: {
          done: 'Tareas actualizadas',
          pending: 'Actualizar tareas',
          pendingAction: 'Actualizando'
        },
        vision_analyze: {
          done: 'Imagen analizada',
          pending: 'Analizar imagen',
          pendingAction: 'Analizando'
        },
        web_extract: {
          done: 'Página web leída',
          pending: 'Leer página web',
          pendingAction: 'Leyendo'
        },
        web_search: {
          done: 'Búsqueda web completada',
          pending: 'Buscar en la web',
          pendingAction: 'Buscando'
        },
        write_file: {
          done: 'Archivo escrito',
          pending: 'Escribir archivo',
          pendingAction: 'Escribiendo'
        }
      }
    }
  },
  prompts: {
    gatewayDisconnected: 'Hermes está sin conexión ahora mismo. Reconéctate y vuelve a enviarlo.',
    reconnect: 'Reconectar',
    sudoSendFailed: 'No se pudo enviar la contraseña sudo',
    secretSendFailed: 'No se pudo enviar el secreto',
    sudoTitle: 'Contraseña de administrador',
    sudoDesc:
      'Revisa el comando antes de introducir tu contraseña de sudo. La contraseña se envía al agente que lo ejecuta y se guarda en caché durante esta sesión.',
    sudoCommandUnavailable: 'Este agente no indicó el comando. Cancela si no puedes verificarlo en la conversación.',
    sudoInstallDesc:
      'Hermes necesita tu contraseña de sudo para instalar los paquetes de Bot Screen (TigerVNC + Xfce) en el host del gateway. Solo se envía a ese host.',
    sudoPlaceholder: 'contraseña sudo',
    secretTitle: 'Se requiere un secreto',
    secretDesc: 'Hermes necesita una credencial para continuar.',
    secretPlaceholder: 'valor secreto',
    vaultUnlockSendFailed: 'No se pudo enviar la contraseña maestra',
    vaultUnlockTitle: (name: string) => `Desbloquear ${name}`,
    vaultUnlockDesc: (name: string) =>
      `El agente quiere iniciar sesión en un sitio con un acceso guardado en ${name}. Introduce tu contraseña maestra para desbloquearlo durante esta sesión: va directamente a ${name} en este equipo y nunca se guarda ni se muestra al agente.`,
    vaultUnlockPlaceholder: 'Contraseña maestra',
    vaultUnlockKeepLocked: 'Mantener bloqueado',
    vaultUnlockConfirm: 'Desbloquear',
    vaultSaveSendFailed: 'No se pudo guardar el acceso',
    vaultSaveTitle: (site: string) => `¿Guardar tu acceso a ${site}?`,
    vaultSaveDesc: (origin: string) =>
      `Hermes llegó a una página de inicio de sesión en ${origin} y no tiene un acceso para ella. Introdúcelo una vez aquí; se cifra en este equipo y se rellena en la página sin que el modelo vea nunca la contraseña.`,
    vaultSaveIdentifierLabel: 'Correo electrónico o nombre de usuario',
    vaultSaveIdentifierPlaceholder: 'tu@ejemplo.com',
    vaultSavePasswordPlaceholder: 'Contraseña',
    vaultSaveFootnote: 'Gestiona los accesos guardados en Configuración → Contraseñas e inicios de sesión.',
    vaultSaveDecline: 'No guardar',
    vaultSaveConfirm: 'Guardar e iniciar sesión',
    vaultCodeSendFailed: 'No se pudo enviar el código',
    vaultCodeTitle: (site: string) => `Código de verificación de ${site}`,
    vaultCodeDesc: (site: string) =>
      `${site} pide un código de un solo uso (SMS, correo o app de autenticación). Introdúcelo aquí y Hermes lo escribe en la página; el modelo nunca lo ve.`,
    vaultCodeLabel: 'Código',
    vaultCodeFootnote:
      'Consejo: guarda la clave del autenticador con este acceso en Configuración → Contraseñas e inicios de sesión y Hermes introducirá los códigos por ti.',
    vaultCodeSkip: 'Omitir',
    vaultCodeConfirm: 'Introducir código'
  },
  desktop: {
    audioReadFailed: 'No se pudo leer el audio grabado',
    sessionUnavailable: 'Sesión no disponible',
    createSessionFailed: 'No se pudo crear una sesión nueva',
    promptFailed: 'Falló el prompt',
    staleSessionTitle: 'Chat desactualizado',
    staleSessionBody:
      'Esta ventana estaba detrás de otra vista del mismo chat. Se cargaron los mensajes más recientes. Envía de nuevo si aún quieres.',
    providerCredentialRequired: 'Añade una credencial de proveedor antes de enviar tu primer mensaje.',
    emptySlashCommand: 'comando slash vacío',
    desktopCommands: 'Comandos de escritorio',
    skillCommandsAvailable: count => `${count} comandos de skill disponibles.`,
    warningLine: message => `advertencia: ${message}`,
    yoloArmed: 'YOLO activado para este chat',
    yoloOff: 'YOLO desactivado',
    yoloSystem: active => `YOLO ${active ? 'activado' : 'desactivado'} para esta sesión`,
    yoloTitle: 'YOLO',
    yoloToggleFailed: 'No se pudo alternar YOLO',
    profileStatus: current =>
      `Perfil: ${current}. Usa /profile <nombre> o el selector de "Nueva sesión" para iniciar un chat en otro perfil.`,
    unknownProfile: 'Perfil desconocido',
    noProfileNamed: (target, available) => `No hay ningún perfil llamado "${target}". Disponibles: ${available}`,
    newChatsProfile: name => `Los chats nuevos usarán el perfil ${name}.`,
    setProfileFailed: 'No se pudo definir el perfil',
    sttDisabled: 'Voz a texto está deshabilitado en la configuración.',
    stopFailed: 'No se pudo detener',
    regenerateFailed: 'No se pudo regenerar',
    editFailed: 'No se pudo editar',
    editTurnUnavailable: 'Este turno ya no está en el historial del servidor (puede haber sido comprimido).',
    resumeFailed: 'No se pudo reanudar',
    readOnlyTranscriptTitle: 'Abierto en modo de solo lectura',
    readOnlyTranscriptBody:
      'Ningún backend conectado reclama todavía este chat antiguo, así que se abrió como transcripción de solo lectura. Su historial está intacto; el envío está desactivado hasta que un backend lo reclame.',
    readOnlyTranscriptSendBlocked:
      'Este chat está abierto como transcripción de solo lectura: el envío está desactivado.',
    resumeStrandedTitle: 'No se pudo cargar esta sesión',
    resumeStrandedBody:
      'No se pudo conectar con esta sesión y se agotaron los reintentos automáticos. Comprueba que el gateway esté en ejecución y vuelve a intentarlo.',
    poolSlotTimeoutBody:
      'Hay demasiados bots en ejecución a la vez para el límite de este equipo. Sube el límite en Configuración → Avanzado o espera a que termine uno y reinténtalo.',
    poolSlotTimeoutOpenSettings: 'Abrir configuración avanzada',
    resumeRetry: 'Reintentar',
    nothingToBranch: 'No hay nada que ramificar',
    branchNeedsChat: 'Inicia o reanuda un chat antes de ramificar.',
    sessionBusy: 'Sesión ocupada',
    branchStopCurrent: 'Detén el turno actual antes de ramificar este chat.',
    branchNoText: 'Este mensaje no tiene texto desde el que ramificar.',
    branchTitle: n => `Borrador: rama n.º ${n}`,
    branchFailed: 'No se pudo ramificar',
    deleteFailed: 'No se pudo eliminar',
    archived: 'Archivado',
    archiveFailed: 'No se pudo archivar',
    restored: 'Restaurado',
    unarchiveFailed: 'No se pudo desarchivar',
    cwdChangeFailed: 'No se pudo cambiar el directorio de trabajo',
    cwdStagedTitle: 'Directorio de trabajo preparado',
    cwdStagedMessage: 'Reinicia el backend de escritorio para aplicar cambios de cwd a esta sesión activa.',
    modelSwitchConfirmBody: 'Este cambio de modelo requiere confirmación.',
    modelSwitchConfirmLabel: 'Cambiar de todos modos',
    modelSwitchConfirmTitle: (model: string) => `¿Cambiar a ${model}?`,
    modelSwitchConfirmTitleFallback: '¿Cambiar de modelo?',
    modelSwitchFailed: 'No se pudo cambiar de modelo',
    modelSwitchKeepLabel: 'Mantener el modelo actual',
    modelSwitchStaleNotice: 'La selección cambió: no se aplicó el cambio de modelo.',
    hydrationSyncing: (profile: string) => `Sincronizando ${profile}\u2026`,
    sessionExported: 'Sesión exportada',
    sessionExportFailed: 'No se pudo exportar la sesión',
    imageSaved: 'Imagen guardada',
    downloadStarted: 'Descarga iniciada',
    restartToUseSaveImage: 'Reinicia Hermes Desktop para usar Guardar imagen.',
    restartToSaveImages: 'Reinicia Hermes Desktop para guardar imágenes',
    imageDownloadFailed: 'Falló la descarga de imagen',
    openImage: 'Abrir imagen',
    downloadImage: 'Descargar imagen',
    savingImage: 'Guardando imagen',
    imagePreviewFailed: 'Falló la vista previa de imagen',
    imageAttach: 'Adjuntar imagen',
    imageWriteFailed: 'No se pudo escribir la imagen en disco.',
    imageAttachFailed: 'No se pudo adjuntar la imagen',
    pastedContent: 'Contenido pegado',
    pasteAttachFailed: 'No se pudo adjuntar el texto pegado',
    attachImages: 'Adjuntar imágenes',
    clipboard: 'Portapapeles',
    noClipboardImage: 'No se encontró ninguna imagen en el portapapeles',
    clipboardPasteFailed: 'No se pudo pegar desde el portapapeles',
    dropFiles: 'Soltar archivos',
    handoff: {
      pickPlatform: 'Elige un destino',
      success: platform => `Transferido a ${platform}. Puedes reanudar aquí cuando quieras.`,
      systemNote: platform => `↻ Transferido a ${platform}; puedes reanudar aquí cuando quieras.`,
      failed: error => `La transferencia falló: ${error}`,
      timedOut:
        'Hermes no pudo llegar a tu conexión de mensajería. Iníciala desde Configuración → Mensajería y vuelve a intentar el traspaso.',
      startMessaging: 'Iniciar mensajería'
    }
  },
  tips: {
    close: 'No volver a mostrar este consejo',
    items: {
      'new-session': {
        title: 'Empieza de cero',
        text: 'Un chat nuevo tiene su propio contexto, terminal y directorio de trabajo.'
      },
      skills: {
        title: 'Enséñale una vez',
        text: 'Las skills son carpetas de instrucciones que Hermes carga cuando el trabajo las necesita.'
      },
      messaging: {
        title: 'Hermes lejos de tu escritorio',
        text: 'Conecta Telegram, Discord, Slack y más: el mismo agente, la misma memoria.'
      },
      artifacts: {
        title: 'Todo lo que ha creado Hermes',
        text: 'Imágenes, archivos y enlaces de cada sesión, indexados en un solo lugar.'
      },
      cron: {
        title: 'Trabajo que se ejecuta solo',
        text: 'Programa un prompt cada hora, cada noche o con una expresión cron.'
      },
      'command-palette': {
        title: 'Un cuadro para todo',
        text: 'Sesiones, configuración, skills y comandos responden a la paleta.'
      },
      profiles: {
        title: 'Los perfiles son independientes',
        text: 'Cada uno es su propio Hermes: sus propias claves, su propia memoria, sus propias sesiones.'
      },
      'composer-mentions': {
        title: 'Adjunta y ordena',
        text: 'Escribe @ para traer un archivo a la conversación y / para ejecutar un comando.'
      },
      'local-runtime-update': {
        title: 'Hay una actualización del motor local',
        text: 'Actualiza el motor que ejecuta tus modelos locales. Las solicitudes locales activas pueden interrumpirse.',
        action: 'Actualizar ahora'
      },
      'local-setup': {
        title: 'Este equipo puede ejecutar modelos localmente',
        text: 'Tu hardware puede servir un modelo local. Los chats se quedan en tu equipo y no cuestan nada.',
        action: 'Configurarlo'
      },
      'right-pane': {
        title: 'El panel de trabajo',
        text: 'Archivos, terminal, revisión y el navegador integrado comparten el lado derecho.'
      }
    }
  },
  errors: {
    genericFailure: 'Algo salió mal',
    boundaryTitle: 'Algo se rompió en la interfaz',
    boundaryDesc: 'La vista encontró un error inesperado. Tus chats y configuración están a salvo.',
    boundaryDetails: 'Detalles',
    sendDiagnostics: 'Enviar diagnóstico',
    reloadWindow: 'Recargar ventana',
    openLogs: 'Abrir registros'
  },
  ui: {
    search: {
      clear: 'Limpiar búsqueda'
    },
    pagination: {
      label: 'paginación',
      previous: 'Anterior',
      previousAria: 'Ir a la página anterior',
      next: 'Siguiente',
      nextAria: 'Ir a la página siguiente'
    },
    sidebar: {
      title: 'Barra lateral',
      description: 'Muestra la barra lateral móvil.',
      toggle: open => `${open ? 'Mostrar' : 'Ocultar'} barra lateral`
    }
  }
} satisfies TranslationOverrides

export const es = defineLocale(esOverrides)
