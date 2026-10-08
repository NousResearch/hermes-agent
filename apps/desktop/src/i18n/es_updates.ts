import type { TranslationOverrides } from './define-locale'

// The updates surface's copy (About page, updates overlay, statusbar version items), composed by es.ts.
export const esUpdates = {
  updates: {
    discontinuedTitle: 'Esta versión de Hermes ya no tiene soporte',
    discontinuedBody:
      'Esta versión de Hermes ya no tiene soporte y podría dejar de funcionar; desinstálala. Tus datos permanecen en el disco.',
    channels: { stable: 'Estable', canary: 'Canary' },
    appName: 'Hermes',
    availableBodyRelease: tag => `La versión ${tag} está lista para instalarse.`,
    releaseAvailable: tag => `La versión ${tag} está disponible.`,
    checkingShort: 'Comprobando…',
    availableBodyAppInstaller:
      'Hay una nueva versión de Hermes. Hermes se cerrará, Windows terminará la actualización y Hermes volverá a abrirse automáticamente.',
    applyingBodyAppInstaller:
      'Hermes se cerrará y Windows terminará la actualización. Hermes volverá a abrirse al finalizar; no tienes que hacer nada.',
    applyingCloseAppInstaller:
      'Esta ventana se cerrará; Windows terminará la actualización y Hermes volverá a abrirse automáticamente.',
    checkUnknownTitleAppInstaller: 'No se pudieron buscar actualizaciones',
    checkUnknownBodyAppInstaller:
      'Windows no pudo buscar actualizaciones ahora. También se instalan automáticamente al reiniciar Hermes.',
    versionDetailsTitle: 'Detalles de la versión',
    versionDetailsBody:
      'Esta instalación se administra fuera de la app. Actualízala de la misma forma en que la instalaste.',
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
    localBranchBehind: count =>
      `Este checkout está en una rama local con sus propios commits: ${count} commit${count === 1 ? '' : 's'} por detrás del main upstream. Actualízalo desde una terminal con \`hermes update\`.`,
    localBranchBehindUnknown:
      'Este checkout está en una rama local con sus propios commits; no se pudo contar su distancia respecto al main upstream.',
    localBranchCurrent: 'Este checkout está en una rama local con sus propios commits y está al día con el main upstream.',
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
    copyFullLog: 'Copiar el registro de cambios completo',
    manualTitle: 'Actualizar desde la terminal',
    manualUnavailableTitle: 'No se puede actualizar desde aquí',
    manualBody:
      'Instalaste Hermes desde la línea de comandos, así que las actualizaciones también se ejecutan ahí. Pega esto en tu terminal:',
    manualPickedUp: 'Hermes usará la nueva versión la próxima vez que lo abras.',
    manualBodyBackend: 'El backend de Hermes se gestiona fuera de esta app. Ejecuta esto en el servidor que lo aloja:',
    manualPickedUpBackend: 'El backend cargará la nueva versión cuando termine la actualización.',
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
  }
} satisfies Pick<TranslationOverrides, 'updates'>
