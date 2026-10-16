import type { PluginsTabCopy } from './types_plugins_tab'

export const esPluginsTab: PluginsTabCopy = {
  desktopHalfPending: 'copiando…',
  desktopHalfPendingTip:
    'Este paquete incluye una mitad de escritorio que todavía no se ha copiado en la app. Usa Volver a escanear o reinicia la app.',
  desktopHalfRemote: 'no disponible (backend remoto)',
  desktopHalfRemoteTip:
    'La mitad de escritorio de este paquete está en el disco del backend remoto, que esta app no puede leer. Para usarla aquí, ejecuta Instalar desde Git con la URL del repositorio del paquete y el destino Escritorio marcado; eso clona la mitad de escritorio en este equipo.',
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
  fleet: {
    updateEverywhere: (count: number) => `Actualizar en todas partes · ${count}`,
    updateHereOnly: (profile: string) => `Solo ${profile}`,
    updateAll: (count: number) => `Actualizar todo · ${count}`,
    updateAllTip: (installs: number, plugins: number) =>
      `${plugins === 1 ? '1 plugin' : `${plugins} plugins`} desactualizados en ${installs === 1 ? '1 perfil' : `${installs} perfiles`} de tus gateways`,
    titleOne: (name: string) => `Actualizar ${name} en todas partes`,
    titleAll: 'Actualizar plugins en todas partes',
    summary: (installs: number, gateways: number) =>
      `${installs === 1 ? '1 perfil' : `${installs} perfiles`} en ${gateways === 1 ? '1 gateway' : `${gateways} gateways`} usan una versión anterior.`,
    summaryDone: (installs: number) =>
      `${installs === 1 ? '1 perfil actualizado' : `${installs} perfiles actualizados`}. Reinicia esos gateways para cargar la nueva versión.`,
    summaryFailed: (ok: number, failed: number) =>
      `${ok} actualizados, ${failed} fallidos. Pasa el cursor sobre un fallo para ver el motivo.`,
    summaryDeclined: (ok: number, declined: number) =>
      `${ok} actualizados. ${declined} no aplicados porque se rechazaron los nuevos permisos.`,
    thisProfile: '· este perfil',
    failed: 'Falló',
    declined: 'No aplicado',
    skippedOffline: 'Sin conexión · omitido',
    skippedSignIn: 'Sesión cerrada · omitido',
    skippedIdle: 'No conectado · omitido',
    skippedUnreadable: 'No se pudo comprobar',
    confirm: (installs: number) => `Actualizar ${installs}`,
    updating: 'Actualizando…',
    done: 'Listo',
    consentBody: (names: string, installs: number) =>
      `La nueva versión de catálogo de ${names} añade capacidades que la versión instalada no tiene. Aprobar la aplica a ${installs === 1 ? '1 perfil' : `${installs} perfiles`}:`,
    finished: (installs: number) =>
      `${installs === 1 ? '1 perfil actualizado' : `${installs} perfiles actualizados`}. Reinicia esos gateways para aplicarlo.`
  }
}
