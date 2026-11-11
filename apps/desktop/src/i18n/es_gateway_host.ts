import type { GatewayHostTranslations } from './types_gateway_host'

// Settings → Machines copy where the connection cannot change, spread into es.ts.
export const esGatewayHost: GatewayHostTranslations = {
  unavailableTitle: 'Ajustes de la máquina no disponibles',
  unavailableDesc:
    'Los ajustes de conexión solo se pueden cambiar desde la app Hermes Desktop en el equipo que la ejecuta.',
  webappHostTitle: 'Host de Hermes',
  webappHostDesc:
    'La Webapp siempre se ejecuta en el host de Hermes que la sirve. Las conexiones guardadas de abajo son la lista de esta máquina, compartida con la app Hermes Desktop; las demás máquinas se alcanzan a través de este host.',
  webappOauthPaste: {
    title: 'Terminar de iniciar sesión',
    description:
      'Si la pestaña de inicio de sesión terminó en una dirección que no carga (este navegador no está en la máquina que ejecuta Hermes), pega esa dirección aquí.',
    placeholder: 'http://127.0.0.1:…/webapp/gateway-connections/oauth/callback?…',
    finish: 'Terminar',
    failed: 'No se pudo terminar el inicio de sesión'
  },
  webAccess: {
    title: 'Acceso web',
    intro:
      'Abre Hermes de este ordenador desde tu teléfono u otro ordenador, con tu cuenta de Nous. Quien inicie sesión puede ejecutar comandos en este ordenador.',
    failed: 'No se pudo cambiar el acceso web',
    running: 'En ejecución',
    stopped: 'Detenido',
    starting: 'Iniciando',
    stopping: 'Deteniendo',
    restartNeeded: 'Reinicio necesario',
    unavailable: 'No disponible',
    notRunningDetail: 'Inícialo para abrir Hermes desde otro dispositivo.',
    needsNous: 'Primero inicia sesión en tu cuenta de Nous en Proveedores.',
    runningLan: 'Abre esta dirección en un dispositivo de la misma red.',
    runningPublic: 'Servido a través de tu URL.',
    runningLocal:
      'Solo este ordenador puede abrirla. Detenla y vuelve a iniciarla para abrirla desde otros dispositivos.',
    pendingOn: 'Guardado. Hermes servirá la aplicación web cuando se reinicie.',
    pendingOff: 'Desactivado, pero Hermes sigue sirviendo la aplicación web hasta que se reinicie.',
    unavailableDetail: reason => `Hermes no pudo servir la aplicación web: ${reason}`,
    restartBusy:
      'Guardado. Hay un chat en curso, así que Hermes no se reinició. Elige «Reiniciar Hermes» cuando termine.',
    restartFailed: 'Guardado, pero Hermes no pudo reiniciarse. Elige «Reiniciar Hermes» para volver a intentarlo.',
    restartFailedBecause: reason => `Hermes no pudo reiniciarse: ${reason}`,
    startTimedOut: 'Hermes no sirvió la aplicación web en 90 segundos.',
    stopTimedOut: 'Hermes no dejó de servir la aplicación web en 90 segundos.',
    startStages: {
      preparing: 'Preparando la aplicación web. Compilarla puede tardar unos minutos.',
      building: 'Compilando la aplicación web. Puede tardar unos minutos.',
      registering: 'Configurando el inicio de sesión de Nous para esta dirección.',
      restarting: 'Reiniciando Hermes para servir la aplicación web.'
    },
    startElapsed: time => `Tiempo transcurrido: ${time}`,
    start: 'Iniciar',
    stop: 'Detener',
    restartGateway: 'Reiniciar Hermes',
    modeLan: 'Esta red',
    modeLanDetail: url => `Los dispositivos de tu wifi abren ${url}. Úsalo solo en una red de confianza.`,
    lanUnencrypted: 'La conexión no está cifrada: otros en esta red podrían leer lo que envías y recibes.',
    noLan: 'Este ordenador no tiene una dirección de red a la que puedan llegar otros dispositivos.',
    modePublic: 'Mi propia URL',
    modePublicDetail: port =>
      `Una dirección HTTPS que ya tengas, como Tailscale Serve o un túnel, que reenvíe a http://127.0.0.1:${port}.`
  }
}
