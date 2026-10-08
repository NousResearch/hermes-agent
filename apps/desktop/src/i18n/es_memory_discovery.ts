import type { MemoryDiscoveryTranslations } from './types_memory_discovery'

export const esMemoryDiscovery = {
  installed: 'Instalados',
  availableToInstall: 'Disponibles para instalar',
  installationRequired: 'Requiere instalación',
  reviewInstall: 'Revisar e instalar',
  exploreAll: 'Explorar todos…',
  missing: 'Ausente',
  installConsent:
    'Instala y habilita el plugin con sus dependencias. El proveedor de memoria activo no cambia hasta que lo elijas explícitamente.',
  builtin: 'Integrado',
  configureElsewhere: 'Configura el proveedor mediante la CLI o actualiza Hermes para guardar sin activar.',
  notReady:
    'Completa la configuración e instala las dependencias. Tras instalar, reinicia el backend y vuelve a intentarlo.',
  useFailed: 'No se pudo usar el proveedor. Revisa su configuración e inténtalo de nuevo.',

  activeProvider: name => `Activo: ${name}`,
  useProvider: 'Usar proveedor',
  loadFailed: 'No se pudieron cargar los proveedores de memoria',
  ownerChanged: 'Vuelve a la conexión y al perfil donde abriste este instalador e inténtalo de nuevo.',
  notDiscovered:
    'El paquete se instaló, pero su proveedor de memoria aún no se ha detectado. Vuelve a los ajustes de memoria para intentarlo de nuevo.',
  installedNotice: 'Proveedor detectado. Configúralo y después elige usarlo explícitamente.',
  backToMemory: 'Volver a los ajustes de memoria',
  connect: 'Conectar',
  reconnect: 'Volver a conectar',
  connectOAuth: 'Conectar con OAuth',
  apiKeySet: 'Clave de API configurada',
  oauthSet: 'OAuth conectado',
  waitingConsent: 'Esperando el consentimiento en el navegador…',
  stopWaiting: 'Dejar de esperar',
  stoppedWaiting: 'Se dejó de esperar. La autorización puede seguir pendiente.',
  startFailed: 'No se pudo iniciar la conexión.',
  connectionFailed: 'La conexión falló.'
} satisfies Partial<MemoryDiscoveryTranslations>
