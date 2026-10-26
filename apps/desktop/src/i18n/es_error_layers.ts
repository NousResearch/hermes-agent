import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

// The error card's layer fallbacks (lib/error-surface-copy.ts): used when a failure code has no own entry.
export const esErrorLayers: Pick<ThreadCopy, 'errorLayerBodies' | 'errorLayers'> = {
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
    provider: 'El servicio de IA no pudo completar esta solicitud. Reinténtalo en un momento o cambia de proveedor.',
    runtime:
      'Hermes tuvo un problema interno al iniciar esta respuesta. Vuelve a enviar el mensaje; si sigue ocurriendo, envía un diagnóstico.',
    streaming: 'La conexión se cortó antes de que terminara la respuesta. Reinténtalo para enviarla de nuevo.'
  }
}
