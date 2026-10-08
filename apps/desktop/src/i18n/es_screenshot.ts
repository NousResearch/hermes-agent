import type { ScreenshotTranslations } from './types_screenshot'

export const esScreenshot: ScreenshotTranslations = {
  leftCommand: 'Izquierda ⌘',
  rightCommand: 'Derecha ⌘',
  enabledTitle: 'Atajo de captura de pantalla',
  enabledDesc:
    'Pulsa las teclas Comando (⌘) izquierda y derecha a la vez desde cualquier app para capturar su ventana frontal y adjuntarla a tu borrador actual de Hermes. Nunca se envía automáticamente. Desactivado por defecto; se aplica solo a este Mac. El contenido de la ventana puede ser confidencial: revisa el adjunto antes de enviarlo.',
  statusTitle: 'Estado del atajo de captura',
  checking: 'Comprobando el atajo de captura…',
  disabled: 'El atajo de captura está desactivado.',
  starting: 'Iniciando la escucha del atajo. Todavía no está listo.',
  ready: 'El atajo está listo. Las capturas se adjuntan a tu borrador actual sin enviarse.',
  inputPermission:
    'El permiso de Monitorización de entrada permite a Hermes detectar las teclas Comando (⌘) izquierda y derecha mientras otra app está activa. Permite Hermes en Ajustes del Sistema → Privacidad y seguridad → Monitorización de entrada, vuelve aquí y reinténtalo.',
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
}
