import type { TranslationOverrides } from './define-locale'
import { esDesktopCommands } from './es_desktop_commands'
import { esSessionCommands } from './es_session_commands'

// Shell notices (remote-display toast, butterbar, keep-awake offer), spread into es.ts.
export const esNotices = {
  remoteDisplayBanner: {
    message: reason =>
      `Renderizado por software activo — se detectó una pantalla remota (${reason}). Se desactivó la aceleración por GPU para evitar parpadeos.`
  },
  butterbar: {
    goTo: (index, total) => `Mostrar aviso ${index} de ${total}`,
    legal: {
      before: 'El uso de Hermes Agent está sujeto a nuestros ',
      terms: 'Términos del servicio',
      between: ' y a nuestra ',
      privacy: 'Política de privacidad',
      after: '.'
    }
  },
  promptNotices: {
    legacySendUnconfirmed:
      'Este servidor no pudo confirmar el envío anterior de este mensaje, así que es posible que ya se haya ejecutado. Revisa la conversación antes de volver a enviarlo.',
    mcpReloadDeferred:
      'Este chat está ocupado: el cambio de MCP se aplica cuando termine su turno y MCP se recargue de nuevo, o al volver a abrir el chat',
    mcpReloadUnavailable: servers => `No se conectó: ${servers}`,
    compactedTurnUnavailable:
      'Ese mensaje se resumió al comprimir el chat, así que ya no se puede restaurar ni editar.',
    staleTurnUnavailable: 'Ese mensaje ya no está en el historial de este chat.',
    sessionCommands: esSessionCommands,
    desktopCommands: esDesktopCommands
  },
  keepAwakeOffer: {
    title: 'Tu equipo entró en suspensión mientras Hermes trabajaba',
    message:
      'Hermes puede mantener tu equipo activo mientras se ejecuta una tarea. La pantalla se sigue apagando, y con batería el equipo aún puede entrar en suspensión.',
    action: 'Mantener activo mientras trabaja'
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
  }
} satisfies Pick<
  TranslationOverrides,
  'remoteDisplayBanner' | 'butterbar' | 'promptNotices' | 'keepAwakeOffer' | 'billingBlock' | 'sendDiagnostics'
>
