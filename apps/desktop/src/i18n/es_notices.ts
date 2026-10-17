import type { TranslationOverrides } from './define-locale'
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
    mcpReloadNotInChat: server => `${server} no está activado en este chat`,
    mcpReloadUnavailable: servers => `No se conectó: ${servers}`,
    sessionCommands: esSessionCommands
  },
  keepAwakeOffer: {
    title: 'Tu equipo entró en suspensión mientras Hermes trabajaba',
    message:
      'Hermes puede mantener tu equipo activo mientras se ejecuta una tarea. La pantalla se sigue apagando, y con batería el equipo aún puede entrar en suspensión.',
    action: 'Mantener activo mientras trabaja'
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'butterbar' | 'promptNotices' | 'keepAwakeOffer'>
