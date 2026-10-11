import type { TranslationOverrides } from './define-locale'

// Shell notices (remote-display toast, butterbar), spread into es.ts.
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
      'Este servidor no pudo confirmar el envío anterior de este mensaje, así que es posible que ya se haya ejecutado. Revisa la conversación antes de volver a enviarlo.'
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'butterbar' | 'promptNotices'>
