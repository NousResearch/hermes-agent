import type { TranslationOverrides } from './define-locale'

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
  keepAwakeOffer: {
    title: 'Tu equipo entró en suspensión mientras Hermes trabajaba',
    message:
      'Hermes puede mantener tu equipo activo mientras se ejecuta una tarea. La pantalla se sigue apagando, y con batería el equipo aún puede entrar en suspensión.',
    action: 'Mantener activo mientras trabaja'
  }
} satisfies Pick<TranslationOverrides, 'remoteDisplayBanner' | 'butterbar' | 'keepAwakeOffer'>
