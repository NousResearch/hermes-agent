import type { TranslationOverrides } from './define-locale'

// Temporary (incognito) chat copy, composed by es.ts.
export const esTemporaryChat = {
  temporaryChat: {
    title: 'Chat temporal',
    description:
      'Los chats temporales no aparecen en tu historial, no usan ni actualizan la memoria y se eliminan al salir del chat.',
    placeholder: 'Pregunta lo que quieras en un chat temporal',
    use: 'Usar chat temporal',
    exit: 'Salir del chat temporal',
    badge: 'Temporal',
    badgeHint: 'No se guarda. Termina cuando lo cierras o tras 10 minutos inactivo mientras estás en otra parte.',
    newChat: 'Nuevo chat temporal',
    unsupportedBackend:
      'Este backend de Hermes aún no admite chats temporales. Actualízalo o desactiva el modo temporal para chatear con normalidad.',
    endChat: 'Terminar chat temporal',
    endTitle: '¿Terminar este chat temporal?',
    endDescription: 'Se elimina para siempre; no se guarda nada.',
    endConfirm: 'Terminar y eliminar',
    returnTo: 'Volver al chat temporal',
    running: 'Aún respondiendo…'
  }
} satisfies TranslationOverrides
