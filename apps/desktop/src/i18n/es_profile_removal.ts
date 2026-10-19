// The profile delete / reset confirmation copy, composed by es.ts as `profiles.removal`.
import type { TranslationOverrides } from './define-locale'

export const esProfileRemoval: NonNullable<TranslationOverrides['profiles']>['removal'] = {
  deleteTitle: '¿Eliminar perfil?',
  deleteDescPrefix: 'Esto eliminará ',
  deleteDescMid: ' y quitará su directorio ',
  deleteDescSuffix: '. Esto no se puede deshacer.',
  deleting: 'Eliminando...',
  resetTitle: '¿Restablecer el perfil predeterminado?',
  resetDesc:
    'El perfil predeterminado no se puede eliminar: su directorio contiene la instalación de Hermes y tus otros perfiles. ' +
    'Restablecerlo borra sus memorias, habilidades, ajustes, SOUL.md y chats, y quita sus tareas cron. ' +
    'Se conservan tus otros perfiles, la instalación y tus claves de API e inicios de sesión. Esto no se puede deshacer.',
  resetAction: 'Restablecer',
  resetting: 'Restableciendo...',
  resetDone: 'Perfil restablecido'
}
