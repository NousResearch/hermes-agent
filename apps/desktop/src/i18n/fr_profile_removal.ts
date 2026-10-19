// The profile delete / reset confirmation copy, composed by fr.ts as `profiles.removal`.
import type { TranslationOverrides } from './define-locale'

export const frProfileRemoval: NonNullable<TranslationOverrides['profiles']>['removal'] = {
  deleteTitle: 'Supprimer le profil ?',
  deleteDescPrefix: 'Cela supprimera ',
  deleteDescMid: ' et supprimera son ',
  deleteDescSuffix: ' répertoire. Cela ne peut pas être annulé.',
  deleting: 'Suppression...',
  resetTitle: 'Réinitialiser le profil par défaut ?',
  resetDesc:
    'Le profil par défaut ne peut pas être supprimé : son dossier contient l’installation de Hermes et vos autres profils. ' +
    'La réinitialisation efface ses mémoires, compétences, réglages, SOUL.md et conversations, et supprime ses tâches cron. ' +
    'Vos autres profils, l’installation ainsi que vos clés API et connexions sont conservés. Cette action est irréversible.',
  resetAction: 'Réinitialiser',
  resetting: 'Réinitialisation...',
  resetDone: 'Profil réinitialisé'
}
