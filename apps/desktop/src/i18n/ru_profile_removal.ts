// The profile delete / reset confirmation copy, composed by ru.ts as `profiles.removal`.
import type { TranslationOverrides } from './define-locale'

export const ruProfileRemoval: NonNullable<TranslationOverrides['profiles']>['removal'] = {
  deleteTitle: 'Удалить профиль?',
  deleteDescPrefix: 'Это удалит ',
  deleteDescMid: ' и сотрёт его ',
  deleteDescSuffix: ' каталог. Это действие необратимо.',
  deleting: 'Удаление...',
  resetTitle: 'Сбросить профиль по умолчанию?',
  resetDesc:
    'Профиль по умолчанию нельзя удалить: в его каталоге находятся установка Hermes и ваши другие профили. ' +
    'Сброс сотрёт его память, навыки, настройки, SOUL.md и чаты, а также удалит его задачи cron. ' +
    'Другие профили, установка, API-ключи и входы в аккаунты сохранятся. Это действие нельзя отменить.',
  resetAction: 'Сбросить',
  resetting: 'Сброс...',
  resetDone: 'Профиль сброшен'
}
