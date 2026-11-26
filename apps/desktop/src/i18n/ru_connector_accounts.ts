import type { TranslationOverrides } from './define-locale'

export const ruConnectorAccounts: NonNullable<TranslationOverrides['connectorsPage']>['accounts'] = {
  heading: 'Аккаунты',
  count: count => `Аккаунтов: ${count}`,
  addAnother: 'Добавить ещё один аккаунт',
  aliasLabel: 'Имя аккаунта',
  aliasHint: 'До 32 строчных латинских букв, цифр и дефисов. Например, work или home.',
  aliasInvalid: 'Используйте строчные латинские буквы, цифры и дефисы. Первый символ — буква или цифра.',
  aliasTaken: 'Это имя уже занято.',
  add: 'Подключить',
  rename: 'Переименовать',
  renameLabel: name => `Новое имя для ${name}`,
  renameGone: 'Этого аккаунта больше нет. Он был удалён или заменён при переподключении.',
  remove: 'Удалить',
  removeTitle: (app, name) => `Удалить аккаунт ${app} «${name}»?`,
  removeBody: 'Hermes перестанет действовать от имени этого аккаунта. Другие аккаунты останутся подключёнными.',
  retiredCount: count => `Заменено: ${count}`,
  status: {
    expired: 'Срок доступа истёк',
    failed: 'Не удалось подключиться',
    inactive: 'Неактивен',
    pending: 'Ожидание входа',
    retired: 'Заменён при переподключении',
    revoked: 'Доступ отозван'
  }
}
