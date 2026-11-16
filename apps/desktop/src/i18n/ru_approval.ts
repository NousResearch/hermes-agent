import type { ApprovalCopy } from './types_approval'

export const ruApproval: Partial<ApprovalCopy> = {
  gatewayDisconnected: 'Шлюз Hermes не подключён',
  sendFailed: 'Не удалось отправить ответ на подтверждение',
  run: 'Выполнить',
  command: 'Команда',
  moreOptions: 'Больше опций подтверждения',
  allowSession: 'Разрешить в этом сеансе',
  alwaysAllowMenu: 'Всегда разрешать…',
  jumpToApproval: 'Нужно подтверждение',
  reject: 'Отклонить',
  alwaysTitle: 'Всегда разрешать эту команду?',
  alwaysDescription: pattern =>
    `Это добавит паттерн «${pattern}» в ваш постоянный список разрешений (~/.hermes/config.yaml). Hermes больше не будет спрашивать о подобных командах — ни в этом сеансе, ни в будущем.`,
  alwaysAllow: 'Всегда разрешать',
  deny: 'Запретить',
  allow: 'Разрешить',
  allowHint: 'До выключения песочницы',
  denyReason: 'Причина запрета',
  denyReasonPlaceholder: 'Объясните Hermes причину (необязательно)',
  access: {
    title: 'Доступ к папке',
    readOnly: 'Только чтение',
    readWrite: 'Чтение и запись',
    alwaysTitle: 'Всегда разрешать эту папку?',
    alwaysDescription: 'Папка добавляется в песочницу этого профиля. Hermes больше не спросит о ней ни в одном чате.'
  },
  network: {
    title: 'Доступ к сайту',
    alwaysTitle: 'Всегда разрешать этот сайт?',
    alwaysDescription:
      'Сайт добавляется в разрешённые сайты этого профиля. Hermes больше не спросит о нём ни в одном чате.'
  }
}
