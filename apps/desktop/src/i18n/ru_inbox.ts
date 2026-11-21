import type { InboxCopy } from './types_inbox'

export const ruInbox: InboxCopy = {
  title: 'Входящие',
  unreadCount: (count: number) => `Непрочитанных: ${count}`,
  loading: 'Загрузка входящих…',
  emptyTitle: 'Результатов пока нет',
  emptyDesc: 'Когда запланированная задача завершится, её результат появится здесь.',
  running: 'Выполняется',
  unread: 'Не прочитано',
  noPreview: 'Завершено без ответа.',
  refresh: 'Обновить',
  refreshing: 'Обновление…',
  markReadFailed: 'Не удалось отметить результат как прочитанный',
  newResultsTitle: (count: number) => `Новых результатов запланированных задач: ${count}`,
  newResultsBody: 'Откройте «Входящие», чтобы прочитать их.'
}
