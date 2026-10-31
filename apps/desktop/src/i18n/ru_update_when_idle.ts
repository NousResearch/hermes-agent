import type { UpdateWhenIdleCopy } from './types_update_when_idle'

export const ruUpdateWhenIdle: UpdateWhenIdleCopy = {
  busyTitle: 'Hermes ещё работает',
  busyBody: 'Обновление перезапускает Hermes и остановит:',
  chats: (n: number) => (n === 1 ? 'Идёт 1 чат' : `Идут чаты: ${n}`),
  cron: (jobs: string) => `Выполняется задание по расписанию (${jobs})`,
  delegation: 'В фоне работают субагенты',
  input: 'Запрос ждёт вашего ответа',
  unknown: 'Hermes не смог подтвердить, что он свободен',
  updateNow: (n: number) => `Обновить сейчас (прервёт ${n})`,
  updateWhenIdle: 'Обновить, когда освободится',
  waitingTitle: 'Обновление ждёт, пока Hermes закончит',
  waitingBody:
    'Обновление начнётся само, когда ничего не будет выполняться. Продолжайте работу; до этого Hermes доступен как обычно.',
  waitingFor: 'Ожидание:',
  waitingIdle: 'Ничего не выполняется. Запуск обновления…',
  waitingUntil: (time: string) => `Если в ${time} Hermes всё ещё занят, обновление будет пропущено.`,
  cancel: 'Отменить обновление',
  timedOut: 'Hermes оставался занят, поэтому обновление не запустилось. Ничего не изменено; попробуйте позже.'
}
