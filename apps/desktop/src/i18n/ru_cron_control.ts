import type { CronControlCopy } from './types_cron_control'

export const ruCronControl: CronControlCopy = {
  stop: 'Остановить',
  stopping: 'Остановка…',
  stopped: 'Запуск отменён',
  finishedFirst: 'Запуск завершился до остановки',
  nothingToStop: 'Нет активного запуска',
  failedStop: 'Не удалось остановить запуск',
  lastCancelled: (reason: string) => (reason ? `Последний запуск отменён: ${reason}` : 'Последний запуск отменён'),
  catchUpLabel: 'Пропущенные запуски',
  catchUpHint: 'Что делать с запусками, пропущенными, пока Hermes не работал.',
  catchUpChoices: {
    default: 'По умолчанию',
    run: 'Запустить один раз после возврата',
    skip: 'Пропустить'
  }
}
