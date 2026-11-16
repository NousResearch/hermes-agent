import type { ClarifyCopy } from './types_clarify'

export const ruClarify: Partial<ClarifyCopy> = {
  notReady: 'Запрос уточнения ещё не готов',
  gatewayDisconnected: 'Шлюз Hermes не подключён',
  sendFailed: 'Не удалось отправить ответ на уточнение',
  loadingQuestion: 'Загрузка вопроса…',
  other: 'Другое (введите ответ)',
  placeholder: 'Введите ваш ответ…',
  skip: 'Пропустить',
  skipped: 'Пропущено',
  noAnswer: 'Нет ответа',
  confirmAndContinueLabel: 'Подтвердить и продолжить',
  singleSelectHint: 'Выберите один',
  multiSelectHint: 'Выберите все подходящие',
  oneQuestion: '1 вопрос',
  questionProgress: (answered, total) => `Ответ дан на ${answered} из ${total}`,
  notDelivered:
    'Этот вопрос не дошёл до приложения, поэтому ответить здесь нельзя. Нажмите «Стоп», чтобы завершить ход, и ответьте в чате.'
}
