import type { Translations } from './types'

export const ruModelOptions: Translations['shell']['modelOptions'] = {
  noOptions: 'Для этой модели нет опций',
  options: 'Опции',
  thinking: 'Размышление',
  fast: 'Быстрая',
  ultrafast: 'Ultrafast',
  useStandardSpeed: 'Использовать стандартную скорость',
  auto: 'Авто',
  cold: 'Холодный',
  speedPolicy: 'Политика скорости',
  effort: 'Усилия',
  minimal: 'Минимально',
  low: 'Низкое',
  medium: 'Среднее',
  high: 'Высокое',
  xhigh: 'Очень высокое',
  max: 'Максимум',
  ultra: 'Ультра',
  sendsOnRoute: (level: string) => `на этом маршруте отправляется ${level}`,
  updateFailed: 'Не удалось обновить опцию модели',
  fastFailed: 'Не удалось обновить быстрый режим'
}
