import type { TranslationOverride } from '@hermes/shared/i18n'

import { ruAuxTasks } from './ru_aux_tasks'
import type { ModelSettingsTranslations } from './types_model_settings'

export const ruModelSettings: TranslationOverride<ModelSettingsTranslations> = {
  moaPreset: 'Пресет',
  moaDescription:
    'Настройте именованные пресеты, которые появятся как модели провайдера Mixture of Agents. Модель-агрегатор выполняет каждый шаг цикла инструментов; почти вся стоимость запуска списывается у её провайдера. По умолчанию референсные модели дают совет один раз на сообщение пользователя.',
  moaAggregator: 'Модель-агрегатор',
  moaAggregatorBilled: 'исполняющая модель · на неё приходится стоимость запуска',
  moaReferenceHint: 'по умолчанию советует один раз за ход',
  setupProviderFallback: 'провайдер',
  setUpProvider: name => `Настроить ${name}`,
  staleAuxBefore: (count, names) => `Вспомогательные задачи (${count}: ${names}) по-прежнему используют `,
  staleAuxAfter: ', а не основную модель.',
  staleAuxOtherProviders: 'других провайдеров',
  moaEnabled: 'Включено',
  moaSetDefault: 'Сделать основным',
  moaNewPresetPlaceholder: 'новый пресет',
  moaAddPreset: 'Добавить пресет',
  customModel: 'Своя модель…',
  customModelPlaceholder: 'ID модели',
  chooseFromList: 'Выбрать из списка',
  moaDefault: 'По умолчанию:',
  moaReferenceToggle: (enabled, index) => `${enabled ? 'Отключить' : 'Включить'} референс ${index}`,
  moaReferenceTitle: index => `Референс ${index}`,
  moaAddReference: 'Добавить референсную модель',
  loading: 'Загрузка конфигурации модели…',
  appliesDesc:
    'Применяется к новым сеансам. Для горячей смены модели в активном чате используйте выборщик модели в композере.',
  provider: 'Провайдер',
  model: 'Модель',
  applying: 'Применение…',
  defaultsLabel: 'По умолчанию',
  reasoning: 'Рассуждения',
  reasoningOff: 'Выкл',
  speed: 'Скорость',
  speedStandard: 'Стандартная',
  defaultsFailed: 'Не удалось сохранить модель по умолчанию',
  auxiliaryTitle: 'Вспомогательные модели',
  resetAllToMain: 'Сбросить всё на основную',
  staleAuxDismiss: 'Больше не показывать',
  auxiliaryDesc:
    'Вспомогательные задачи по умолчанию выполняются основной моделью. Назначьте отдельную модель любой задаче, чтобы переопределить.',
  setToMain: 'На основную',
  change: 'Изменить',
  autoUseMain: 'авто · использовать основную модель',
  inheritMainEffort: 'наследовать · усилие основной модели',
  inheritsFrom: task => `наследует ${task}`,
  followTask: task => `Как ${task}`,
  providerDefault: '(по умолчанию провайдера)',
  fallbackAdd: 'Добавить запасную',
  fallbackEmpty: 'Запасных моделей нет — используется модель по умолчанию, если она не падает.',
  notInCatalog: 'нет в списке моделей этого провайдера — вызовы могут уходить на запасную.',
  moaTitle: 'Смесь агентов',
  tasks: ruAuxTasks,
  subagents: {
    inherit: 'Наследовать от основного агента',
    modelOnly: 'Своя модель (учётные данные основного агента)',
    providerLabel: 'Провайдер субагентов',
    modelLabel: 'Модель субагентов',
    direct: 'Прямой эндпоинт',
    customProvider: 'Свой провайдер...',
    customProviderLabel: 'Свой провайдер субагентов',
    parentModel: 'Использовать модель основного агента',
    parentModelWarning:
      'Выбранный провайдер должен поддерживать модель основного агента. Маршрут только с провайдером может не сработать, если модель несовместима.',
    directActive:
      'Активен прямой эндпоинт, он важнее сохранённого провайдера. Пока выбран этот режим, эндпоинт и учётные данные сохраняются.',
    directClear:
      'Применение этого маршрута удалит прямой эндпоинт и его сохранённый ключ. Остальные настройки запросов сохранятся.',
    catalogFailed: 'Каталог моделей недоступен. Введите провайдера и модель вручную или повторите попытку.',
    draftHint: 'Изменения остаются локальными, пока вы их не примените.'
  }
}
