import type { DelegationRoutingTranslations } from './types_delegation_routing'

export const ruDelegationRouting: DelegationRoutingTranslations = {
  title: 'Маршрутизация',
  settingsTitle: 'Параметры субагентов по умолчанию',
  intro: 'Как каждый субагент получает свою модель.',
  modes: {
    default: { label: 'По умолчанию', description: 'Все субагенты работают на модели субагента ниже.' },
    model_map: {
      label: 'Маршруты моделей',
      description: 'Назовите несколько моделей. Агент выбирает одну для каждого субагента.'
    },
    profile_routes: {
      label: 'Маршруты профилей',
      description: 'Субагенты работают как один из ваших профилей — с его моделью, навыками и ключами.'
    }
  },
  inUse: 'Используется',
  plugin: 'Плагин',
  pluginMissing: 'Маршруты моделей и маршруты профилей предоставляет плагин Advanced Delegation.',
  pluginDisabled: 'Advanced Delegation установлен, но отключён для этого профиля.',
  pluginNotLoaded: 'Advanced Delegation включён, но Hermes ещё не загрузил его. Перезапустите Hermes, чтобы завершить.',
  installPlugin: 'Установить плагин',
  installUnavailable:
    'Advanced Delegation пока нет в каталоге плагинов. Установите его в разделе Возможности › Плагины.',
  enablePlugin: 'Включить',
  enableFailed: 'Не удалось включить Advanced Delegation',
  modeFailed: 'Не удалось изменить маршрутизацию субагентов',
  loadFailed: 'Не удалось загрузить маршрутизацию субагентов',
  blockedPill: 'Не работает',
  blockedDetail:
    'Этот режим сохранён, поэтому Hermes отклоняет каждого субагента, а не переходит молча на «По умолчанию».',
  blockedMissing: 'Субагенты не запустятся, пока не установлен Advanced Delegation.',
  blockedDisabled: 'Субагенты не запустятся, пока Advanced Delegation не включён для этого профиля.',
  blockedNotLoaded:
    'Субагенты не запустятся, пока Hermes не загрузит Advanced Delegation. Перезапустите Hermes, чтобы завершить.',
  blockedUnavailable: label =>
    `Субагенты не запустятся, пока ${label} не сможет работать здесь. Его плагин сообщает, что не может.`,
  blockedUnregistered: name =>
    `Субагенты не запустятся, пока не установлен и не включён плагин, предоставляющий «${name}».`,
  thirdPartyDescription: 'Предоставлен плагином. Его настройки находятся в этом плагине.',

  modelRoutesTitle: 'Маршруты моделей',
  modelRoutesEmpty: 'Маршрутов пока нет. Пока вы не добавите маршрут, субагенты используют модель основного агента.',
  addRoute: 'Добавить маршрут',
  editRoute: 'Изменить',
  makeDefault: 'Сделать основным',
  deleteRoute: 'Удалить',
  defaultBadge: 'Основной',
  routeActions: name => `Действия для ${name}`,
  deleteConfirmTitle: name => `Удалить маршрут «${name}»?`,
  deleteConfirmBody: 'Субагенты больше не смогут его выбрать.',
  deleteConfirmNextDefault: name => `Это основной маршрут, поэтому основным станет «${name}».`,
  deleteConfirmAction: 'Удалить маршрут',
  saveFailed: 'Не удалось сохранить маршруты',
  baseUrlRoute: url => `эндпоинт ${url}`,

  dialogNew: 'Новый маршрут модели',
  dialogEdit: 'Изменить маршрут модели',
  dialogDesc: 'Агент читает название и описание, чтобы решить, какой маршрут выберет субагент.',
  nameLabel: 'Название',
  namePlaceholder: 'например, fast',
  nameHint: 'Коротко, без пробелов. Агент выбирает маршруты по этому названию.',
  descriptionLabel: 'Когда использовать',
  descriptionPlaceholder: 'например, Дёшево и быстро: поиск, сводки, простые правки.',
  descriptionHint: 'Одна строка, которую агент читает перед выбором.',
  providerLabel: 'Провайдер',
  providerPlaceholder: 'Выберите провайдера',
  modelLabel: 'Модель',
  effortLabel: 'Глубина рассуждений',
  effortInherit: 'Как у субагента по умолчанию',
  tierLabel: 'Скорость',
  tierNormal: 'Обычная',
  tierFast: 'Быстрая',
  tierUltrafast: 'Сверхбыстрая',
  tierHint: 'Только для моделей, которые это поддерживают.',
  optional: 'необязательно',
  errors: { required: 'Обязательно', spaces: 'Без пробелов', duplicate: 'Это название уже использует другой маршрут' },
  saveRoute: 'Сохранить маршрут',
  createRoute: 'Добавить маршрут',

  profileRoutesTitle: 'Профили, от имени которых могут работать субагенты',
  profileRoutesHint: 'Каждый отмеченный профиль приносит свою модель, навыки, персону и ключи.',
  profileDescriptionPlaceholder: 'Когда использовать (необязательно)',
  profileDefault: 'Основной',
  profileEmpty: 'Других профилей пока нет. Создайте профиль на панели профилей.',
  profileMissing: name => `${name} (профиль больше не существует)`,

  inactiveTitle: 'Не используется в этом режиме',
  inactiveModelMap: route =>
    `Маршруты моделей выбирают модель для каждого субагента. Субагенты, не выбравшие маршрут, используют «${route}».`,
  inactiveProfiles: profile =>
    `Маршруты профилей выбирают профиль для каждого субагента. Субагенты, не выбравшие профиль, работают как «${profile}».`,
  inactiveModelMapNoDefault:
    'Маршруты моделей выбирают модель для каждого субагента. Субагенты, не выбравшие маршрут, используют модель основного агента.',
  inactiveProfilesNoDefault:
    'Маршруты профилей выбирают профиль для каждого субагента. Субагенты, не выбравшие профиль, используют модель основного агента.',
  inactivePlugin: label => `${label} выбирает модель для каждого субагента, поэтому эта настройка не применяется.`,
  useDefault: 'Переключить на «По умолчанию»'
}
