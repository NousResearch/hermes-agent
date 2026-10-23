import type { TranslationOverrides } from './define-locale'

export const ruOnboarding: TranslationOverrides['onboarding'] = {
  headerTitle: 'Настроим для вас Hermes Agent',
  headerDesc: 'Подключите провайдера модели, чтобы начать общение. Большинство вариантов — в один клик.',
  preparingInstall: 'Hermes завершает установку. Обычно это занимает меньше минуты при первом запуске.',
  starting: 'Запускаем Hermes…',
  setupSlowTitle: 'Настройка занимает больше времени, чем обычно.',
  setupSlowBody: 'Hermes всё ещё запускается в фоновом режиме.',
  continueWithoutSetup: 'Продолжить без настройки',
  lookingUpProviders: 'Ищем провайдеров...',
  collapse: 'Свернуть',
  otherProviders: 'Другие провайдеры',
  haveApiKey: 'У меня есть API-ключ',
  chooseLater: 'Выберу провайдера позже',
  recommended: 'Рекомендуется',
  connected: 'Подключено',
  featuredPitch: 'Одна подписка, 300+ передовых моделей — рекомендуемый способ запускать Hermes',
  fireworksPitch: 'Прямой API моделей — передовые модели на хостинге Fireworks',
  openRouterPitch: 'Один ключ, сотни моделей — надёжный вариант по умолчанию',
  apiKeyOptions: {
    fireworks: {
      short: 'прямой API моделей',
      description: 'Прямой доступ к моделям на хостинге Fireworks AI.'
    },
    openrouter: {
      short: 'один ключ, много моделей',
      description: 'Сотни моделей за одним ключом. Хороший вариант по умолчанию для новых установок.'
    },
    openai: { short: 'модели класса GPT', description: 'Прямой доступ к моделям OpenAI.' },
    gemini: { short: 'модели Gemini', description: 'Прямой доступ к моделям Google Gemini.' },
    xai: { short: 'модели Grok', description: 'Прямой доступ к моделям xAI Grok.' },
    local: {
      short: 'self-hosted',
      description:
        'Укажите Hermes локальный или self-hosted OpenAI-совместимый endpoint (vLLM, llama.cpp, Ollama и т.д.).'
    }
  },
  backToSignIn: 'Назад ко входу',
  getKey: 'Получить ключ',
  replaceCurrent: 'Заменить текущее значение',
  pasteApiKey: 'Вставьте API-ключ',
  localApiKeyPlaceholder: 'API-ключ (необязательно — только если ваш endpoint его требует)',
  couldNotSave: 'Не удалось сохранить учётные данные.',
  connecting: 'Подключение',
  update: 'Обновить',
  flowSubtitles: {
    pkce: 'Откроет браузер для входа, затем продолжит здесь',
    device_code: 'Откроет страницу подтверждения в браузере — Hermes подключится автоматически',
    external: 'Войдите один раз в терминале, затем вернитесь в чат'
  },
  startingSignIn: provider => `Начинаем вход для ${provider}...`,
  verifyingCode: provider => `Проверяем ваш код через ${provider}...`,
  connectedProvider: provider => `${provider} подключён`,
  connectedPicking: provider => `${provider} подключён. Выбираем модель по умолчанию...`,
  signInFailed: 'Вход не удался. Попробуйте снова.',
  pickDifferentProvider: 'Выбрать другого провайдера',
  signInWith: provider => `Войти через ${provider}`,
  openedBrowser: provider => `Мы открыли ${provider} в вашем браузере.`,
  authorizeThere: 'Авторизуйте Hermes там.',
  copyAuthCode: 'Скопируйте код авторизации и вставьте его ниже.',
  pasteAuthCode: 'Вставьте код авторизации',
  reopenAuthPage: 'Открыть страницу авторизации снова',
  waitingAuthorize: 'Ждём вашей авторизации...',
  externalPending: provider =>
    `${provider} входит через собственный CLI. Выполните эту команду в терминале, затем вернитесь и выберите «Я вошёл»:`,
  signedIn: 'Я вошёл',
  deviceCodeOpened: provider => `Мы открыли ${provider} в вашем браузере. Введите там этот код:`,
  reopenVerification: 'Открыть страницу подтверждения снова',
  copy: 'Копировать',
  defaultModel: 'Модель по умолчанию',
  freeTier: 'Бесплатный тариф',
  pro: 'Pro',
  free: 'Free',
  price: (input, output) => `${input} вход / ${output} выход за Mtok`,
  change: 'Изменить',
  startChatting: 'Начать',
  docs: provider => `Документация ${provider}`
}
