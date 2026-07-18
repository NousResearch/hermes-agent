import { defineFieldCopy } from './field-copy'

export const RU_FIELD_LABELS: Record<string, string> = defineFieldCopy({
  model: 'Модель по умолчанию',
  modelContextLength: 'Окно контекста',
  fallbackProviders: 'Резервные модели',
  toolsets: 'Включённые наборы инструментов',
  timezone: 'Часовой пояс',
  display: {
    personality: 'Стиль общения',
    showReasoning: 'Блоки рассуждений'
  },
  agent: {
    maxTurns: 'Максимум ходов агента',
    imageInputMode: 'Вложения изображений',
    apiMaxRetries: 'Повторные запросы к API',
    serviceTier: 'Уровень обслуживания',
    toolUseEnforcement: 'Обязательное использование инструментов'
  },
  terminal: {
    cwd: 'Рабочая директория',
    backend: 'Среда выполнения команд',
    timeout: 'Время ожидания команды',
    persistentShell: 'Постоянная командная оболочка',
    envPassthrough: 'Передача переменных окружения',
    dockerImage: 'Образ Docker',
    singularityImage: 'Образ Singularity',
    modalImage: 'Образ Modal',
    daytonaImage: 'Образ Daytona'
  },
  fileReadMaxChars: 'Ограничение чтения файлов',
  toolOutput: {
    maxBytes: 'Ограничение вывода терминала',
    maxLines: 'Строк на странице вывода',
    maxLineLength: 'Ограничение длины строки'
  },
  codeExecution: {
    mode: 'Режим выполнения кода'
  },
  approvals: {
    mode: 'Режим подтверждений',
    timeout: 'Время ожидания подтверждения',
    mcpReloadConfirm: 'Подтверждать перезагрузку MCP'
  },
  commandAllowlist: 'Разрешённые команды',
  security: {
    redactSecrets: 'Скрывать секреты',
    allowPrivateUrls: 'Разрешить внутренние URL'
  },
  browser: {
    allowPrivateUrls: 'Внутренние URL в браузере',
    autoLocalForPrivateUrls: 'Локальный браузер для внутренних URL'
  },
  checkpoints: {
    enabled: 'Контрольные точки файлов',
    maxSnapshots: 'Ограничение контрольных точек'
  },
  voice: {
    recordKey: 'Горячая клавиша голосового ввода',
    maxRecordingSeconds: 'Максимальная длительность записи',
    autoTts: 'Озвучивать ответы'
  },
  stt: {
    enabled: 'Распознавание речи',
    echoTranscripts: 'Публиковать расшифровки',
    provider: 'Провайдер распознавания речи',
    local: {
      model: 'Локальная модель распознавания',
      language: 'Язык распознавания'
    },
    openai: {
      model: 'Модель распознавания OpenAI'
    },
    groq: {
      model: 'Модель распознавания Groq'
    },
    mistral: {
      model: 'Модель распознавания Mistral'
    },
    elevenlabs: {
      modelId: 'Модель распознавания ElevenLabs',
      languageCode: 'Язык ElevenLabs',
      tagAudioEvents: 'Отмечать звуковые события',
      diarize: 'Разделять речь по говорящим'
    }
  },
  tts: {
    provider: 'Провайдер синтеза речи',
    edge: {
      voice: 'Голос Edge'
    },
    openai: {
      model: 'Модель синтеза речи OpenAI',
      voice: 'Голос OpenAI'
    },
    elevenlabs: {
      voiceId: 'Голос ElevenLabs',
      modelId: 'Модель ElevenLabs'
    },
    xai: {
      voiceId: 'Голос xAI (Grok)',
      language: 'Язык xAI'
    },
    minimax: {
      model: 'Модель синтеза речи MiniMax',
      voiceId: 'Голос MiniMax'
    },
    mistral: {
      model: 'Модель синтеза речи Mistral',
      voiceId: 'Голос Mistral'
    },
    gemini: {
      model: 'Модель синтеза речи Gemini',
      voice: 'Голос Gemini'
    },
    neutts: {
      model: 'Модель NeuTTS',
      device: 'Устройство NeuTTS'
    },
    kittentts: {
      model: 'Модель KittenTTS',
      voice: 'Голос KittenTTS'
    },
    piper: {
      voice: 'Голос Piper'
    }
  },
  memory: {
    memoryEnabled: 'Долговременная память',
    userProfileEnabled: 'Профиль пользователя',
    memoryCharLimit: 'Объём памяти',
    userCharLimit: 'Объём профиля',
    provider: 'Провайдер памяти'
  },
  context: {
    engine: 'Механизм управления контекстом'
  },
  compression: {
    enabled: 'Автоматическое сжатие',
    threshold: 'Порог сжатия',
    targetRatio: 'Целевая степень сжатия',
    protectLastN: 'Недавние сообщения без сжатия'
  },
  delegation: {
    model: 'Модель субагента',
    provider: 'Провайдер субагента',
    maxIterations: 'Максимум ходов субагента',
    maxConcurrentChildren: 'Параллельные субагенты',
    childTimeoutSeconds: 'Время ожидания субагента',
    reasoningEffort: 'Уровень рассуждений субагента'
  },
  updates: {
    nonInteractiveLocalChanges: 'Локальные изменения при обновлении из приложения'
  }
})

export const RU_FIELD_DESCRIPTIONS: Record<string, string> = defineFieldCopy({
  model: 'Используется в новых чатах, если не выбрать другую модель в поле ввода.',
  modelContextLength:
    'Укажите 0, чтобы использовать автоматически определённый размер окна контекста выбранной модели.',
  fallbackProviders: 'Пары «провайдер:модель», используемые по очереди при сбое основной модели.',
  display: {
    personality: 'Стиль общения ассистента по умолчанию для новых сессий.',
    showReasoning: 'Показывать разделы с рассуждениями, если их передаёт бэкенд.'
  },
  timezone:
    'Используется, когда Hermes требуется местное время. Оставьте поле пустым, чтобы использовать часовой пояс системы.',
  agent: {
    imageInputMode: 'Определяет, как вложенные изображения передаются модели.',
    maxTurns: 'Максимальное количество ходов с вызовами инструментов, после которого Hermes останавливает выполнение.'
  },
  terminal: {
    cwd: 'Папка проекта по умолчанию для работы с инструментами и терминалом.',
    persistentShell: 'Сохранять состояние командной оболочки между командами, если это поддерживает среда выполнения.',
    envPassthrough: 'Переменные окружения, передаваемые при выполнении инструментов.',
    dockerImage: 'Образ контейнера для среды выполнения Docker.',
    singularityImage: 'Образ для среды выполнения Singularity.',
    modalImage: 'Образ для среды выполнения Modal.',
    daytonaImage: 'Образ для среды выполнения Daytona.'
  },
  codeExecution: {
    mode: 'Определяет, насколько строго выполнение кода ограничено текущим проектом.'
  },
  fileReadMaxChars: 'Максимальное количество символов, которое Hermes может прочитать из файла за один запрос.',
  approvals: {
    mode: 'Определяет, как Hermes обрабатывает команды, требующие явного подтверждения.',
    timeout: 'Время ожидания ответа на запрос подтверждения.'
  },
  security: {
    redactSecrets: 'По возможности скрывать обнаруженные секреты в данных, доступных модели.'
  },
  checkpoints: {
    enabled: 'Создавать точки восстановления перед изменением файлов.'
  },
  memory: {
    memoryEnabled: 'Сохранять долговременные сведения, которые могут пригодиться в будущих сессиях.',
    userProfileEnabled: 'Вести компактный профиль предпочтений пользователя.'
  },
  context: {
    engine: 'Стратегия ведения длинных диалогов при приближении к ограничению контекста.'
  },
  compression: {
    enabled: 'Сжимать старую часть контекста при увеличении объёма диалога.'
  },
  voice: {
    autoTts: 'Автоматически озвучивать ответы ассистента.'
  },
  tts: {
    xai: {
      voiceId: 'Идентификатор голоса xAI, например eve, или пользовательского голоса.',
      language: 'Код языка речи, например en.'
    },
    neutts: {
      device: 'Локальное устройство для инференса NeuTTS.'
    }
  },
  stt: {
    enabled: 'Включить локальное или облачное распознавание речи.',
    echoTranscripts: 'Отправлять в чат необработанную расшифровку голосового сообщения с отметкой 🎙️.',
    elevenlabs: {
      languageCode:
        'Необязательный код языка ISO 639-3. Оставьте поле пустым для автоматического определения в ElevenLabs.'
    }
  },
  updates: {
    nonInteractiveLocalChanges:
      'При обновлении Hermes из приложения без запроса в терминале сохранять локальные изменения исходного кода во временном хранилище либо удалять их. При обновлении из терминала Hermes всегда запрашивает действие.'
  }
})
