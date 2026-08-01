// Russian translations for field labels and descriptions.
// Used by ru.ts instead of the English defaults from constants.ts.
// Keys use camelCase matching the defineFieldCopy output from constants.ts.

import { defineFieldCopy } from './field-copy'

export const RU_FIELD_LABELS: Record<string, string> = defineFieldCopy({
  model: 'Модель по умолчанию',
  modelContextLength: 'Окно контекста',
  fallbackProviders: 'Резервные модели',
  toolsets: 'Включённые наборы инструментов',
  timezone: 'Часовой пояс',
  display: {
    personality: 'Личность',
    showReasoning: 'Блоки рассуждений'
  },
  desktop: {
    repoScanEnabled: 'Автоматическое обнаружение репозиториев',
    repoScanRoots: 'Корни обнаружения репозиториев',
    repoScanExcludePaths: 'Исключённые пути репозиториев'
  },
  agent: {
    maxTurns: 'Макс. шагов агента',
    imageInputMode: 'Вложения изображений',
    apiMaxRetries: 'Повторы API',
    serviceTier: 'Уровень сервиса',
    toolUseEnforcement: 'Принудительное использование инструментов'
  },
  terminal: {
    cwd: 'Рабочая директория',
    backend: 'Бэкенд выполнения',
    timeout: 'Тайм-аут команды',
    persistentShell: 'Постоянная оболочка',
    envPassthrough: 'Передача переменных окружения',
    dockerImage: 'Docker-образ',
    singularityImage: 'Singularity-образ',
    modalImage: 'Modal-образ',
    daytonaImage: 'Daytona-образ'
  },
  fileReadMaxChars: 'Лимит чтения файла',
  toolOutput: {
    maxBytes: 'Лимит вывода терминала',
    maxLines: 'Лимит страницы файла',
    maxLineLength: 'Лимит длины строки'
  },
  codeExecution: {
    mode: 'Режим выполнения кода'
  },
  approvals: {
    mode: 'Режим подтверждений',
    timeout: 'Тайм-аут подтверждения',
    mcpReloadConfirm: 'Подтверждать перезагрузку MCP'
  },
  commandAllowlist: 'Белый список команд',
  security: {
    redactSecrets: 'Скрывать секреты',
    allowPrivateUrls: 'Разрешить приватные URL'
  },
  browser: {
    allowPrivateUrls: 'Приватные URL в браузере',
    autoLocalForPrivateUrls: 'Локальный браузер для приватных URL'
  },
  checkpoints: {
    enabled: 'Контрольные точки файлов',
    maxSnapshots: 'Лимит контрольных точек'
  },
  voice: {
    recordKey: 'Горячая клавиша голоса',
    maxRecordingSeconds: 'Макс. длина записи',
    autoTts: 'Читать ответы вслух'
  },
  stt: {
    enabled: 'Речь в текст',
    echoTranscripts: 'Эхо транскриптов',
    provider: 'Провайдер STT',
    local: {
      model: 'Локальная модель транскрипции',
      language: 'Язык транскрипции'
    },
    openai: {
      model: 'Модель OpenAI STT'
    },
    groq: {
      model: 'Модель Groq STT'
    },
    mistral: {
      model: 'Модель Mistral STT'
    },
    elevenlabs: {
      modelId: 'Модель ElevenLabs STT',
      languageCode: 'Язык ElevenLabs',
      tagAudioEvents: 'Теги аудио-событий',
      diarize: 'Диаризация спикеров'
    }
  },
  tts: {
    provider: 'Провайдер TTS',
    edge: {
      voice: 'Голос Edge'
    },
    openai: {
      model: 'Модель OpenAI TTS',
      voice: 'Голос OpenAI'
    },
    elevenlabs: {
      voiceId: 'Голос ElevenLabs',
      modelId: 'Модель ElevenLabs'
    },
    xai: {
      voiceId: 'Голос xAI (Grok)',
      language: 'Язык xAI',
      speed: 'Скорость воспроизведения xAI',
      autoSpeechTags: 'Авто-теги речи xAI',
      optimizeStreamingLatency: 'Оптимизация задержки потока xAI',
      sampleRate: 'Частота дискретизации xAI',
      bitRate: 'Битрейт xAI'
    },
    minimax: {
      model: 'Модель MiniMax TTS',
      voiceId: 'Голос MiniMax'
    },
    mistral: {
      model: 'Модель Mistral TTS',
      voiceId: 'Голос Mistral'
    },
    gemini: {
      model: 'Модель Gemini TTS',
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
    },
    deepinfra: {
      model: 'Модель DeepInfra TTS',
      voice: 'Голос DeepInfra'
    }
  },
  memory: {
    memoryEnabled: 'Постоянная память',
    userProfileEnabled: 'Профиль пользователя',
    memoryCharLimit: 'Бюджет памяти',
    userCharLimit: 'Бюджет профиля',
    provider: 'Провайдер памяти'
  },
  context: {
    engine: 'Движок контекста'
  },
  compression: {
    enabled: 'Авто-сжатие',
    threshold: 'Порог сжатия',
    targetRatio: 'Целевое сжатие',
    protectLastN: 'Защищённые последние сообщения',
    abortOnSummaryFailure: 'Прервать при ошибке резюмирования',
    inPlace: 'Сжатие на месте'
  },
  cron: {
    provider: 'Провайдер cron',
    wrapResponse: 'Оборачивать ответ'
  },
  gateway: {
    messageTimestamps: {
      enabled: 'Отметки времени сообщений'
    },
    maxInboundMediaBytes: 'Макс. размер входящих медиа',
    strict: 'Строгий режим шлюза',
    trustRecentFiles: 'Доверять недавним файлам',
    trustRecentFilesSeconds: 'Секунды доверия недавним файлам'
  },
  streaming: {
    enabled: 'Стриминг',
    transport: 'Транспорт стриминга',
    editInterval: 'Интервал редактирования',
    bufferThreshold: 'Порог буфера'
  },
  sessions: {
    autoPrune: 'Авто-очистка сессий',
    retentionDays: 'Дни хранения сессий'
  },
  onboarding: {
    profileBuild: 'Сборка профиля при запуске'
  },
  updates: {
    preUpdateBackup: 'Резервная копия перед обновлением',
    backupKeep: 'Хранить резервные копии',
    nonInteractiveLocalChanges: 'Неинтерактивные локальные изменения'
  },
  lsp: {
    enabled: 'LSP включён',
    waitMode: 'Режим ожидания LSP',
    waitTimeout: 'Тайм-аут ожидания LSP',
    installStrategy: 'Стратегия установки LSP'
  },
  xSearch: {
    model: 'Модель поиска X',
    timeoutSeconds: 'Тайм-аут поиска X',
    retries: 'Повторы поиска X'
  },
  pasteCollapseThreshold: 'Порог свёртки вставки',
  pasteCollapseThresholdFallback: 'Запасной порог свёртки',
  pasteCollapseCharThreshold: 'Символьный порог свёртки',
  computerUse: {
    cuaTelemetry: 'Телеметрия CUA'
  },
  kanban: {
    dispatchInGateway: 'Диспетчеризация в шлюзе',
    dispatchIntervalSeconds: 'Интервал диспетчеризации',
    failureLimit: 'Лимит ошибок',
    autoDecompose: 'Авто-разбиение',
    autoDecomposePerTick: 'Авто-разбиение за тик',
    dispatchStaleTimeoutSeconds: 'Тайм-аут устаревшей диспетчеризации'
  },
  skills: {
    externalDirs: 'Внешние каталоги навыков',
    templateVars: 'Шаблонные переменные',
    inlineShell: 'Встроенный shell',
    inlineShellTimeout: 'Тайм-аут встроенного shell',
    guardAgentCreated: 'Защита создания агента',
    writeApproval: 'Одобрение записи',
    creationNudgeInterval: 'Интервал подсказок создания'
  },
  curator: {
    enabled: 'Куратор включён',
    intervalHours: 'Интервал куратора',
    minIdleHours: 'Мин. время простоя',
    staleAfterDays: 'Устаревание после дней',
    archiveAfterDays: 'Архивация после дней',
    consolidate: 'Консолидация',
    pruneBuiltins: 'Обрезка встроенных',
    backup: {
      enabled: 'Резервное копирование куратора',
      keep: 'Хранить резервные копии куратора'
    }
  },
  groupSessionsPerUser: 'Группировать сессии по пользователю',
  reasoning: 'Рассуждения',
  reasoningEffort: 'Усилие рассуждений',
  sessionReset: {
    atHour: 'Сброс сессии в час',
    idleMinutes: 'Минуты простоя для сброса',
    mode: 'Режим сброса сессии'
  },
  delegation: {
    model: 'Модель делегирования',
    provider: 'Провайдер делегирования',
    baseUrl: 'Базовый URL делегирования',
    apiKey: 'API-ключ делегирования',
    inheritMcpToolsets: 'Наследовать MCP-наборы',
    maxIterations: 'Макс. итераций делегирования',
    childTimeoutSeconds: 'Тайм-аут дочернего делегирования',
    reasoningEffort: 'Усилие рассуждений делегирования',
    maxConcurrentChildren: 'Макс. параллельных дочерних',
    maxSpawnDepth: 'Макс. глубина порождения',
    orchestratorEnabled: 'Оркестратор делегирования',
    subagentAutoApprove: 'Авто-одобрение подагентов'
  },
  privacy: {
    redactPii: 'Скрывать PII'
  },
  promptCaching: {
    cacheTtl: 'Время жизни кэша промптов'
  },
  mcpDiscoveryTimeout: 'Тайм-аут обнаружения MCP',
  toolLoopGuardrails: {
    warningsEnabled: 'Предупреждения цикла инструментов',
    hardStopEnabled: 'Жёсткая остановка цикла'
  },
  modelCatalog: {
    enabled: 'Каталог моделей',
    url: 'URL каталога моделей',
    ttlHours: 'Время жизни каталога'
  },
  network: {
    forceIpv4: 'Принудительный IPv4'
  }
})

export const RU_FIELD_DESCRIPTIONS: Record<string, string> = defineFieldCopy({
  model: 'Основная модель для ответов. Fallback-модели используются при ошибке.',
  modelContextLength: 'Максимальное количество токенов в окне контекста.',
  fallbackProviders: 'Дополнительные провайдеры на случай отказа основного.',
  toolsets: 'Наборы инструментов, доступные агенту.',
  timezone: 'Часовой пояс для отображения времени.',
  display: {
    personality: 'Файл личности (AGENTS.md / SOUL.md).',
    showReasoning: 'Показывать блоки рассуждений модели.'
  },
  desktop: {
    repoScanEnabled: 'Сканировать локальные папки для поиска Git-репозиториев.',
    repoScanRoots: 'Папки для сканирования. Оставьте пустым для сканирования домашней директории.',
    repoScanExcludePaths: 'Папки и их потомки для исключения при обнаружении репозиториев.'
  },
  agent: {
    imageInputMode: 'Как обрабатывать вложенные изображения.',
    maxTurns: 'Максимальное количество итераций агента за один запрос.'
  },
  terminal: {
    cwd: 'Рабочая директория для команд терминала.',
    backend: 'Где выполнять команды: локально, Docker, SSH.',
    timeout: 'Сколько секунд ждать завершения команды.',
    persistentShell: 'Сохранять состояние оболочки между командами.',
    envPassthrough: 'Какие переменные окружения передавать в терминал.',
    dockerImage: 'Docker-образ для выполнения команд.',
    singularityImage: 'Singularity-образ для выполнения команд.',
    modalImage: 'Modal-образ для выполнения команд.',
    daytonaImage: 'Daytona-образ для выполнения команд.'
  },
  fileReadMaxChars: 'Максимум символов при чтении файла.',
  approvals: {
    mode: 'Когда запрашивать подтверждение перед выполнением.',
    timeout: 'Сколько ждать ответа пользователя.'
  },
  codeExecution: {
    mode: 'Насколько строго выполнение кода ограничено текущим проектом.'
  },
  security: {
    redactSecrets: 'Автоматически скрывать API-ключи и пароли в выводе.'
  },
  checkpoints: {
    enabled: 'Создавать снимки отката перед редактированием файлов.'
  },
  memory: {
    memoryEnabled: 'Сохранять долговременную память для будущих сессий.',
    userProfileEnabled: 'Поддерживать компактный профиль предпочтений пользователя.'
  },
  context: {
    engine: 'Стратегия управления длинными разговорами вблизи лимита контекста.'
  },
  compression: {
    enabled: 'Резюмировать старый контекст при увеличении разговора.'
  },
  voice: {
    autoTts: 'Автоматически озвучивать ответы ассистента.'
  },
  tts: {
    xai: {
      voiceId: 'Голос xAI (Grok) TTS.',
      language: 'Язык xAI TTS.',
      speed: 'Скорость воспроизведения xAI.',
      autoSpeechTags: 'Авто-теги речи xAI.',
      optimizeStreamingLatency: 'Оптимизация задержки потока xAI.',
      sampleRate: 'Частота дискретизации xAI.',
      bitRate: 'Битрейт xAI.'
    },
    neutts: {
      device: 'Устройство для NeuTTS (CPU/GPU).'
    }
  },
  stt: {
    enabled: 'Включить распознавание речи.',
    echoTranscripts: 'Отправлять транскрипцию голосовых сообщений обратно в чат.',
    elevenlabs: {
      languageCode: 'Язык для ElevenLabs.'
    }
  },
  updates: {
    nonInteractiveLocalChanges: 'Что делать с локальными изменениями при обновлении.'
  }
})
