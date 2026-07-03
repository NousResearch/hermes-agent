import { codiconIcon } from '@/components/ui/codicon'
import { Brain, type IconComponent, Lock, MessageCircle, Mic, Monitor, Moon, Palette, Sun, Wrench } from '@/lib/icons'
import type { ThemeMode } from '@/themes/context'

import { defineFieldCopy } from './field-copy'
import type { DesktopConfigSection } from './types'

// Provider group definitions used to fold raw env-var names like
// ``XAI_API_KEY`` into a single "xAI" card with a friendly label, short
// description, and signup URL. Membership is determined by longest
// prefix match (see ``providerGroup`` in helpers.ts) so more specific
// prefixes (``MINIMAX_CN_``) correctly beat their general parents
// (``MINIMAX_``). New providers should be added here so they get their
// own card in Settings → Keys instead of being lumped into "Other".
interface ProviderPrefix {
  prefix: string
  name: string
  /** Optional one-line tagline shown beneath the group name. */
  description?: string
  /** Optional canonical signup/console URL surfaced from the card header. */
  docsUrl?: string
  /** Lower numbers float to the top of the providers list. */
  priority: number
}

export const EMPTY_SELECT_VALUE = '__hermes_empty__'
export const CONTROL_TEXT = 'text-xs'

export const PROVIDER_GROUPS: ProviderPrefix[] = [
  {
    prefix: 'NOUS_',
    name: 'Nous Portal',
    description: 'Hosted Hermes & Nous-trained models',
    docsUrl: 'https://portal.nousresearch.com',
    priority: 0
  },
  {
    prefix: 'OPENROUTER_',
    name: 'OpenRouter',
    description: 'Aggregator for hundreds of frontier models',
    docsUrl: 'https://openrouter.ai/keys',
    priority: 1
  },
  {
    prefix: 'ANTHROPIC_',
    name: 'Anthropic',
    description: 'Claude API access (Sonnet, Opus, Haiku)',
    docsUrl: 'https://console.anthropic.com/settings/keys',
    priority: 2
  },
  {
    prefix: 'XAI_',
    name: 'xAI',
    description: 'Grok models (use OAuth for SuperGrok / Premium+)',
    docsUrl: 'https://console.x.ai/',
    priority: 3
  },
  {
    prefix: 'GOOGLE_',
    name: 'Gemini',
    description: 'Google AI Studio (Gemini 1.5 / 2.0 / 2.5)',
    docsUrl: 'https://aistudio.google.com/app/apikey',
    priority: 4
  },
  { prefix: 'GEMINI_', name: 'Gemini', priority: 4 },
  {
    prefix: 'DEEPSEEK_',
    name: 'DeepSeek',
    description: 'Direct DeepSeek API (V3.x, R1)',
    docsUrl: 'https://platform.deepseek.com/api_keys',
    priority: 5
  },
  {
    prefix: 'DASHSCOPE_',
    name: 'DashScope (Qwen)',
    description: 'Alibaba Cloud DashScope — Qwen and multi-vendor models',
    docsUrl: 'https://modelstudio.console.alibabacloud.com/',
    priority: 6
  },
  { prefix: 'HERMES_QWEN_', name: 'DashScope (Qwen)', priority: 6 },
  {
    prefix: 'GLM_',
    name: 'GLM / Z.AI',
    description: 'Zhipu GLM-4.6 and Z.AI hosted endpoints',
    docsUrl: 'https://z.ai/',
    priority: 7
  },
  { prefix: 'ZAI_', name: 'GLM / Z.AI', priority: 7 },
  { prefix: 'Z_AI_', name: 'GLM / Z.AI', priority: 7 },
  {
    prefix: 'KIMI_',
    name: 'Kimi / Moonshot',
    description: 'Moonshot Kimi K2 / coding endpoints',
    docsUrl: 'https://platform.moonshot.cn/',
    priority: 8
  },
  {
    prefix: 'KIMI_CN_',
    name: 'Kimi (China)',
    description: 'Moonshot China endpoint',
    docsUrl: 'https://platform.moonshot.cn/',
    priority: 9
  },
  {
    prefix: 'MINIMAX_',
    name: 'MiniMax',
    description: 'MiniMax-M2 and Hailuo international endpoints',
    docsUrl: 'https://www.minimax.io/',
    priority: 10
  },
  {
    prefix: 'MINIMAX_CN_',
    name: 'MiniMax (China)',
    description: 'MiniMax mainland China endpoint',
    docsUrl: 'https://www.minimaxi.com/',
    priority: 11
  },
  {
    prefix: 'HF_',
    name: 'Hugging Face',
    description: 'Inference Providers — 20+ open models via router.huggingface.co',
    docsUrl: 'https://huggingface.co/settings/tokens',
    priority: 12
  },
  {
    prefix: 'OPENCODE_ZEN_',
    name: 'OpenCode Zen',
    description: 'Pay-as-you-go access to curated coding models',
    docsUrl: 'https://opencode.ai/auth',
    priority: 13
  },
  {
    prefix: 'OPENCODE_GO_',
    name: 'OpenCode Go',
    description: '$10/month subscription for open coding models',
    docsUrl: 'https://opencode.ai/auth',
    priority: 14
  },
  {
    prefix: 'NVIDIA_',
    name: 'NVIDIA NIM',
    description: 'build.nvidia.com or your own local NIM endpoint',
    docsUrl: 'https://build.nvidia.com/',
    priority: 15
  },
  {
    prefix: 'OLLAMA_',
    name: 'Ollama Cloud',
    description: 'Cloud-hosted open models from ollama.com',
    docsUrl: 'https://ollama.com/settings',
    priority: 16
  },
  {
    prefix: 'LM_',
    name: 'LM Studio',
    description: 'Local LM Studio server (OpenAI-compatible)',
    docsUrl: 'https://lmstudio.ai/docs/local-server',
    priority: 17
  },
  {
    prefix: 'STEPFUN_',
    name: 'StepFun',
    description: 'StepFun Step Plan coding models',
    docsUrl: 'https://platform.stepfun.com/',
    priority: 18
  },
  {
    prefix: 'XIAOMI_',
    name: 'Xiaomi MiMo',
    description: 'MiMo-V2.5 and Xiaomi proprietary models',
    docsUrl: 'https://platform.xiaomimimo.com',
    priority: 19
  },
  {
    prefix: 'ARCEEAI_',
    name: 'Arcee AI',
    description: 'Arcee-hosted small + medium models',
    docsUrl: 'https://chat.arcee.ai/',
    priority: 20
  },
  { prefix: 'ARCEE_', name: 'Arcee AI', priority: 20 },
  {
    prefix: 'GMI_',
    name: 'GMI Cloud',
    description: 'GMI Cloud GPU + model serving',
    docsUrl: 'https://www.gmicloud.ai/',
    priority: 21
  },
  {
    prefix: 'AZURE_FOUNDRY_',
    name: 'Azure Foundry',
    description: 'Azure AI Foundry custom endpoints (OpenAI / Anthropic-compatible)',
    docsUrl: 'https://ai.azure.com/',
    priority: 22
  },
  {
    prefix: 'AWS_',
    name: 'AWS Bedrock',
    description: 'Authenticate via AWS profile + region',
    docsUrl: 'https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html',
    priority: 23
  }
]

export const BUILTIN_PERSONALITIES = [
  'helpful',       // Полезный
  'concise',       // Лаконичный
  'technical',     // Технический
  'creative',      // Творческий
  'teacher',       // Учитель
  'kawaii',        // Кавайный
  'catgirl',       // Кошко-девочка
  'pirate',        // Пират
  'shakespeare',   // Шекспир
  'surfer',        // Сёрфер
  'noir',          // Нуар
  'uwu',           // Uwu
  'philosopher',   // Философ
  'hype'           // Хайп
]

// Schema-side select overrides for desktop-relevant enum fields whose
// backend schema only declares a string type.
export const ENUM_OPTIONS: Record<string, string[]> = {
  'agent.image_input_mode': ['auto', 'native', 'text'],
  'approvals.mode': ['manual', 'smart', 'off'],
  'code_execution.mode': ['project', 'strict'],
  'context.engine': ['compressor', 'default', 'custom'],
  'delegation.reasoning_effort': ['', 'minimal', 'low', 'medium', 'high', 'xhigh'],
  'memory.provider': ['', 'builtin', 'hindsight', 'honcho'],
  'terminal.backend': ['local', 'docker', 'singularity', 'modal', 'daytona', 'ssh'],
  'stt.elevenlabs.model_id': ['scribe_v2', 'scribe_v1'],
  'stt.local.model': ['tiny', 'base', 'small', 'medium', 'large-v3'],
  'stt.provider': ['local', 'groq', 'openai', 'mistral', 'xai', 'elevenlabs'],
  'tts.openai.voice': ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
  'tts.provider': ['edge', 'elevenlabs', 'openai', 'xai', 'minimax', 'mistral', 'gemini', 'neutts', 'kittentts', 'piper'],
  'stt.openai.model': ['whisper-1', 'gpt-4o-mini-transcribe', 'gpt-4o-transcribe'],
  'stt.mistral.model': ['voxtral-mini-latest', 'voxtral-mini-2602'],
  'tts.openai.model': ['gpt-4o-mini-tts', 'tts-1', 'tts-1-hd'],
  'tts.elevenlabs.model_id': ['eleven_multilingual_v2', 'eleven_turbo_v2_5', 'eleven_flash_v2_5'],
  'tts.neutts.device': ['cpu', 'cuda', 'mps'],
  'updates.non_interactive_local_changes': ['stash', 'discard']
}

export const FIELD_LABELS: Record<string, string> = defineFieldCopy({
  model: 'Модель по умолчанию',
  modelContextLength: 'Размер контекста',
  fallbackProviders: 'Резервные модели',
  toolsets: 'Включённые наборы инструментов',
  timezone: 'Часовой пояс',
  display: {
    personality: 'Персона',
    showReasoning: 'Блоки рассуждений'
  },
  agent: {
    maxTurns: 'Макс. шагов агента',
    imageInputMode: 'Изображения',
    apiMaxRetries: 'Повторные попытки API',
    serviceTier: 'Уровень обслуживания',
    toolUseEnforcement: 'Контроль вызова инструментов'
  },
  terminal: {
    cwd: 'Рабочая директория',
    backend: 'Среда выполнения',
    timeout: 'Таймаут команды',
    persistentShell: 'Постоянная оболочка',
    envPassthrough: 'Передача переменных окружения',
    dockerImage: 'Образ Docker',
    singularityImage: 'Образ Singularity',
    modalImage: 'Образ Modal',
    daytonaImage: 'Образ Daytona'
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
    mode: 'Режим одобрения',
    timeout: 'Таймаут одобрения',
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
    maxRecordingSeconds: 'Макс. длительность записи',
    autoTts: 'Читать ответы вслух'
  },
  stt: {
    enabled: 'Распознавание речи',
    provider: 'Провайдер распознавания',
    local: {
      model: 'Локальная модель транскрибации',
      language: 'Язык транскрибации'
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
      tagAudioEvents: 'Разметка аудио-событий',
      diarize: 'Диаризация спикеров'
    }
  },
  tts: {
    provider: 'Провайдер синтеза речи',
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
      language: 'Язык xAI'
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
    targetRatio: 'Цель сжатия',
    protectLastN: 'Защитить последние сообщения'
  },
  delegation: {
    model: 'Модель подагентов',
    provider: 'Провайдер подагентов',
    maxIterations: 'Лимит ходов подагента',
    maxConcurrentChildren: 'Параллельные подагенты',
    childTimeoutSeconds: 'Таймаут подагента',
    reasoningEffort: 'Усилие рассуждений подагента'
  },
  updates: {
    nonInteractiveLocalChanges: 'Локальные изменения при обновлении'
  }
})

export const FIELD_DESCRIPTIONS: Record<string, string> = defineFieldCopy({
  model: 'Используется для новых чатов, если не выбрана другая модель в компоновщике.',
  modelContextLength: 'Оставьте 0 для использования определённого размера контекста модели.',
  fallbackProviders: 'Резервные записи провайдер:модель при ошибке основной модели.',
  display: {
    personality: 'Стиль ассистента по умолчанию для новых сессий.',
    showReasoning: 'Показывать секции рассуждений, когда бэкенд их предоставляет.'
  },
  timezone: 'Используется, когда Hermes нуждается в локальном времени. Пусто = системный часовой пояс.',
  agent: {
    imageInputMode: 'Управляет тем, как вложенные изображения отправляются модели.',
    maxTurns: 'Верхний предел ходов с вызовом инструментов до остановки выполнения.'
  },
  terminal: {
    cwd: 'Папка проекта по умолчанию для работы с инструментами и терминалом.',
    persistentShell: 'Сохранять состояние оболочки между командами, если бэкенд поддерживает.',
    envPassthrough: 'Переменные окружения для передачи в выполнение инструментов.',
    dockerImage: 'Образ контейнера при выполнении в Docker.',
    singularityImage: 'Образ при выполнении в Singularity.',
    modalImage: 'Образ при выполнении в Modal.',
    daytonaImage: 'Образ при выполнении в Daytona.'
  },
  codeExecution: {
    mode: 'Строгость ограничения выполнения кода текущим проектом.'
  },
  fileReadMaxChars: 'Максимум символов, которые Hermes может прочитать из одного файла.',
  approvals: {
    mode: 'Как Hermes обрабатывает команды, требующие явного одобрения.',
    timeout: 'Как долго ожидается одобрение до таймаута.'
  },
  security: {
    redactSecrets: 'Скрывать обнаруженные секреты из видимого моделью содержимого, когда возможно.'
  },
  checkpoints: {
    enabled: 'Создавать снимки отката перед редактированием файлов.'
  },
  memory: {
    memoryEnabled: 'Сохранять постоянные воспоминания для помощи в будущих сессиях.',
    userProfileEnabled: 'Поддерживать компактный профиль предпочтений пользователя.'
  },
  context: {
    engine: 'Стратегия управления длинными разговорами около лимита контекста.'
  },
  compression: {
    enabled: 'Резюмировать старый контекст при больших разговорах.'
  },
  voice: {
    autoTts: 'Автоматически озвучивать ответы ассистента.'
  },
  tts: {
    xai: {
      voiceId: 'ID голоса xAI (например, eve) или пользовательский ID.',
      language: 'Код языка, например ru.'
    },
    neutts: {
      device: 'Локальное устройство для NeuTTS.'
    }
  },
  stt: {
    enabled: 'Включить локальное или через провайдера распознавание речи.',
    elevenlabs: {
      languageCode: 'ISO-639-3 код языка. Пусто = автоопределение ElevenLabs.'
    }
  },
  updates: {
    nonInteractiveLocalChanges:
      'При обновлении Hermes из приложения (без терминала) сохранять локальные изменения (stash) или отбрасывать (discard). Обновления через терминал всегда спрашивают.'
  }
})

// Curated desktop config surface: only fields a user might tune from the app.
export const SECTIONS: DesktopConfigSection[] = [
  {
    id: 'model',
    label: 'Model',
    icon: codiconIcon('symbol-namespace'),
    keys: ['model_context_length', 'fallback_providers']
  },
  {
    id: 'chat',
    label: 'Chat',
    icon: MessageCircle,
    keys: ['display.personality', 'timezone', 'display.show_reasoning', 'agent.image_input_mode']
  },
  {
    id: 'appearance',
    label: 'Appearance',
    icon: Palette,
    keys: []
  },
  {
    id: 'workspace',
    label: 'Workspace',
    icon: Monitor,
    keys: [
      'terminal.cwd',
      'code_execution.mode',
      'terminal.persistent_shell',
      'terminal.env_passthrough',
      'file_read_max_chars'
    ]
  },
  {
    id: 'safety',
    label: 'Safety',
    icon: Lock,
    keys: [
      'approvals.mode',
      'approvals.timeout',
      'approvals.mcp_reload_confirm',
      'command_allowlist',
      'security.redact_secrets',
      'security.allow_private_urls',
      'browser.allow_private_urls',
      'browser.auto_local_for_private_urls',
      'checkpoints.enabled'
    ]
  },
  {
    id: 'memory',
    label: 'Memory & Context',
    icon: Brain,
    keys: [
      'memory.memory_enabled',
      'memory.user_profile_enabled',
      'memory.memory_char_limit',
      'memory.user_char_limit',
      'memory.provider',
      'context.engine',
      'compression.enabled',
      'compression.threshold',
      'compression.target_ratio',
      'compression.protect_last_n'
    ]
  },
  {
    id: 'voice',
    label: 'Voice',
    icon: Mic,
    keys: [
      'tts.provider',
      'stt.enabled',
      'stt.provider',
      'voice.auto_tts',
      'tts.edge.voice',
      'tts.openai.model',
      'tts.openai.voice',
      'tts.elevenlabs.voice_id',
      'tts.elevenlabs.model_id',
      'tts.xai.voice_id',
      'tts.xai.language',
      'tts.minimax.model',
      'tts.minimax.voice_id',
      'tts.mistral.model',
      'tts.mistral.voice_id',
      'tts.gemini.model',
      'tts.gemini.voice',
      'tts.neutts.model',
      'tts.neutts.device',
      'tts.kittentts.model',
      'tts.kittentts.voice',
      'tts.piper.voice',
      'stt.local.model',
      'stt.local.language',
      'stt.openai.model',
      'stt.groq.model',
      'stt.mistral.model',
      'stt.elevenlabs.model_id',
      'stt.elevenlabs.language_code',
      'stt.elevenlabs.tag_audio_events',
      'stt.elevenlabs.diarize',
      'voice.record_key',
      'voice.max_recording_seconds'
    ]
  },
  {
    id: 'advanced',
    label: 'Advanced',
    icon: Wrench,
    keys: [
      'toolsets',
      'terminal.backend',
      'terminal.timeout',
      'terminal.docker_image',
      'terminal.singularity_image',
      'terminal.modal_image',
      'terminal.daytona_image',
      'tool_output.max_bytes',
      'tool_output.max_lines',
      'tool_output.max_line_length',
      'checkpoints.max_snapshots',
      'agent.max_turns',
      'agent.api_max_retries',
      'agent.service_tier',
      'agent.tool_use_enforcement',
      'delegation.model',
      'delegation.provider',
      'delegation.max_iterations',
      'delegation.max_concurrent_children',
      'delegation.child_timeout_seconds',
      'delegation.reasoning_effort',
      'updates.non_interactive_local_changes'
    ]
  }
]

export interface ModeOption {
  id: ThemeMode
  label: string
  icon: IconComponent
}

export const MODE_OPTIONS: ModeOption[] = [
  { id: 'light', label: 'Light', icon: Sun },
  { id: 'dark', label: 'Dark', icon: Moon },
  { id: 'system', label: 'System', icon: Monitor }
]
