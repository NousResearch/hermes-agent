import { FIELD_DESCRIPTIONS, FIELD_LABELS } from '@/app/settings/constants'

import { defineLocale } from './define-locale'

function translateSettingsCopy(value: string): string {
  const exact: Record<string, string> = {
    Model: 'Модель',
    Provider: 'Провайдер',
    Toolsets: 'Наборы инструментов',
    Timezone: 'Часовой пояс',
    Backend: 'Бэкенд',
    Memory: 'Память',
    Security: 'Безопасность',
    Browser: 'Браузер',
    Voice: 'Голос',
    Terminal: 'Терминал',
    Enabled: 'Включено',
    Disabled: 'Отключено',
    Automatic: 'Автоматически'
  }
  if (exact[value]) return exact[value]
  return value
    .replace(/Enable/g, 'Включить')
    .replace(/Disable/g, 'Отключить')
    .replace(/Enabled/g, 'Включено')
    .replace(/Disabled/g, 'Отключено')
    .replace(/Default/g, 'По умолчанию')
    .replace(/Provider/g, 'Провайдер')
    .replace(/Model/g, 'Модель')
    .replace(/API key/g, 'API-ключ')
    .replace(/API/g, 'API')
    .replace(/Base URL/g, 'Base URL')
    .replace(/Context/g, 'Контекст')
    .replace(/Memory/g, 'Память')
    .replace(/Profile/g, 'Профиль')
    .replace(/Terminal/g, 'Терминал')
    .replace(/Browser/g, 'Браузер')
    .replace(/Security/g, 'Безопасность')
    .replace(/Voice/g, 'Голос')
    .replace(/Timeout/g, 'Тайм-аут')
    .replace(/Working directory/g, 'Рабочая директория')
    .replace(/command/g, 'команда')
    .replace(/commands/g, 'команды')
    .replace(/tool/g, 'инструмент')
    .replace(/tools/g, 'инструменты')
    .replace(/session/g, 'сессия')
    .replace(/sessions/g, 'сессии')
    .replace(/Use /g, 'Использовать ')
    .replace(/Show /g, 'Показывать ')
    .replace(/Hide /g, 'Скрывать ')
    .replace(/Maximum/g, 'Максимум')
    .replace(/Minimum/g, 'Минимум')
    .replace(/seconds/g, 'секунд')
}

function translateSettingsRecord(record: Record<string, string>): Record<string, string> {
  return Object.fromEntries(Object.entries(record).map(([key, value]) => [key, translateSettingsCopy(value)]))
}


const EXACT_UI_TRANSLATIONS: Record<string, string> = {
  'Default assistant style for new sessions.': 'Стиль общения ассистента по умолчанию для новых сессий.',
  'Show reasoning sections when the backend provides them.': 'Показывать блоки рассуждений, когда бэкенд их предоставляет.',
  'Used when Hermes needs local time context. Blank uses the system timezone.': 'Используется, когда Hermes нужен локальный часовой пояс. Пустое значение означает системный часовой пояс.',
  'Controls how image attachments are sent to the model.': 'Управляет тем, как вложенные изображения отправляются в модель.',
  'Used for new chats unless you pick a different model in the composer.': 'Используется для новых чатов, если вы не выберете другую модель в редакторе.',
  "Leave at 0 to use the selected model's detected context window.": 'Оставьте 0, чтобы использовать определённое окно контекста выбранной модели.',
  'Backup provider:model entries to try if the default model fails.': 'Резервные записи provider:model, которые будут пробоваться при сбое модели по умолчанию.',
  'Upper bound for tool-calling turns before Hermes stops a run.': 'Верхняя граница ходов с вызовами инструментов перед остановкой запуска Hermes.',
  'Default project folder for tool and terminal work.': 'Папка проекта по умолчанию для инструментов и терминала.',
  'Keep shell state between commands when the backend supports it.': 'Сохраняет состояние оболочки между командами, если бэкенд это поддерживает.',
  'Environment variables to pass into tool execution.': 'Переменные окружения, передаваемые при выполнении инструментов.',
  'Container image used when the execution backend is Docker.': 'Образ контейнера, используемый при бэкенде выполнения Docker.',
  'Image used when the execution backend is Singularity.': 'Образ, используемый при бэкенде выполнения Singularity.',
  'Image used when the execution backend is Modal.': 'Образ, используемый при бэкенде выполнения Modal.',
  'Image used when the execution backend is Daytona.': 'Образ, используемый при бэкенде выполнения Daytona.',
  'How strictly code execution is scoped to the current project.': 'Насколько строго выполнение кода ограничено текущим проектом.',
  'Maximum characters Hermes can read from one file request.': 'Максимум символов, которые Hermes может прочитать за один запрос файла.',
  'How Hermes handles commands that need explicit approval.': 'Как Hermes обрабатывает команды, требующие явного подтверждения.',
  'How long approval prompts wait before timing out.': 'Как долго запросы подтверждения ждут до тайм-аута.',
  'Hide detected secrets from model-visible content when possible.': 'По возможности скрывает найденные секреты из содержимого, видимого модели.',
  'Create rollback snapshots before file edits.': 'Создаёт снимки для отката перед редактированием файлов.',
  'Save durable memories that can help future sessions.': 'Сохраняет долговременные воспоминания, полезные для будущих сессий.',
  'Maintain a compact profile of user preferences.': 'Поддерживает компактный профиль предпочтений пользователя.',
  'Strategy for managing long conversations near the context limit.': 'Стратегия управления длинными диалогами около лимита контекста.',
  'Summarize older context when conversations get large.': 'Сжимает старый контекст, когда диалоги становятся большими.',
  'Automatically speak assistant responses.': 'Автоматически озвучивает ответы ассистента.',
  'xAI voice ID (e.g. eve) or a custom voice ID.': 'ID голоса xAI (например, eve) или пользовательский ID голоса.',
  'Spoken language code, e.g. en.': 'Код языка речи, например ru.',
  'Local inference device for NeuTTS.': 'Локальное устройство инференса для NeuTTS.',
  'Enable local or provider-backed speech transcription.': 'Включает локальную или провайдерскую транскрибацию речи.',
  'Optional ISO-639-3 language code. Blank lets ElevenLabs auto-detect.': 'Необязательный код языка ISO-639-3. Пустое значение позволяет ElevenLabs определить язык автоматически.',
  'When Hermes updates itself from the app (no terminal prompt), keep local source edits (stash) or throw them away (discard). Terminal updates always ask.': 'Когда Hermes обновляется из приложения без запроса в терминале, сохранять локальные правки исходников (stash) или отбрасывать их (discard). Обновления из терминала всегда спрашивают.',
  'Applies to new sessions. Use the model picker in the composer to hot-swap the active chat.': 'Применяется к новым сессиям. Для быстрой смены модели в текущем чате используйте выбор модели в редакторе.',
  'Default Model': 'Модель по умолчанию',
  'Context Window': 'Окно контекста',
  'Fallback Models': 'Резервные модели',
  'Enabled Toolsets': 'Включённые наборы инструментов',
  Personality: 'Стиль общения',
  'Reasoning Blocks': 'Блоки рассуждений',
  'Max Agent Steps': 'Максимум шагов агента',
  'Image Attachments': 'Вложения изображений',
  'API Retries': 'Повторы API',
  'Service Tier': 'Уровень сервиса',
  'Tool-Use Enforcement': 'Контроль использования инструментов',
  'Working Directory': 'Рабочая директория',
  'Execution Backend': 'Бэкенд выполнения',
  'Command Timeout': 'Тайм-аут команды',
  'Persistent Shell': 'Постоянная оболочка',
  'Environment Passthrough': 'Передача переменных окружения',
  'Docker Image': 'Образ Docker',
  'Singularity Image': 'Образ Singularity',
  'Modal Image': 'Образ Modal',
  'Daytona Image': 'Образ Daytona',
  'File Read Limit': 'Лимит чтения файла',
  'Terminal Output Limit': 'Лимит вывода терминала',
  'File Page Limit': 'Лимит страницы файла',
  'Line Length Limit': 'Лимит длины строки',
  'Code Execution Mode': 'Режим выполнения кода',
  'Approval Mode': 'Режим подтверждений',
  'Approval Timeout': 'Тайм-аут подтверждения',
  'Confirm MCP Reloads': 'Подтверждать перезагрузку MCP',
  'Command Allowlist': 'Разрешённые команды',
  'Redact Secrets': 'Скрывать секреты',
  'Allow Private URLs': 'Разрешить приватные URL',
  'Browser Private URLs': 'Приватные URL в браузере',
  'Local Browser For Private URLs': 'Локальный браузер для приватных URL',
  'File Checkpoints': 'Контрольные точки файлов',
  'Checkpoint Limit': 'Лимит контрольных точек',
  'Voice Shortcut': 'Горячая клавиша голоса',
  'Max Recording Length': 'Максимальная длина записи',
  'Read Responses Aloud': 'Зачитывать ответы вслух',
  'Speech To Text': 'Распознавание речи',
  'Speech-To-Text Provider': 'Провайдер распознавания речи',
  'Local Transcription Model': 'Локальная модель транскрибации',
  'Transcription Language': 'Язык транскрибации',
  'OpenAI STT Model': 'Модель OpenAI STT',
  'Groq STT Model': 'Модель Groq STT',
  'Mistral STT Model': 'Модель Mistral STT',
  'ElevenLabs STT Model': 'Модель ElevenLabs STT',
  'ElevenLabs Language': 'Язык ElevenLabs',
  'Tag Audio Events': 'Помечать аудиособытия',
  'Speaker Diarization': 'Разделение по спикерам',
  'Text-To-Speech Provider': 'Провайдер синтеза речи',
  'Edge Voice': 'Голос Edge',
  'OpenAI TTS Model': 'Модель OpenAI TTS',
  'OpenAI Voice': 'Голос OpenAI',
  'ElevenLabs Voice': 'Голос ElevenLabs',
  'ElevenLabs Model': 'Модель ElevenLabs',
  'xAI (Grok) Voice': 'Голос xAI (Grok)',
  'xAI Language': 'Язык xAI',
  'MiniMax TTS Model': 'Модель MiniMax TTS',
  'MiniMax Voice': 'Голос MiniMax',
  'Mistral TTS Model': 'Модель Mistral TTS',
  'Mistral Voice': 'Голос Mistral',
  'Gemini TTS Model': 'Модель Gemini TTS',
  'Gemini Voice': 'Голос Gemini',
  'NeuTTS Model': 'Модель NeuTTS',
  'NeuTTS Device': 'Устройство NeuTTS',
  'KittenTTS Model': 'Модель KittenTTS',
  'KittenTTS Voice': 'Голос KittenTTS',
  'Piper Voice': 'Голос Piper',
  'Persistent Memory': 'Постоянная память',
  'User Profile': 'Профиль пользователя',
  'Memory Budget': 'Бюджет памяти',
  'Profile Budget': 'Бюджет профиля',
  'Memory Provider': 'Провайдер памяти',
  'Context Engine': 'Движок контекста',
  'Auto-Compression': 'Автосжатие',
  'Compression Threshold': 'Порог сжатия',
  'Compression Target': 'Целевой объём сжатия',
  'Protected Recent Messages': 'Защищённые последние сообщения',
  'Subagent Model': 'Модель субагента',
  'Subagent Provider': 'Провайдер субагента',
  'Subagent Turn Limit': 'Лимит ходов субагента',
  'Parallel Subagents': 'Параллельные субагенты',
  'Subagent Timeout': 'Тайм-аут субагента',
  'Subagent Reasoning Effort': 'Уровень рассуждений субагента',
  'In-App Update Local Changes': 'Локальные изменения при обновлении из приложения',
  Light: 'Светлая',
  Dark: 'Тёмная',
  System: 'Системная',
  Auto: 'Авто',
  Native: 'Нативно',
  Text: 'Текст',
  Manual: 'Вручную',
  Smart: 'Умно',
  'Archived sessions': 'Архивные сессии',
  'Loading archived sessions…': 'Загрузка архивных сессий…',
  'Archived chats are hidden from the sidebar but keep all their messages. Ctrl/⌘-click a chat in the sidebar to archive it.': 'Архивные чаты скрыты из боковой панели, но сохраняют все сообщения. Ctrl/⌘-клик по чату в боковой панели архивирует его.',
  'Nothing archived': 'Архив пуст',
  'Archive a chat to hide it here.': 'Архивируйте чат, чтобы скрыть его здесь.',
  Unarchive: 'Разархивировать',
  'Delete permanently': 'Удалить навсегда',
  'Default project directory': 'Рабочая папка проекта по умолчанию',
  'New sessions start in this folder unless you pick another. Leave it unset to use your home directory.': 'Новые сессии запускаются в этой папке, если вы не выберете другую. Оставьте пустым, чтобы использовать домашнюю папку.',
  'Default project directory updated — start a new chat (Ctrl/⌘+N) for it to take effect': 'Рабочая папка по умолчанию обновлена — создайте новый чат (Ctrl/⌘+N), чтобы применить изменение',
  'Close agents': 'Закрыть агентов',
  'Live subagent activity for the current turn.': 'Живая активность субагентов текущего хода.',
  'No live subagents': 'Нет активных субагентов',
  'When a turn delegates work, child agents stream their progress here.': 'Когда ход делегирует работу, дочерние агенты показывают прогресс здесь.',
  'Close command center': 'Закрыть командный центр',
  'Search sessions, views, and actions': 'Поиск сессий, разделов и действий',
  'Install theme...': 'Установить тему…',
  'Search the VS Code Marketplace...': 'Поиск в VS Code Marketplace…',
  'Could not reach the Marketplace.': 'Не удалось подключиться к Marketplace.',
  'No matching themes.': 'Подходящие темы не найдены.',
  'Navigate': 'Перейти',
  'Messaging gateway running': 'Шлюз мессенджеров работает',
  'Messaging gateway stopped': 'Шлюз мессенджеров остановлен',
  'Restart gateway': 'Перезапустить шлюз',
  'Gateway restart failed.': 'Не удалось перезапустить шлюз.',
  'Action started, waiting for status...': 'Действие запущено, ожидание статуса…',
  'Loading status...': 'Загрузка статуса…',
  'No usage in the last ': 'Нет использования за последние ',
  'No daily activity.': 'Нет активности по дням.',
  'No model usage yet.': 'Использования моделей пока нет.',
  'No skill activity yet.': 'Активности навыков пока нет.',
  'New session': 'Новая сессия',
  'Skills & Tools': 'Навыки и инструменты',
  Messaging: 'Мессенджеры',
  Artifacts: 'Артефакты',
  'Search sessions…': 'Поиск сессий…',
  'Search sessions...': 'Поиск сессий…',
  Pinned: 'Закреплённые',
  PINNED: 'ЗАКРЕПЛЁННЫЕ',
  'Shift-click a chat to pin': 'Shift-клик по чату закрепляет его',
  Results: 'Результаты',
  Sessions: 'Сессии',
  'Cron jobs': 'Cron-задачи',
  'No workspace': 'Без рабочей области',
  Loading: 'Загрузка',
  'Loading…': 'Загрузка…',
  'Load more': 'Загрузить ещё',
  Pin: 'Закрепить',
  Unpin: 'Открепить',
  Export: 'Экспорт',
  Rename: 'Переименовать',
  Archive: 'Архивировать',
  'New window': 'Новое окно',
  'Session actions': 'Действия сессии',
  'Session running': 'Сессия выполняется',
  'Needs your input': 'Нужен ваш ввод',
  'Waiting for your answer': 'Ожидает вашего ответа',
  Renamed: 'Переименовано',
  'Rename failed': 'Не удалось переименовать',
  'Rename session': 'Переименовать сессию',
  'Untitled session': 'Сессия без названия',
  now: 'сейчас',
  Model: 'Модель',
  Provider: 'Провайдер',
  Providers: 'Провайдеры',
  Gateway: 'Шлюз',
  Appearance: 'Внешний вид',
  Settings: 'Настройки',
  Language: 'Язык',
  Notifications: 'Уведомления',
  Safety: 'Безопасность',
  Advanced: 'Расширенные',
  Workspace: 'Рабочая область',
  Voice: 'Голос',
  'Memory & Context': 'Память и контекст',
  'Archived Chats': 'Архив чатов',
  About: 'О приложении',
  'Tools & Keys': 'Инструменты и ключи',
  Accounts: 'Аккаунты',
  'API keys': 'API-ключи',
  Tools: 'Инструменты',
  'Close settings': 'Закрыть настройки',
  'Export config': 'Экспорт конфигурации',
  'Import config': 'Импорт конфигурации',
  'Reset to defaults': 'Сбросить по умолчанию',
  'Nothing to configure': 'Нечего настраивать',
  'Settings failed to load': 'Не удалось загрузить настройки',
  'Autosave failed': 'Автосохранение не удалось',
  'Config imported': 'Конфигурация импортирована',
  'Invalid config JSON': 'Некорректный JSON конфигурации',
  'None': 'Нет',
  '(none)': '(нет)',
  'Not set': 'Не задано',
  'Apply': 'Применить',
  'Back': 'Назад',
  'Save': 'Сохранить',
  'Cancel': 'Отмена',
  'Change': 'Изменить',
  'Choose': 'Выбрать',
  'Clear': 'Очистить',
  'Close': 'Закрыть',
  'Confirm': 'Подтвердить',
  'Connect': 'Подключить',
  'Connected': 'Подключено',
  'Disconnect': 'Отключить',
  'Collapse': 'Свернуть',
  'Continue': 'Продолжить',
  'Copy': 'Копировать',
  'Delete': 'Удалить',
  'Done': 'Готово',
  'Error': 'Ошибка',
  'Failed': 'Сбой',
  'Free': 'Бесплатно',
  'Refresh': 'Обновить',
  'Remove': 'Удалить',
  'Replace': 'Заменить',
  'Retry': 'Повторить',
  'Run': 'Запустить',
  'Send': 'Отправить',
  'Set': 'Задать',
  'Skip': 'Пропустить',
  'Update': 'Обновить',
  'On': 'Вкл.',
  'Off': 'Выкл.',
  Skills: 'Навыки',
  Toolsets: 'Наборы инструментов',
  All: 'Все',
  'No skills found': 'Навыки не найдены',
  'No toolsets found': 'Наборы инструментов не найдены',
  'No description.': 'Описание отсутствует.',
  Configured: 'Настроено',
  'Needs keys': 'Нужны ключи',
  Agents: 'Агенты',
  Running: 'Выполняется',
  Streaming: 'Потоковая передача',
  Files: 'Файлы',
  'Command palette': 'Командная палитра',
  'Command Center': 'Командный центр',
  'Go to': 'Перейти к',
  'Go to session': 'Перейти к сессии',
  'Change theme...': 'Сменить тему…',
  'Change color mode...': 'Сменить цветовой режим…',
  'No matching results found.': 'Совпадений не найдено.',
  'Pin session': 'Закрепить сессию',
  'Unpin session': 'Открепить сессию',
  'Export session': 'Экспорт сессии',
  'Delete session': 'Удалить сессию',
  'No sessions yet.': 'Сессий пока нет.',
  'Recent logs': 'Последние логи',
  'No logs loaded yet.': 'Логи ещё не загружены.',
  'Daily tokens': 'Токены по дням',
  input: 'ввод',
  output: 'вывод',
  'Top models': 'Популярные модели',
  'Top skills': 'Популярные навыки',
  'Add provider': 'Добавить провайдера',
  'No authenticated providers.': 'Нет авторизованных провайдеров.',
  'No models found.': 'Модели не найдены.',
  current: 'текущая:',
  unknown: 'неизвестно',
  'Switch model': 'Сменить модель',
  'Search models': 'Поиск моделей',
  'Edit Models…': 'Настроить модели…',
  'Refresh Models': 'Обновить модели',
  Fast: 'Быстро',
  Medium: 'Средне',
  Minimal: 'Минимально',
  High: 'Высоко',
  Max: 'Максимум',
  Effort: 'Усилие',
  Options: 'Параметры',
  Thinking: 'Думает',
  'Window controls': 'Управление окном',
  'Pane controls': 'Управление панелями',
  'App controls': 'Управление приложением',
  'Right sidebar': 'Правая боковая панель',
  'File system': 'Файловая система',
  Terminal: 'Терминал',
  'No folder selected': 'Папка не выбрана',
  'Open folder': 'Открыть папку',
  'Refresh tree': 'Обновить дерево',
  Empty: 'Пусто',
  Unreadable: 'Нечитаемо',
  Preview: 'Предпросмотр',
  'Preview unavailable': 'Предпросмотр недоступен',
  'Open preview': 'Открыть предпросмотр',
  'Copy URL': 'Копировать URL',
  'Copy path': 'Копировать путь',
  Chat: 'Чат',
  Message: 'Сообщение',
  Stop: 'Остановить',
  Speaking: 'Говорит',
  Transcribing: 'Распознавание',
  Muted: 'Отключено',
  Listening: 'Слушает',
  Attach: 'Прикрепить',
  Queued: 'В очереди',
  Edit: 'Редактировать',
  Next: 'Далее',
  Images: 'Изображения',
  Folder: 'Папка',
  'Paste image': 'Вставить изображение',
  'Prompt snippets': 'Фрагменты промптов',
  'Drop files to attach': 'Перетащите файлы для прикрепления',
  'Approval needed': 'Требуется подтверждение',
  Command: 'Команда',
  Reject: 'Отклонить',
  'Always allow': 'Всегда разрешать',
  'Input needed': 'Требуется ввод',
  'Type your answer…': 'Введите ответ…',
  Other: 'Другое',
  'Something went wrong': 'Что-то пошло не так',
  'Reload window': 'Перезагрузить окно',
  pagination: 'пагинация',
  Prev: 'Назад'
}

function hasCyrillic(value: string): boolean {
  return /[А-Яа-яЁё]/.test(value)
}

function shouldPreserveAscii(value: string): boolean {
  const trimmed = value.trim()
  if (!trimmed) return true
  if (/^https?:\/\//.test(trimmed)) return true
  if (/^[A-Z0-9_]{3,}$/.test(trimmed)) return true
  if (/^[\w.-]+\/[\w.-]+$/.test(trimmed)) return true
  if (/^[\w.-]+\.[\w.-]+$/.test(trimmed)) return true
  return false
}

function translateLooseEnglish(value: string): string {
  if (hasCyrillic(value) || shouldPreserveAscii(value)) return value
  const exact = EXACT_UI_TRANSLATIONS[value]
  if (exact) return exact

  let out = value
  const replacements: Array<[RegExp, string]> = [
    [/\bnew sessions\b/gi, 'новые сессии'],
    [/\bsessions\b/gi, 'сессии'],
    [/\bsession\b/gi, 'сессия'],
    [/\bagents\b/gi, 'агенты'],
    [/\bagent\b/gi, 'агент'],
    [/\bmodels\b/gi, 'модели'],
    [/\bmodel\b/gi, 'модель'],
    [/\bproviders\b/gi, 'провайдеры'],
    [/\bprovider\b/gi, 'провайдер'],
    [/\bconfiguration\b/gi, 'конфигурация'],
    [/\bconfig\b/gi, 'конфигурация'],
    [/\bcomposer\b/gi, 'редактор'],
    [/\bcurrent\b/gi, 'текущий'],
    [/\bactive\b/gi, 'активный'],
    [/\barchived\b/gi, 'архивные'],
    [/\barchive\b/gi, 'архив'],
    [/\bchat\b/gi, 'чат'],
    [/\bchats\b/gi, 'чаты'],
    [/\bmessages\b/gi, 'сообщения'],
    [/\bmessage\b/gi, 'сообщение'],
    [/\bwindow\b/gi, 'окно'],
    [/\bwindows\b/gi, 'окна'],
    [/\bsidebar\b/gi, 'боковая панель'],
    [/Search/g, 'Поиск'],
    [/Loading/g, 'Загрузка'],
    [/Failed to load/g, 'Не удалось загрузить'],
    [/Could not load/g, 'Не удалось загрузить'],
    [/Could not save/g, 'Не удалось сохранить'],
    [/Could not copy/g, 'Не удалось скопировать'],
    [/Could not update/g, 'Не удалось обновить'],
    [/Failed/g, 'Сбой'],
    [/Error/g, 'Ошибка'],
    [/Settings/g, 'Настройки'],
    [/Provider/g, 'Провайдер'],
    [/Providers/g, 'Провайдеры'],
    [/Model/g, 'Модель'],
    [/Models/g, 'Модели'],
    [/Session/g, 'Сессия'],
    [/Sessions/g, 'Сессии'],
    [/Profile/g, 'Профиль'],
    [/Profiles/g, 'Профили'],
    [/Message/g, 'Сообщение'],
    [/Messages/g, 'Сообщения'],
    [/Toolsets/g, 'Наборы инструментов'],
    [/Tools/g, 'Инструменты'],
    [/Skills/g, 'Навыки'],
    [/Agents/g, 'Агенты'],
    [/Artifacts/g, 'Артефакты'],
    [/Notifications/g, 'Уведомления'],
    [/Gateway/g, 'Шлюз'],
    [/Terminal/g, 'Терминал'],
    [/Preview/g, 'Предпросмотр'],
    [/File/g, 'Файл'],
    [/Files/g, 'Файлы'],
    [/Folder/g, 'Папка'],
    [/Voice/g, 'Голос'],
    [/Memory/g, 'Память'],
    [/Context/g, 'Контекст'],
    [/Security/g, 'Безопасность'],
    [/Browser/g, 'Браузер'],
    [/Theme/g, 'Тема'],
    [/Color/g, 'Цвет'],
    [/Mode/g, 'Режим'],
    [/Default/g, 'По умолчанию'],
    [/Advanced/g, 'Расширенные'],
    [/Appearance/g, 'Внешний вид'],
    [/Workspace/g, 'Рабочая область'],
    [/Update/g, 'Обновить'],
    [/Install/g, 'Установить'],
    [/Create/g, 'Создать'],
    [/Remove/g, 'Удалить'],
    [/Delete/g, 'Удалить'],
    [/Save/g, 'Сохранить'],
    [/Copy/g, 'Копировать'],
    [/Open/g, 'Открыть'],
    [/Close/g, 'Закрыть'],
    [/Show/g, 'Показать'],
    [/Hide/g, 'Скрыть'],
    [/Enable/g, 'Включить'],
    [/Disable/g, 'Отключить'],
    [/Enabled/g, 'Включено'],
    [/Disabled/g, 'Отключено'],
    [/Running/g, 'Выполняется'],
    [/Ready/g, 'Готово'],
    [/Done/g, 'Готово'],
    [/Retry/g, 'Повторить'],
    [/Refresh/g, 'Обновить'],
    [/Connect/g, 'Подключить'],
    [/Connected/g, 'Подключено'],
    [/Disconnect/g, 'Отключить'],
    [/Unknown/g, 'Неизвестно'],
    [/Current/g, 'Текущий'],
    [/Recent/g, 'Недавние'],
    [/Archived/g, 'Архивные'],
    [/No /g, 'Нет '],
    [/New /g, 'Новая '],
    [/Add /g, 'Добавить '],
    [/Change /g, 'Изменить '],
    [/Select /g, 'Выбрать '],
    [/Choose /g, 'Выбрать '],
    [/Run /g, 'Запустить '],
    [/Send /g, 'Отправить '],
    [/Go to /g, 'Перейти к '],
    [/Sign in/g, 'Войти'],
    [/Sign out/g, 'Выйти'],
    [/not found/gi, 'не найдены'],
    [/unavailable/gi, 'недоступно'],
    [/required/gi, 'требуется'],
    [/optional/gi, 'необязательно'],
    [/empty/gi, 'пусто'],
    [/local/gi, 'локальный'],
    [/remote/gi, 'удалённый']
  ]
  for (const [pattern, replacement] of replacements) out = out.replace(pattern, replacement)
  return out
}

function localizeEnglishFallbacks<T>(value: T): T {
  if (typeof value === 'string') return translateLooseEnglish(value) as T
  if (typeof value === 'function') {
    return ((...args: any[]) => translateLooseEnglish(String((value as any)(...args)))) as T
  }
  if (Array.isArray(value)) return value.map(item => localizeEnglishFallbacks(item)) as T
  if (value && typeof value === 'object') {
    const result: Record<string, unknown> = {}
    for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
      result[key] = localizeEnglishFallbacks(child)
    }
    return result as T
  }
  return value
}

export const ru = localizeEnglishFallbacks(defineLocale({
  common: {
    apply: 'Применить',
    back: 'Назад',
    save: 'Сохранить',
    saving: 'Сохранение…',
    cancel: 'Отмена',
    change: 'Изменить',
    choose: 'Выбрать',
    clear: 'Очистить',
    close: 'Закрыть',
    collapse: 'Свернуть',
    confirm: 'Подтвердить',
    connect: 'Подключить',
    connecting: 'Подключение',
    continue: 'Продолжить',
    copied: 'Скопировано',
    copy: 'Копировать',
    copyFailed: 'Не удалось скопировать',
    delete: 'Удалить',
    docs: 'Документация',
    done: 'Готово',
    error: 'Ошибка',
    failed: 'Сбой',
    free: 'Бесплатно',
    loading: 'Загрузка…',
    notSet: 'Не задано',
    refresh: 'Обновить',
    remove: 'Удалить',
    replace: 'Заменить',
    retry: 'Повторить',
    run: 'Запустить',
    send: 'Отправить',
    set: 'Задать',
    skip: 'Пропустить',
    update: 'Обновить',
    on: 'Вкл.',
    off: 'Выкл.'
  },

  boot: {
    ready: 'Hermes Desktop готов',
    desktopBootFailedWithMessage: (message: any) => `Не удалось запустить Desktop: ${message}`,
    steps: {
      connectingGateway: 'Подключение к live-шлюзу Desktop',
      loadingSettings: 'Загрузка настроек Hermes',
      loadingSessions: 'Загрузка недавних сессий',
      startingDesktopConnection: 'Запуск подключения Desktop',
      startingHermesDesktop: 'Запуск Hermes Desktop…'
    },
    errors: {
      backgroundExited: 'Фоновый процесс Hermes завершился.',
      backgroundExitedDuringStartup: 'Фоновый процесс Hermes завершился во время запуска.',
      backendStopped: 'Бэкенд остановлен',
      desktopBootFailed: 'Не удалось запустить Desktop',
      gatewaySignInRequired: 'Требуется вход в шлюз',
      ipcBridgeUnavailable: 'Desktop IPC bridge недоступен.'
    },
    failure: {
      title: 'Hermes не удалось запустить',
      description:
        'Фоновый шлюз не запустился. Попробуйте один из вариантов восстановления ниже. Чаты и настройки не будут удалены.',
      remoteTitle: 'Требуется вход в удалённый шлюз',
      remoteDescription:
        'Сессия удалённого шлюза истекла. Войдите снова, чтобы переподключиться. Чаты и настройки не будут удалены.',
      retry: 'Повторить',
      repairInstall: 'Восстановить установку',
      useLocalGateway: 'Использовать локальный шлюз',
      openLogs: 'Открыть логи',
      repairHint: 'Восстановление повторно запускает установщик и на новой машине может занять несколько минут.',
      remoteSignInHint: 'Откроет окно входа в шлюз. Чтобы перейти на встроенный бэкенд, выберите локальный шлюз.',
      hideRecentLogs: 'Скрыть последние логи',
      showRecentLogs: 'Показать последние логи',
      signedInTitle: 'Вход выполнен',
      signedInMessage: 'Повторное подключение к удалённому шлюзу…',
      signInIncompleteTitle: 'Вход не завершён',
      signInIncompleteMessage: 'Окно входа закрылось до завершения аутентификации.',
      signInFailed: 'Не удалось войти',
      signInToRemoteGateway: 'Войти в удалённый шлюз',
      signInWithProvider: (provider: any) => `Войти через ${provider}`,
      identityProvider: 'провайдер идентификации'
    }
  },

  notifications: {
    region: 'Уведомления',
    hide: 'Скрыть',
    show: 'Показать',
    more: (count: any) => `Ещё уведомлений: ${count}`,
    clearAll: 'Очистить всё',
    dismiss: 'Закрыть уведомление',
    details: 'Подробности',
    copyDetail: 'Копировать подробности',
    copyDetailFailed: 'Не удалось скопировать подробности уведомления',
    backendOutOfDateTitle: 'Бэкенд устарел',
    backendOutOfDateMessage:
      'Бэкенд Hermes старее этой сборки Desktop и может работать некорректно. Обновите его, чтобы версии совпадали.',
    updateHermes: 'Обновить Hermes',
    updateReadyTitle: 'Обновление готово',
    updateReadyMessage: (count: any) => `Доступно новых изменений: ${count}.`,
    seeWhatsNew: 'Что нового',
    errors: {
      elevenLabsNeedsKey: 'Для ElevenLabs STT нужен ELEVENLABS_API_KEY.',
      elevenLabsRejectedKey: 'ElevenLabs отклонил API-ключ (401).',
      methodNotAllowed:
        'Desktop-бэкенд отклонил запрос (405 Method Not Allowed). Попробуйте перезапустить Hermes Desktop.',
      microphonePermission: 'Доступ к микрофону запрещён.',
      openaiRejectedApiKey: 'OpenAI отклонил API-ключ.',
      openaiRejectedApiKeyWithStatus: (status: any) => `OpenAI отклонил API-ключ (${status} invalid_api_key).`,
      openaiTtsNeedsKey: 'Для OpenAI TTS нужен VOICE_TOOLS_OPENAI_KEY или OPENAI_API_KEY.'
    },
    voice: {
      configureSpeechToText: 'Настройте распознавание речи, чтобы использовать голосовой режим.',
      couldNotStartSession: 'Не удалось начать голосовую сессию',
      microphoneAccessDenied: 'Доступ к микрофону запрещён.',
      microphoneConstraintsUnsupported: 'Ограничения микрофона не поддерживаются этим устройством.',
      microphoneFailed: 'Ошибка микрофона',
      microphoneInUse: 'Микрофон уже используется другим приложением.',
      microphonePermissionDenied: 'Разрешение на микрофон отклонено.',
      microphoneStartFailed: 'Не удалось начать запись с микрофона.',
      microphoneUnsupported: 'Эта среда не поддерживает запись с микрофона.',
      noMicrophone: 'Микрофон не найден.',
      noSpeechDetected: 'Речь не обнаружена',
      playbackFailed: 'Не удалось воспроизвести голос',
      recordingFailed: 'Не удалось записать голос',
      transcriptionFailed: 'Не удалось распознать голос',
      transcriptionUnavailable: 'Распознавание голоса пока недоступно.',
      tryRecordingAgain: 'Попробуйте записать ещё раз.',
      unavailable: 'Голос недоступен'
    },
    native: {
      approvalTitle: 'Требуется подтверждение',
      approveAction: 'Одобрить',
      rejectAction: 'Отклонить',
      inputTitle: 'Требуется ввод',
      inputBody: 'Hermes ждёт вашего ответа.',
      turnDoneTitle: 'Hermes завершил работу',
      turnDoneBody: 'Ответ готов.',
      turnErrorTitle: 'Ошибка выполнения',
      backgroundDoneTitle: 'Фоновая задача завершена',
      backgroundFailedTitle: 'Фоновая задача завершилась с ошибкой'
    }
  },

  remoteDisplayBanner: {
    message: (reason: any) => `Активен программный рендеринг — обнаружен удалённый дисплей (${reason}). Аппаратное ускорение отключено во избежание мерцания.`,
    dismiss: 'Закрыть'
  },

  titlebar: {
    hideSidebar: 'Скрыть боковую панель',
    showSidebar: 'Показать боковую панель',
    search: 'Поиск',
    searchTitle: 'Поиск сессий, разделов и действий',
    swapSidebarSides: 'Поменять боковые панели местами',
    swapSidebarSidesTitle: 'Поменять местами список сессий и файловый браузер',
    hideRightSidebar: 'Скрыть правую панель',
    showRightSidebar: 'Показать правую панель',
    muteHaptics: 'Выключить тактильный отклик',
    unmuteHaptics: 'Включить тактильный отклик',
    openSettings: 'Открыть настройки',
    openKeybinds: 'Горячие клавиши'
  },

  keybinds: {
    title: 'Горячие клавиши',
    subtitle: (open: any) => `Нажмите на сочетание, чтобы переназначить · ${open} снова откроет эту панель.`,
    rebind: 'Переназначить',
    reset: 'Сбросить по умолчанию',
    resetAll: 'Сбросить всё',
    pressKey: 'Нажмите клавишу…',
    set: 'задано',
    conflictWith: (label: any) => `Также назначено на «${label}»`,
    categories: {
      composer: 'Редактор',
      profiles: 'Профили',
      session: 'Сессия',
      navigation: 'Навигация',
      view: 'Вид'
    },
    actions: {
      'keybinds.openPanel': 'Открыть горячие клавиши',
      'nav.commandPalette': 'Открыть командную палитру',
      'nav.commandCenter': 'Открыть командный центр',
      'nav.settings': 'Открыть настройки',
      'nav.profiles': 'Открыть профили',
      'nav.skills': 'Открыть навыки',
      'nav.messaging': 'Открыть мессенджеры',
      'nav.artifacts': 'Открыть артефакты',
      'nav.cron': 'Открыть задачи по расписанию',
      'nav.agents': 'Открыть агентов',
      'session.new': 'Новая сессия',
      'session.newWindow': 'Новая сессия в окне',
      'session.next': 'Следующая сессия',
      'session.prev': 'Предыдущая сессия',
      'session.slot.1': 'Переключиться на недавнюю сессию 1',
      'session.slot.2': 'Переключиться на недавнюю сессию 2',
      'session.slot.3': 'Переключиться на недавнюю сессию 3',
      'session.slot.4': 'Переключиться на недавнюю сессию 4',
      'session.slot.5': 'Переключиться на недавнюю сессию 5',
      'session.slot.6': 'Переключиться на недавнюю сессию 6',
      'session.slot.7': 'Переключиться на недавнюю сессию 7',
      'session.slot.8': 'Переключиться на недавнюю сессию 8',
      'session.slot.9': 'Переключиться на недавнюю сессию 9',
      'session.focusSearch': 'Поиск сессий',
      'session.togglePin': 'Закрепить / открепить текущую сессию',
      'composer.focus': 'Фокус на редактор',
      'composer.modelPicker': 'Открыть выбор модели',
      'view.toggleSidebar': 'Переключить боковую панель сессий',
      'view.toggleRightSidebar': 'Переключить файловый браузер',
      'view.showFiles': 'Показать файловый браузер',
      'view.showTerminal': 'Показать терминал',
      'view.terminalSelection': 'Отправить выделенное в терминале в редактор',
      'view.closePreviewTab': 'Закрыть вкладку предпросмотра',
      'view.flipPanes': 'Поменять боковые панели местами',
      'appearance.toggleMode': 'Переключить светлую / тёмную тему',
      'profile.default': 'Переключиться на профиль по умолчанию',
      'profile.switch.1': 'Переключиться на профиль 1',
      'profile.switch.2': 'Переключиться на профиль 2',
      'profile.switch.3': 'Переключиться на профиль 3',
      'profile.switch.4': 'Переключиться на профиль 4',
      'profile.switch.5': 'Переключиться на профиль 5',
      'profile.switch.6': 'Переключиться на профиль 6',
      'profile.switch.7': 'Переключиться на профиль 7',
      'profile.switch.8': 'Переключиться на профиль 8',
      'profile.switch.9': 'Переключиться на профиль 9',
      'profile.switch.10': 'Переключиться на профиль 10',
      'profile.switch.11': 'Переключиться на профиль 11',
      'profile.switch.12': 'Переключиться на профиль 12',
      'profile.switch.13': 'Переключиться на профиль 13',
      'profile.switch.14': 'Переключиться на профиль 14',
      'profile.switch.15': 'Переключиться на профиль 15',
      'profile.switch.16': 'Переключиться на профиль 16',
      'profile.switch.17': 'Переключиться на профиль 17',
      'profile.switch.18': 'Переключиться на профиль 18',
      'profile.next': 'Следующий профиль',
      'profile.prev': 'Предыдущий профиль',
      'profile.toggleAll': 'Переключить вид всех профилей',
      'profile.create': 'Создать профиль',
      'composer.send': 'Отправить сообщение',
      'composer.newline': 'Вставить новую строку',
      'composer.steer': 'Направить текущий ход',
      'composer.sendQueued': 'Отправить следующую очередь',
      'composer.mention': 'Сослаться на файлы, папки, URL',
      'composer.slash': 'Палитра slash-команд',
      'composer.help': 'Краткая справка',
      'composer.history': 'Цикл поп-ап / истории',
      'composer.cancel': 'Закрыть поп-ап / отменить запуск'
    }
  },

  language: {
    label: 'Язык',
    description: 'Выберите язык интерфейса Desktop.',
    saving: 'Сохранение языка…',
    saveError: 'Не удалось изменить язык',
    switchTo: 'Сменить язык',
    searchPlaceholder: 'Поиск языков…',
    noResults: 'Языки не найдены'
  },

  settings: {
    closeSettings: 'Закрыть настройки',
    exportConfig: 'Экспорт конфигурации',
    importConfig: 'Импорт конфигурации',
    resetToDefaults: 'Сбросить по умолчанию',
    resetConfirm: 'Сбросить все настройки до значений Hermes по умолчанию?',
    exportFailed: 'Экспорт не удался',
    resetFailed: 'Сброс не удался',
    nav: {
      providers: 'Провайдеры',
      providerAccounts: 'Аккаунты',
      providerApiKeys: 'API-ключи',
      gateway: 'Шлюз',
      apiKeys: 'Инструменты и ключи',
      keysTools: 'Инструменты',
      keysSettings: 'Настройки',
      mcp: 'MCP',
      archivedChats: 'Архив чатов',
      about: 'О приложении',
      notifications: 'Уведомления'
    },
    notifications: {
      title: 'Уведомления',
      intro:
        'Нативные уведомления рабочего стола, отдельные от всплывающих сообщений в приложении. Они локальны для устройства — каждый компьютер хранит свои настройки.',
      enableAll: 'Включить уведомления',
      enableAllDesc: 'Главный переключатель. Отключите, чтобы заглушить все уведомления ниже.',
      focusedHint: 'Оповещения о завершении срабатывают только тогда, когда Hermes работает в фоновом режиме.',
      kinds: {
        approval: {
          label: 'Требуется подтверждение',
          description: 'Команда ожидает вашего одобрения или отклонения.'
        },
        input: {
          label: 'Требуется ввод',
          description: 'Hermes задал вопрос или ему нужен пароль или секрет.'
        },
        turnDone: {
          label: 'Ответ готов',
          description: 'Ход завершился, пока Hermes был в фоновом режиме.'
        },
        turnError: {
          label: 'Ошибка хода',
          description: 'Ход завершился с ошибкой.'
        },
        backgroundDone: {
          label: 'Фоновая задача завершена',
          description: 'Фоновая команда терминала завершена.'
        }
      },
      test: 'Отправить тестовое уведомление',
      testTitle: 'Hermes',
      testBody: 'Уведомления работают.',
      testSent: 'Тест отправлен. Если ничего не появилось, проверьте разрешения на уведомления в ОС и режим "Не беспокоить".',
      testUnsupported: 'Эта система не поддерживает нативные уведомления.',
      completionSoundTitle: 'Звук завершения',
      completionSoundDesc: 'Воспроизводится по завершении хода агента. Выберите предустановку и прослушайте ее здесь.',
      completionSoundPreview: 'Предпросмотр'
    },
    sections: {
      model: 'Модель',
      chat: 'Чат',
      appearance: 'Внешний вид',
      workspace: 'Рабочее пространство',
      safety: 'Безопасность',
      memory: 'Память и контекст',
      voice: 'Голос',
      advanced: 'Расширенные'
    },
    searchPlaceholder: {
      about: 'О Hermes Desktop',
      config: 'Поиск настроек...',
      gateway: 'Подключение к шлюзу...',
      keys: 'Поиск API-ключей...',
      mcp: 'Поиск MCP-серверов...',
      sessions: 'Поиск в архиве сессий...'
    },
    modeOptions: {
      light: { label: 'Светлая', description: 'Светлые поверхности рабочего стола' },
      dark: { label: 'Тёмная', description: 'Рабочее пространство с низким уровнем бликов' },
      system: { label: 'Системная', description: 'Следовать настройкам ОС' }
    },
    appearance: {
      title: 'Внешний вид',
      intro:
        'Это настройки отображения только для десктопной версии. Режим управляет яркостью; тема — палитрой акцентов и стилем чата.',
      colorMode: 'Цветовой режим',
      colorModeDesc: 'Выберите фиксированный режим или позвольте Hermes следовать системным настройкам.',
      toolViewTitle: 'Отображение вызовов инструментов',
      toolViewDesc: 'Продуктовый режим скрывает необработанные данные инструментов; Технический показывает полный ввод/вывод.',
      translucencyTitle: 'Полупрозрачность окна',
      translucencyDesc: 'Просмотр рабочего стола через все окно. Только для macOS и Windows.',
      product: 'Продуктовый',
      productDesc: 'Понятное отображение активности инструментов с краткими сводками.',
      technical: 'Технический',
      technicalDesc: 'Включает необработанные аргументы/результаты инструментов и низкоуровневые детали.',
      themeTitle: 'Тема',
      themeDesc: 'Только палитры для десктопа. Выбранный режим применяется поверх.',
      themeProfileNote: (profile: any) => `Сохранено для профиля ${profile} — каждый профиль имеет свою тему.`,
      installTitle: 'Установить из VS Code',
      installDesc:
        'Вставьте ID расширения из Marketplace (например, dracula-theme.theme-dracula), чтобы преобразовать его цветовую тему в палитру для десктопа.',
      installPlaceholder: 'publisher.extension',
      installButton: 'Установить',
      installing: 'Установка…',
      installError: 'Не удалось установить эту тему.',
      installed: (name: any) => `Установлена «${name}».`,
      removeTheme: 'Удалить тему',
      importedBadge: 'Импортировано'
    },
    fieldLabels: translateSettingsRecord(FIELD_LABELS),
    fieldDescriptions: translateSettingsRecord(FIELD_DESCRIPTIONS),
    about: {
      heading: 'Hermes Desktop',
      version: (value: any) => `Версия ${value}`,
      versionUnavailable: 'Версия недоступна',
      updates: 'Обновления',
      checkNow: 'Проверить сейчас',
      checking: 'Проверка…',
      seeWhatsNew: 'Что нового',
      releaseNotes: 'Примечания к выпуску',
      onLatest: 'У вас последняя версия.',
      installing: 'Обновление устанавливается.',
      cantUpdate: 'Эта сборка не может обновляться из приложения.',
      cantReach: 'Не удалось связаться с сервером обновлений.',
      tapCheck: 'Нажмите "Проверить сейчас" для поиска обновлений.',
      updateReady: (count: any) => `Готово новое обновление (включает ${count} изменен${count === 1 ? 'ие' : 'ий'}).`,
      lastChecked: (age: any) => `Последняя проверка ${age}`,
      justNowSuffix: ' · только что',
      automaticUpdates: 'Автоматические обновления',
      automaticUpdatesDesc:
        'Hermes автоматически проверяет наличие обновлений в фоновом режиме и сообщает, когда они готовы.',
      branchCommit: (branch: any, commit: any) => `Ветка ${branch} · Коммит ${commit}`,
      never: 'никогда',
      justNow: 'только что',
      minAgo: (count: any) => `${count} мин назад`,
      hoursAgo: (count: any) => `${count} ч назад`,
      daysAgo: (count: any) => `${count} дн назад`
    },
    config: {
      none: 'Нет',
      noneParen: '(нет)',
      notSet: 'Не установлено',
      commaSeparated: 'значения через запятую',
      loading: 'Загрузка конфигурации Hermes...',
      emptyTitle: 'Нечего настраивать',
      emptyDesc: 'В этом разделе нет настраиваемых параметров.',
      failedLoad: 'Не удалось загрузить настройки',
      autosaveFailed: 'Автосохранение не удалось',
      imported: 'Конфигурация импортирована',
      invalidJson: 'Неверный JSON конфигурации'
    },
    credentials: {
      pasteKey: 'Вставить ключ',
      pasteLabelKey: (label: any) => `Вставить ключ ${label}`,
      optional: 'Опционально',
      enterValueFirst: 'Сначала введите значение.',
      couldNotSave: 'Не удалось сохранить учетные данные.',
      remove: 'Удалить',
      or: 'или',
      escToCancel: 'esc для отмены',
      getKey: 'Получить ключ',
      saving: 'Сохранение'
    },
    envActions: {
      actionsFor: (label: any) => `Действия для ${label}`,
      credentialActions: 'Действия с учетными данными',
      docs: 'Документация',
      hideValue: 'Скрыть значение',
      revealValue: 'Показать значение',
      replace: 'Заменить',
      set: 'Установить',
      clear: 'Очистить'
    },
    gateway: {
      loading: 'Загрузка настроек шлюза...',
      unavailableTitle: 'Настройки шлюза недоступны',
      unavailableDesc: 'Мост IPC рабочего стола не предоставляет доступ к настройкам шлюза.',
      title: 'Подключение к шлюзу',
      envOverride: 'переопределение env',
      intro:
        'По умолчанию Hermes Desktop запускает собственный локальный шлюз. Используйте удаленный шлюз, если хотите, чтобы это приложение управляло уже запущенным бэкендом Hermes на другой машине или за доверенным прокси. Выберите профиль ниже, чтобы назначить ему свой удаленный хост.',
      appliesTo: 'Применяется к',
      allProfiles: 'Всем профилям',
      defaultConnection: 'Соединение по умолчанию для каждого профиля, у которого нет собственного переопределения.',
      profileConnection: (profile: any) =>
        `Соединение, используемое только тогда, когда активен профиль «${profile}». Установите его в Локальный, чтобы наследовать настройки по умолчанию.`,
      envOverrideTitle: 'Переменные окружения управляют этой сессией рабочего стола.',
      envOverrideDesc:
        'Сбросьте HERMES_DESKTOP_REMOTE_URL и HERMES_DESKTOP_REMOTE_TOKEN, чтобы использовать сохраненные настройки ниже.',
      localTitle: 'Локальный шлюз',
      localDesc: 'Запустить частный бэкенд Hermes на localhost. Это стандартный вариант, работающий офлайн.',
      remoteTitle: 'Удаленный шлюз',
      remoteDesc:
        'Подключить эту оболочку рабочего стола к удаленному бэкенду Hermes. Хостинговые шлюзы используют OAuth или имя пользователя и пароль; саморазмещенные могут использовать токен сессии.',
      remoteUrlTitle: 'URL удаленного шлюза',
      remoteUrlDesc: 'Базовый URL для удаленного бэкенда панели управления. Поддерживаются префиксы пути, например /hermes.',
      probing: 'Проверка способа аутентификации этого шлюза…',
      probeError: 'Пока не удалось подключиться к этому шлюзу. Проверьте URL — метод аутентификации появится после ответа.',
      signedIn: 'Вход выполнен',
      signIn: 'Войти',
      signOut: 'Выйти',
      signInWith: (provider: any) => `Войти через ${provider}`,
      authTitle: 'Аутентификация',
      authSignedInPassword:
        'Этот шлюз использует имя пользователя и пароль. Вы вошли в систему; сессия обновляется автоматически.',
      authSignedInOauth: 'Этот шлюз использует OAuth. Вы вошли в систему; сессия обновляется автоматически.',
      authNeedsPassword: 'Этот шлюз использует имя пользователя и пароль. Войдите, чтобы авторизовать это десктопное приложение.',
      authNeedsOauth: (provider: any) => `Этот шлюз использует OAuth. Войдите через ${provider}, чтобы авторизовать это десктопное приложение.`,
      tokenTitle: 'Токен сессии',
      tokenDesc: 'Токен сессии панели управления, используемый для доступа к REST и WebSocket. Оставьте поле пустым, чтобы сохранить текущий токен.',
      existingToken: (value: any) => `Существующий токен ${value}`,
      savedToken: 'сохранен',
      pasteSessionToken: 'Вставить токен сессии',
      testRemote: 'Тест удаленного подключения',
      saveForRestart: 'Сохранить для следующего перезапуска',
      saveAndReconnect: 'Сохранить и переподключиться',
      diagnostics: 'Диагностика',
      diagnosticsDesc: 'Показать desktop.log в вашем файловом менеджере — полезно, когда шлюз не запускается.',
      openLogs: 'Открыть логи',
      incompleteTitle: 'Настройки удаленного шлюза неполны',
      incompleteSignIn: 'Введите URL удаленного шлюза и войдите, прежде чем переключаться на удаленный.',
      incompleteToken: 'Введите URL удаленного шлюза и токен сессии, прежде чем переключаться на удаленный.',
      incompleteSignInTest: 'Введите URL удаленного шлюза и войдите перед тестированием.',
      incompleteTokenTest: 'Введите URL удаленного шлюза и токен сессии перед тестированием.',
      enterUrlFirst: 'Сначала введите URL удаленного шлюза.',
      restartingTitle: 'Перезапуск подключения к шлюзу',
      savedTitle: 'Настройки шлюза сохранены',
      restartingMessage: 'Hermes Desktop переподключится, используя сохраненные настройки.',
      savedMessage: 'Сохранено для следующего перезапуска.',
      connectedTo: (baseUrl: any, version: any) => `Подключено к ${baseUrl}${version ? ` · Hermes ${version}` : ''}`,
      reachableTitle: 'Удаленный шлюз доступен',
      signedOutTitle: 'Выход выполнен',
      signedOutMessage: 'Сессия удаленного шлюза очищена.'
    },
    keys: {
      loading: 'Загрузка ключей API...',
      title: 'Настройки ключей API',
      intro:
        'Эти ключи хранятся локально в файле .env вашего Hermes. Каждое поле здесь соответствует переменной окружения.',
      category: {
        provider: 'Провайдеры моделей',
        messaging: 'Мессенджеры',
        tool: 'Инструменты',
        setting: 'Настройки'
      }
    },
    mcp: {
      title: 'Серверы MCP',
      intro:
        'Добавьте сюда пользовательские наборы инструментов, размещенные на серверах MCP (Multi-provider Capability Protocol). Каждый сервер предоставляет одну или несколько возможностей, которые Hermes может вызывать.',
      loading: 'Загрузка серверов MCP...',
      addServer: 'Добавить сервер',
      serverAddress: 'Адрес сервера',
      serverName: 'Имя сервера',
      serverNamePlaceholder: 'Мой набор инструментов',
      serverAddressPlaceholder: 'localhost:8080',
      removeServer: 'Удалить сервер',
      noServers: 'Нет настроенных серверов MCP.'
    },
    archivedSessions: {
      title: 'Архивные сессии',
      intro:
        'Это сессии, которые вы вручную удалили из списка недавних. Они остаются доступными здесь. Вы можете возобновить любую из них, чтобы вернуть ее в основной список, или удалить навсегда.',
      loading: 'Загрузка архивных сессий...',
      searchPlaceholder: 'Поиск сессий...',
      noArchivedSessions: 'Нет архивных сессий.',
      delete: 'Удалить навсегда',
      resume: 'Возобновить',
      deleteConfirm: 'Вы уверены, что хотите навсегда удалить эту сессию?',
      deleteConfirmAll: 'Вы уверены, что хотите навсегда удалить все архивные сессии?',
      deleteAll: 'Удалить все'
    }
  },

  skills: {
    title: 'Навыки',
    titleWithCount: (count: any) => `Навыки (${count})`,
    intro:
      'Навыки — это инструкции, которым следует Hermes. Они хранятся как локальные текстовые файлы и могут быть созданы, изменены или удалены.',
    searchPlaceholder: 'Поиск навыков...',
    loading: 'Загрузка навыков...',
    loadError: 'Не удалось загрузить навыки',
    noSkills: 'Нет установленных навыков',
    noFilteredSkills: 'Нет навыков, соответствующих вашему поиску',
    newSkill: 'Новый навык',
    agentCreated: 'Создано агентом',
    official: 'Официальный',
    marketplace: 'Маркетплейс',
    local: 'Локальный',
    enableAll: 'Включить все',
    disableAll: 'Отключить все',
    enabled: 'Включено',
    disabled: 'Отключено',
    edit: 'Редактировать',
    delete: 'Удалить',
    confirmDelete: 'Вы уверены, что хотите удалить этот навык?',
    view: 'Просмотр',
    source: 'Источник',
    skillSource: (source: any) => `Источник навыка: ${source}`
  },

  agents: {
    title: 'Агенты',
    titleWithCount: (count: any) => `Агенты (${count})`,
    intro: 'Агенты — это настроенные экземпляры Hermes, каждый со своей собственной целью и возможностями. Управляйте ими здесь.',
    searchPlaceholder: 'Поиск агентов...',
    loading: 'Загрузка агентов...',
    loadError: 'Не удалось загрузить агентов',
    noAgents: 'Нет доступных агентов',
    noFilteredAgents: 'Нет агентов, соответствующих вашему поиску',
    newAgent: 'Новый агент',
    edit: 'Редактировать',
    delete: 'Удалить',
    confirmDelete: 'Вы уверены, что хотите удалить этого агента?'
  },

  commandCenter: {
    title: 'Командный центр',
    intro:
      'Командный центр предоставляет быстрый доступ к действиям и навигации. Используйте его для поиска сессий, переключения видов и выполнения команд.',
    searchPlaceholder: 'Поиск сессий, видов и команд...',
    noResults: 'Результаты не найдены',
    session: 'Сессия',
    view: 'Вид',
    action: 'Действие'
  },

  messaging: {
    title: 'Мессенджеры',
    titleWithCount: (count: any) => `Мессенджеры (${count})`,
    intro:
      'Подключайте Hermes к вашим любимым платформам для обмена сообщениями. Каждая платформа настраивается индивидуально.',
    loading: 'Загрузка платформ...',
    loadError: 'Не удалось загрузить платформы',

    platforms: {
      discord: 'Discord',
      slack: 'Slack',
      telegram: 'Telegram',
      whatsapp: 'WhatsApp'
    },
    status: {
      connected: 'Подключено',
      disconnected: 'Отключено',
      connecting: 'Подключение...',
      error: 'Ошибка'
    },
    lastMessage: (time: any) => `Последнее сообщение: ${time}`,
    configure: 'Настроить',
    disconnect: 'Отключить',
    connect: 'Подключить',
    noPlatforms: 'Платформы обмена сообщениями не настроены'
  },

  profiles: {
    title: 'Профили',
    titleWithCount: (count: any) => `Профили (${count})`,
    intro:
      'Профили — это изолированные экземпляры Hermes, каждый со своими настройками, памятью и сессиями. Используйте их для разделения рабочих пространств.',
    searchPlaceholder: 'Поиск профилей...',
    loading: 'Загрузка профилей...',
    loadError: 'Не удалось загрузить профили',
    noProfiles: 'Профили не найдены',
    noFilteredProfiles: 'Нет профилей, соответствующих вашему поиску',
    newProfile: 'Новый профиль',
    edit: 'Редактировать',
    delete: 'Удалить',
    confirmDelete: 'Вы уверены, что хотите удалить этот профиль?',
    switch: 'Переключиться',
    active: 'Активен',
    default: 'По умолчанию'
  },

  cron: {
    title: 'Задачи по расписанию',
    titleWithCount: (count: any) => `Задачи (${count})`,
    intro: 'Задачи по расписанию (cron) позволяют автоматически запускать Hermes с заданными интервалами.',
    searchPlaceholder: 'Поиск задач...',
    loading: 'Загрузка задач...',
    loadError: 'Не удалось загрузить задачи',
    noTasks: 'Нет задач по расписанию',
    noFilteredTasks: 'Нет задач, соответствующих вашему поиску',
    newTask: 'Новая задача',
    edit: 'Редактировать',
    delete: 'Удалить',
    confirmDelete: 'Вы уверены, что хотите удалить эту задачу?',
    runNow: 'Запустить сейчас',
    lastRun: 'Последний запуск',
    nextRun: 'Следующий запуск',
    never: 'никогда',
    schedule: 'Расписание',
    enabled: 'Включено'
  },

  artifacts: {
    title: 'Артефакты',
    titleWithCount: (count: any) => `Артефакты (${count})`,
    intro: 'Артефакты — это файлы, созданные Hermes. Они могут включать код, документы, изображения и многое другое.',
    searchPlaceholder: 'Поиск артефактов...',
    loading: 'Загрузка артефактов...',
    loadError: 'Не удалось загрузить артефакты',
    noArtifacts: 'Артефакты не найдены',
    noFilteredArtifacts: 'Нет артефактов, соответствующих вашему поиску',
    open: 'Открыть',
    download: 'Скачать',
    delete: 'Удалить',
    confirmDelete: 'Вы уверены, что хотите удалить этот артефакт?',
    view: 'Просмотр',
    created: 'Создано'
  },

  sidebar: {
    nav: {
      'new-session': 'Новая сессия',
      skills: 'Навыки и инструменты',
      messaging: 'Мессенджеры',
      artifacts: 'Артефакты'
    },
    searchAria: 'Поиск сессий',
    searchPlaceholder: 'Поиск сессий…',
    clearSearch: 'Очистить поиск',
    noMatch: (query: any) => `Нет сессий по запросу «${query}».`,
    results: 'Результаты',
    pinned: 'Закреплённые',
    sessions: 'Сессии',
    cronJobs: 'Cron-задачи',
    groupAriaGrouped: 'Показать сессии единым списком',
    groupAriaUngrouped: 'Группировать сессии по рабочей области',
    groupTitleGrouped: 'Разгруппировать сессии',
    groupTitleUngrouped: 'Группировать по рабочей области',
    allPinned: 'Здесь всё закреплено. Открепите чат, чтобы он появился в недавних.',
    shiftClickHint: 'Shift-клик по чату закрепляет его',
    noWorkspace: 'Без рабочей области',
    newSessionIn: (label: any) => `Новая сессия в ${label}`,
    reorderWorkspace: (label: any) => `Изменить порядок рабочей области ${label}`,
    showMoreIn: (count: any, label: any) => `Показать ещё ${count} в ${label}`,
    loading: 'Загрузка…',
    loadMore: 'Загрузить ещё',
    loadCount: (step: any) => `Загрузить ещё ${step}`,
    row: {
      pin: 'Закрепить',
      unpin: 'Открепить',
      copyId: 'Копировать ID',
      export: 'Экспорт',
      rename: 'Переименовать',
      archive: 'Архивировать',
      newWindow: 'Новое окно',
      copyIdFailed: 'Не удалось скопировать ID сессии',
      actionsFor: (title: any) => `Действия для ${title}`,
      sessionActions: 'Действия сессии',
      sessionRunning: 'Сессия выполняется',
      needsInput: 'Нужен ваш ввод',
      waitingForAnswer: 'Ожидает вашего ответа',
      handoffOrigin: (platform: any) => `Передано из ${platform}`,
      renamed: 'Переименовано',
      renameFailed: 'Не удалось переименовать',
      renameTitle: 'Переименовать сессию',
      renameDesc: 'Дайте этому чату понятное название. Оставьте пустым, чтобы очистить.',
      untitledPlaceholder: 'Сессия без названия',
      ageNow: 'сейчас',
      ageDay: 'д',
      ageHour: 'ч',
      ageMin: 'м'
    }
  },

  composer: {
    placeholder: 'Спросите что-нибудь или введите / для команд',
    editPlaceholder: 'Отредактируйте запрос или нажмите Esc, чтобы отменить',
    slashCommands: 'Slash-команды',
    mention: 'Упомянуть',
    mentions: 'Упоминания',
    noResults: 'Нет результатов',
    resolvedMentions: (count: any) => `Разрешённые упоминания: ${count}`,
    rerun: 'Повторить',
    edit: 'Редактировать',
    stop: 'Остановить',
    stopping: 'Остановка…',
    wakingProfile: (profile: any) => `Запуск профиля ${profile}…`,
    placeholderStarting: 'Запуск Hermes…',
    placeholderReconnecting: 'Переподключение к Hermes…',
    placeholderFollowUp: 'Отправить продолжение',
    newSessionPlaceholders: [
      'Что будем строить?',
      'Дайте Hermes задачу',
      'Что у вас на уме?',
      'Опишите, что нужно сделать',
      'Чем займёмся?',
      'Спросите что угодно',
      'Начните с цели'
    ],
    followUpPlaceholders: [
      'Отправьте продолжение',
      'Добавьте контекст',
      'Уточните запрос',
      'Что дальше?',
      'Продолжим',
      'Развейте идею',
      'Изменить или продолжить'
    ],
    startVoice: 'Начать голосовой диалог',
    queueMessage: 'Поставить сообщение в очередь',
    steer: 'Направить текущий запуск',
    send: 'Отправить',
    speaking: 'Говорит',
    transcribing: 'Распознавание',
    thinking: 'Думает',
    muted: 'Микрофон выключен',
    listening: 'Слушает',
    muteMic: 'Выключить микрофон',
    unmuteMic: 'Включить микрофон',
    stopListening: 'Остановить прослушивание и отправить',
    stopShort: 'Стоп',
    endConversation: 'Завершить голосовой диалог',
    endShort: 'Завершить',
    stopDictation: 'Остановить диктовку',
    transcribingDictation: 'Распознавание диктовки',
    voiceDictation: 'Голосовая диктовка',
    lookupLoading: 'Поиск…',
    lookupNoMatches: 'Совпадений нет.',
    lookupTry: 'Попробуйте',
    lookupOr: 'или',
    commonCommands: 'Частые команды',
    hotkeys: 'Горячие клавиши',
    helpFooter: 'открывает полную панель · Backspace закрывает',
    commandDescs: {
      '/help': 'полный список команд и горячих клавиш',
      '/clear': 'начать новую сессию',
      '/resume': 'возобновить прошлую сессию',
      '/details': 'управление детализацией транскрипта',
      '/copy': 'скопировать выделение или последний ответ ассистента',
      '/quit': 'выйти из Hermes'
    },
    hotkeyDescs: {
      'composer.mention': 'сослаться на файлы, папки, URL или git',
      'composer.slash': 'палитра slash-команд',
      'composer.help': 'эта краткая справка (Delete закрывает)',
      'composer.sendNewline': 'отправить · Shift+Enter для новой строки',
      'composer.sendQueued': 'отправить следующий ход из очереди',
      'keybinds.openPanel': 'все сочетания клавиш',
      'composer.cancel': 'закрыть всплывающее окно · отменить запуск',
      'composer.history': 'переключать всплывающее окно / историю'
    },
    attachUrlTitle: 'Прикрепить URL',
    attachUrlDesc: 'Hermes загрузит страницу и добавит её как контекст для этого хода.',
    urlPlaceholder: 'https://example.com/post',
    urlHintPre: 'Укажите полный URL, например ',
    attach: 'Прикрепить',
    queued: (count: any) => `В очереди: ${count}`,
    attachmentOnly: 'Ход только с вложением',
    emptyTurn: 'Пустой ход',
    attachments: (count: any) => `${count} вложен${count === 1 ? 'ие' : 'ий'}`,
    editingInComposer: 'Редактирование в редакторе',
    editingQueuedInComposer: 'Редактирование хода из очереди',
    queueEdit: 'Редактировать',
    queueSendNext: 'Далее',
    queueSend: 'Отправить',
    voice: {
      stop: 'Остановить запись',
      start: 'Начать запись',
      listening: 'Прослушивание…'
    },
    errors: {
      mentionInvalid: 'Одно или несколько упоминаний недействительны и были удалены.'
    },
    history: {
      title: 'История',
      empty: 'История пуста',
      recent: 'Недавние',
      clear: 'Очистить'
    }
  },

  statusStack: {
    thinking: 'Думаю...',
    botTyping: 'Hermes печатает...',
    userTyping: 'Вы печатаете...',
    shell: 'Терминал',
    running: 'Выполняется',
    pending: 'Ожидание'
  },

  updates: {
    title: 'Доступно обновление',
    body: (version: any, changes: any) =>
      `Hermes Desktop версии ${version} готов к установке. Этот выпуск включает ${changes} изменен${
        changes === 1 ? 'ие' : 'ий'
      }.`,
    notes: 'Примечания к выпуску',
    install: 'Установить сейчас',
    later: 'Позже'
  },

  install: {
    title: 'Установка...',
    body: 'Hermes устанавливает обновление. Приложение перезапустится после завершения.',
    restarting: 'Перезапуск...'
  },

  onboarding: {
    title: 'Добро пожаловать в Hermes Desktop',
    welcome: 'Добро пожаловать!',
    intro:
      'Hermes — это ваш персональный AI-ассистент, созданный для совместной работы. Давайте настроим несколько ключевых параметров, чтобы начать.',
    modelTitle: 'Выберите вашу модель',
    modelDescription: 'Выберите модель, которую вы хотите использовать. Вы можете изменить это в любое время в настройках.',
    modelFamily: 'Семейство моделей',
    selectModel: 'Выберите модель',
    apiKey: 'API-ключ',
    finish: 'Завершить',
    allSet: 'Все готово!',
    allSetDescription: 'Ваша настройка завершена. Hermes готов к работе. Откройте новый чат или изучите настройки.'
  },

  modelPicker: {
    title: 'Выбор модели',
    searchPlaceholder: 'Поиск моделей...',
    noResults: 'Модели не найдены',
    featured: 'Рекомендуемые',
    recentlyUsed: 'Недавно использованные',
    allModels: 'Все модели',
    provider: 'Провайдер',
    hidden: 'скрыто',
    manageVisibility: 'Управление видимостью'
  },

  modelVisibility: {
    title: 'Видимость моделей',
    intro: 'Скройте модели, которые вы не используете, чтобы упростить список.',
    searchPlaceholder: 'Поиск моделей...',
    noResults: 'Модели не найдены',
    showAll: 'Показать все',
    hideAll: 'Скрыть все'
  },

  shell: {
    title: 'Терминал',
    searchPlaceholder: 'Поиск вывода терминала...',
    noOutput: 'Нет вывода терминала',
    clear: 'Очистить',
    sendCommand: 'Отправить в редактор',

    // Новые переводы
    terminalUnavailable: 'Терминал недоступен',
    terminalUnavailableBody: 'Бэкенд терминала не смог запуститься. Проверьте ваши настройки.',
    openSettings: 'Открыть настройки',
    retry: 'Повторить',
    live: 'В ЭФИРЕ',
    defunct: 'НЕ РАБОТАЕТ'
  },

  rightSidebar: {
    title: 'Правая боковая панель',
    files: 'Файлы',
    code: 'Код',
    browser: 'Браузер',
    noFiles: 'Нет открытых файлов',
    openFile: 'Открыть файл',

    // Новые переводы
    search: 'Поиск',
    openFilePlaceholder: 'Путь к файлу...',
    searchFilesPlaceholder: 'Поиск по имени файла...',
    fileActions: {
        newFile: 'Новый файл',
        newFolder: 'Новая папка',
        rename: 'Переименовать',
        delete: 'Удалить',
        refresh: 'Обновить',
        openInCode: 'Открыть в VS Code',
        copyPath: 'Копировать путь',
        copyRelativePath: 'Копировать относительный путь'
    },
    confirmDelete: {
        title: 'Подтвердите удаление',
        body: (path: any) => `Вы уверены, что хотите удалить ${path}? Это действие нельзя отменить.`,
        confirm: 'Удалить'
    },
    rename: {
        title: 'Переименовать',
        newName: 'Новое имя...'
    },
    newFile: {
        title: 'Новый файл',
        name: 'Имя файла...'
    },
    newFolder: {
        title: 'Новая папка',
        name: 'Имя папки...'
    },

    empty: 'Рабочая область пуста',
    emptyDesc: 'Откройте папку, чтобы начать работу с файлами.',
    openFolder: 'Открыть папку'
  },

  preview: {
    title: 'Предпросмотр',
    unsupported: 'Тип файла не поддерживается',
    unsupportedBody: 'Этот файл нельзя просмотреть.',
    loading: 'Загрузка...',
    error: 'Не удалось загрузить файл',
    editInVscode: 'Редактировать в VS Code',
    openInNewTab: 'Открыть в новой вкладке',

    // Новые переводы
    image: 'Изображение',
    lines: 'строки'
  },

  assistant: {
    requestApproval: 'Запрос на одобрение',
    requestInput: 'Запрос на ввод',
    requestSecret: 'Запрос секрета',
    toolAuth: 'Аутентификация инструмента',
    approve: 'Одобрить',
    reject: 'Отклонить',
    enter: 'Ввести',

    // Новые переводы
    input: {
        placeholder: 'Введите ваш ответ...'
    },
    secret: {
        placeholder: 'Введите секрет...'
    },
    approval: {
        reason: 'Причина',
        toolName: 'Имя инструмента',
        rawArgs: 'Аргументы',
        parsedArgs: 'Разобранные аргументы'
    },
    auth: {
        title: (provider: any) => `Аутентификация в ${provider}`,
        message: (provider: any) => `Hermes необходимо аутентифицироваться в ${provider}, чтобы продолжить.`,
        signIn: 'Войти'
    },

    running: 'Выполняется',
    toolCalls: 'Вызовы инструментов',
    thinking: 'Думаю...'
  },

  prompts: {
    title: 'Подсказки',
    newPrompt: 'Новая подсказка',
    searchPlaceholder: 'Поиск подсказок...',
    noPrompts: 'Подсказки не найдены',

    // Новые переводы
    empty: 'Нет сохраненных подсказок',
    emptyDesc: 'Сохраняйте часто используемые подсказки для быстрого доступа.',
    create: 'Создать подсказку',
    edit: {
        title: 'Редактировать подсказку',
        name: 'Имя',
        namePlaceholder: 'Моя подсказка...',
        content: 'Содержимое',
        contentPlaceholder: 'Содержимое подсказки...'
    },
    confirmDelete: {
        title: 'Подтвердите удаление',
        body: (name: any) => `Вы уверены, что хотите удалить подсказку "${name}"?`,
        confirm: 'Удалить'
    },
    use: 'Использовать',
    editAction: 'Редактировать',
    deleteAction: 'Удалить'
  },

  desktop: {
    unmatchedConfirmation: {
      title: 'Несохраненные изменения',
      body: 'У вас есть несохраненные изменения. Вы уверены, что хотите закрыть?',
      confirm: 'Закрыть',
      cancel: 'Отмена'
    }
  },

  errors: {
    title: 'Ошибка',
    connectionFailed: 'Не удалось подключиться к шлюзу Hermes.',
    connectionFailedBody:
      'Убедитесь, что Hermes запущен, и проверьте настройки шлюза.',
    reconnect: 'Переподключиться',
    openSettings: 'Открыть настройки',

    // Новые переводы
    unknown: 'Произошла неизвестная ошибка.',
    restart: 'Перезапустить приложение'
  },

  ui: {
    // Этот раздел для общих элементов интерфейса, которые не подходят под другие категории
    theme: {
      dark: 'Тёмная',
      light: 'Светлая',
      system: 'Системная'
    },
    select: {
      placeholder: 'Выберите опцию...'
    },
    search: {
      clear: 'Очистить поиск'
    },
    filter: 'Фильтр...'
  }
} as any))
