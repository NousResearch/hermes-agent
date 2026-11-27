import type { enBuiltinToolsets } from './en_builtin_toolsets'

export const ruBuiltinToolsets = {
  web: { label: 'Поиск в интернете', description: 'Поиск страниц и извлечение их содержимого.' },
  browser: {
    label: 'Управление браузером',
    description: 'Открывать страницы, нажимать кнопки, вводить текст и прокручивать страницу.'
  },
  terminal: { label: 'Терминал и процессы', description: 'Запускать команды и управлять запущенными процессами.' },
  file: { label: 'Работа с файлами', description: 'Читать, записывать, редактировать и искать файлы.' },
  code_execution: { label: 'Выполнение кода', description: 'Запускать программный код.' },
  vision: { label: 'Анализ изображений', description: 'Распознавать и описывать содержимое изображений.' },
  video: { label: 'Анализ видео', description: 'Анализировать видеозаписи с помощью модели, поддерживающей видео.' },
  image_gen: { label: 'Создание изображений', description: 'Создавать изображения по описанию.' },
  video_gen: { label: 'Создание видео', description: 'Создавать видео по тексту, изображению или образцу.' },
  x_search: {
    label: 'Поиск в X (Twitter)',
    description: 'Искать публикации в X. Требуется вход через xAI или ключ XAI_API_KEY.'
  },
  tts: { label: 'Синтез речи', description: 'Озвучивать текст.' },
  stt: {
    label: 'Распознавание речи',
    description: 'Преобразовывать в текст голосовые сообщения и речь в голосовом режиме.'
  },
  skills: { label: 'Навыки', description: 'Просматривать навыки и управлять ими.' },
  todo: { label: 'Планирование задач', description: 'Составлять список задач и отмечать выполненные.' },
  kanban: { label: 'Доска задач', description: 'Подключать инструменты канбан-доски для этой платформы.' },
  memory: { label: 'Память', description: 'Сохранять сведения между чатами.' },
  context_engine: {
    label: 'Управление контекстом',
    description: 'Использовать инструменты активного модуля контекста.'
  },
  session_search: { label: 'Поиск по чатам', description: 'Искать информацию в прошлых разговорах.' },
  connections: {
    label: 'Подключения',
    description: 'Использовать инструменты подключённых сервисов и авторизовывать учётные записи.'
  },
  clarify: { label: 'Уточняющие вопросы', description: 'Запрашивать у пользователя недостающую информацию.' },
  delegation: { label: 'Делегирование задач', description: 'Поручать задачи вспомогательным агентам.' },
  cronjob: {
    label: 'Задачи по расписанию',
    description: 'Создавать, запускать и приостанавливать задачи, менять расписание и подключать навыки.'
  },
  discord: { label: 'Общение в Discord', description: 'Читать сообщения, искать участников и создавать ветки.' },
  discord_admin: {
    label: 'Управление сервером Discord',
    description: 'Просматривать каналы и роли, закреплять сообщения и назначать роли.'
  },
  computer_use: {
    label: 'Управление компьютером (macOS, Windows, Linux)',
    description: 'Управлять рабочим столом в фоновом режиме через cua-driver.'
  }
} satisfies Record<keyof typeof enBuiltinToolsets, { label: string; description: string }>
