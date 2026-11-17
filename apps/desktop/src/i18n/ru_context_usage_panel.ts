import type { ContextUsagePanelCopy } from './types_context_usage_panel'

export const ruContextUsagePanel = {
  categories: {
    conversation: 'Диалог',
    mcp: 'MCP',
    memory: 'Память',
    rules: 'Правила',
    skills: 'Навыки',
    subagent_definitions: 'Определения субагентов',
    system_prompt: 'Системный промпт',
    tool_definitions: 'Определения инструментов'
  },
  compact: {
    action: 'Сжать диалог',
    busyHint: 'Сначала завершите или остановите текущий ответ',
    confirm: 'Сжать',
    hint: 'Сожмите старый контекст, чтобы освободить место. Разговор продолжится здесь же.',
    question: 'Сжать этот диалог в краткое изложение?',
    questionNamed: title => `Сжать «${title}» в краткое изложение?`,
    working: 'Сжимаю…'
  },
  empty: 'Данных контекста пока нет',
  loading: 'Загрузка разбивки…',
  percentFull: percent => `${percent}% занято`,
  title: 'Использование контекста',
  tokenSummary: (used, max) => `${used} / ${max} токенов`,
  tabs: { inspector: 'Инспектор', usage: 'Использование' },
  inspector: {
    changes: {
      model_changed: (before, after) => `Модель изменена: ${before} → ${after}`,
      prompt_changed: 'Промпт изменён',
      provider_changed: (before, after) => `Провайдер изменён: ${before} → ${after}`,
      tool_added: name => `Инструмент добавлен: ${name}`,
      tool_removed: name => `Инструмент удалён: ${name}`
    },
    error: 'Не удалось прочитать, что видел агент',
    lineage: count => `Цепочка сжатий · ${count} сегм.`,
    loading: 'Чтение сохранённого снимка…',
    mcpServers: 'MCP-серверы',
    model: 'Модель',
    noChanges: 'Без изменений',
    noPrompt: 'Сохранённого промпта пока нет: он сохраняется после первого хода.',
    none: 'Нет',
    promptChars: count => `${count} симв.`,
    provider: 'Провайдер',
    segment: index => `Сегмент ${index}`,
    sourceLive: 'Активный агент',
    sourceStored: 'Сохранённый снимок',
    systemPrompt: 'Системный промпт',
    tools: count => `Инструменты (${count})`
  }
} satisfies ContextUsagePanelCopy
