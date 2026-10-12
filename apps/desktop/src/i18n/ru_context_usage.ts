import type { TranslationOverride } from '@hermes/shared/i18n'

import type { Translations } from './types'

export const ruContextUsagePanel: TranslationOverride<Translations['shell']['statusbar']['contextUsagePanel']> = {
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
  empty: 'Данных контекста пока нет',
  loading: 'Загрузка разбивки…',
  percentFull: percent => `${percent}% занято`,
  title: 'Использование контекста',
  tokenSummary: (used, max) => `${used} / ${max} токенов`
}
