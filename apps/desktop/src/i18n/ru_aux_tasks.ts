import type { AuxTaskCopyMap } from './types_aux_tasks'

export const ruAuxTasks: AuxTaskCopyMap = {
  vision: { label: 'Зрение', hint: 'Анализ изображений' },
  compression: { label: 'Сжатие', hint: 'Компрессия контекста' },
  skills_hub: { label: 'Хаб навыков', hint: 'Поиск навыков' },
  approval: { label: 'Одобрение', hint: 'Умное авто-одобрение' },
  mcp: { label: 'MCP', hint: 'Маршрутизация MCP-инструментов' },
  title_generation: { label: 'Ген. заголовка', hint: 'Заголовки сеансов' },
  voice_chat: { label: 'Голосовой чат', hint: 'Ответы в голосовом режиме' },
  curator: { label: 'Куратор', hint: 'Просмотр использования навыков' },
  review: {
    label: 'Разбор',
    hint: 'Субагент-рецензент команды /review'
  },
  triage_specifier: {
    label: 'Уточнение задачи',
    hint: 'Детализация спецификации канбан-задачи'
  },
  kanban_decomposer: {
    label: 'Разбиение канбан-задач',
    hint: 'Разбиение задачи на подзадачи'
  },
  profile_describer: {
    label: 'Описание профиля',
    hint: 'Автоматическое описание профилей'
  }
}
