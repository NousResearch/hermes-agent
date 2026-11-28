import type { AuxTaskCopyMap } from './types_aux_tasks'

export const ruAuxTasks: AuxTaskCopyMap = {
  vision: { label: 'Зрение', hint: 'Анализ изображений' },
  compression: { label: 'Сжатие', hint: 'Компрессия контекста' },
  approval: { label: 'Одобрение', hint: 'Умное авто-одобрение' },
  mcp: { label: 'MCP', hint: 'Маршрутизация MCP-инструментов' },
  title_generation: { label: 'Ген. заголовка', hint: 'Заголовки сеансов' },
  voice_chat: { label: 'Голосовой чат', hint: 'Ответы в голосовом режиме' },
  curator: { label: 'Куратор', hint: 'Просмотр использования навыков' },
  memory_query_rewrite: { label: 'Переписывание запросов памяти', hint: 'Запросы поиска по памяти' },
  tts_audio_tags: { label: 'Аудиотеги TTS', hint: 'Вставка тегов для Gemini TTS' },
  group_router: { label: 'Маршрутизатор групп', hint: 'Кто отвечает в группах с автомаршрутизацией' },
  goal_judge: { label: 'Судья целей', hint: 'Составление и проверка контрактов /goal' },
  background_review: { label: 'Фоновый обзор', hint: 'Обновление памяти и навыков после хода' },
  monitor: { label: 'Монитор', hint: 'Оценка важных писем' },
  moa_reference: { label: 'Референс MoA', hint: 'Референсные модели смеси агентов' },
  moa_aggregator: { label: 'Агрегатор MoA', hint: 'Шаг агрегации смеси агентов' }
}
