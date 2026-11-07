import type { FieldCopyTree } from '@/app/settings/field-copy'

// Settings > Memory & Context > Context & compression copy for this locale.
export const ruCompressionFieldLabels: FieldCopyTree = {
  context: {
    engine: 'Движок контекста'
  },
  compression: {
    enabled: 'Авто-сжатие',
    threshold: 'Порог сжатия',
    codexGpt55Autoraise: 'Автоповышение сжатия Codex',
    targetRatio: 'Целевое сжатие',
    protectLastN: 'Защищённые недавние сообщения',
    warmHandoff: 'Тёплая передача'
  },
  auxiliary: {
    compression: {
      timeout: 'Таймаут модели сжатия (с)'
    }
  }
}

export const ruCompressionFieldDescriptions: FieldCopyTree = {
  context: {
    engine: 'Стратегия управления длинными диалогами у предела контекста.'
  },
  compression: {
    enabled: 'Сжимать более старый контекст, когда диалоги становятся большими.',
    codexGpt55Autoraise: 'Повышает порог сжатия до 85% для поддерживаемых моделей ChatGPT Codex OAuth.',
    warmHandoff:
      'Основная модель пишет сводку сжатия поверх своего кэшированного промпта, поэтому сервер с кэшированием промптов читает только новые сообщения. Быстрее, когда для сжатия используется та же модель. Авто: только если сжатие использует основную модель и сервер сообщает о кэшированных токенах. Вкл: всегда пробовать. Выкл: всегда использовать модель сжатия. При любой ошибке используется обычная сводка.'
  },
  auxiliary: {
    compression: {
      timeout:
        'Сколько секунд ждать вспомогательную модель сжатия за один вызов (по умолчанию 120). Увеличьте для медленных локальных моделей.'
    }
  }
}
