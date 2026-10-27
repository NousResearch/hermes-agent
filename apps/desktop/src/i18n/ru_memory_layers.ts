import type { MemoryLayersTranslations } from './types_memory_layers'

export const ruMemoryLayers: MemoryLayersTranslations = {
  activeProvider: 'Активный провайдер памяти',
  noProvider: 'Нет — только встроенная память',
  noProviderDesc:
    'Hermes использует только встроенный слой ниже. Выберите внешнего провайдера в поле «Провайдер памяти», чтобы добавить его.',
  providerDesc: name => `${name} добавляет собственные воспоминания в каждую сессию поверх встроенного слоя.`,
  status: {
    ready: 'готов',
    needs_config: 'нужна настройка',
    unavailable: 'недоступен',
    missing: 'не установлен'
  },
  builtinLayer: 'Встроенный слой',
  alwaysOn: 'Всегда включён',
  builtinDesc:
    'MEMORY.md и USER.md загружаются в каждую сессию — с внешним провайдером или без него. Провайдер дополняет этот слой, но никогда его не заменяет.'
}
