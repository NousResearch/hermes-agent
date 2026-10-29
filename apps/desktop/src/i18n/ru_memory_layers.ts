import type { MemoryLayersTranslations } from './types_memory_layers'

export const ruMemoryLayers: MemoryLayersTranslations = {
  activeProvider: 'Активный провайдер памяти',
  selectedProvider: 'Выбранный провайдер памяти',
  selectedDesc: current => `Ещё не активен: сеансы продолжают использовать ${current}, пока этот выбор не сохранён.`,
  builtinMemory: 'только встроенную память',
  noProvider: 'Нет — только встроенная память',
  noProviderDesc:
    'Hermes использует только встроенный слой ниже. Чтобы добавить внешний провайдер, выберите его в Настройки › Память и контекст.',
  providerDesc: name => `${name} добавляет собственные воспоминания в каждый сеанс поверх встроенного слоя.`,
  providerNotReadyDesc: name =>
    `${name} — сохранённый провайдер, но он не добавляет воспоминаний, пока не готов. Встроенный слой работает в любом случае.`,
  providerUnknownDesc: name => `${name} — сохранённый провайдер; Hermes не смог узнать, готов ли он.`,
  statusUnavailable: 'Состояние памяти недоступно',
  statusUnavailableDesc: 'Не удалось узнать, какой провайдер памяти активен и готов ли он.',
  retry: 'Повторить',
  status: {
    ready: 'готов',
    needs_config: 'требуется настройка',
    unavailable: 'недоступен',
    missing: 'не установлен'
  },
  builtinLayer: 'Встроенный слой',
  onByDefault: 'Включён по умолчанию',
  on: 'Вкл.',
  off: 'Выкл.',
  onlyFile: file => `только ${file}`,
  builtinDesc:
    'MEMORY.md и USER.md — собственная память Hermes, независимая от внешних провайдеров. По умолчанию оба файла загружаются в сеансы; переключатели памяти и профиля пользователя в Настройки › Память и контекст отключают каждый из них. Провайдер дополняет этот слой и никогда его не заменяет.'
}
