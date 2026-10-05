export const ruModelMenu = {
  search: 'Поиск моделей',
  noModels: 'Модели не найдены',
  editModels: 'Изменить модели…',
  followDefault: 'Использовать модель по умолчанию из настроек',
  refreshModels: 'Обновить модели',
  favorites: 'Избранное',
  addFavorite: 'Добавить в избранное',
  removeFavorite: 'Убрать из избранного',
  favoriteShortcut: '⇧ Клик',
  fast: 'Быстрая',
  free: 'бесплатно',
  cacheRead: 'чтение из кэша',
  priceTitle: (input: string, output: string, cache: string) =>
    `Вход ${input}/Mtok · Выход ${output}/Mtok` + (cache ? ` · Чтение из кэша ${cache}/Mtok` : ''),
  limited: 'Лимит',
  limitedUntil: (time: string) => `Лимит до ${time}`,
  limitedTip: (provider: string, time: null | string) =>
    time
      ? `${provider} исчерпал лимит использования. Он сбросится в ${time}; модель на потом можно выбрать уже сейчас.`
      : `${provider} исчерпал лимит использования. Модель на время после сброса можно выбрать уже сейчас.`,
  modelResets: (time: string) => `снова в ${time}`,
  modelLimitedTip: (time: string) =>
    `Эта модель исчерпала собственный лимит и вернётся в ${time}. Остальные модели здесь работают.`,
  usageLeft: (percent: number, time: null | string) =>
    time ? `Осталось ${percent}% · сброс в ${time}` : `Осталось ${percent}%`,
  usageTip: (provider: string) => `${provider} почти исчерпал лимит использования.`,
  usageWindow: (label: string, percent: number, time: null | string) =>
    time ? `${label}: осталось ${percent}%, сброс в ${time}` : `${label}: осталось ${percent}%`
}
