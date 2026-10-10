/** settings.config shard — keepAwake + managed-scope copy, split out of
 *  ru.ts (FILE_LINES cap). */
export const ruSettingsConfig = {
  keepAwakeTitle: 'Не давать компьютеру засыпать',
  keepAwakeDesc:
    'Не давать этому компьютеру засыпать. «Во время работы» держит его только пока выполняется ход — ночные запуски переживут сон, а ноутбук не останется включённым всю неделю. Экран по‑прежнему может гаснуть.',
  keepAwakeOff: 'Выкл',
  keepAwakeWhileWorking: 'Во время работы',
  keepAwakeAlways: 'Всегда',
  managedFieldHint: 'Управляется администратором{source} — только чтение',
  managedRejectedTitle: 'Некоторые параметры не были сохранены',
  managedRejectedNotice: 'Не сохранено (управляется администратором): {keys}',
}
