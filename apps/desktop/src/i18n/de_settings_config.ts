/** settings.config shard — keepAwake + managed-scope copy, split out of
 *  de.ts (FILE_LINES cap). */
export const deSettingsConfig = {
  keepAwakeTitle: 'Computer wach halten',
  keepAwakeDesc:
    'Verhindert, dass dieser Rechner in den Ruhezustand wechselt. „Während der Arbeit“ gilt nur, solange ein Durchlauf läuft: Läufe über Nacht laufen weiter, ohne den Laptop die ganze Woche wach zu halten. Der Bildschirm kann trotzdem abdunkeln.',
  keepAwakeOff: 'Aus',
  keepAwakeWhileWorking: 'Während der Arbeit',
  keepAwakeAlways: 'Immer',
  managedFieldHint: 'Von Ihrem Administrator verwaltet{source} — schreibgeschützt',
  managedRejectedTitle: 'Einige Einstellungen wurden nicht gespeichert',
  managedRejectedNotice: 'Nicht gespeichert (von Ihrem Administrator verwaltet): {keys}',
}
