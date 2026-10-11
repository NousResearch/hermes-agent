/** settings.config shard — keepAwake + managed-scope copy, split out of
 *  en.ts (FILE_LINES cap). */
export const enSettingsConfig = {
  keepAwakeTitle: 'Keep computer awake',
  keepAwakeDesc:
    'Stop this machine from sleeping. "While working" holds it only while a turn is in flight, so overnight runs survive without pinning the laptop awake all week. The display can still dim.',
  keepAwakeOff: 'Off',
  keepAwakeWhileWorking: 'While working',
  keepAwakeAlways: 'Always',
  managedFieldHint: 'Managed by your administrator{source} — read-only',
  managedRejectedTitle: 'Some settings were not saved',
  managedRejectedNotice: 'Not saved (managed by your administrator): {keys}',
}
