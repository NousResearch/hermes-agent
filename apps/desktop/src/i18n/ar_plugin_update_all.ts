import type { PluginUpdateAllTranslations } from './types_plugin_update'

/** Plugins tab: "Update all" and the auto-update switch. */
export const arPluginUpdateAll: PluginUpdateAllTranslations = {
  updateAll: 'تحديث الكل',
  updatingAll: 'جارٍ التحديث…',
  updateAllTip: 'فحص كل الإضافات المثبتة وتطبيق التحديثات التي لا تحتاج إلى مراجعة',
  updateAllDone: (updated: number, review: number, failed: number) =>
    `تم تحديث ${updated}` + (review ? `، ${review} تحتاج إلى مراجعة` : '') + (failed ? `، فشل ${failed}` : ''),
  updateAllNone: 'كل الإضافات محدّثة',
  updateAllFailed: 'تعذّر تحديث الإضافات',
  autoUpdate: 'التحديث التلقائي',
  autoUpdateTip: (hours: number) =>
    `تطبيق تحديثات الإضافات تلقائيًا عندما يفحصها Hermes (كل ${hours} ساعة). تُحمَّل الشيفرة المحدّثة في الجلسات الجديدة.`,
  autoUpdateFailed: 'تعذّر تغيير التحديث التلقائي للإضافات'
}
