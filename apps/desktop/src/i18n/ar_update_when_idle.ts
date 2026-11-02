import type { UpdateWhenIdleCopy } from './types_update_when_idle'

export const arUpdateWhenIdle: UpdateWhenIdleCopy = {
  busyTitle: 'Hermes لا يزال يعمل',
  busyBody: 'التحديث يعيد تشغيل Hermes، وهذا يوقف:',
  chats: (n: number) => (n === 1 ? 'محادثة واحدة قيد التشغيل' : `${n} محادثات قيد التشغيل`),
  cron: (jobs: string) => `مهمة مجدولة قيد التشغيل (${jobs})`,
  delegation: 'وكلاء فرعيون يعملون في الخلفية',
  input: 'سؤال ينتظر إجابتك',
  unknown: 'تعذّر على Hermes التأكد من أنه في وضع الخمول',
  updateNow: (n: number) => `التحديث الآن (يقاطع ${n})`,
  updateWhenIdle: 'التحديث عند الخمول',
  waitingTitle: 'التحديث ينتظر حتى ينتهي Hermes',
  waitingBody: 'يبدأ التحديث تلقائيًا عندما لا يعمل شيء. تابع عملك؛ يبقى Hermes متاحًا حتى ذلك الحين.',
  waitingFor: 'في انتظار:',
  waitingIdle: 'لا شيء قيد التشغيل. جارٍ بدء التحديث…',
  waitingUntil: (time: string) => `إذا بقي Hermes مشغولًا عند ${time}، فسيتم تخطي التحديث.`,
  cancel: 'إلغاء التحديث',
  timedOut: 'بقي Hermes مشغولًا، لذلك لم يُنفَّذ التحديث. لم يتغير شيء؛ حاول لاحقًا.'
}
