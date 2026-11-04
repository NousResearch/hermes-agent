import type { CronControlCopy } from './types_cron_control'

export const arCronControl: CronControlCopy = {
  stop: 'إيقاف',
  stopping: 'جارٍ الإيقاف…',
  stopped: 'أُلغي التشغيل',
  finishedFirst: 'انتهى التشغيل قبل إيقافه',
  nothingToStop: 'لا يوجد تشغيل جارٍ',
  failedStop: 'تعذّر إيقاف التشغيل',
  lastCancelled: (reason: string) => (reason ? `أُلغي آخر تشغيل: ${reason}` : 'أُلغي آخر تشغيل'),
  catchUpLabel: 'التشغيلات الفائتة',
  catchUpHint: 'ما يحدث للتشغيلات الفائتة أثناء توقف Hermes.',
  catchUpChoices: {
    default: 'استخدام الإعداد الافتراضي',
    run: 'تشغيل مرة واحدة عند العودة',
    skip: 'تخطيها'
  }
}
