import type { FirstRunCopy } from './types_first_run'

export const arFirstRun: FirstRunCopy = {
  beats: {
    name: 'الاسم',
    theme: 'لون التمييز',
    layout: 'التخطيط',
    connectors: 'التطبيقات',
    plugins: 'الإضافات',
    tour: 'الجولة',
    build: 'المهمة الأولى'
  },
  namePlaceholder: 'اسمك',
  useName: name => `استخدام «${name}»`,
  install: 'تثبيت',
  installed: 'مثبّت',
  loading: 'جارٍ التحميل…',
  noPlugins: 'لا توجد إضافات لعرضها الآن.',
  noneChosen: 'لا شيء',
  noAnswer: 'لا توجد إجابة',
  answerRefused: 'لم تُقبل الإجابة',
  tours: {
    quick: 'جولة سريعة',
    full: 'أرني كل شيء',
    skip: 'تخطَّ، لنصنع شيئًا'
  },
  skippedAll: 'تم تخطي الإعداد',
  skip: 'تخطي الإعداد',
  skipTip: 'إنهاء الإعداد الموجَّه وفتح محادثة جديدة.',
  skipFailed: 'تعذّر تخطي الإعداد',
  slashOffTitle: 'أوامر الشرطة المائلة غير متاحة أثناء الإعداد',
  slashOffBody: 'ستعمل كالمعتاد بعد انتهاء الإعداد.',
  localModel: {
    title: 'يمكن لهذا الكمبيوتر تشغيل نموذج محليًا',
    text: (model, size) => `${model} (${size}) يناسب أجهزتك. يعمل مجانًا، وتبقى محادثاتك على هذا الكمبيوتر.`,
    setUp: 'إعداده',
    notNow: 'ليس الآن'
  },
  intro: {
    greeting: 'مرحبًا، أنا Hermes.',
    plan: 'لنجهّز كل شيء لك، ثم ننجز معًا شيئًا رائعًا.'
  },
  rerun: {
    title: 'تشغيل الإعداد الأولي مرة أخرى',
    description: 'يفتح محادثة إعداد جديدة تمر بخطوات التشغيل الأول باستخدام نموذجك الحالي.',
    action: 'تشغيل مرة أخرى',
    failed: 'تعذّر بدء الإعداد الأولي'
  }
}
