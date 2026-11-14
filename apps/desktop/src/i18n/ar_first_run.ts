import type { FirstRunCopy } from './types_first_run'

export const arFirstRun: FirstRunCopy = {
  guide: {
    title: 'إعداد Hermes',
    resume: 'استئناف الإعداد',
    upgradeRequired:
      'يحتاج Hermes على هذا الجهاز إلى تحديث للإعداد دون جلسة. شغّل hermes update على ذلك الجهاز، وأعد تشغيل Hermes، ثم أعد المحاولة. محادثاتك لم تتغير.',
    intro: 'بعض التفضيلات الاختيارية، ثم محادثتك المعتادة. لا يؤدي الإعداد إلى إنشاء محادثة.',
    appearance: 'المظهر',
    model: 'اختر نموذجًا',
    ready: 'نموذجك جاهز.',
    providerNeeded: 'اربط مزودًا أو استخدم النموذج المجاني قبل إرسال رسالة.',
    freeModel: 'استخدام النموذج المجاني',
    freeUnavailable: 'النموذج المجاني غير متاح هنا. اربط مزودًا أو اختر واحدًا لاحقًا.',
    providers: 'ربط مزود',
    finish: 'إنهاء الإعداد',
    skip: 'تخطي الإعداد',
    back: 'رجوع',
    check: 'التحقق من الجاهزية',
    checking: 'جارٍ التحقق…',
    privacy: 'مشاركة مقاييس الاستخدام اختيارية.'
  },
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
