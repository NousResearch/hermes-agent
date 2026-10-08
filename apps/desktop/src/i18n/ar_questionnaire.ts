import type { Translations } from './types'

export const arQuestionnaire: Translations['questionnaire'] = {
  skipSetup: 'تخطي الإعداد',
  back: 'رجوع',
  trailLabel: 'إجاباتك',
  changeAnswer: 'تغيير هذه الإجابة',
  retry: 'حاول مرة أخرى',
  saveFailed: 'تعذّر تسجيل اكتمال الإعداد',
  skipped: 'تم التخطي',
  otherLabel: 'إجابة أخرى',
  kinds: {
    accent: 'لون التمييز',
    layout: 'التخطيط',
    local: 'نموذج محلي',
    apps: 'التطبيقات على هذا الجهاز',
    connectors: 'الموصلات'
  },
  name: {
    greeting: 'مرحبًا، أنا Hermes.',
    question: 'بماذا أناديك؟',
    placeholder: 'اسمك'
  },
  accent: {
    title: 'اختر لونًا',
    custom: 'لون مخصص'
  },
  layout: {
    title: 'كيف تريد أن تعمل؟',
    basic: 'أساسي',
    basicDetail: 'للتحدث مع Hermes.',
    elite: 'متقدم',
    eliteDetail: 'للمطورين: الطرفية والملفات والفروقات.'
  },
  local: {
    titleSpark: 'يمكن لجهاز Spark تشغيل Hermes دون اتصال',
    title: kind => `يمكن لهذا الـ ${kind} تشغيل نموذج محليًا`,
    download: model => `تنزيل ${model}`,
    downloadDetail: 'يتم التنزيل في الخلفية عند بدء محادثتك الأولى.',
    notNow: 'ليس الآن',
    notNowDetail: 'يبقى في قائمة النماذج باسم «تشغيل محليًا».',
    chip: 'الشريحة',
    gpu: 'GPU',
    memory: 'الذاكرة',
    os: 'نظام التشغيل',
    memoryValue: gb => `${gb} GB`
  },
  apps: {
    title: 'استخدام Hermes داخل هذه التطبيقات؟',
    plugin: 'إضافة',
    pluginNotRunning: 'إضافة · التطبيق غير قيد التشغيل',
    approve: 'توافق على كل تثبيت في محادثتك الأولى.',
    found: 'تم العثور عليها على هذا الجهاز. التطبيقات غير المثبتة لا تظهر.'
  },
  connectors: {
    title: 'ربط تطبيقاتك؟',
    checking: 'جارٍ التحقق من التطبيقات التي يمكن لحسابك ربطها…',
    offered: 'هذه هي القائمة التي يتيحها حسابك حاليًا. تسجّل الدخول إلى كل منها في محادثتك الأولى.'
  },
  task: {
    title: 'بماذا نبدأ؟',
    question: 'اختر واحدة، أو اكتب ما تريد.',
    blender: 'بناء مشهد في Blender',
    nvidia: 'ضبط بطاقة GPU',
    broadcast: 'إعداد الميكروفون والكاميرا',
    brief: 'ملخص يومي',
    tidy: 'ترتيب مجلد التنزيلات'
  },
  tour: {
    title: 'هل تريد جولة؟',
    question: 'نظرة سريعة بعد ذلك؟',
    quick: 'أرني المكان',
    quickDetail: 'بضع محطات، أقل من دقيقة.',
    none: 'سأكتشف بنفسي',
    noneDetail: 'يمكنك طلب جولة من Hermes لاحقًا.'
  },
  start: {
    title: name => (name ? `جاهز يا ${name}.` : 'جاهز.'),
    sub: 'هذه أول رسالة يتلقاها Hermes. يظهر الطلب أولًا، ويأتي الباقي تحته.',
    aboutMe: 'عني:',
    beforeYouStart: 'قبل أن تبدأ:',
    noTask: 'لم تُختر مهمة: سيسألك Hermes عمّا تريد فعله.',
    action: 'ابدأ',
    waiting: 'جارٍ تجهيز حسابك المجاني…',
    failed: 'تعذّر بدء محادثتك الأولى'
  },
  summary: {
    custom: 'مخصص',
    local: model => `محلي: ${model}`,
    localNo: 'محلي: ليس الآن',
    noApps: 'لا تطبيقات',
    noConnectors: 'لا موصلات',
    more: (label, count) => `${label} +${count}`,
    tour: 'جولة',
    noTour: 'بلا جولة'
  },
  status: {
    settingUp: 'جارٍ إعداد حسابك المجاني',
    stillSettingUp: 'لا يزال إعداد حسابك المجاني جاريًا',
    unavailable: 'الحساب المجاني غير متاح',
    ready: 'Nous · الخطة المجانية',
    downloadingEngine: 'جارٍ تنزيل المحرك المحلي',
    downloading: model => `جارٍ تنزيل ${model}`
  },
  settings: {
    title: 'تشغيل الإعداد مجددًا',
    description: 'يعيد فتح أسئلة التشغيل الأول. تُطبَّق إجاباتك على الملف الشخصي الافتراضي.',
    action: 'تشغيل الإعداد',
    failed: 'تعذّر بدء الإعداد'
  }
}
