import type { MemoryDiscoveryTranslations } from './types_memory_discovery'

export const arMemoryDiscovery = {
  installed: 'مثبت',
  availableToInstall: 'متاح للتثبيت',
  installationRequired: 'التثبيت مطلوب',
  reviewInstall: 'مراجعة وتثبيت',
  exploreAll: 'استكشاف الكل…',
  missing: 'مفقود',
  installConsent: 'يثبّت الإضافة ويفعّلها مع اعتمادياتها. لا يتغير مزوّد الذاكرة النشط حتى تختاره صراحةً.',
  builtin: 'مدمج',
  configureElsewhere: 'أعدّ المزوّد عبر CLI أو حدّث Hermes للحفظ دون تفعيل.',
  notReady: 'أكمل الإعداد وثبّت الاعتماديات. بعد التثبيت أعد تشغيل الخلفية ثم حاول مجدداً.',
  useFailed: 'تعذّر استخدام المزوّد. تحقّق من الإعدادات وأعد المحاولة.',

  activeProvider: name => `النشط: ${name}`,
  useProvider: 'استخدام المزوّد',
  loadFailed: 'تعذّر تحميل مزوّدي الذاكرة',
  ownerChanged: 'عُد إلى الاتصال والملف الشخصي اللذين فتحت منهما برنامج التثبيت، ثم حاول مجددًا.',
  notDiscovered: 'تم تثبيت الحزمة، لكن لم يُكتشف مزوّد الذاكرة بعد. عُد إلى إعدادات الذاكرة لإعادة المحاولة.',
  installedNotice: 'تم اكتشاف المزوّد. أعدّه أولاً، ثم اختر استخدامه صراحةً.',
  backToMemory: 'العودة إلى إعدادات الذاكرة',
  connect: 'اتصال',
  reconnect: 'إعادة الاتصال',
  connectOAuth: 'الاتصال عبر OAuth',
  apiKeySet: 'تم تعيين مفتاح API',
  oauthSet: 'OAuth متصل',
  waitingConsent: 'بانتظار الموافقة في المتصفح…',
  stopWaiting: 'إيقاف الانتظار',
  stoppedWaiting: 'توقف الانتظار. قد يظل التفويض قيد الانتظار.',
  startFailed: 'تعذّر بدء الاتصال.',
  connectionFailed: 'فشل الاتصال.'
} satisfies Partial<MemoryDiscoveryTranslations>
