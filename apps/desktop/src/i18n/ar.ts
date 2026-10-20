import { arArtifacts } from './ar_artifacts'
import { arAssistant } from './ar_assistant'
import { arBoot } from './ar_boot'
import { arCapabilities } from './ar_capabilities'
import { arChat } from './ar_chat'
import { arChrome } from './ar_chrome'
import { arCommandCenter } from './ar_command_center'
import { arCommon } from './ar_common'
import { arConnectors } from './ar_connectors'
import { arDiagnostics } from './ar_diagnostics'
import { arFirstRun } from './ar_first_run'
import { arSettings } from './ar_settings'
import { defineLocale, type TranslationOverrides } from './define-locale'

export const arOverrides = {

  catalog: {
    add: 'إضافة',
    added: 'تمت الإضافة',
    discover: 'اكتشف',
    featured: 'مميزة',
    explorePlugins: 'استكشف الإضافات',
    exploreSkills: 'استكشف المهارات',
    mostStarred: 'الأكثر حصولًا على نجوم',
    newest: 'الأحدث',
    recentlyUpdated: 'المحدثة مؤخرًا',
    alphabetical: 'الاسم',
    sortBy: 'ترتيب حسب',
    seeAll: 'عرض الكل',
    related: 'عناصر مشابهة',
    tags: 'الوسوم',
    screenshots: 'لقطات الشاشة',
    listView: 'عرض القائمة',
    cardView: 'عرض البطاقات',
    installTitle: (name: string) => `تثبيت «${name}»؟`,
    installDescription: 'ستتوفر هذه المهارة في الجلسات الجديدة. ثبّت من المصادر التي تثق بها فقط.',
    installTo: 'التثبيت في',
    thisComputer: 'هذا الكمبيوتر',
    installing: 'جارٍ التثبيت…',
    installComplete: (name: string) => `تم تثبيت «${name}»`,
    destinationChanged: 'تغيرت وجهة التثبيت. أغلق هذا الحوار وافتح رابط التثبيت مجددًا.',
    installed: 'المثبتة',
    searchSkills: 'البحث في المهارات',
    searchPlugins: 'البحث في الإضافات',
    allSources: 'كل المصادر',
    allCategories: 'كل الفئات',
    about: 'نبذة',
    author: 'المؤلف',
    source: 'المصدر',
    category: 'الفئة',
    version: 'الإصدار',
    platforms: 'المنصات',
    requires: 'المتطلبات',
    tools: 'الأدوات',
    hooks: 'الخطافات',
    middleware: 'البرمجيات الوسيطة',
    commands: 'الأوامر',
    license: 'الترخيص',
    addedDate: 'تاريخ الإضافة',
    updatedDate: 'تاريخ التحديث',
    repository: 'المستودع',
    documentation: 'التوثيق',
    noResults: 'لا توجد نتائج مطابقة',
    tryAnother: 'جرّب بحثًا آخر أو امسح عوامل التصفية.',
    clearFilters: 'مسح عوامل التصفية',
    filters: 'عوامل التصفية',
    loadFailed: 'تعذر تحميل الكتالوج',
    retry: 'حاول مرة أخرى',
    more: 'عرض المزيد',
    pinned: 'التزام تمت مراجعته',
    snapshotHint: 'من كتالوج Hermes. لا يتصل التصفح بمستودعات المصدر مطلقًا.',
    installHint: 'راجع المصدر قبل التثبيت. تسري التغييرات على الجلسات الجديدة.',
    results: (count: number) => `النتائج: ${count.toLocaleString('ar')}`,
    back: 'العودة إلى النتائج'
  },
  connectors: {
    directory: {
      suggested: 'مقترحة',
      allApps: 'كل التطبيقات',
      receiptConnected: apps => `متصلة: ${apps}`,
      receiptNone: 'لم يتم ربط تطبيقات جديدة',
      receiptTimedOut: 'أُغلقت بعد 5 دقائق',
      finishInBrowser: 'أكمل في المتصفح',
      signInFailed: 'تعذّر الاتصال. انقر لإعادة المحاولة'
    }
  },
  sharedMetrics: arCommon.sharedMetrics,
  firstRun: arFirstRun,
  externalOpenFailed: arChrome.externalOpenFailed,
  sessionImport: arConnectors.sessionImport,
  sendDiagnostics: arDiagnostics.sendDiagnostics,
  common: arCommon.common,
  fileMenu: arChrome.fileMenu,
  boot: arBoot.boot,
  notifications: arDiagnostics.notifications,
  remoteDisplayBanner: arBoot.remoteDisplayBanner,
  butterbar: arBoot.butterbar,
  promptNotices: arBoot.promptNotices,
  keepAwakeOffer: arBoot.keepAwakeOffer,
  titlebar: arChrome.titlebar,
  keybinds: arChrome.keybinds,
  language: arSettings.language,
  settings: arSettings.settings,
  skills: arCapabilities.skills,
  agents: arCapabilities.agents,
  commandCenter: arCommandCenter.commandCenter,
  messaging: arCommandCenter.messaging,
  profiles: arCommandCenter.profiles,
  modelAssignment: arSettings.modelAssignment,
  cron: arCommandCenter.cron,
  artifacts: arArtifacts.artifacts,
  artifactCard: arArtifacts.artifactCard,
  artifactPreview: arArtifacts.artifactPreview,
  sidebar: arChrome.sidebar,
  composer: {
    ...arChat.composer,
    redirect: 'إعادة توجيه التشغيل الحالي',
    queueLostNote: 'فُقد الدور أثناء إعادة التشغيل',
    restoreImageDraft: 'استعادة مسودة الصورة',
    queueCancelFailed: 'تعذّر إزالة الدور المنتظر',
    queueLostDiscard: 'تجاهل',
    queueLostDiscardTip: 'أعادت البوابة التشغيل أثناء هذا الدور ولا يمكن إكماله. تجاهله لتستمر الأدوار المنتظرة خلفه.'
  },
  statusStack: arChat.statusStack,
  updates: arBoot.updates,
  install: arBoot.install,
  onboarding: arBoot.onboarding,
  modelPicker: arSettings.modelPicker,
  modelVisibility: arSettings.modelVisibility,
  shell: arChrome.shell,
  rightSidebar: arChrome.rightSidebar,
  preview: arArtifacts.preview,
  interfaceMode: arSettings.interfaceMode,
  zones: arChrome.zones,
  contextMenu: arChrome.contextMenu,
  assistant: arAssistant.assistant,
  prompts: arChat.prompts,
  desktop: arChat.desktop,
  errors: arDiagnostics.errors,
  tips: arChat.tips,
  ui: arCommon.ui,
  freeTier: {
    offer: {
      heading: 'واصل مع Hermes',
      body: 'أنت تستخدم الحصة المجانية. إذا واصلت استخدام Hermes فستبدأ بمواجهة حدود الاستخدام. سجّل الدخول بحساب Nous مجاني للحصول على حصة أكبر.',
      signIn: 'تسجيل الدخول',
      notNow: 'ليس الآن'
    }
  }
} satisfies TranslationOverrides

export const ar = defineLocale(arOverrides)
