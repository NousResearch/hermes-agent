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
  temporaryChat: {
    title: 'محادثة مؤقتة',
    description: 'لا تظهر المحادثات المؤقتة في سجلك، ولا تستخدم الذاكرة أو تحدّثها، وتُحذف عند مغادرة المحادثة.',
    placeholder: 'اسأل أي شيء في محادثة مؤقتة',
    use: 'استخدام محادثة مؤقتة',
    exit: 'الخروج من المحادثة المؤقتة',
    badge: 'مؤقتة',
    badgeHint: 'غير محفوظة. تنتهي عند إغلاقها أو بعد 10 دقائق من الخمول أثناء غيابك.',
    newChat: 'محادثة مؤقتة جديدة',
    unsupportedBackend: 'لا يدعم خادم Hermes هذا المحادثات المؤقتة بعد. حدّثه أو أوقف الوضع المؤقت للمحادثة بشكل عادي.',
    endChat: 'إنهاء المحادثة المؤقتة',
    endTitle: 'إنهاء هذه المحادثة المؤقتة؟',
    endDescription: 'ستُحذف نهائيًا ولن يُحفظ منها شيء.',
    endConfirm: 'إنهاء وحذف',
    returnTo: 'العودة إلى المحادثة المؤقتة',
    running: 'لا يزال يرد…'
  },
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
